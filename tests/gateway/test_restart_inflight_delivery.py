"""A restart or a stop must not cut a ledgered final reply before its ledger mark.

Carry of NousResearch/hermes-agent#92589. A turn frees its work slot before its final reply is
sent, so the restart wait (``gateway.restart_when_idle.await_idle``) and the ``stop()`` drain saw an
idle gateway, the teardown cut the send after the platform had the message but before
``mark_delivered()``, and the next boot swept the ``attempting`` row and sent the reply again as
"♻️ Recovered reply".
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger
from gateway.platforms.base import MessageEvent, SendResult
from gateway.restart_when_idle import await_idle, request_service_restart
from tests.gateway.final_delivery_test_helpers import (
    REPLY, authority_gateway, boot_sweep_of_the_next_process, held_after_a_while, ledger_rows, make_event,
    make_gateway, process, quiet_stop_teardown, settle)
from tests.gateway.restart_test_helpers import RestartTestAdapter


def _sampler(runner, adapter, marked, samples, *, auto_ack=True):
    """Sample the restart wait's count every loop tick until *marked*; with ``auto_ack``, release
    the platform ACK once the send has been on the platform for a while."""
    async def sample():
        while not marked.is_set():
            samples.append(runner._awaitable_work_count())
            if auto_ack and len(samples) > 50 and adapter.on_platform.is_set():
                adapter.ack.set()
            await asyncio.sleep(0)
    return sample


def _watch_mark_delivered(monkeypatch):
    marked, real_mark = threading.Event(), delivery_ledger.mark_delivered

    def mark_delivered(obligation_id):
        real_mark(obligation_id)
        marked.set()
    monkeypatch.setattr(delivery_ledger, "mark_delivered", mark_delivered)
    return marked


def _chat_event(source, message_id="m1"):
    return MessageEvent(text="hi", source=source, message_id=message_id)


# --- Step 1: the incident ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_restart_during_the_final_send_waits_for_its_ledger_mark_and_nothing_is_resent(monkeypatch):
    runner, adapter = make_gateway()
    seen_at_stop = {}

    async def stop(**_kwargs):
        # What the real stop() does to an adapter mid-send: its teardown cancels the processing task.
        seen_at_stop["rows"] = ledger_rows()
        processing.cancel()

    runner.stop = stop
    processing = process(runner, adapter, make_event())
    await asyncio.wait_for(adapter.on_platform.wait(), 5)  # the user already has the reply

    runner.request_restart(detached=False, via_service=True)  # SIGUSR1: the drain-first reload
    if await held_after_a_while(runner._restart_task):
        adapter.ack.set()
        await asyncio.wait_for(runner._restart_task, 5)
    await asyncio.gather(processing, return_exceptions=True)

    assert seen_at_stop["rows"] == [("delivered", REPLY)], "stop() ran before the reply's ledger mark"
    assert boot_sweep_of_the_next_process(monkeypatch) == [], "the next boot sends the reply again"
    assert adapter.sent == [REPLY]


# --- R1: no gap from the turn to the ledger mark ----------------------------------------------

@pytest.mark.asyncio
async def test_the_reply_is_work_without_a_gap_from_the_running_turn_to_its_ledger_mark(tmp_path, monkeypatch):
    marked, samples = _watch_mark_delivered(monkeypatch), []

    async def run_turn(_event):
        runner._sampler = asyncio.create_task(_sampler(runner, adapter, marked, samples)())  # from the turn on
        return REPLY

    runner, adapter, authority, source = await authority_gateway(tmp_path, monkeypatch, run_turn)
    await asyncio.wait_for(adapter._process_message_background(_chat_event(source), "s"), 10)
    await asyncio.wait_for(runner._sampler, 5)
    await settle(adapter, authority)

    assert adapter.sent == [REPLY]
    assert samples and min(samples) >= 1, "the restart wait saw an idle gateway before the ledger mark"
    assert runner._awaitable_work_count() == 0


# --- R2: the claim ends on every path of the send --------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["delivered", "send_error", "platform_exception", "cancelled"])
async def test_the_claim_ends_on_every_path_of_the_send(outcome):
    runner, adapter = make_gateway()
    if outcome == "send_error":
        adapter.result = SendResult(success=False, error="Forbidden: bot was blocked by the user")
    elif outcome == "platform_exception":
        adapter.raises = RuntimeError("platform exploded")
    processing = process(runner, adapter, make_event())
    await asyncio.wait_for(adapter.on_platform.wait(), 5)
    assert runner._awaitable_work_count() == 1

    if outcome == "cancelled":
        processing.cancel()
    adapter.ack.set()
    await asyncio.gather(processing, return_exceptions=True)

    assert runner._awaitable_work_count() == 0 and runner._active_work_count() == 0
    assert await asyncio.wait_for(await_idle(runner), 5) is True
    state = {"delivered": "delivered", "send_error": "failed"}.get(outcome, "attempting")
    assert [row[0] for row in ledger_rows()] == [state]


# --- R3: every planned stop that waits for idle ----------------------------------------------

def _sigusr1(runner):
    from gateway.run import _start_gateway_make_restart_signal_handler
    _start_gateway_make_restart_signal_handler(runner)()  # systemd reload / hermes gateway restart


PLANNED_STOPS = {
    "sigusr1": _sigusr1,
    # /restart under systemd or launchd (``is_gateway_supervisor_process``): exit 75, relaunched.
    "restart_supervised": lambda runner: runner.request_restart(detached=False, via_service=True),
    # /restart with no service manager: a detached helper relaunches after the wait.
    "restart_unsupervised": lambda runner: runner.request_restart(detached=True, via_service=False),
    "update_pause": request_service_restart,  # the updater's pause-for-update drain
}


@pytest.mark.asyncio
@pytest.mark.parametrize("planned_stop", sorted(PLANNED_STOPS))
async def test_every_planned_stop_waits_for_the_reply_ledger_mark(planned_stop):
    runner, adapter = make_gateway()
    order = []
    runner.stop = AsyncMock(side_effect=lambda **_: order.append(("stop", ledger_rows())))
    runner._launch_detached_restart_command = AsyncMock(side_effect=lambda: order.append(("helper", ledger_rows())))
    processing = process(runner, adapter, make_event())
    await asyncio.wait_for(adapter.on_platform.wait(), 5)

    PLANNED_STOPS[planned_stop](runner)
    assert await held_after_a_while(runner._restart_task)
    assert order == []
    adapter.ack.set()
    await asyncio.wait_for(runner._restart_task, 5)
    await processing

    assert order and all(rows == [("delivered", REPLY)] for _, rows in order)
    assert [name for name, _ in order] == (["helper", "stop"] if planned_stop == "restart_unsupervised" else ["stop"])


# --- The SIGTERM / systemctl stop path (hard restart, OS reboot) ------------------------------

@pytest.mark.asyncio
async def test_a_sigterm_stop_waits_for_the_reply_ledger_mark_and_nothing_is_resent(monkeypatch):
    quiet_stop_teardown(monkeypatch)
    runner, adapter = make_gateway()
    processing = process(runner, adapter, make_event())
    await asyncio.wait_for(adapter.on_platform.wait(), 5)

    runner._stop_requested_by_signal = True  # what the SIGTERM handler sets before it calls stop()
    stopping = asyncio.create_task(runner.stop())
    if await held_after_a_while(stopping):
        adapter.ack.set()
    await asyncio.wait_for(stopping, 10)
    processing.cancel()  # the process exits now: whatever did not finish is lost
    await asyncio.gather(processing, return_exceptions=True)

    assert ledger_rows() == [("delivered", REPLY)], "stop() cut the send before its ledger mark"
    assert boot_sweep_of_the_next_process(monkeypatch) == [], "the next boot sends the reply again"


@pytest.mark.asyncio
async def test_a_hung_send_holds_a_sigterm_stop_no_longer_than_the_cron_floor(monkeypatch):
    quiet_stop_teardown(monkeypatch)
    runner, adapter = make_gateway()
    runner._cron_drain_timeout = 0.3  # the floor stop() may wait on (30 s by default, under the leash)
    processing = process(runner, adapter, make_event())  # the ACK never comes
    await asyncio.wait_for(adapter.on_platform.wait(), 5)

    await asyncio.wait_for(runner.stop(), 10)  # the stop goes on: it is not held past the floor
    processing.cancel()
    await asyncio.gather(processing, return_exceptions=True)
    assert runner._active_work_count() == 0


# --- R4: a hung send holds the waits at most agent.gateway_timeout ----------------------------

@pytest.mark.asyncio
async def test_a_hung_send_holds_the_restart_until_the_gateway_timeout_and_no_longer(monkeypatch):
    from gateway import final_delivery_claims
    monkeypatch.delenv("HERMES_AGENT_TIMEOUT", raising=False)  # 1800 s
    clock = [1000.0]
    monkeypatch.setattr(final_delivery_claims, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    runner, adapter = make_gateway()
    processing = process(runner, adapter, make_event())  # the ACK never comes
    await asyncio.wait_for(adapter.on_platform.wait(), 5)

    clock[0] += 1799.9
    assert runner._awaitable_work_count() == 1 and runner._drain_work_counts()[5] == 1
    clock[0] += 0.1
    assert runner._awaitable_work_count() == 0 and runner._drain_work_counts()[5] == 0
    assert await asyncio.wait_for(await_idle(runner), 5) is True  # the stop goes on
    assert runner._active_work_count() == 1  # still listed, as a wedged turn is
    assert [unit["wedged"] for unit in runner._describe_active_work() if unit["kind"] == "final_delivery"] == [True]
    processing.cancel()
    await asyncio.gather(processing, return_exceptions=True)
    assert runner._active_work_count() == 0


# --- R5: the active-work listing names the send ----------------------------------------------

@pytest.mark.asyncio
async def test_the_active_work_verb_lists_the_send_while_it_runs():
    from gateway.run_active_work_verb import active_work_verb
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit
    runner, adapter = make_gateway()
    verb = active_work_verb(runner, asyncio.get_running_loop())
    event = make_event()
    processing = process(runner, adapter, event)
    await asyncio.wait_for(adapter.on_platform.wait(), 5)

    during = await asyncio.to_thread(verb, {})
    adapter.ack.set()
    await processing
    after = await asyncio.to_thread(verb, {})

    units = [unit for unit in during["active_work"] if unit["kind"] == "final_delivery"]
    assert during["active_agents"] == 1
    assert [(unit["session"], unit["platform"]) for unit in units] == [
        (runner._session_key_for_source(event.source), "telegram")]
    assert after["active_agents"] == 0 and after["active_work"] == []
    assert describe_active_work_unit(units[0]).startswith(f"final reply to {units[0]['session']} being sent")


# --- R6: no restart pending ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_without_a_restart_the_reply_goes_out_once_and_its_row_is_delivered():
    runner, adapter = make_gateway(RestartTestAdapter())  # an immediate ACK
    await process(runner, adapter, make_event())

    assert adapter.sent == [REPLY]
    assert ledger_rows() == [("delivered", REPLY)]
    assert runner._active_work_count() == 0 and runner._restart_task_started is False


@pytest.mark.asyncio
async def test_a_reply_the_ledger_does_not_keep_never_holds_a_restart():
    runner, adapter = make_gateway()
    processing = process(runner, adapter, make_event(text="/status"))  # a command reply: never ledgered
    await asyncio.wait_for(adapter.on_platform.wait(), 5)
    assert runner._awaitable_work_count() == 0 and ledger_rows() == []
    adapter.ack.set()
    await processing


# --- The busy follow-up path (dispatch_shared_busy) -------------------------------------------

@pytest.mark.asyncio
async def test_a_busy_follow_up_reply_holds_the_restart_through_its_delivery_copy(tmp_path, monkeypatch):
    from gateway.final_delivery_claims import final_delivery_units
    from gateway.session_ingress import dispatch_shared_busy
    marked, samples = _watch_mark_delivered(monkeypatch), []
    ack_released = asyncio.Event()

    async def run_turn(_event):
        runner._sampler = asyncio.create_task(_sampler(runner, adapter, ack_released, samples, auto_ack=False)())
        return REPLY

    runner, adapter, authority, source = await authority_gateway(tmp_path, monkeypatch, run_turn)
    await asyncio.wait_for(dispatch_shared_busy(adapter, _chat_event(source, "f1"), "s"), 10)
    await asyncio.wait_for(adapter.on_platform.wait(), 10)

    assert len(final_delivery_units(runner)) == 1, "one claim, on the delivery copy, while it sends"
    restart_wait = asyncio.create_task(await_idle(runner))
    assert await held_after_a_while(restart_wait), "the restart wait did not hold for the follow-up's send"
    ack_released.set()  # the sampler stops; the ACK is released below
    adapter.ack.set()
    assert await asyncio.wait_for(restart_wait, 5) is True
    await settle(adapter, authority)

    assert marked.is_set() and adapter.sent == [REPLY] and ledger_rows() == [("delivered", REPLY)]
    assert samples and min(samples) >= 1, "the restart wait saw an idle gateway before the send"
    assert final_delivery_units(runner) == []


# --- Upstream #92589 cases, adapted -----------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_replies_are_counted_one_by_one():
    from tests.waiting import async_wait_until
    runner, adapter = make_gateway()
    adapter.acks = {"1": asyncio.Event(), "2": asyncio.Event()}
    first_task = process(runner, adapter, make_event("in-1", chat_id="1"))
    second_task = process(runner, adapter, make_event("in-2", chat_id="2"))
    await async_wait_until(lambda: len(adapter.sent) == 2, what="both replies on the platform")
    assert runner._awaitable_work_count() == 2

    adapter.acks["1"].set()
    await first_task
    assert runner._awaitable_work_count() == 1  # the released claim never takes the other one along
    adapter.acks["2"].set()
    await second_task
    assert runner._awaitable_work_count() == 0
