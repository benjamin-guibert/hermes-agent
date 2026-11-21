"""The stop-replay window never fsyncs on the event loop once per refused message.

Field report (macOS gateway, load 95): a SIGUSR1 restart drained for 49.5 s and the shutdown
watchdog then forced the exit at 58 s. Its last dump showed the event loop in
``StopReplayWindow.save`` -> ``atomic_json_write`` (fsync), called from ``defer_refused_inbound``
<- ``handle_message`` for EACH message the stop refused. Here every write takes 2 s (a loaded
disk) and 10 messages arrive during the stop: the loop must stay responsive, the writes coalesce,
the whole stop stays far inside launchd's 60 s ``ExitTimeOut``, and the final window is on disk
when the adapter teardown returns. ``discord.stop_replay_write_off_loop: false`` keeps the
synchronous write per message.
"""
import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

from tests.gateway.fixtures.loop_lag import LoopLag
from tests.gateway.test_discord_stop_replay import (
    FakeChannel, dispatched_ids, live_event, make_adapter, make_message, restart, state_path,
)
from tests.waiting import async_wait_until

WRITE_SECONDS = 2.0
REFUSED = 10


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    return tmp_path


@pytest.fixture
def slow_writes(monkeypatch):
    """``atomic_json_write`` that takes WRITE_SECONDS per call (fsync on a loaded host)."""
    import utils

    real, calls = utils.atomic_json_write, []

    def slow(path, payload, **kwargs):
        calls.append(threading.current_thread() is threading.main_thread())
        threading.Event().wait(WRITE_SECONDS)  # health: allow HX013 -- a slow fsync IS the stall under test
        return real(path, payload, **kwargs)

    monkeypatch.setattr(utils, "atomic_json_write", slow)
    return calls


async def _stop_with_refused_messages(stopping, channel):
    stopping.set_message_handler(AsyncMock(return_value=None))
    messages = [make_message(900 + i, channel) for i in range(REFUSED)]
    stopping.set_message_handler(None)  # drain_gateway_runtime withdraws admission
    for message in messages:  # inbound during the drain: each one is refused and deferred
        await stopping.handle_message(live_event(stopping, message))
    await stopping.cancel_background_tasks()  # the adapter teardown
    return [m.id for m in messages]


@pytest.mark.asyncio
async def test_refused_messages_during_a_stop_never_fsync_on_the_loop(_env, slow_writes):
    channel = FakeChannel()
    stopping = make_adapter(channel)
    lag = LoopLag()
    await async_wait_until(lambda: lag.ticks >= 2, timeout=5)
    started = lag.loop.time()

    refused = await _stop_with_refused_messages(stopping, channel)

    elapsed = lag.loop.time() - started
    worst = lag.stop()
    assert worst < 1.0, f"the event loop stalled {worst:.2f}s on the stop-replay writes"
    assert not any(slow_writes), "a stop-replay write ran on the event loop thread"
    assert len(slow_writes) <= 3, f"{len(slow_writes)} writes for {REFUSED} refused messages"
    assert elapsed < 3 * WRITE_SECONDS + 2, f"the stop took {elapsed:.1f}s"
    # Durable when the teardown returns: the next process replays every refused message once.
    state = json.loads(state_path(_env).read_text())
    assert sorted(int(mid) for mid in state["deferred"]) == refused
    restarted = await restart(channel)
    assert dispatched_ids(restarted) == refused


@pytest.mark.asyncio
async def test_the_off_switch_keeps_one_synchronous_write_per_refused_message(_env, slow_writes, monkeypatch):
    monkeypatch.setattr("tests.gateway.test_discord_stop_replay_write_off_loop.WRITE_SECONDS", 0.0)
    channel = FakeChannel()
    stopping = make_adapter(channel, extra={"stop_replay_write_off_loop": False})

    refused = await _stop_with_refused_messages(stopping, channel)

    assert slow_writes == [True] * (1 + REFUSED)  # the withdraw, then one per message, on the loop
    assert sorted(int(mid) for mid in json.loads(state_path(_env).read_text())["deferred"]) == refused


TEARDOWN_BOUND = 2.0  # seconds, the per-adapter teardown bound (HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT)


@pytest.mark.asyncio
@pytest.mark.parametrize("write_seconds, inside", [(0.8, True), (1.2, False)], ids=["just-under", "just-over"])
async def test_the_final_window_against_the_teardown_bound(_env, slow_writes, monkeypatch, write_seconds, inside):
    """Two writes at most follow the withdrawal (the running one, then the coalesced rest), measured
    from the teardown start: inside the bound the teardown returns with the window on disk; over it
    the teardown moves on (no hang) and the worker still lands the full window."""
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    monkeypatch.setattr("tests.gateway.test_discord_stop_replay_write_off_loop.WRITE_SECONDS", write_seconds)
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", str(TEARDOWN_BOUND))
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    stopping.set_message_handler(None)
    messages = [make_message(900 + i, channel) for i in range(REFUSED)]
    for message in messages:
        await stopping.handle_message(live_event(stopping, message))
    runner = object.__new__(GatewayRunner)
    loop = asyncio.get_running_loop()

    started = loop.time()
    await runner._bounded_adapter_teardown(stopping, Platform.DISCORD)
    elapsed = loop.time() - started

    def deferred():
        return sorted(int(mid) for mid in json.loads(state_path(_env).read_text())["deferred"])

    if inside:
        assert deferred() == [m.id for m in messages]  # on disk when the teardown returns
    else:
        assert elapsed < 2 * TEARDOWN_BOUND + 1, f"the teardown hung {elapsed:.1f}s"
        assert len(deferred()) < REFUSED  # the bound tripped: the coalesced write was still running
        await async_wait_until(lambda: state_path(_env).exists() and len(deferred()) == REFUSED, timeout=10)
    assert len(slow_writes) == 2 and not any(slow_writes)
