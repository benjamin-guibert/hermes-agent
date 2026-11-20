"""Every final-delivery claim ends with the adapter lifecycle that holds it.

The idle gate of an agent host reads the ``active-work`` verb and treats every listed unit as busy,
a ``wedged`` one too, so a claim that is never released would block every later reload of that
gateway. The session drain claims a reply for the waiting adapter event before the delivery-ledger
bracket runs (``gateway.final_delivery_claims.claim_owed_reply``); these paths end that lifecycle
without the bracket, or with no reply at all.
"""

import asyncio

import pytest

from gateway.final_delivery_claims import final_delivery_units
from gateway.platforms.base import MessageEvent
from tests.gateway.final_delivery_test_helpers import REPLY, authority_gateway, settle


async def _lifecycle(tmp_path, monkeypatch, reply, *, before_send=None, during_turn=None):
    """One adapter lifecycle over a real session turn that answers *reply*. ``before_send(adapter,
    runner)`` runs where the adapter prepares the reply, before the ledger bracket; ``during_turn``
    runs inside the turn. Returns the runner, the adapter and the processing task."""
    seen = {}

    async def run_turn(event):
        if during_turn is not None:
            return await during_turn(event, seen)
        return reply

    runner, adapter, authority, source = await authority_gateway(tmp_path, monkeypatch, run_turn)
    adapter.ack.set()  # replies and error notices go out at once
    seen.update(runner=runner, adapter=adapter)
    if before_send is not None:
        extract = adapter._extract_response_content

        async def prepare(*args, **kwargs):
            await before_send(adapter, runner, seen)
            return await extract(*args, **kwargs)
        adapter._extract_response_content = prepare
    processing = asyncio.create_task(
        adapter._process_message_background(MessageEvent(text="hi", source=source, message_id="m1"), "s"))
    adapter._background_tasks.add(processing)
    processing.add_done_callback(adapter._background_tasks.discard)
    return runner, adapter, authority, processing, seen


def _assert_no_claim_left(runner):
    assert final_delivery_units(runner) == [], "a claim outlived its adapter lifecycle"
    assert runner._awaitable_work_count() == 0 and runner._active_work_count() == 0, runner._describe_active_work()


@pytest.mark.asyncio
async def test_cancelled_after_the_turn_handed_over_its_reply(tmp_path, monkeypatch):
    reached = asyncio.Event()

    async def before_send(_adapter, runner, seen):
        seen["claims"] = len(final_delivery_units(runner))
        reached.set()
        await asyncio.Event().wait()  # the adapter task is cancelled here

    runner, adapter, authority, processing, seen = await _lifecycle(tmp_path, monkeypatch, REPLY, before_send=before_send)
    await asyncio.wait_for(reached.wait(), 10)
    processing.cancel()
    await settle(adapter, authority)

    assert seen["claims"] == 1, "no claim was live before the ledger bracket"
    assert adapter.sent == []
    _assert_no_claim_left(runner)


@pytest.mark.asyncio
async def test_raises_after_the_turn_handed_over_its_reply(tmp_path, monkeypatch):
    async def before_send(_adapter, runner, seen):
        seen["claims"] = len(final_delivery_units(runner))
        raise RuntimeError("reply preparation failed")

    runner, adapter, authority, _processing, seen = await _lifecycle(
        tmp_path, monkeypatch, REPLY, before_send=before_send)
    await settle(adapter, authority)

    assert seen["claims"] == 1, "no claim was live before the ledger bracket"
    assert REPLY not in adapter.sent  # only the turn-error notice went out
    _assert_no_claim_left(runner)


@pytest.mark.asyncio
async def test_cancelled_while_the_turn_runs(tmp_path, monkeypatch):
    turn_started, finish_turn = asyncio.Event(), asyncio.Event()

    async def during_turn(_event, _seen):
        turn_started.set()
        await finish_turn.wait()
        return REPLY

    runner, adapter, authority, processing, _seen = await _lifecycle(
        tmp_path, monkeypatch, REPLY, during_turn=during_turn)
    await asyncio.wait_for(turn_started.wait(), 10)
    processing.cancel()  # the waiting adapter goes away; the turn runs on in its session
    await asyncio.gather(processing, return_exceptions=True)
    finish_turn.set()
    await settle(adapter, authority)

    _assert_no_claim_left(runner)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [None, ""], ids=["empty", "silent_no_reply"])
async def test_a_turn_that_sends_no_reply(tmp_path, monkeypatch, reply):
    # ``None``: nothing to send (streamed, queued). ``""``: what the turn hands the adapter for an
    # intentional NO_REPLY / [SILENT] (gateway.run_turn_persistence suppresses the marker).
    runner, adapter, authority, _processing, _seen = await _lifecycle(tmp_path, monkeypatch, reply)
    await settle(adapter, authority)

    assert adapter.sent == []
    _assert_no_claim_left(runner)


@pytest.mark.asyncio
async def test_an_interrupted_turn_whose_stale_reply_is_suppressed(tmp_path, monkeypatch):
    turns = []

    async def during_turn(_event, seen):
        turns.append(1)
        if len(turns) == 1:  # a follow-up arrived and interrupted this turn while it ran
            adapter = seen["adapter"]
            adapter._active_sessions["s"].set()
            adapter._pending_messages["s"] = MessageEvent(
                text="and also this", source=_event.source, message_id="m2")
            return REPLY  # the stale answer: handed over (claimed), then suppressed by the adapter
        return None

    runner, adapter, authority, _processing, _seen = await _lifecycle(
        tmp_path, monkeypatch, REPLY, during_turn=during_turn)
    await settle(adapter, authority)

    assert len(turns) == 2 and adapter.sent == []
    _assert_no_claim_left(runner)
