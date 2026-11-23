"""Interrupted real ledger finalization keeps guarded Discord text progress."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.run import GatewayRunner
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.gateway.test_discord_free_response import adapter as adapter_fixture, FakeThread, FakeTextChannel
from tests.gateway.test_queued_final_ledger import _fresh_db, _rows

adapter = adapter_fixture


class Killed(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", ["after-ack", "after-last-ack", "during-post", "recovered-after-ack", "recovered-during-post", "recovered-refused"])
async def test_crash_boundary_keeps_only_confirmed_progress(adapter, monkeypatch, crash):
    parent = FakeTextChannel(channel_id=4000)
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.send = AsyncMock()
    acknowledged = []
    phase = ["initial"]

    async def post(**kwargs):
        if crash == "recovered-refused" and phase[0] == "recovery":
            class Refused(Exception):
                status = 403
            raise Refused("send refused")
        if ((crash == "during-post" and len(acknowledged) == 1)
                or (crash == "recovered-during-post" and phase[0] == "recovery")):
            raise Killed()
        acknowledged.append(kwargs["content"])
        return SimpleNamespace(id=9000 + len(acknowledged))

    parent.send = AsyncMock(side_effect=post)
    def wire(target):
        target._resolve_channel = AsyncMock(side_effect=lambda channel_id: parent if str(channel_id) == "4000" else thread)
        target._client = SimpleNamespace(user=SimpleNamespace(id=999))
        return target
    wire(adapter)
    original_checkpoint = ledger.checkpoint_delivery_progress

    def checkpoint(obligation_id, progress):
        original_checkpoint(obligation_id, progress)
        if (crash == "after-last-ack" and phase[0] == "initial" and progress["resume_safe"]
                and progress["message_ids"] and not progress["remaining_chunks"]):
            raise Killed()
        if progress["resume_safe"] and progress["remaining_chunks"]:
            if phase[0] == "initial" and crash not in ("during-post", "after-last-ack") and len(progress["message_ids"]) == 1:
                raise Killed()
            if phase[0] == "recovery" and crash == "recovered-after-ack" and len(progress["message_ids"]) == 2:
                raise Killed()

    monkeypatch.setattr(ledger, "checkpoint_delivery_progress", checkpoint)
    source = SessionSource(platform=Platform.DISCORD, chat_id="4001", thread_id="4001", user_id="6000")
    event = MessageEvent(text="request", source=source, message_id="7001")
    with pytest.raises(Killed):
        await adapter.send_final_ledgered(event, "crash-session", "x" * 5100, {"thread_id": "4001"}, reply_to="7001")
    first = _rows()[0]
    assert first["state"] == "attempting"
    obligation = first["obligation_id"]
    assert first["chat_id"] == "4001"
    saved = json.loads(ledger.get_delivery_receipt(obligation))
    assert saved["message_ids"] == [str(9001 + index) for index in range(len(acknowledged))]
    assert saved["chat_id"] == "4000"
    assert saved["progress_contract"] == "discord.archived_redirect.v1"
    assert saved["resume_safe"] is (crash != "during-post")
    planned_posts = [*acknowledged, *saved["remaining_chunks"]]
    monkeypatch.setattr(ledger, "_owner_alive", lambda *args: False)
    owner = [303]
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (999999, owner[0]))

    async def recover():
        fresh = wire(DiscordAdapter(adapter.config))
        runner = SimpleNamespace(_obligation_adapter=AsyncMock(return_value=fresh),
                                 _arm_flood_timers_for_waiting_rows=AsyncMock())
        return await GatewayRunner._redeliver_claimed_obligations(runner, ledger.sweep_recoverable())

    phase[0] = "recovery"
    if crash.startswith("recovered"):
        if crash == "recovered-refused":
            assert await recover() == 0
        else:
            with pytest.raises(Killed):
                await recover()
        saved = json.loads(ledger.get_delivery_receipt(obligation))
        expected_ids = ["9001", "9002"] if crash.endswith("after-ack") else ["9001"]
        assert saved["message_ids"] == expected_ids
        assert saved["resume_safe"] is (crash.endswith("after-ack") or crash == "recovered-refused")
        phase[0] = "second-recovery"
        owner[0] += 1
    count = await recover()
    if crash.endswith("during-post"):
        assert count == 0
        assert _rows()[0]["state"] == "failed"
    else:
        assert count == 1
        assert _rows()[0]["state"] == "delivered"
        assert acknowledged == planned_posts
        assert sum(text.count("<#4001>") for text in acknowledged) == 1
        assert sum(text.count("x") for text in acknowledged) == 5100
        final = json.loads(ledger.get_delivery_receipt(obligation))
        assert len(final["message_ids"]) == len(acknowledged)
    assert len(_rows()) == 1 and _rows()[0]["obligation_id"] == obligation
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_discord_recovery_ignores_discord_receipt(monkeypatch):
    from gateway.platforms.base import SendResult
    ledger.record_obligation(obligation_id="legacy", session_key="legacy-session", platform="telegram",
                             chat_id="5000", thread_id=None, content="ordinary reply")
    ledger.mark_failed("legacy", "retry", json.dumps({"progress_contract": "discord.archived_redirect.v1"}))
    monkeypatch.setattr(ledger, "_owner_alive", lambda *args: False)
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (999999, 303))
    unrelated = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True, message_id="9001")))
    runner = SimpleNamespace(_obligation_adapter=AsyncMock(return_value=unrelated),
                             _arm_flood_timers_for_waiting_rows=AsyncMock())
    assert await GatewayRunner._redeliver_claimed_obligations(runner, ledger.sweep_recoverable()) == 1
    unrelated.send.assert_awaited_once()
    assert unrelated.send.await_args.kwargs["chat_id"] == "5000"
    assert unrelated.send.await_args.kwargs["content"].endswith("ordinary reply")
    assert _rows()[0]["state"] == "delivered"


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", ['{}', '{', '{"progress_contract":"discord.archived_redirect.v2"}',
                                     '{"progress_contract":"discord.archived_redirect.v1","resume_safe":true}'])
async def test_invalid_progress_is_never_a_safe_redirect(adapter, receipt):
    adapter._resolve_channel = AsyncMock()
    result = await adapter.resume_delivery_receipt(receipt)
    assert not result.success
    adapter._resolve_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkpoint_failure_prevents_post_and_whole_fallback(adapter, monkeypatch):
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock()
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.send = AsyncMock()
    adapter._resolve_channel = AsyncMock(return_value=thread)
    source = SessionSource(platform=Platform.DISCORD, chat_id="4001", thread_id="4001", user_id="6000")
    event = MessageEvent(text="request", source=source, message_id="7001")
    def refuse_checkpoint(*args):
        raise RuntimeError("storage refused")
    monkeypatch.setattr(ledger, "checkpoint_delivery_progress", refuse_checkpoint)
    result, _ = await adapter.send_final_ledgered(event, "storage-session", "reply", {"thread_id": "4001"}, reply_to="7001")
    assert not result.success and result.raw_response["progress_storage_failed"]
    parent.send.assert_not_awaited()
    thread.send.assert_not_awaited()
    assert _rows()[0]["state"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("old_receipt", [False, True])
async def test_ordinary_discord_failure_keeps_normal_recovery(adapter, monkeypatch, old_receipt):
    class Refused(Exception):
        status = 403
    channel = FakeTextChannel(channel_id=5000)
    channel.send = AsyncMock(side_effect=Refused("send refused"))
    adapter._resolve_channel = AsyncMock(return_value=channel)
    source = SessionSource(platform=Platform.DISCORD, chat_id="5000", user_id="6000")
    event = MessageEvent(text="request", source=source, message_id="7001")
    result, _ = await adapter.send_final_ledgered(event, "ordinary-session", "ordinary reply", {}, reply_to="7001")
    assert not result.success
    row = _rows()[0]
    assert row["state"] == "failed"
    assert ledger.get_delivery_receipt(row["obligation_id"]) is None
    if old_receipt:
        ledger.mark_failed(row["obligation_id"], "send refused", json.dumps({"chat_id": "5000", "message_ids": []}))
    channel.send.side_effect = None
    channel.send.return_value = SimpleNamespace(id=9001)
    channel.send.reset_mock()
    fresh = DiscordAdapter(adapter.config)
    fresh._client = SimpleNamespace(user=SimpleNamespace(id=999))
    fresh._resolve_channel = AsyncMock(return_value=channel)
    monkeypatch.setattr(ledger, "_owner_alive", lambda *args: False)
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (999999, 303))
    runner = SimpleNamespace(_obligation_adapter=AsyncMock(return_value=fresh),
                             _arm_flood_timers_for_waiting_rows=AsyncMock())
    assert await GatewayRunner._redeliver_claimed_obligations(runner, ledger.sweep_recoverable()) == 1
    channel.send.assert_awaited_once()
    assert channel.send.await_args.kwargs["content"].endswith("ordinary reply")
    assert _rows()[0]["obligation_id"] == row["obligation_id"]
    assert _rows()[0]["state"] == "delivered"
