"""Live ingress provenance reaches the ordinary final text delivery boundary."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import _mark_notify_metadata, _thread_metadata_for_event
from tests.gateway.test_discord_free_response import adapter as adapter_fixture, FakeThread, FakeTextChannel, make_message

adapter = adapter_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("case,allowed", [
    ("human", True), ("before", False), ("equal", False), ("bot", False),
    ("webhook", False), ("recovery", False), ("internal", False),
    ("mismatch", False), ("rearchive", False), ("missing", False), ("merged", False), ("background-task", False), ("forged", False),
])
async def test_ingress_to_final_reply(adapter, monkeypatch, case, allowed):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(return_value=SimpleNamespace(id=9002))
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    thread.archive_timestamp = epoch
    thread.send = AsyncMock(return_value=SimpleNamespace(id=9001))
    message = make_message(channel=thread, content="input")
    message.author.bot = case == "bot"
    message.webhook_id = 8000 if case == "webhook" else None
    message.created_at = epoch + timedelta(seconds=1)
    if case in ("before", "equal"):
        message.created_at = epoch - timedelta(seconds=1) if case == "before" else epoch
    adapter._resolve_channel = AsyncMock(return_value=thread)
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    assert await adapter._handle_message(message, recovered=case == "recovery")
    event = adapter.handle_message.await_args.args[0]
    if case == "merged":
        event.absorb_turn_input(event)
    if case == "internal":
        event.internal = True
    if case == "mismatch":
        event.reply_anchor_override = "9999"
    if case == "rearchive":
        thread.archive_timestamp = epoch + timedelta(seconds=2)
    if case == "missing":
        thread.archive_timestamp = None
    if case == "forged":
        event.reply_trigger = {"human": True, "thread_id": "4001"}
    metadata = _mark_notify_metadata(_thread_metadata_for_event(event))

    async def ledger_send(event, session_key, text, metadata, **kwargs):
        send = adapter.send(event.source.chat_id, text, reply_to=kwargs["reply_to"], metadata=metadata)
        result = await asyncio.create_task(send) if case == "background-task" else await send
        return result, adapter

    adapter.send_final_ledgered = ledger_send
    results = []
    await adapter._send_final_text(event, "test", "reply", metadata, False, 0, results.append)
    assert results[0].success
    assert thread.send.await_count == int(allowed)
    assert parent.send.await_count == int(not allowed)
    # A later bot/background send cannot reuse a prior final reply's metadata.
    thread.send.reset_mock()
    parent.send.reset_mock()
    await adapter.send(event.source.chat_id, "background", reply_to=str(message.id), metadata=metadata)
    thread.send.assert_not_awaited()
    parent.send.assert_awaited_once()
