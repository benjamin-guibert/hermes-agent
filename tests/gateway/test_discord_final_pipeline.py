"""Real final ledger, retry and document delivery with fake Discord transport."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_free_response import adapter as adapter_fixture, FakeThread, FakeTextChannel, make_message
from tests.gateway.test_queued_final_ledger import _fresh_db, _rows, _holder
import json

adapter = adapter_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["human", "bot", "rearchive", "replacement", "stale-active", "refresh-error", "retry"])
async def test_final_text_and_document(adapter, monkeypatch, tmp_path, kind):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(return_value=SimpleNamespace(id=9002, attachments=[object()]))
    if kind == "retry":
        class DefinitivelyUnsent(PermissionError):
            status = 403
        parent.send.side_effect = [DefinitivelyUnsent("send refused"),
                                  SimpleNamespace(id=9002, attachments=[object()]),
                                  SimpleNamespace(id=9003, attachments=[object()])]
    thread = FakeThread(channel_id=4001, parent=parent)
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    thread.archived, thread.archive_timestamp = True, epoch
    thread.send = AsyncMock(return_value=SimpleNamespace(id=9001, attachments=[object()]))
    message = make_message(channel=thread, content="request")
    message.author.bot = kind in ("bot", "retry")
    message.webhook_id = None
    message.created_at = epoch + timedelta(seconds=1)
    assert await adapter._handle_message(message)
    event = adapter.handle_message.await_args.args[0]
    if kind == "rearchive":
        thread.archive_timestamp = epoch + timedelta(seconds=2)
    cached = thread
    if kind == "stale-active":
        cached = FakeThread(channel_id=4001, parent=parent)
        cached.archived = False
        thread.archive_timestamp = epoch + timedelta(seconds=2)
    live = adapter
    if kind == "replacement":
        from plugins.platforms.discord.adapter import DiscordAdapter
        live = DiscordAdapter(adapter.config)
        live._client = SimpleNamespace(user=adapter._client.user)
        adapter.gateway_runner = SimpleNamespace(_delivery_adapter_for=lambda source: live)
    live._resolve_channel = AsyncMock(return_value=cached)
    live._client.fetch_channel = AsyncMock(return_value=thread)
    if kind == "refresh-error":
        live._client.fetch_channel.side_effect = RuntimeError("refresh refused")
    file = tmp_path / "result.txt"
    file.write_text("result")
    adapter._message_handler = AsyncMock(return_value=f"reply\nMEDIA:{file}")
    adapter._start_typing_refresh = lambda *a: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *a: None
    await adapter._process_message_background(event, "pipeline-session")
    if kind == "refresh-error":
        thread.send.assert_not_awaited()
        parent.send.assert_not_awaited()
        assert _rows()[0]["state"] != "delivered"
    else:
        destination = thread if kind in ("human", "replacement") else parent
        other = parent if destination is thread else thread
        assert destination.send.await_count == (3 if kind == "retry" else 2)
        other.send.assert_not_awaited()
        assert destination.send.await_args_list[-1].kwargs["files"]
        if destination is parent:
            assert "<#4001>" in destination.send.await_args_list[-2].kwargs["content"]
            assert not destination.send.await_args_list[-1].kwargs["content"]
        assert _rows()[0]["state"] == "delivered"
        assert _rows()[0]["session_key"] == "pipeline-session"
        from gateway.delivery_ledger import compute_obligation_id
        assert _rows()[0]["obligation_id"] == compute_obligation_id("pipeline-session", str(event.message_id), "reply")
        with _holder()._read_ctx() as connection:
            receipt = json.loads(connection.execute("SELECT delivery_receipt FROM delivery_obligations").fetchone()[0])
        assert receipt["chat_id"] == str(destination.id)
        assert receipt["message_ids"] == ["9001" if destination is thread else "9002"]
    assert live._client.fetch_channel.await_count == 1


@pytest.mark.asyncio
async def test_reply_file_cleanup_uses_actual_parent(adapter, tmp_path):
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(return_value=SimpleNamespace(id=9002, attachments=[]))
    deleted = AsyncMock()
    parent.get_partial_message = lambda message_id: SimpleNamespace(delete=deleted)
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.send = AsyncMock()
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    adapter._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else thread)
    path = tmp_path / "reply.txt"
    path.write_text("reply")
    result = await adapter.send_reply_document("4001", str(path), metadata={"thread_id": "4001"})
    assert not result.success and result.message_id is None
    assert result.raw_response["chat_id"] == "4000"
    assert result.raw_response["deleted_message_ids"] == ["9002"]
    assert result.raw_response["message_ids"] == []
    deleted.assert_awaited_once()
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_parent_text_does_not_replay_visible_head(adapter):
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(side_effect=[SimpleNamespace(id=9002), PermissionError("send refused"), SimpleNamespace(id=9003)])
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.send = AsyncMock()
    adapter._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else thread)
    result = await adapter._send_with_retry("4001", "first " * 600, metadata={"thread_id": "4001"}, base_delay=0)
    assert result.success
    assert result.raw_response["message_ids"] == ["9002", "9003"]
    assert result.raw_response["chat_id"] == "4000"
    assert parent.send.await_count == 3
    assert "<#4001>" in parent.send.await_args_list[0].kwargs["content"]
    assert "<#4001>" not in parent.send.await_args_list[-1].kwargs["content"]
    assert parent.send.await_args_list[1].kwargs["content"] == parent.send.await_args_list[2].kwargs["content"]
    thread.send.assert_not_awaited()


def test_receipt_column_migration_preserves_existing_obligation():
    import sqlite3
    from persistence.sqlite.gateway._delivery_ledger import _ensure_obligations_table
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE delivery_obligations (obligation_id TEXT PRIMARY KEY, adapter_profile TEXT)")
        connection.execute("INSERT INTO delivery_obligations VALUES ('old', NULL)")
        _ensure_obligations_table(connection)
        _ensure_obligations_table(connection)
        assert connection.execute("SELECT obligation_id, delivery_receipt FROM delivery_obligations").fetchall() == [("old", None)]


@pytest.mark.asyncio
async def test_uncertain_resumed_tail_drops_old_safe_flag(adapter):
    from gateway.platforms.base import SendResult
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(side_effect=TimeoutError("remote acknowledgement lost"))
    adapter._resolve_channel = AsyncMock(return_value=parent)
    prior = SendResult(success=False, raw_response={
        "chat_id": "4000", "thread_id": None, "message_ids": ["9002"],
        "partial_overflow": True, "remaining_chunks": ["tail"], "resume_safe": True})
    resumed = await adapter._resume_partial_send("4001", prior)
    assert not resumed.success
    assert resumed.raw_response["message_ids"] == ["9002"]
    assert resumed.raw_response["resume_safe"] is False
    assert await adapter._resume_partial_send("4001", resumed) is None
    parent.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("progress_case", ["safe", "uncertain", "malformed"])
async def test_failed_parent_progress_survives_new_adapter(adapter, monkeypatch, progress_case):
    from gateway import delivery_ledger as ledger
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.run import GatewayRunner
    from plugins.platforms.discord.adapter import DiscordAdapter

    class Refused(Exception):
        status = 403

    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(side_effect=[SimpleNamespace(id=9002), Refused("forbidden")])
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.send = AsyncMock()
    adapter._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else thread)
    source = SessionSource(platform=Platform.DISCORD, chat_id="4001", thread_id="4001", user_id="6000")
    event = MessageEvent(text="request", source=source, message_id="7001")
    result, _ = await adapter.send_final_ledgered(event, "recovery-session", "first " * 600,
                                               {"thread_id": "4001"}, reply_to="7001")
    assert not result.success
    row = _rows()[0]
    assert row["state"] == "failed" and row["chat_id"] == "4001"
    receipt = json.loads(ledger.get_delivery_receipt(row["obligation_id"]))
    assert receipt["chat_id"] == "4000" and receipt["message_ids"] == ["9002"]
    tail = list(receipt["remaining_chunks"])
    if progress_case == "malformed":
        ledger.mark_failed(row["obligation_id"], "invalid progress", '{"resume_safe":true}')
    if progress_case == "uncertain":
        parent.send.side_effect = TimeoutError("acknowledgement lost")
    else:
        parent.send.side_effect = None
        parent.send.return_value = SimpleNamespace(id=9003)
    parent.send.reset_mock()
    # A new transport and owner have no event or human trigger from the first process.
    fresh = DiscordAdapter(adapter.config)
    fresh._client = SimpleNamespace(user=adapter._client.user)
    fresh._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else thread)
    monkeypatch.setattr(ledger, "_owner_alive", lambda *args: False)
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (999999, 303))
    claimed = ledger.sweep_recoverable()
    assert len(claimed) == 1 and claimed[0]["obligation_id"] == row["obligation_id"]
    runner = SimpleNamespace(_obligation_adapter=AsyncMock(return_value=fresh),
                             _arm_flood_timers_for_waiting_rows=AsyncMock())
    count = await GatewayRunner._redeliver_claimed_obligations(runner, claimed)
    assert len(_rows()) == 1 and _rows()[0]["obligation_id"] == row["obligation_id"]
    thread.send.assert_not_awaited()
    if progress_case == "safe":
        assert count == 1 and _rows()[0]["state"] == "delivered"
        assert [call.kwargs["content"] for call in parent.send.await_args_list] == tail
        saved = json.loads(ledger.get_delivery_receipt(row["obligation_id"]))
        assert saved["message_ids"] == ["9002", "9003"]
    elif progress_case == "uncertain":
        assert count == 0 and parent.send.await_count == 1
        saved = json.loads(ledger.get_delivery_receipt(row["obligation_id"]))
        assert saved["resume_safe"] is False and saved["message_ids"] == ["9002"]
        parent.send.reset_mock()
        assert await GatewayRunner._redeliver_claimed_obligations(runner, claimed) == 0
        parent.send.assert_not_awaited()
    else:
        assert count == 0
        parent.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_reply_file_loss_cleans_parent_then_falls_back_with_link(adapter):
    from gateway.config import Platform
    from gateway.platforms.base import _mark_notify_metadata
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    parent = FakeTextChannel(channel_id=4000)
    deleted = AsyncMock()
    parent.get_partial_message = lambda message_id: SimpleNamespace(delete=deleted)
    sent = []

    async def post(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(id=9000 + len(sent), attachments=[])

    parent.send = AsyncMock(side_effect=post)
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.send = AsyncMock()
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    adapter._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else thread)
    adapter.MAX_SPLIT_MESSAGES = 2
    source = SessionSource(platform=Platform.DISCORD, chat_id="4001", thread_id="4001", user_id="6000")
    event = MessageEvent(text="request", source=source, message_id="7001")
    results = []
    await adapter._send_final_text(event, "reply-file-session", "line " * 1500,
                                  _mark_notify_metadata({"thread_id": "4001"}), False, 0, results.append)
    assert results[0].success
    assert sent[0]["files"]
    deleted.assert_awaited_once()
    fallback = [post["content"] for post in sent[1:]]
    assert fallback and "<#4001>" in fallback[0]
    assert sum("<#4001>" in text for text in fallback) == 1
    assert results[0].raw_response["chat_id"] == "4000"
    assert _rows()[0]["state"] == "delivered"
    thread.send.assert_not_awaited()
