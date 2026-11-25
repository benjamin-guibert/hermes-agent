"""Guarded native media uses real Discord delivery methods and fake SDK/HTTP."""
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.delivery_ledger import ledger_enabled as real_ledger_enabled

from tests.gateway.test_discord_free_response import adapter as adapter_fixture, FakeThread, FakeTextChannel, make_message
from tests.gateway.test_discord_audio_delivery import _ogg_opus
from tests.gateway.test_queued_final_ledger import _fresh_db, _rows

adapter = adapter_fixture


@pytest.fixture
def media_transport(adapter, monkeypatch):
    from plugins.platforms.discord import adapter as platform
    monkeypatch.setattr(platform.discord.http, "Route", lambda method, path, **values: SimpleNamespace(
        method=method, url=path.format(**values)))
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(return_value=SimpleNamespace(id=9001, attachments=[object()]))
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    thread.archive_timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    thread.send = AsyncMock(return_value=SimpleNamespace(id=9002, attachments=[object()]))
    adapter._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else thread)
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    adapter._client.http = SimpleNamespace(request=AsyncMock(return_value={"id": "9003"}))
    return adapter, parent, thread


@pytest.mark.asyncio
async def test_invalid_voice_has_no_orphan_prefix(media_transport, tmp_path):
    adapter, parent, _thread = media_transport
    result = await adapter.send_voice("4001", str(tmp_path / "missing.ogg"), metadata={"thread_id": "4001"})
    assert not result.success
    parent.send.assert_not_awaited()
    adapter._client.http.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_uncertain_native_voice_does_not_replay_as_file(media_transport, tmp_path):
    adapter, parent, _thread = media_transport
    path = tmp_path / "voice.ogg"
    path.write_bytes(_ogg_opus(2.0))
    adapter._client.http.request.side_effect = TimeoutError("acknowledgement lost")
    result = await adapter.send_voice("4001", str(path), metadata={"thread_id": "4001"})
    assert not result.success
    assert result.raw_response["chat_id"] == "4000"
    assert result.raw_response["message_ids"] == ["9001"]
    assert result.raw_response["ambiguous"]
    assert parent.send.await_count == 1
    assert "file" not in parent.send.await_args.kwargs
    adapter._client.http.request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "definitive", "file"])
async def test_voice_receipts_preserve_mode_and_prefix_ids(media_transport, tmp_path, mode):
    adapter, parent, thread = media_transport
    path = tmp_path / "voice.ogg"
    path.write_bytes(_ogg_opus(2.0))
    if mode == "definitive":
        class Refused(Exception):
            status = 400
        adapter._client.http.request.side_effect = Refused("native flag refused")
    result = await adapter.send_voice("4001", str(path), caption="caption", metadata={"thread_id": "4001"}, is_voice=mode != "file")
    assert result.success and result.raw_response["chat_id"] == "4000"
    if mode == "success":
        request = adapter._client.http.request.await_args
        assert json.loads(request.kwargs["form"][0]["value"])["flags"] == 8192
        assert request.kwargs["form"][1]["value"] == path.read_bytes()
        assert result.raw_response["message_ids"] == ["9001", "9003"]
        assert result.message_id == "9003"
    elif mode == "file":
        adapter._client.http.request.assert_not_awaited()
        assert parent.send.await_args.kwargs["files"]
    else:
        assert parent.send.await_count == 2
        assert parent.send.await_args.kwargs["file"]
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["definitive", "unknown"])
async def test_image_batch_partial_has_ids_and_no_head_replay(media_transport, tmp_path, failure):
    adapter, parent, thread = media_transport
    images = []
    for index in range(11):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(b"image")
        images.append((path.as_uri(), "caption"))
    class Refused(Exception):
        status = 403
    parent.send.side_effect = [SimpleNamespace(id=9001, attachments=[object()]),
                              Refused("refused") if failure == "definitive" else TimeoutError("acknowledgement lost")]
    result = await adapter.send_multiple_images("4001", images, metadata={"thread_id": "4001"})
    assert not result.success
    assert result.raw_response["chat_id"] == "4000"
    assert result.raw_response["message_ids"] == ["9001"]
    assert result.raw_response["ambiguous"] is (failure == "unknown")
    assert parent.send.await_count == 2
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("author", ["human", "bot"])
@pytest.mark.parametrize("media", ["mixed", "images", "tts"])
@pytest.mark.parametrize("state", ["archived", "active", "stale", "opt-out"])
async def test_real_final_native_media_pipeline(media_transport, monkeypatch, tmp_path, author, media, state, caplog):
    import logging
    from pathlib import Path
    from gateway.platforms.event import MessageType
    adapter, parent, thread = media_transport
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    message = make_message(channel=thread, content="request")
    message.author.bot = author == "bot"
    message.webhook_id = None
    message.created_at = thread.archive_timestamp + timedelta(seconds=1)
    assert await adapter._handle_message(message)
    event = adapter.handle_message.await_args.args[0]
    if state == "active":
        thread.archived = False
    if state == "stale":
        cached = FakeThread(channel_id=4001, parent=parent)
        cached.archived = False
        cached.send = AsyncMock(side_effect=AssertionError("stale transport used"))
        adapter._resolve_channel = AsyncMock(side_effect=lambda target: parent if str(target) == "4000" else cached)
    if state == "opt-out":
        adapter.config.extra["redirect_archived_thread_messages"] = False
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    voice = tmp_path / "voice.ogg"
    voice.write_bytes(_ogg_opus(2.0))
    response = f"MEDIA:{image}"
    if media == "mixed":
        response = f"reply\nMEDIA:{image}\n[[audio_as_voice]]\nMEDIA:{voice}"
    if media == "tts":
        event.message_type = MessageType.VOICE
        adapter._auto_tts_default = True
        response = "spoken reply"
        monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
        def generate(text, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(_ogg_opus(2.0))
            return json.dumps({"success": True, "file_path": output_path})
        monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", generate)
    adapter._message_handler = AsyncMock(return_value=response)
    adapter._start_typing_refresh = lambda *args: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *args: None
    await adapter._process_message_background(event, "native-session")
    allowed = author == "human" or state in ("active", "opt-out")
    destination, other = (thread, parent) if allowed else (parent, thread)
    other.send.assert_not_awaited()
    assert destination.send.await_count >= 1
    if media != "images":
        adapter._client.http.request.assert_awaited_once()
        payload = json.loads(adapter._client.http.request.await_args.kwargs["form"][0]["value"])
        assert payload["flags"] == 8192
        assert adapter._client.http.request.await_args.args[0].url == f"/channels/{destination.id}/messages"
    captions = [call.kwargs.get("content") or "" for call in destination.send.await_args_list]
    assert sum(text.count("<#4001>") for text in captions) == (0 if allowed else 1)
    assert len([r for r in caplog.records if "archived thread redirect" in r.getMessage()]) == (0 if allowed else 1)
    assert adapter._client.fetch_channel.await_count == (0 if state == "opt-out" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_url_media_preserves_target_and_unknown_does_not_fallback(media_transport, monkeypatch, public_dns, failure):
    from unittest.mock import MagicMock
    adapter, parent, thread = media_transport
    response = SimpleNamespace(status=200, headers={"Content-Type": "image/png"}, read=AsyncMock(return_value=b"image bytes"))
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=False)
    session = SimpleNamespace(get=lambda *args, **kwargs: response_context)
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session_context)
    if failure:
        parent.send.side_effect = TimeoutError("acknowledgement lost")
    result = await adapter.send_image("4001", "https://example.invalid/image.png", caption="caption", metadata={"thread_id": "4001"})
    assert result.success is not failure
    assert result.raw_response["chat_id"] == "4000"
    parent.send.assert_awaited_once()
    assert parent.send.await_args.kwargs["file"]
    assert parent.send.await_args.kwargs["content"] == "<#4001>\ncaption"
    if failure:
        assert result.raw_response["ambiguous"]
    else:
        assert result.raw_response["message_ids"] == ["9001"]
    thread.send.assert_not_awaited()



@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["human", "bot", "opt-out"])
async def test_explicit_yaml_ledger_off_is_not_a_redirect_dependency(media_transport, monkeypatch, tmp_path, kind):
    from gateway import delivery_ledger as ledger
    from gateway.platforms.base import _thread_metadata_for_event, _mark_notify_metadata
    adapter, parent, thread = media_transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("gateway:\n  delivery_ledger: false\n")
    monkeypatch.setattr(ledger, "ledger_enabled", real_ledger_enabled)
    assert not ledger.ledger_enabled()
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    message = make_message(channel=thread, content="request")
    message.author.bot = kind != "human"
    message.webhook_id = None
    message.created_at = thread.archive_timestamp + timedelta(seconds=1)
    assert await adapter._handle_message(message)
    event = adapter.handle_message.await_args.args[0]
    if kind == "opt-out":
        adapter.config.extra["redirect_archived_thread_messages"] = False
    results = []
    await adapter._send_final_text(event, "ledger-off-session", "reply", _mark_notify_metadata(_thread_metadata_for_event(event)),
                                  False, 0, results.append)
    assert results[0].success
    destination, other = (parent, thread) if kind == "bot" else (thread, parent)
    destination.send.assert_awaited_once()
    other.send.assert_not_awaited()
    assert _rows() == []


@pytest.mark.asyncio
async def test_native_reply_file_unknown_does_not_fallback_to_text(media_transport):
    adapter, parent, thread = media_transport
    adapter.MAX_SPLIT_MESSAGES = 2
    parent.send.side_effect = TimeoutError("acknowledgement lost")
    result = await adapter._send_with_retry("4001", "x" * 6100, metadata={"thread_id": "4001"})
    assert not result.success and result.raw_response["ambiguous"]
    assert result.raw_response["chat_id"] == "4000"
    parent.send.assert_awaited_once()
    assert parent.send.await_args.kwargs["files"]
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_voice_has_no_prefix_or_upload(media_transport, tmp_path):
    adapter, parent, thread = media_transport
    path = tmp_path / "huge.ogg"
    with path.open("wb") as file:
        file.seek(adapter._discord_upload_limit_bytes(thread))
        file.write(b"x")
    result = await adapter.send_voice("4001", str(path), metadata={"thread_id": "4001"})
    assert not result.success
    parent.send.assert_not_awaited()
    adapter._client.http.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_image_batch_definitive_rejection_keeps_safe_fallback_ids(media_transport, tmp_path):
    adapter, parent, _thread = media_transport
    class Refused(Exception):
        status = 400
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        path.write_bytes(b"image")
    parent.send.side_effect = [Refused("batch rejected"), SimpleNamespace(id=9002, attachments=[object()]),
                              SimpleNamespace(id=9003, attachments=[object()])]
    result = await adapter.send_multiple_images("4001", [(path.as_uri(), "caption") for path in paths], metadata={"thread_id": "4001"})
    assert result.success
    assert result.raw_response["message_ids"] == ["9002", "9003"]
    assert result.raw_response["chat_id"] == "4000"
    assert parent.send.await_count == 3
    texts = [call.kwargs.get("content") or "" for call in parent.send.await_args_list[1:]]
    assert sum(text.count("<#4001>") for text in texts) == 1


def test_real_cron_native_voice_keeps_mode_and_partial_receipt(media_transport, tmp_path):
    import asyncio
    import threading
    from gateway.config import GatewayConfig, Platform
    from gateway.inbound_reply_context import reply_pointer
    from cron.scheduler_delivery import _deliver_via_live_adapter
    from tests.gateway.test_discord_outbound_operation import target
    adapter, _parent, thread = media_transport
    config = GatewayConfig(platforms={Platform.DISCORD: adapter.config})
    loop = asyncio.new_event_loop()
    worker = threading.Thread(target=loop.run_forever)
    worker.start()
    try:
        path = tmp_path / "report.ogg"
        path.write_bytes(_ogg_opus(2.0))
        state = target(config, adapter.config, loop, adapter)
        errors, media_errors, unverified = [], [], []
        assert _deliver_via_live_adapter(state, "", [(str(path), True)], target_errors=errors, delivery_errors=media_errors, unverified_targets=unverified)
        assert not errors and not media_errors
        assert state.delivery_receipt["message_ids"] == ["9001", "9003"]
        assert state.delivery_receipt["chat_id"] == "4000"
        request = adapter._client.http.request.await_args
        assert request.args[0].url == "/channels/4000/messages"
        assert json.loads(request.kwargs["form"][0]["value"])["flags"] == 8192
        assert "earlier scheduled cron delivery" in reply_pointer("voice", replied_to=("discord", "4000", "9003"))
        thread.send.assert_not_awaited()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        worker.join(timeout=5)
        loop.close()


@pytest.mark.asyncio
async def test_rejected_image_batch_then_next_batch_has_one_link(media_transport, tmp_path):
    adapter, parent, _thread = media_transport
    class Refused(Exception):
        status = 400
    paths = [tmp_path / f"image-{index}.png" for index in range(11)]
    for path in paths:
        path.write_bytes(b"image")
    parent.send.side_effect = [Refused("batch rejected")] + [
        SimpleNamespace(id=9100 + index, attachments=[object()]) for index in range(11)]
    result = await adapter.send_multiple_images("4001", [(path.as_uri(), "caption") for path in paths],
                                                metadata={"thread_id": "4001"})
    assert result.success
    assert result.raw_response["message_ids"] == [str(9100 + index) for index in range(11)]
    assert result.raw_response["chat_id"] == "4000"
    texts = [call.kwargs.get("content") or "" for call in parent.send.await_args_list[1:]]
    assert sum(text.count("<#4001>") for text in texts) == 1
