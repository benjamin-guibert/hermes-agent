"""Registered and cron sends share a fresh route across text and media."""
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.outbound.send import _send_to_platform
from gateway.platform_registry import PlatformEntry, platform_registry
from plugins.platforms.discord import adapter as discord_platform
from tests.tools.test_discord_send_message_caption import _resp


class API:
    def __init__(self):
        self.gets, self.posts = [], []
        self.states = {
            "4001": {"id": "4001", "type": 11, "parent_id": "4000", "thread_metadata": {"archived": True}},
            "4000": {"id": "4000", "type": 0},
            "5000": {"id": "5000", "type": 0},
        }

    @staticmethod
    def context(response):
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    def session(self, **kwargs):
        session = MagicMock()
        session.get = self.get
        session.post = self.post
        return self.context(session)

    def get(self, url, **kwargs):
        self.gets.append(url)
        state = self.states.get(url.rsplit("/", 1)[-1])
        return self.context(_resp(200 if isinstance(state, dict) else 403, state or {}))

    def post(self, url, **kwargs):
        payload = kwargs.get("json", {})
        files = []
        for field in getattr(kwargs.get("data"), "_fields", []):
            if field[0].get("name") == "payload_json":
                payload = json.loads(field[2])
            if field[0].get("name", "").startswith("files["):
                files.append(field[2].read())
        self.posts.append((url, payload, files))
        response = _resp(200, {"id": str(9000 + len(self.posts)), "attachments": [{"id": "1"}]})
        response.headers = {}
        return self.context(response)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    discord_platform.register(SimpleNamespace(register_platform=lambda **kwargs: platform_registry.register(PlatformEntry(**kwargs))))
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *args, **kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    api = API()
    monkeypatch.setattr("aiohttp.ClientSession", api.session)

    def load(enabled=True, home=None):
        directory = home or tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(directory))
        switch = "" if enabled is None else f"  redirect_archived_thread_messages: {str(enabled).lower()}\n"
        (directory / "config.yaml").write_text(
            "platforms:\n  discord:\n    enabled: true\n    token: fixture-token\ndiscord:\n" + switch)
        config = load_gateway_config()
        return config, config.platforms[Platform.DISCORD]
    return api, load


@pytest.mark.asyncio
async def test_registered_long_text_and_file_one_route(configured, tmp_path, caplog):
    api, load = configured
    _config, pconfig = load(None)
    path = tmp_path / "result.txt"
    path.write_bytes(b"original file")
    text = "original <@6000> " + "x" * 4100
    with caplog.at_level(logging.INFO):
        result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", text, thread_id="4001", media_files=[(str(path), False)])
    assert result["success"]
    assert result["chat_id"] == "4000" and result["thread_id"] is None
    assert sum(url.endswith("/4001") for url in api.gets) == 1
    assert sum(url.endswith("/4000") for url in api.gets) == 1
    assert all(url.endswith("/4000/messages") for url, _, _ in api.posts)
    texts = [payload.get("content", "") for _, payload, _ in api.posts]
    assert sum(text.count("<#4001>") for text in texts) == 1
    assert "<@6000>" in "".join(texts)
    assert any(files == [b"original file"] for _, _, files in api.posts)
    assert len(result["delivered_message_ids"]) == len(api.posts)
    assert len([record for record in caplog.records if "archived thread redirect" in record.getMessage()]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [None, False])
async def test_yaml_opt_out_registered_and_live_construction(configured, switch):
    api, load = configured
    _config, pconfig = load(switch)
    live = platform_registry.get("discord").adapter_factory(pconfig)
    assert live.config.extra.get("redirect_archived_thread_messages", True) is (switch is not False)
    result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", "reply", thread_id="4001")
    assert result["success"]
    target = "4001" if switch is False else "4000"
    assert api.posts[0][0].endswith(f"/{target}/messages")
    if switch is False:
        assert api.gets == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [None, {"id": "4001", "type": 11},
    {"id": "9999", "type": 11, "thread_metadata": {"archived": True}},
    {"id": "4001", "type": 0},
    {"id": "4001", "type": 11, "parent_id": "9999", "thread_metadata": {"archived": True}}])
async def test_registered_refuses_invalid_state_without_post(configured, state):
    api, load = configured
    _, pconfig = load()
    api.states["4001"] = state
    result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", "reply", thread_id="4001")
    assert result.get("success") is not True
    assert api.posts == []


def target(config, pconfig, loop=None, live=None):
    from cron.scheduler_delivery import _TargetDelivery
    from gateway.delivery import DeliveryTransport
    return _TargetDelivery(
        job={"id": "fixture-job", "name": "report"}, platform=Platform.DISCORD, platform_name="discord",
        chat_id="5000", thread_id="4001", transport=DeliveryTransport(live, pconfig, Platform.DISCORD) if live else None,
        pconfig=pconfig, runtime_adapter=live, target_adapters={Platform.DISCORD: live} if live else {},
        config=config, loop=loop, notify_delivery=True, origin={}, origin_target=False, origin_user_id=None,
        is_dm_target=False, mirror_text="Cronjob Response: report", mirror_this_target=False,
        in_channel_surface=False, inchannel_continuable=False, opened_thread_id=None)


@pytest.mark.parametrize("switch", [None, False])
@pytest.mark.parametrize("mode", ["long", "media-only", "caption-only", "get-refused"])
def test_external_cron_real_registered_sender(configured, tmp_path, switch, mode):
    from cron.scheduler_delivery import _deliver_standalone
    from gateway.inbound_reply_context import reply_pointer
    api, load = configured
    config, pconfig = load(switch)
    path = tmp_path / "report.txt"
    path.write_bytes(b"report file")
    state = target(config, pconfig)
    text = "Cronjob Response: report\n<@6000> " + "x" * 4100
    if mode == "media-only":
        text = ""
    if mode == "caption-only":
        text = "Cronjob Response: caption <@6000>"
    if mode == "get-refused":
        api.states["4001"] = None
    errors, delivery_errors = [], []
    delivered = _deliver_standalone(state, text, [(str(path), False)], errors, delivery_errors)
    if switch is not False and mode == "get-refused":
        assert not delivered and errors
        assert api.posts == []
        return
    assert delivered and not errors and not delivery_errors
    destination = "4001" if switch is False else "4000"
    assert all(url.endswith(f"/{destination}/messages") for url, _, _ in api.posts)
    assert sum(url.endswith("/4001") for url in api.gets) == (0 if switch is False else 1)
    if mode != "media-only":
        assert "earlier scheduled cron delivery" in reply_pointer("quoted tail", replied_to=("discord", destination, "9001"))


@pytest.mark.parametrize("switch", [None, False])
@pytest.mark.parametrize("mode", ["long", "media-only", "unknown", "refresh-error"])
def test_live_cron_refreshes_once_and_delivers_text_plus_file(configured, tmp_path, monkeypatch, switch, mode, caplog):
    import threading
    from cron.scheduler_delivery import _deliver_via_live_adapter
    from gateway.inbound_reply_context import reply_pointer
    from tests.gateway.test_discord_free_response import FakeThread, FakeTextChannel, FakeForumChannel
    _api, load = configured
    caplog.set_level(logging.INFO)
    config, pconfig = load(switch)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel)
    parent = FakeTextChannel(channel_id=4000)
    parent.send = AsyncMock(return_value=SimpleNamespace(id=9001, attachments=[object()]))
    fresh = FakeThread(channel_id=4001, parent=parent)
    fresh.archived = None if mode == "unknown" else True
    fresh.send = AsyncMock()
    cached = FakeThread(channel_id=4001, parent=parent)
    cached.archived = False
    cached.send = AsyncMock(return_value=SimpleNamespace(id=9001, attachments=[object()]))
    live = platform_registry.get("discord").adapter_factory(pconfig)
    live._client = SimpleNamespace(get_channel=lambda channel_id: cached if channel_id == 4001 else parent,
                                   fetch_channel=AsyncMock(return_value=fresh))
    if mode == "refresh-error":
        live._client.fetch_channel.side_effect = RuntimeError("state unavailable")
    loop = asyncio.new_event_loop()
    worker = threading.Thread(target=loop.run_forever)
    worker.start()
    try:
        path = tmp_path / "report.txt"
        path.write_bytes(b"report file")
        state = target(config, pconfig, loop, live)
        errors, delivery_errors, unverified = [], [], []
        text = "" if mode == "media-only" else "Cronjob Response: report\n<@6000> " + "x" * 4100
        delivered = _deliver_via_live_adapter(state, text, [(str(path), False)], target_errors=errors, delivery_errors=delivery_errors, unverified_targets=unverified)
        if switch is not False and mode in ("unknown", "refresh-error"):
            assert not delivered
            parent.send.assert_not_awaited()
            cached.send.assert_not_awaited()
            fresh.send.assert_not_awaited()
            live._client.fetch_channel.assert_awaited_once()
            return
        assert delivered and not errors and not delivery_errors
        assert live._client.fetch_channel.await_count == (0 if switch is False else 1)
        fresh.send.assert_not_awaited()
        destination = cached if switch is False else parent
        other = parent if switch is False else cached
        other.send.assert_not_awaited()
        if mode == "media-only":
            assert destination.send.await_count == 1
        else:
            assert destination.send.await_count >= 4
        assert destination.send.await_args.kwargs["files"]
        texts = [call.kwargs.get("content") or "" for call in destination.send.await_args_list]
        assert sum(text.count("<#4001>") for text in texts) == (0 if switch is False else 1)
        assert state.delivery_receipt["chat_id"] == str(destination.id)
        assert state.delivery_receipt["thread_id"] == ("4001" if switch is False else None)
        redirects = [record for record in caplog.records if "archived thread redirect" in record.getMessage()]
        assert len(redirects) == (0 if switch is False else 1)
        assert "earlier scheduled cron delivery" in reply_pointer("quoted tail", replied_to=("discord", str(destination.id), "9001"))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        worker.join(timeout=5)
        loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "caption <@6000>"])
async def test_registered_media_first_keeps_prefix_and_caption(configured, tmp_path, text):
    api, load = configured
    _, pconfig = load()
    path = tmp_path / "image.png"
    path.write_bytes(b"image bytes")
    result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", text, thread_id="4001", media_files=[(str(path), False)])
    assert result["success"]
    assert len(api.posts) == 1
    assert api.posts[0][1]["content"] == "<#4001>\n" + text
    assert api.posts[0][2] == [b"image bytes"]


@pytest.mark.asyncio
async def test_registered_concurrent_and_subsequent_operations_are_isolated(configured, tmp_path):
    from gateway.platforms.event import current_reply_trigger
    api, load = configured
    _, first = load(home=tmp_path / "first-profile")
    _, second = load(False, home=tmp_path / "second-profile")
    token = current_reply_trigger.set((object(), asyncio.current_task(), object(), {}))
    try:
        results = await asyncio.gather(
            _send_to_platform(Platform.DISCORD, first, "5000", "first", thread_id="4001"),
            _send_to_platform(Platform.DISCORD, second, "5000", "second", thread_id="4001"))
        assert all(result["success"] for result in results)
        assert sum(payload.get("content", "").count("<#4001>") for _, payload, _ in api.posts) == 1
        api.posts.clear()
        await _send_to_platform(Platform.DISCORD, first, "5000", "third", thread_id="4001")
        assert api.posts[0][1]["content"] == "<#4001>\nthird"
        assert sum(url.endswith("/4001") for url in api.gets) == 2
        assert current_reply_trigger.get()[0] is not None
    finally:
        current_reply_trigger.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("thread", [False, True])
async def test_registered_active_and_normal_controls(configured, thread):
    api, load = configured
    _, pconfig = load()
    if thread:
        api.states["4001"]["thread_metadata"]["archived"] = False
    result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", "ordinary <@6000>", thread_id="4001" if thread else None)
    assert result["success"]
    assert api.posts[0][0].endswith("/4001/messages" if thread else "/5000/messages")
    assert api.posts[0][1]["content"] == "ordinary <@6000>"


@pytest.mark.asyncio
async def test_registered_allowed_mentions_control_all_chunks_and_file(configured, tmp_path):
    api, load = configured
    _, pconfig = load()
    path = tmp_path / "report.txt"
    path.write_bytes(b"file")
    controls = {"parse": [], "replied_user": False}
    result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", "<@6000> " + "x" * 4100,
                                     thread_id="4001", media_files=[(str(path), False)], mentions=controls)
    assert result["success"]
    assert all(payload.get("allowed_mentions") == controls for _, payload, _ in api.posts)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "timeout"])
async def test_registered_partial_or_unknown_does_not_replay_head(configured, monkeypatch, failure):
    api, load = configured
    _, pconfig = load()
    original_post = api.post
    attempted = []
    def post(url, **kwargs):
        attempted.append(url)
        if len(attempted) == 2:
            if failure == "timeout":
                raise TimeoutError("acknowledgement lost")
            return api.context(_resp(403, {"message": "send refused"}))
        return original_post(url, **kwargs)
    monkeypatch.setattr(api, "post", post)
    result = await _send_to_platform(Platform.DISCORD, pconfig, "5000", "x" * 4100, thread_id="4001")
    assert result.get("success") is not True
    assert result["delivery"] == "partial"
    assert result["delivered_message_ids"]
    assert len(attempted) == 2
    assert all(url.endswith("/4000/messages") for url in attempted)
    assert len(api.posts) == 1
    assert api.posts[0][1]["content"].count("<#4001>") == 1


def test_external_cron_partial_timeout_keeps_actual_receipt_without_replay(configured, monkeypatch):
    from cron.scheduler_delivery import _deliver_standalone
    from gateway.inbound_reply_context import reply_pointer
    api, load = configured
    config, pconfig = load()
    original_post = api.post
    attempted = []
    def post(url, **kwargs):
        attempted.append(url)
        if len(attempted) == 2:
            raise TimeoutError("acknowledgement lost")
        return original_post(url, **kwargs)
    monkeypatch.setattr(api, "post", post)
    state = target(config, pconfig)
    errors, delivery_errors = [], []
    assert _deliver_standalone(state, "Cronjob Response: report\n" + "x" * 4100, [], errors, delivery_errors)
    assert state.standalone_in_flight and delivery_errors
    assert state.delivery_receipt["chat_id"] == "4000"
    assert state.delivery_receipt["delivery"] == "partial"
    assert len(attempted) == 2 and len(api.posts) == 1
    assert all(url.endswith("/4000/messages") for url in attempted)
    assert "earlier scheduled cron delivery" in reply_pointer("tail", replied_to=("discord", "4000", "9001"))
