"""Discord inbound that reaches an adapter the runner retired for a reconnect is held, then dispatched
once by the replacement adapter (carry for v0.23.3; the Discord side of the #132829 hand-over).

The incident: a text batch debounced on the old adapter while its WebSocket health check fired. The
runner retired the adapter, the batch flushed ~1.8 s later into admission, which found no Discord
adapter (``AuthorityRefusal('not_found')``), and the message was lost.

Contract:
S1 A text batch pending when the runner retires the adapter is not lost: the replacement runs its turn.
R1 Held events run once, in arrival order, through the replacement, bound to it.
R2 A failed reconnect candidate hands the hold back; a batch flushing after the hand-over forwards.
R3 A non-batched inbound (attachment) dispatched by the retired adapter is held the same way.
R4 A held message runs once: no re-admission by a replay, and a later stop does not replay it.
R5 The hold is capped at 64 (oldest dropped) and drops what is older than 30 min at hand-over.
R6 A gateway stop keeps today's path: the stop replay defers the batch, nothing is held.
R7 One INFO line per hold and one per redispatch, without message text.
R8 ``discord.reconnect_hold_inbound: false`` keeps the base behaviour (the batch is lost).
"""

import asyncio
import datetime as dt
import json
import logging
import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult, cache_image_from_bytes_async
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from hermes_constants import get_hermes_home
from tests.gateway.conftest import _ensure_discord_mock
from tests.waiting import async_wait_until

_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter

USER = "4242"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
BOT = SimpleNamespace(id=999, name="bot", display_name="bot", bot=True)


class Harness:
    """A real GatewayRunner with native admission and one Discord adapter, and the turns it ran."""

    def __init__(self, runner, authority, monkeypatch, extra):
        self.runner, self.authority, self.monkeypatch, self.extra = runner, authority, monkeypatch, extra
        self.turns: list[str] = []  # text of each turn the runner ran
        self.sent: list[tuple[str, str]] = []  # (adapter tag, content)

    def adapter(self, tag):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token", extra=dict(self.extra)))
        async def close():
            return None

        adapter._client = SimpleNamespace(user=BOT, get_channel=lambda cid: None, guilds=[], close=close)
        adapter._ready_event.set()
        adapter._running = True
        adapter._text_batch_delay_seconds = 0.05
        adapter._text_batch_split_delay_seconds = 0.05
        adapter._hold_tag = tag
        adapter.gateway_runner = self.runner

        async def send(chat_id, content, **_):
            self.sent.append((tag, content))
            return SendResult(success=True, message_id="s")

        async def send_typing(*_a, **_kw):
            return None

        self.monkeypatch.setattr(adapter, "send", send)
        self.monkeypatch.setattr(adapter, "send_typing", send_typing)
        self.runner._wire_adapter_handlers(adapter)
        return adapter

    def event(self, adapter, text, message_id, *, chat="5001", media=None):
        source = adapter.build_source(chat_id=chat, chat_type="group", user_id=USER, message_id=message_id)
        channel = SimpleNamespace(id=int(chat))
        raw = SimpleNamespace(id=int(message_id), channel=channel)
        event = MessageEvent(text=text, message_type=MessageType.PHOTO if media else MessageType.TEXT,
                             source=source, raw_message=raw, message_id=message_id,
                             timestamp=dt.datetime.now(dt.UTC))
        if media:
            event.media_urls, event.media_types = [media], ["image/png"]
        return event

    async def retire(self, adapter):
        """The WebSocket health check's fatal: the runner pops, queues and disconnects the adapter."""
        adapter._set_fatal_error("discord_websocket_health_stale", "stale", retryable=True)
        await self.runner._handle_adapter_fatal_error_impl(adapter)

    async def reconnect(self, candidate, *, succeed=True):
        """One reconnect-watcher pass that builds ``candidate`` and connects it (or fails)."""
        async def create(*_a, **_kw):
            return candidate

        async def connect(adapter, *_a, **_kw):
            if not succeed:
                adapter._set_fatal_error("connect_failed", "nope", retryable=True)
            return succeed

        self.monkeypatch.setattr(self.runner, "_create_adapter_off_loop_deps", create)
        self.monkeypatch.setattr(self.runner, "_connect_adapter_with_timeout", connect)
        self.runner._failed_platforms[Platform.DISCORD]["next_retry"] = 0
        await self.runner._reconnect_failed_platform(Platform.DISCORD, time.monotonic())


@pytest.fixture
def make_harness(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", USER)
    made = []

    async def build(extra=None):
        extra = dict(extra or {})
        config = GatewayConfig(sessions_dir=tmp_path / "sessions",
                               platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake-token",
                                                                           extra=extra)})
        runner = GatewayRunner(config)
        h = Harness(runner, None, monkeypatch, extra)
        old = h.adapter("old")
        runner.adapters = {Platform.DISCORD: old}
        runner.delivery_router.adapters = runner.adapters
        h.authority = await initialize_session_authority(runner, profile_id="default", instance_id="dc")
        monkeypatch.setattr(runner, "_ensure_reconnect_watcher_running", lambda: None)

        async def turn(event, *_args, **_kwargs):
            h.turns.append(event.text)
            execution_result.get()["result"] = {"final_response": "ok", "messages": []}
            return "ok"

        monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
        made.append(h)
        return h, old

    yield build
    for h in made:
        h.authority.persistence.close()


async def settle(adapter):
    """Wait until ``adapter`` has flushed its text batches and finished every admission it started."""
    await async_wait_until(lambda: not adapter._pending_text_batch_tasks, what="text batch flushed")
    await async_wait_until(lambda: all(t.done() for t in adapter._background_tasks), timeout=30,
                           what="admission of the flushed batch")


@pytest.mark.asyncio
async def test_batch_pending_at_the_retire_runs_on_the_replacement(make_harness):
    """Contract: S1"""
    h, old = await make_harness()
    old._enqueue_text_event(h.event(old, "wake up", "1001"))
    await h.retire(old)
    await settle(old)
    assert h.turns == []  # nothing runs while no Discord adapter is published
    new = h.adapter("new")
    await h.reconnect(new)
    await async_wait_until(lambda: h.turns, what="the held message's turn")
    await async_wait_until(lambda: h.sent, what="the reply")
    assert h.turns == ["wake up"]
    assert h.sent == [("new", "ok")]


async def hold_batch(h, adapter, text, message_id, **kw):
    """One text batch that flushes on ``adapter`` (retired: it is held)."""
    adapter._enqueue_text_event(h.event(adapter, text, message_id, **kw))
    await settle(adapter)


async def all_turns(h, n):
    await async_wait_until(lambda: len(h.turns) >= n and len(h.sent) >= n, timeout=30, what=f"{n} turn(s)")
    for adapter in h.runner.adapters.values():
        await async_wait_until(lambda a=adapter: all(t.done() for t in a._background_tasks), timeout=30,
                               what="every admitted turn finished")
    return h.turns


@pytest.mark.asyncio
async def test_held_messages_run_once_in_arrival_order_on_the_replacement(make_harness):
    """Contract: R1"""
    h, old = await make_harness()
    await h.retire(old)
    await hold_batch(h, old, "first", "1001")
    await hold_batch(h, old, "second", "1002", chat="5002")
    new = h.adapter("new")
    dispatched, handle = [], new.handle_message

    async def spy(event):
        dispatched.append(event.text)
        await handle(event)

    h.monkeypatch.setattr(new, "handle_message", spy)
    await h.reconnect(new)
    assert sorted(await all_turns(h, 2)) == ["first", "second"]
    assert dispatched == ["first", "second"]
    assert h.sent == [("new", "ok"), ("new", "ok")]


@pytest.mark.asyncio
async def test_failed_candidate_hands_the_hold_back_and_a_late_flush_forwards(make_harness):
    """Contract: R2"""
    h, old = await make_harness()
    await h.retire(old)
    await hold_batch(h, old, "before", "1001")
    await h.reconnect(h.adapter("failed"), succeed=False)
    assert h.turns == []
    await h.reconnect(h.adapter("new"))
    await all_turns(h, 1)
    await hold_batch(h, old, "after the hand-over", "1002")  # flushes on the retired instance
    assert await all_turns(h, 2) == ["before", "after the hand-over"]
    assert h.sent == [("new", "ok"), ("new", "ok")]


@pytest.mark.asyncio
async def test_attachment_dispatched_by_the_retired_adapter_is_held(make_harness):
    """Contract: R3"""
    h, old = await make_harness()
    await h.retire(old)
    image = await cache_image_from_bytes_async(PNG_BYTES, ".png")
    await old.handle_message(h.event(old, "look", "1001", media=image))  # attachments skip the batch
    await settle(old)
    assert h.turns == []
    await h.reconnect(h.adapter("new"))
    assert await all_turns(h, 1) == ["look"]
    assert h.sent == [("new", "ok")]


@pytest.mark.asyncio
async def test_a_held_message_is_not_replayed_or_readmitted_afterwards(make_harness):
    """Contract: R4"""
    h, old = await make_harness()
    assert old._dedup.is_duplicate("1001") is False  # on_message claimed the live message
    old._enqueue_text_event(h.event(old, "wake up", "1001"))
    await h.retire(old)
    await settle(old)
    new = h.adapter("new")
    await h.reconnect(new)
    await all_turns(h, 1)
    channel = SimpleNamespace(id=5001)
    replayed = SimpleNamespace(id=1001, channel=channel, created_at=dt.datetime.now(dt.UTC))
    assert await new._replay_one(replayed) is False  # a resume / recovery replay of the same message
    new.set_message_handler(None)  # a planned stop right after: its window must skip the message
    await new.cancel_background_tasks()  # the teardown: the window is on disk when it returns
    state = json.loads((get_hermes_home() / "gateway" / "discord_stop_replay.json").read_text())
    assert "1001" in state["skip"] and "1001" not in state["deferred"]
    assert await all_turns(h, 1) == ["wake up"]


@pytest.mark.asyncio
async def test_hold_is_capped_and_the_oldest_is_dropped(make_harness, caplog):
    """Contract: R5 (cap)"""
    h, old = await make_harness()
    await h.retire(old)
    with caplog.at_level(logging.WARNING):
        for n in range(65):
            await old.handle_message(h.event(old, f"m{n}", str(2000 + n), chat=str(6000 + n)))
    assert sum("Held-inbound queue full (64)" in r.getMessage() for r in caplog.records) == 1
    await h.reconnect(h.adapter("new"))
    turns = await all_turns(h, 64)
    assert sorted(turns) == sorted(f"m{n}" for n in range(1, 65))


@pytest.mark.asyncio
@pytest.mark.parametrize("age, answered", [(1800 - 5, True), (1800 + 5, False)])
async def test_hold_older_than_30_min_from_the_retire_is_dropped(make_harness, caplog, age, answered):
    """Contract: R5 (age, at the edge)"""
    h, old = await make_harness()
    await h.retire(old)
    h.runner._failed_platforms[Platform.DISCORD]["queued_at"] = time.monotonic() - age
    await hold_batch(h, old, "late", "1001")
    new = h.adapter("new")
    with caplog.at_level(logging.WARNING):
        await h.reconnect(new)
        if answered:
            assert await all_turns(h, 1) == ["late"]
        else:
            await async_wait_until(lambda: any("Dropping 1 held inbound" in r.getMessage() for r in caplog.records),
                                   what="the drop warning")
            assert h.turns == []


@pytest.mark.asyncio
async def test_a_gateway_stop_keeps_the_stop_replay_path(make_harness, caplog):
    """Contract: R6"""
    h, old = await make_harness()
    await old.handle_message(h.event(old, "healthy", "1000"))
    assert await all_turns(h, 1) == ["healthy"]
    assert h.sent == [("old", "ok")]  # one healthy adapter: nothing held
    old._enqueue_text_event(h.event(old, "at the stop", "1001"))
    h.runner._draining = True
    with caplog.at_level(logging.INFO):
        old.set_message_handler(None)  # drain_gateway_runtime withdraws admission, then teardown
        await old.cancel_background_tasks()
    state = json.loads((get_hermes_home() / "gateway" / "discord_stop_replay.json").read_text())
    assert state["deferred"] == {"1001": "5001"}
    assert not any("Holding inbound" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_hold_and_redispatch_log_one_line_each_without_text_or_ids(make_harness, caplog):
    """Contract: R7"""
    h, old = await make_harness()
    await h.retire(old)
    with caplog.at_level(logging.INFO):
        await hold_batch(h, old, "secret words", "1001")
        await hold_batch(h, old, "more", "1002", chat="5002")
        await h.reconnect(h.adapter("new"))
        await all_turns(h, 2)
    lines = [r.getMessage() for r in caplog.records if "held" in r.getMessage().lower() or "Holding" in r.getMessage()]
    assert lines == ["[Discord] Holding inbound for the replacement adapter (12 chars)",
                     "[Discord] Holding inbound for the replacement adapter (4 chars)",
                     "[Discord] Redispatched 2 held inbound message(s) after reconnect"]


@pytest.mark.asyncio
async def test_switch_off_keeps_the_base_behaviour(make_harness, caplog):
    """Contract: R8"""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from plugins.platforms.discord.adapter import _apply_yaml_config
    assert DEFAULT_CONFIG["discord"]["reconnect_hold_inbound"] is True
    assert _apply_yaml_config({}, {"reconnect_hold_inbound": False})["reconnect_hold_inbound"] is False
    h, old = await make_harness({"reconnect_hold_inbound": False})
    old._enqueue_text_event(h.event(old, "wake up", "1001"))
    await h.retire(old)
    with caplog.at_level(logging.INFO):
        await settle(old)
        await h.reconnect(h.adapter("new"))
    assert any("Error handling message: not_found" in r.getMessage() for r in caplog.records)
    assert not any("Holding inbound" in r.getMessage() for r in caplog.records)
    assert h.turns == []
