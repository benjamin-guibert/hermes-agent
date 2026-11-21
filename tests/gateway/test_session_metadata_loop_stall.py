"""A one-entry change never rewrites the whole routing index, and never waits on the event loop.

Field report (macOS gateway under load): every inbound message ran a ``pre_gateway_dispatch``
plugin hook that stamps ONE session's metadata (``set_session_metadata``). That stamp went through
``_update_entry`` -> the full ``_save()``: json.dumps of every entry, a DELETE + executemany over
every ``gateway_routing`` row and an fsynced multi-MB ``sessions.json``, all on the event loop
thread and under the state.db writer mutex (held 8.9 s and 25.1 s). Discord heartbeats were
blocked, the websocket was declared unhealthy and the reply to the owner was lost.

The contract pinned here:

* a change of one existing entry that keeps its key -> session_id mapping persists ONE row
  (no full rewrite), and the row on disk is the same as the full rewrite wrote;
* on an event loop thread that write never waits for a writer mutex another thread holds: the
  loop stays responsive and the row lands once the mutex frees.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.loop_lag import LoopLag
from tests.waiting import async_wait_until, wait_until

ENTRIES = 6000
# The biggest per-entry field on the field host (9.4 MB sessions.json for 6287 entries).
_PIN = {"text": "x" * 1400}


def _source(user_id: str = "owner") -> SessionSource:
    return SessionSource(platform=Platform.DISCORD, chat_id=f"chat-{user_id}", chat_type="dm", user_id=user_id)


def _store(tmp_path, monkeypatch, entries: int = ENTRIES) -> tuple[SessionStore, str]:
    """A store with *entries* routing entries persisted (one real, the rest copies under new keys)."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    key = store.get_or_create_session(_source()).session_key
    with store._lock:
        seed = store._entries[key]
        for i in range(entries - 1):
            other = f"{key}:peer-{i}"
            store._entries[other] = dataclasses.replace(
                seed, session_key=other, session_id=f"{seed.session_id}-{i}", prompt_pin=dict(_PIN),
                metadata={})
        store._save()
    return store, key


def _row(store: SessionStore, key: str) -> dict:
    rows = store.persistence.routing.load_routing_entries(routing_scope=store._routing_scope())
    return json.loads(rows[key])


def _spy_full_rewrite(store: SessionStore, monkeypatch) -> list[str]:
    calls: list[str] = []
    routing = store.persistence.routing
    replace, mirror = routing.replace_routing_entries, store._save_sessions_json
    monkeypatch.setattr(routing, "replace_routing_entries",
                        lambda *a, **k: (calls.append("replace_routing_entries"), replace(*a, **k))[1])
    monkeypatch.setattr(store, "_save_sessions_json",
                        lambda *a, **k: (calls.append("_save_sessions_json"), mirror(*a, **k))[1])
    return calls


def test_metadata_stamp_on_a_big_index_writes_one_row(tmp_path, monkeypatch):
    store, key = _store(tmp_path, monkeypatch)
    calls = _spy_full_rewrite(store, monkeypatch)
    try:
        started = time.monotonic()
        assert store.set_session_metadata(key, "reset_policy_seen", "2026-11-21T10:00:00") is True
        elapsed = time.monotonic() - started

        assert calls == []
        assert elapsed < 2.0, f"a one-entry stamp on {ENTRIES} entries took {elapsed:.2f}s"
        durable = _row(store, key)
        assert durable["metadata"]["reset_policy_seen"] == "2026-11-21T10:00:00"
        assert durable == store._entries[key].to_dict()  # same row the full rewrite would write
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("change", ["suspend", "resume_pending", "yolo"])
def test_flag_changes_of_one_entry_write_one_row(tmp_path, monkeypatch, change):
    store, key = _store(tmp_path, monkeypatch, entries=50)
    calls = _spy_full_rewrite(store, monkeypatch)
    try:
        apply = {"suspend": lambda: store.suspend_session(key),
                 "resume_pending": lambda: store.mark_resume_pending(key),
                 "yolo": lambda: store.set_session_yolo(key, True)}[change]
        assert apply() is True
        assert calls == []
        assert _row(store, key) == store._entries[key].to_dict()

        store.close_all_db_handles()
        restarted = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        restarted._ensure_loaded()
        assert restarted._entries[key].to_dict() == _row(store, key)
        restarted.close_all_db_handles()
    finally:
        store.close_all_db_handles()


def _hold_writer_mutex(store: SessionStore):
    """Hold the state.db writer mutex from another thread through the real write path."""
    holder = raw_holder(store.persistence)
    held, release = threading.Event(), threading.Event()

    def slow_write(conn):
        held.set()
        release.wait(30)
        conn.execute("SELECT 1")

    thread = threading.Thread(target=lambda: holder._execute_write(slow_write), name="asyncio_13")
    thread.start()
    assert held.wait(5), "holder never took the writer mutex"
    return thread, release


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_metadata_stamp_never_blocks_the_loop(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from hermes_cli.plugins import get_plugin_manager

    store, key = _store(tmp_path, monkeypatch, entries=ENTRIES)

    def maybe_reset(event, gateway, session_store, **_):  # the plugin's shape: a sync stamp per message
        session_store.set_session_metadata(key, "reset_policy_seen", "2026-11-21T10:00:00")

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(maybe_reset)
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    event = MessageEvent(text="hi", message_id="m1", source=_source())

    thread, release = _hold_writer_mutex(store)
    lag = LoopLag()
    try:
        await async_wait_until(lambda: lag.ticks >= 2, timeout=5)  # the timer chain runs
        result = await asyncio.wait_for(runner._hm_pre_gateway_dispatch_hook(event, event.source), 30)
        after = lag.ticks
        await async_wait_until(lambda: lag.ticks >= after + 5, timeout=5)  # and keeps running
        worst = lag.stop()
        assert result is event
        assert worst < 1.0, f"the event loop stalled {worst:.2f}s behind another writer"
        assert store.get_session_metadata(key, "reset_policy_seen") == "2026-11-21T10:00:00"
    finally:
        release.set()
        thread.join(30)
        manager._hooks["pre_gateway_dispatch"].remove(maybe_reset)
    try:
        # The row lands once the writer mutex frees.
        assert wait_until(lambda: _row(store, key)["metadata"].get("reset_policy_seen") == "2026-11-21T10:00:00",
                          timeout=30)
    finally:
        store.close_all_db_handles()
