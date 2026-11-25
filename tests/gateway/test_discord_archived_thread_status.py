"""A status notice never reopens an archived Discord thread.

Discord unarchives an unlocked thread when a message is sent to it, so a late non-conversational
send (the background review summary, tool progress, "Working — N min", restart and update notices)
reopened a thread the owner had just closed. ``discord.skip_status_in_archived_threads`` (default
on) skips those sends; conversational sends still reopen the thread, because a reply means the work
continues.
"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.conftest import _ensure_discord_mock

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config

PARENT_ID = "4000"
THREAD_ID = "4001"
STATUS = {"thread_id": THREAD_ID, "non_conversational": True}
SKIP_LINE = f"[Discord] Skipped status notice to archived thread {THREAD_ID}"


class FakeThread:
    """Stands in for ``discord.Thread``; ``archived`` is only set when the test knows it."""

    def __init__(self, channel_id, **attrs):
        self.id = int(channel_id)
        self.send = AsyncMock(return_value=SimpleNamespace(id=9001))
        for name, value in attrs.items():
            setattr(self, name, value)


@pytest.fixture
def make_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _make(channel, **extra):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
        adapter._client = SimpleNamespace(
            get_channel=MagicMock(return_value=channel), fetch_channel=AsyncMock(return_value=channel),
        )
        return adapter

    return _make


def _skip_lines(caplog):
    return [r for r in caplog.records if r.getMessage() == SKIP_LINE]


@pytest.mark.asyncio
async def test_status_notice_to_archived_thread_is_a_delivered_no_op(make_adapter, caplog):
    thread = FakeThread(THREAD_ID, archived=True)
    adapter = make_adapter(thread, missed_message_backfill={"enabled": True})

    with caplog.at_level(logging.INFO):
        result = await adapter.send(PARENT_ID, "💾 Self-improvement review: memory updated", metadata=STATUS)

    thread.send.assert_not_awaited()
    assert result.success is True
    assert result.message_id is None
    assert result.retryable is False
    assert result.error is None
    skip = _skip_lines(caplog)
    assert len(skip) == 1 and skip[0].levelno == logging.INFO
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_skipped_status_notice_writes_no_recovery_ledger_row(make_adapter):
    thread = FakeThread(THREAD_ID, archived=True)
    adapter = make_adapter(thread, missed_message_backfill={"enabled": True})

    await adapter.send(PARENT_ID, "Working — 3 min", reply_to="777", metadata=STATUS)

    thread.send.assert_not_awaited()
    row = adapter._with_discord_recovery_db(
        lambda conn: conn.execute("SELECT status FROM discord_messages WHERE message_id='777'").fetchone()
    )
    assert row is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel, metadata, extra",
    [
        pytest.param(FakeThread(THREAD_ID, archived=False), STATUS, {}, id="active-thread"),
        pytest.param(
            SimpleNamespace(id=int(THREAD_ID), archived=True, send=AsyncMock(return_value=SimpleNamespace(id=9001))),
            STATUS, {}, id="not-a-thread",
        ),
        pytest.param(FakeThread(THREAD_ID, archived=True), {"thread_id": THREAD_ID}, {}, id="conversational"),
        pytest.param(
            FakeThread(THREAD_ID, archived=True), STATUS, {"skip_status_in_archived_threads": False},
            id="switch-off",
        ),
        pytest.param(
            FakeThread(THREAD_ID, archived=True), STATUS, {"skip_status_in_archived_threads": "false"},
            id="switch-off-string",
        ),
        pytest.param(FakeThread(THREAD_ID), STATUS, {}, id="archived-unknown"),
    ],
)
async def test_send_goes_out_unless_status_to_known_archived_thread(make_adapter, caplog, channel, metadata, extra):
    adapter = make_adapter(channel, **extra)

    with caplog.at_level(logging.INFO):
        result = await adapter.send(PARENT_ID, "hello", metadata=metadata)

    requires_guard = isinstance(channel, FakeThread) and getattr(channel, "archived", None) is not False
    if requires_guard:
        channel.send.assert_not_awaited()
        assert result.success is False
        assert "delivery refused" in result.error
    else:
        channel.send.assert_awaited_once()
        assert result.success is True and result.message_id == "9001"
    assert not _skip_lines(caplog)


@pytest.mark.parametrize(
    "discord_cfg, expected",
    [({"skip_status_in_archived_threads": False}, {"skip_status_in_archived_threads": False}), ({}, None)],
)
def test_yaml_key_seeds_adapter_extra(discord_cfg, expected):
    seeded = _apply_yaml_config({}, discord_cfg) or {}
    assert ({k: seeded[k] for k in ("skip_status_in_archived_threads",) if k in seeded} or None) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("skip,redirect,state", [(True, True, "archived"), (False, True, "archived"),
    (True, False, "archived"), (False, False, "archived"), (True, True, "unknown"), (True, True, "error")])
async def test_fresh_archive_status_is_noop_before_parent_route(make_adapter, skip, redirect, state):
    parent = SimpleNamespace(id=4000, send=AsyncMock(return_value=SimpleNamespace(id=9002)))
    cached = FakeThread(THREAD_ID, archived=False, parent=parent, parent_id=4000)
    fresh = FakeThread(THREAD_ID, archived=None if state == "unknown" else True, parent=parent, parent_id=4000)
    adapter = make_adapter(cached, skip_status_in_archived_threads=skip,
                           redirect_archived_thread_messages=redirect)
    adapter._client.fetch_channel = AsyncMock(return_value=fresh)
    if state == "error":
        adapter._client.fetch_channel.side_effect = RuntimeError("state unavailable")
    result = await adapter.send(PARENT_ID, "Restart status", metadata=STATUS)
    if not redirect:
        assert result.success and result.message_id == "9001"
        adapter._client.fetch_channel.assert_not_awaited()
        cached.send.assert_awaited_once()
    else:
        adapter._client.fetch_channel.assert_awaited_once_with(int(THREAD_ID))
        cached.send.assert_not_awaited()
        fresh.send.assert_not_awaited()
        if state != "archived":
            assert not result.success
        elif skip:
            assert result.success and result.message_id is None
            assert result.raw_response["skipped"] == "archived_thread"
        else:
            assert result.success and result.raw_response["chat_id"] == "4000"
            parent.send.assert_awaited_once()
    if skip or not redirect or state != "archived":
        parent.send.assert_not_awaited()
