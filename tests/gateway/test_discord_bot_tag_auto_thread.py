"""A bot handoff that auto-threads keeps its continuation chunks in the tag's thread.

Another bot posts a tag (``<@receiver> brief``) and then, top-level in the same channel, a few
unmentioned chunks (here: one file each). The bot-tag window admits the chunks; they must then
run in the thread the tag opened, not each open a thread (and a session) of their own, and a
chunk that arrives while the tag's batch is pending joins that batch, files included."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_discord_free_response import (
    FakeDMChannel,
    FakeForumChannel,
    FakeTextChannel,
    FakeThread,
    discord_platform,
    make_message,
)
from plugins.platforms.discord.adapter import DiscordAdapter

CHANNEL_ID = 321
BOT_AUTHOR_ID = 42


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)
    for var in ("DISCORD_NO_THREAD_CHANNELS", "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS",
                "DISCORD_AUTO_THREAD", "DISCORD_BOTS_REQUIRE_INLINE_MENTION"):
        monkeypatch.delenv(var, raising=False)
    # The receiving agent's config from the incident: mention-gated, bots by mention, and a
    # free-response channel that still auto-threads each top-level message.
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", str(CHANNEL_ID))
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_AUTO_THREAD", "true")

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    adapter._ready_event.set()
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter.handle_message = AsyncMock()
    adapter._cache_discord_document = AsyncMock(side_effect=lambda att, ext: f"body of {att.filename}".encode())

    created: list = []

    async def _create_thread(message):
        thread = FakeThread(channel_id=5000 + len(created), name="handoff", parent=message.channel)
        created.append(thread)
        return thread

    adapter._auto_create_thread = AsyncMock(side_effect=_create_thread)
    return adapter


def _bot_message(channel, msg_id: int, content: str = "", *, files=()):
    message = make_message(channel=channel, content=content)
    message.id = msg_id
    message.author = SimpleNamespace(id=BOT_AUTHOR_ID, display_name="Finn", name="Finn", bot=True)
    message.attachments = [
        SimpleNamespace(id=msg_id * 10, filename=name, content_type="text/plain", size=32,
                        url=f"https://cdn.discordapp.com/attachments/{msg_id}/{name}")
        for name in files
    ]
    return message


def _tag(adapter, channel, msg_id: int = 100):
    tag = _bot_message(channel, msg_id, f"<@{adapter._client.user.id}> New shared skill request: see the files")
    tag.mentions = [adapter._client.user]
    return tag


async def _flush_pending(adapter) -> None:
    await asyncio.wait_for(asyncio.gather(*adapter._pending_text_batch_tasks.values()), timeout=10.0)


@pytest.mark.asyncio
async def test_attachment_chunks_join_the_tags_thread_and_first_turn(adapter):
    channel = FakeTextChannel(channel_id=CHANNEL_ID)
    files = ("decision-cards.md", "ideas_reactions.py", "digest_post.py")
    messages = [_tag(adapter, channel)] + [
        _bot_message(channel, 101 + i, files=(name,)) for i, name in enumerate(files)
    ]

    # Discord delivers each message as its own task: the chunks arrive while the tag's
    # thread is still being created.
    results = await asyncio.gather(*(adapter._dispatch_discord_message(m) for m in messages))
    await _flush_pending(adapter)

    assert results == [True, True, True, True]
    assert adapter._auto_create_thread.await_count == 1
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.thread_id == "5000"
    assert event.source.chat_id == "5000"
    assert event.source.parent_chat_id == str(CHANNEL_ID)
    assert event.text.startswith("New shared skill request: see the files")
    assert len(event.media_urls) == len(files)
    for name in files:
        assert f"[Content of {name}]" in event.text


@pytest.mark.asyncio
async def test_chunk_after_the_batch_flushed_runs_in_the_tags_session(adapter):
    channel = FakeTextChannel(channel_id=CHANNEL_ID)

    assert await adapter._dispatch_discord_message(_tag(adapter, channel)) is True
    for key in list(adapter._pending_text_batches):
        await adapter._flush_text_batch_now(key)
    adapter.handle_message.assert_awaited_once()

    late = _bot_message(channel, 101, files=("lib.py",))
    assert await adapter._dispatch_discord_message(late) is True

    assert adapter._auto_create_thread.await_count == 1
    assert adapter.handle_message.await_count == 2
    tag_event, late_event = (call.args[0] for call in adapter.handle_message.await_args_list)
    assert late_event.source.chat_id == tag_event.source.chat_id == "5000"
    assert late_event.source.thread_id == tag_event.source.thread_id
    assert late_event.source.parent_chat_id == tag_event.source.parent_chat_id
    assert adapter._event_session_key(late_event) == adapter._event_session_key(tag_event)
    # Only the opening event of a thread carries the auto-thread markers (title lane).
    assert late_event.source.auto_thread_created is False
    assert "[Content of lib.py]" in late_event.text
