"""A top-level Discord post that pings the owner opens its own thread (``discord.owner_ping_thread``).

The owner answers a ping in place: a top-level answer would open an unrelated auto-thread whose
session never saw the post. After a send to a guild text channel whose text mentions a human in
the allowed-users set, the sender opens a public thread on the first delivered message. Both
senders run this: the live adapter (agent replies, cron through the gateway) and the REST sender
(``hermes send``, the send_message tool, cron without a gateway), which may run in another process.

So the thread is handed to the gateway through a per-thread record file, not process memory: on
the first inbound message in that thread the gateway marks it as a bot thread (the auto-thread
mark, so the mention gate admits replies after a restart) and seeds its session with the post text
the way a continuable cron thread is seeded. A thread-create failure never fails the send.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from gateway.inbound_reply_context import CRON_DELIVERY_PREFIX

logger = logging.getLogger(__name__)

CONFIG_KEY = "owner_ping_thread"
RAW_KEY = "owner_ping_thread_id"  # added to the send result's raw data when a thread opened
# Set by the chunk-resume path: a resumed chunk is a later chunk, never the post's first message.
CONTINUATION_KEY = "_owner_ping_continuation"
FALLBACK_NAME = "Reply here"
THREAD_ALREADY_CREATED = 160004  # Discord: "A thread has already been created for this message"
CREATE_TIMEOUT_SECONDS = 5.0
_RECORD_DIR = "discord_owner_ping_threads"
_RECORD_MAX_AGE_SECONDS = 30 * 86400
_SEED_LABEL = "[Your earlier Discord post; the owner answers it in this thread]"
_TOP_LEVEL_REST_TYPES = (0, 5)  # GUILD_TEXT, GUILD_ANNOUNCEMENT
_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ANY_MENTION = re.compile(r"<(?:@[!&]?|#)\d+>|<a?:\w+:\d+>")
_LINE_MARKS = re.compile(r"^(?:\s*(?:#{1,6}|-#|>{1,3}|[-*+]|\d+[.)])\s+)+")
_INLINE_MARKS = re.compile(r"\*\*|__|~~|\|\||[*_`]")
_rest_bot_flags: dict[str, bool] = {}


def yaml_extra(discord_cfg: dict) -> dict:
    """``config.extra`` seed for the ``discord.owner_ping_thread`` YAML key (no env bridge)."""
    return {CONFIG_KEY: discord_cfg[CONFIG_KEY]} if CONFIG_KEY in discord_cfg else {}


def enabled(extra: Any) -> bool:
    return (extra or {}).get(CONFIG_KEY, True) is not False


def _candidates(text: str, owner_ids: Any, self_id: Any) -> list[str]:
    owners = {str(uid) for uid in owner_ids or () if str(uid).isdigit()}
    found = dict.fromkeys(m for m in _USER_MENTION.findall(text or "") if m in owners and m != str(self_id))
    return list(found)


def _clean_line(line: str) -> str:
    line = _ANY_MENTION.sub("", line)
    line = _INLINE_MARKS.sub("", _LINE_MARKS.sub("", line.strip()))
    words = line.split()
    while words and not any(ch.isalnum() for ch in words[0]):
        words.pop(0)  # emoji-only prefix ("🔔", "⚠️ —")
    return " ".join(words)


def thread_name(text: str) -> str:
    """First non-empty line, cleaned; a wrapped cron delivery is named after its job. Max 80."""
    lines = (text or "").splitlines()
    name = ""
    if lines and lines[0].startswith(CRON_DELIVERY_PREFIX):
        name = _clean_line(lines[0][len(CRON_DELIVERY_PREFIX):])
    for line in lines if not name else ():
        name = _clean_line(line)
        if name:
            break
    if not name:
        return FALLBACK_NAME
    return name if len(name) <= 80 else name[:77] + "..."


# --- the hand-off record the gateway reads on the thread's first inbound message ---

def _record_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _RECORD_DIR


def _record_path(thread_id: str) -> Path:
    return _record_dir() / f"{int(thread_id)}.json"


def _write_record(thread_id: str, chat_id: str, text: str) -> None:
    from utils.files import atomic_json_write
    folder = _record_dir()
    folder.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - _RECORD_MAX_AGE_SECONDS
    for old in folder.glob("*.json"):  # a thread nobody answered in a month is not waited on
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            continue
    atomic_json_write(_record_path(thread_id), {"chat_id": str(chat_id), "text": text}, indent=None)


async def _hand_off(thread_id: str, chat_id: str, text: str) -> None:
    try:
        await asyncio.to_thread(_write_record, thread_id, chat_id, text)
    except OSError as exc:  # the thread is open; only its session seed is lost
        logger.warning("Discord owner-ping thread %s: cannot store the post for its session: %s", thread_id, exc)


# --- live adapter path ---

async def _is_human(client: Any, user_id: str) -> bool:
    user = client.get_user(int(user_id))
    if user is None:
        user = await client.fetch_user(int(user_id))
    return getattr(user, "bot", True) is False


async def _create_live_thread(adapter: Any, channel: Any, text: str, message_id: str) -> Optional[str]:
    from plugins.platforms.discord import adapter as facade
    client = adapter._client
    top_level = (facade.discord.ChannelType.text, facade.discord.ChannelType.news)
    if getattr(channel, "type", None) not in top_level:
        return None  # a thread, DM, forum or voice channel
    owners = _candidates(text, adapter._allowed_user_ids, getattr(client.user, "id", None))
    if not owners or not any([await _is_human(client, uid) for uid in owners]):
        return None
    try:
        thread = await channel.get_partial_message(int(message_id)).create_thread(
            name=thread_name(text), auto_archive_duration=1440)
    except facade.discord.HTTPException as exc:
        if getattr(exc, "code", None) != THREAD_ALREADY_CREATED:
            raise
        return str(message_id)  # a thread on a message has the message's id
    return str(thread.id)


async def open_live_thread(adapter: Any, channel: Any, text: str, result: Any, metadata: Any) -> None:
    """After a successful live send: open the owner-ping thread on its first message (never raises)."""
    message_id = result.message_id
    if (not enabled(adapter.config.extra) or not result.success or not message_id
            or (metadata or {}).get(CONTINUATION_KEY) or (metadata or {}).get("_redirect_progress")):
        return
    try:  # R5: the post is delivered; nothing here may fail, delay or repeat the send
        thread_id = await asyncio.wait_for(
            _create_live_thread(adapter, channel, text, message_id), CREATE_TIMEOUT_SECONDS)
        if thread_id is None:
            return
        await adapter._threads.mark_async(thread_id)
        await _hand_off(thread_id, str(channel.id), text)
    except Exception as exc:  # R5: the send is done; any thread failure logs one WARNING
        logger.warning("Discord owner-ping thread on message %s failed: %s", message_id, exc or type(exc).__name__,
                       exc_info=True)
        return
    result.raw_response[RAW_KEY] = thread_id


# --- REST path ---

def _rest_owner_ids(pconfig: Any) -> set:
    from plugins.platforms.discord.adapter import DiscordAdapter, _clean_discord_id, _scoped_gate_env
    raw = _scoped_gate_env("DISCORD_ALLOWED_USERS")
    extra = getattr(pconfig, "extra", None) or {}
    if not raw:
        raw = extra.get("allow_from") or extra.get("allowed_users")
    return {_clean_discord_id(str(entry)) for entry in DiscordAdapter._gate_csv_set(raw)}


async def _rest_json(session: Any, method: str, url: str, **kwargs: Any) -> tuple[int, dict]:
    from plugins.platforms.discord.adapter import (
        _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES, _standalone_read_json_limited)
    async with getattr(session, method)(url, **kwargs) as response:
        try:
            data = await _standalone_read_json_limited(response, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES)
        except ValueError:
            data = {}
        return response.status, data


async def _rest_is_human(session: Any, user_id: str, api_url: Any, headers: dict, req_kw: dict) -> bool:
    if user_id not in _rest_bot_flags:
        status, data = await _rest_json(session, "get", api_url(f"users/{user_id}"), headers=headers, **req_kw)
        if status != 200:
            return False
        _rest_bot_flags[user_id] = bool(data.get("bot"))
    return _rest_bot_flags[user_id] is False


async def _create_rest_thread(aiohttp: Any, pconfig: Any, chat_id: str, text: str, message_id: str,
                              api_url: Any, headers: dict, sess_kw: dict, req_kw: dict) -> Optional[str]:
    owners = _candidates(text, _rest_owner_ids(pconfig), None)
    if not owners:
        return None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CREATE_TIMEOUT_SECONDS), **sess_kw) as session:
        status, channel = await _rest_json(session, "get", api_url(f"channels/{chat_id}"), headers=headers, **req_kw)
        if status != 200 or channel.get("type") not in _TOP_LEVEL_REST_TYPES:
            return None
        if not any([await _rest_is_human(session, uid, api_url, headers, req_kw) for uid in owners]):
            return None
        status, data = await _rest_json(
            session, "post", api_url(f"channels/{chat_id}/messages/{message_id}/threads"), headers=headers,
            json={"name": thread_name(text), "auto_archive_duration": 1440}, **req_kw)
    if status in (200, 201) and data.get("id"):
        return str(data["id"])
    if data.get("code") == THREAD_ALREADY_CREATED:
        return str(message_id)
    raise RuntimeError(f"Discord API error ({status}): {data.get('message') or data}")


async def open_rest_thread(aiohttp: Any, pconfig: Any, text: str, result: dict, api_url: Any,
                           headers: dict, sess_kw: dict, req_kw: dict) -> None:
    """After a successful REST send: open the owner-ping thread on its first message (never raises)."""
    message_ids = result.get("message_ids") or []
    chat_id = result.get("chat_id")
    if (not enabled(getattr(pconfig, "extra", None)) or not result.get("success") or result.get("thread_id")
            or not message_ids or not chat_id or not (text or "").strip()):
        return
    try:  # R5: the post is delivered; nothing here may fail, delay or repeat the send
        thread_id = await asyncio.wait_for(
            _create_rest_thread(aiohttp, pconfig, str(chat_id), text, str(message_ids[0]), api_url,
                                headers, sess_kw, req_kw), CREATE_TIMEOUT_SECONDS)
        if thread_id is None:
            return
        await _hand_off(thread_id, str(chat_id), text)
    except Exception as exc:  # R5: the send is done; any thread failure logs one WARNING
        logger.warning("Discord owner-ping thread on message %s failed: %s", message_ids[0], exc or type(exc).__name__,
                       exc_info=True)
        return
    result[RAW_KEY] = thread_id


# --- gateway side: the first inbound message in a handed-off thread ---

def _seed_session(adapter: Any, source: Any, thread_id: str, text: str) -> None:
    from gateway.mirror import mirror_to_session
    store = getattr(adapter, "_session_store", None)
    if store is None:
        return
    entry = store.get_or_create_session(source)
    if store.load_transcript(entry.session_id):
        return  # already seeded (a continuable cron thread) or a conversation is under way
    if not mirror_to_session("discord", thread_id, f"{_SEED_LABEL}\n{text}", source_label="discord_owner_ping",
                             thread_id=thread_id, role="user", session_id=entry.session_id):
        logger.warning("Discord owner-ping thread %s: the post did not reach its session", thread_id)


def _take_record(adapter: Any, source: Any, thread_id: str) -> None:
    path = _record_path(thread_id)
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return
    try:
        _seed_session(adapter, source, thread_id, str(record.get("text") or ""))
    finally:
        path.unlink(missing_ok=True)


async def adopt_thread(adapter: Any, message: Any) -> None:
    """Before ingress gates: a message in a handed-off owner-ping thread marks it and seeds its session."""
    from plugins.platforms.discord import adapter as facade
    channel = getattr(message, "channel", None)
    if (not isinstance(channel, facade.discord.Thread) or not enabled(adapter.config.extra)
            or getattr(message, "author", None) == adapter._client.user):
        return
    thread_id = str(channel.id)
    if not await asyncio.to_thread(os.path.exists, _record_path(thread_id)):
        return
    await adapter._threads.mark_async(thread_id)
    author = message.author
    guild = getattr(message, "guild", None)
    source = adapter.build_source(
        chat_id=thread_id, chat_name=adapter._format_thread_chat_name(channel), chat_type="thread",
        user_id=str(author.id), user_name=getattr(author, "display_name", None), thread_id=thread_id,
        guild_id=str(guild.id) if guild else None, parent_chat_id=adapter._get_parent_channel_id(channel))
    try:
        await asyncio.to_thread(_take_record, adapter, source, thread_id)
    except Exception as exc:  # a seed failure never drops the owner's reply
        logger.warning("Discord owner-ping thread %s: seeding its session failed: %s", thread_id, exc, exc_info=True)
