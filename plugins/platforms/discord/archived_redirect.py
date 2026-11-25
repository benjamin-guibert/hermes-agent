"""Resolve closed-thread delivery before sending any message chunks."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from gateway.platforms.event import current_reply_trigger
from gateway.outbound.operation import outbound_operation
from plugins.platforms.discord.adapter_archived_status import skip_status_in_archived_thread
from typing import Any

logger = logging.getLogger(__name__)
CONFIG_KEY = "redirect_archived_thread_messages"


@dataclass(frozen=True)
class LiveReplyTrigger:
    issuer: Any
    message_id: str
    thread_id: str
    author_id: str
    created_at: datetime


def capture_live_trigger(adapter: Any, message: Any, *, recovered: bool) -> LiveReplyTrigger | None:
    """Capture only an actual live human post in its original thread."""
    from plugins.platforms.discord import adapter as facade
    author = getattr(message, "author", None)
    created_at = getattr(message, "created_at", None)
    if (recovered or not isinstance(message.channel, facade.discord.Thread)
            or getattr(author, "bot", None) is not False
            or getattr(message, "webhook_id", None) is not None
            or not isinstance(created_at, datetime) or created_at.tzinfo is None
            or not getattr(author, "id", None) or not getattr(message, "id", None)):
        return None
    return LiveReplyTrigger(adapter, str(message.id), str(message.channel.id), str(author.id), created_at)


def permits_current_human_reply(adapter: Any, channel: Any, reply_to: Any, metadata: Any) -> bool:
    trigger = (metadata or {}).get("reply_trigger")
    archived_at = getattr(channel, "archive_timestamp", None)
    scope = current_reply_trigger.get()
    return (isinstance(trigger, LiveReplyTrigger) and isinstance(scope, tuple) and len(scope) == 4
            and scope[0] is trigger and scope[1] is asyncio.current_task() and scope[2] is adapter and str(reply_to) == trigger.message_id
            and str(channel.id) == trigger.thread_id
            and isinstance(archived_at, datetime) and archived_at.tzinfo is not None
            and trigger.created_at > archived_at)


async def refresh_thread_for_send(adapter, channel):
    from plugins.platforms.discord import adapter as facade
    if adapter.config.extra.get(CONFIG_KEY, True) is False or not isinstance(channel, facade.discord.Thread):
        return channel, None
    scope = current_reply_trigger.get()
    operation = outbound_operation.get()
    routes = scope[3] if isinstance(scope, tuple) and len(scope) == 4 and scope[1] is asyncio.current_task() else operation
    if routes is not None:
        key = (id(adapter), str(channel.id))
        if key not in routes:
            try:
                refreshed = await adapter._client.fetch_channel(int(channel.id))
                if not isinstance(refreshed, facade.discord.Thread) or str(refreshed.id) != str(channel.id):
                    raise ValueError("Discord thread identity mismatch")
                routes[key] = refreshed
            except Exception:
                logger.debug("Discord thread refresh failed", exc_info=True)
                routes[key] = ValueError("Cannot refresh Discord thread state; delivery refused")
        if isinstance(routes[key], Exception):
            raise routes[key]
        channel = routes[key]
    return channel, routes


async def resolve_text_send(adapter, channel, content, reply_to, metadata, nonconversational):
    skipped = skip_status_in_archived_thread(adapter.config.extra, channel, nonconversational)
    if skipped is None:
        channel, _routes = await refresh_thread_for_send(adapter, channel)
        skipped = skip_status_in_archived_thread(adapter.config.extra, channel, nonconversational)
    if skipped is not None:
        return channel, content, False, skipped
    channel, content, redirected = await redirect_archived_send(adapter, channel, content, reply_to, metadata)
    return channel, content, redirected, None


async def redirect_archived_send(adapter: Any, channel: Any, content: str, reply_to: Any = None, metadata: Any = None) -> tuple[Any, str, bool]:
    """Return the actual destination and text for one logical send."""
    from plugins.platforms.discord import adapter as facade
    if adapter.config.extra.get(CONFIG_KEY, True) is False or not isinstance(channel, facade.discord.Thread):
        return channel, content, False
    channel, routes = await refresh_thread_for_send(adapter, channel)
    archived = getattr(channel, "archived", None)
    if type(archived) is not bool:
        raise ValueError("Unknown Discord thread archive state; delivery refused")
    if not archived:
        return channel, content, False
    if permits_current_human_reply(adapter, channel, reply_to, metadata):
        return channel, content, False
    parent = getattr(channel, "parent", None)
    parent_id = getattr(channel, "parent_id", None)
    if not parent_id:
        raise ValueError("Cannot resolve archived thread parent; delivery refused")
    if parent is None or str(parent.id) != str(parent_id):
        parent = await adapter._resolve_channel(str(parent_id))
    if (parent is None or str(parent.id) != str(parent_id) or str(parent.id) == str(channel.id)
            or isinstance(parent, facade.discord.Thread)):
        raise ValueError("Cannot resolve archived thread parent; delivery refused")
    if adapter._is_forum_parent(parent):
        raise ValueError("Archived thread parent cannot accept a top-level message; delivery refused")
    scoped = routes is not None
    if not scoped or "redirect_logged" not in routes:
        logger.info("Discord archived thread redirect %s to %s", channel.id, parent.id)
        if scoped:
            routes["redirect_logged"] = True
    prefix = "" if scoped and routes.get("prefix_sent") else f"<#{channel.id}>\n"
    return parent, prefix + content, True


def mark_redirect_delivered() -> None:
    scope = current_reply_trigger.get()
    if isinstance(scope, tuple) and len(scope) == 4 and scope[1] is asyncio.current_task():
        scope[3]["prefix_sent"] = True
    operation = outbound_operation.get()
    if operation is not None:
        operation["prefix_sent"] = True


PROGRESS_CONTRACT = "discord.archived_redirect.v1"


class ProgressStorageError(Exception):
    """A guarded POST cannot proceed without durable progress."""


async def checkpoint_redirect_progress(progress: dict) -> None:
    from gateway.delivery_ledger import current_delivery_obligation, checkpoint_delivery_progress
    scope = current_delivery_obligation.get()
    if scope is None or scope[1] is not asyncio.current_task():
        return
    if scope[0] is None and len(scope) > 2 and scope[2] is True:
        return
    if scope[0] is None:
        raise ProgressStorageError("Redirected delivery has no durable obligation; delivery refused")
    try:
        await asyncio.to_thread(checkpoint_delivery_progress, scope[0], progress)
    except Exception as exc:
        raise ProgressStorageError("Cannot store redirected delivery progress; delivery refused") from exc


def make_redirect_progress(channel, original_target, chunks, redirected, metadata):
    inherited = (metadata or {}).get("_redirect_progress")
    if not (redirected or inherited):
        return None
    return {"progress_contract": PROGRESS_CONTRACT,
            "origin_thread_id": original_target if redirected else inherited["origin_thread_id"],
            "chat_id": str(channel.id), "thread_id": None,
            "message_ids": list((inherited or {}).get("message_ids", [])),
            "remaining_chunks": chunks, "resume_safe": True, "partial_overflow": False}


async def send_guarded_chunks(adapter, channel, chunks, message_ids, reply_to, redirected, progress, metadata):
    reference = adapter._reply_reference_for_send(reply_to, channel)
    later = list(((metadata or {}).get("_redirect_progress") or {}).get("later_chunks", []))
    for index, chunk in enumerate(chunks):
        chunk_reference = reference if adapter._reply_to_mode == "all" or index == 0 else None
        if progress is not None:
            progress.update(remaining_chunks=chunks[index:] + later, resume_safe=False)
            await checkpoint_redirect_progress(progress)
        try:
            message = await channel.send(content=chunk, reference=chunk_reference)
        except Exception as exc:
            if chunk_reference is None or not adapter._is_reply_reference_rejected(exc):
                raise
            logger.warning("Discord reply reference rejected for %s", reply_to)
            reference = None
            message = await channel.send(content=chunk, reference=None)
        message_ids.append(str(message.id))
        if progress is not None:
            progress["message_ids"].append(str(message.id))
            progress.update(remaining_chunks=chunks[index + 1:] + later,
                            resume_safe=True, partial_overflow=bool(progress["message_ids"]))
            await checkpoint_redirect_progress(progress)
        if redirected:
            mark_redirect_delivered()


async def failed_guarded_chunks(result, progress, error, message_ids, chunks, chat_id, thread_id):
    safe = isinstance(error, PermissionError) or getattr(error, "status", None) in (400, 403, 404, 429)
    if progress is not None:
        progress["resume_safe"] = safe and not isinstance(error, ProgressStorageError)
        if isinstance(error, ProgressStorageError):
            progress["progress_storage_failed"] = True
        else:
            try:
                await checkpoint_redirect_progress(progress)
            except ProgressStorageError:
                progress.update(resume_safe=False, progress_storage_failed=True)
        result.raw_response = progress
        result.message_id = progress["message_ids"][0] if progress["message_ids"] else None
    elif chunks:
        result.message_id = message_ids[0] if message_ids else None
        result.raw_response = {"message_ids": message_ids, "chat_id": chat_id, "thread_id": thread_id,
                               "partial_overflow": bool(message_ids), "remaining_chunks": chunks[len(message_ids):],
                               "resume_safe": safe}
    return result
