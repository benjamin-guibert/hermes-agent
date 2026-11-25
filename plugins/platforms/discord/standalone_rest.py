"""Discord REST pacing for the out-of-process sender (``hermes send``, the send_message tool, cron).

Without the live client there is no discord.py rate-limit bucket, so a long text sent as many
messages (a reply whose file upload Discord refused) must honour Discord's limits itself: a 429
waits ``retry_after`` and retries, and a bucket left at ``X-RateLimit-Remaining: 0`` waits
``X-RateLimit-Reset-After`` before the next message (#152012).
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
from dataclasses import dataclass

from gateway.outbound.operation import outbound_operation
import time
from typing import Any

logger = logging.getLogger(__name__)
_MAX_RATE_LIMIT_RETRIES = 5
_bucket_open_at: dict[str, float] = {}  # url -> monotonic time its bucket refills


def api_url(path: str) -> str:
    """``https://discord.com/api/v10/<path>``, from discord.py's ``Route.BASE`` when installed."""
    try:
        from discord.http import Route
        base = Route.BASE
    except ImportError:
        base = "https://discord.com/api/v10"
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _seconds(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 60.0))
    except (TypeError, ValueError):
        return 1.0


async def post_paced(session: Any, url: str, read: Any, error_prefix: str, **kwargs: Any) -> tuple:
    """``POST url`` once the bucket is open; a 429 sleeps ``retry_after`` and retries. Returns
    ``read(resp, error_prefix)`` (``(data, err)``) of the final answer."""
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        wait = _bucket_open_at.get(url, 0.0) - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        async with session.post(url, **kwargs) as resp:
            headers = getattr(resp, "headers", None) or {}
            if str(headers.get("X-RateLimit-Remaining", "")) == "0":
                _bucket_open_at[url] = time.monotonic() + _seconds(headers.get("X-RateLimit-Reset-After"))
            if resp.status == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                retry_after = headers.get("Retry-After")
                try:
                    import json
                    retry_after = json.loads(await resp.read() or b"{}").get("retry_after", retry_after)
                except (ValueError, AttributeError):  # an unreadable 429 body still waits the header value
                    logger.debug("Discord 429 body unreadable; waiting Retry-After %s", retry_after)
                _bucket_open_at[url] = time.monotonic() + _seconds(retry_after)
                continue
            return await read(resp, error_prefix)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class RestDeliveryRoute:
    chat_id: str
    thread_id: str | None
    message: str
    caption: str | None
    redirected: bool
    cache: dict
    mentions: Any


async def _channel_state(aiohttp, target, cache, route_key, headers, sess_kw, req_kw, reader, limit):
    key = (route_key, str(target))
    if key not in cache:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **sess_kw) as session:
                async with session.get(api_url(f"channels/{target}"), headers=headers, **req_kw) as response:
                    if response.status != 200:
                        raise ValueError("Cannot resolve Discord channel state; delivery refused")
                    state = await reader(response, limit)
                    if str(state.get("id")) != str(target):
                        raise ValueError("Discord channel identity mismatch; delivery refused")
                    cache[key] = state
        except Exception as exc:
            raise ValueError("Cannot resolve Discord channel state; delivery refused") from exc
    return cache[key]


async def resolve_rest_route(aiohttp, pconfig, chat_id, thread_id, message, caption, mentions,
                             headers, sess_kw, req_kw, reader, limit, remember_forum):
    operation = outbound_operation.get()
    cache = operation if operation is not None else {}
    route_key = ("rest", id(pconfig))
    mention_key = (route_key, "mention_controls")
    if isinstance(mentions, dict):
        cache[mention_key] = mentions
    else:
        mentions = cache.get(mention_key)
    if (getattr(pconfig, "extra", None) or {}).get("redirect_archived_thread_messages", True) is False:
        return RestDeliveryRoute(chat_id, thread_id, message, caption, False, cache, mentions)
    target = thread_id or chat_id
    state = await _channel_state(aiohttp, target, cache, route_key, headers, sess_kw, req_kw, reader, limit)
    channel_type = state.get("type")
    if type(channel_type) is not int or channel_type not in (0, 1, 3, 5, 10, 11, 12, 15, 16):
        raise ValueError("Unknown Discord channel type; delivery refused")
    if thread_id and channel_type not in (10, 11, 12):
        raise ValueError("Discord thread state does not match target; delivery refused")
    if channel_type not in (10, 11, 12):
        remember_forum(chat_id, channel_type == 15)
        return RestDeliveryRoute(chat_id, thread_id, message, caption, False, cache, mentions)
    archived = (state.get("thread_metadata") or {}).get("archived")
    if type(archived) is not bool:
        raise ValueError("Unknown Discord thread archive state; delivery refused")
    if not archived:
        return RestDeliveryRoute(chat_id, thread_id, message, caption, False, cache, mentions)
    parent = state.get("parent_id")
    if not parent or str(parent) == str(target):
        raise ValueError("Cannot resolve archived thread parent; delivery refused")
    parent_state = await _channel_state(aiohttp, parent, cache, route_key, headers, sess_kw, req_kw, reader, limit)
    if parent_state.get("type") not in (0, 5):
        raise ValueError("Archived thread parent cannot accept a top-level message; delivery refused")
    remember_forum(str(parent), False)
    if not cache.get("redirect_logged"):
        logger.info("Discord archived thread redirect %s to %s", target, parent)
        cache["redirect_logged"] = True
    if not cache.get("prefix_sent"):
        prefix = f"<#{target}>\n"
        if message:
            message = prefix + message
        else:
            caption = prefix + (caption or "")
    message, caption = fit_redirected_caption(message, caption)
    return RestDeliveryRoute(str(parent), None, message, caption, True, cache, mentions)


def fit_redirected_caption(message, caption):
    if caption and len(caption) > 2000:
        return caption, None
    return message, caption


async def post_rest_text(session, url, chunks, reader, headers, request_kwargs, mentions, redirect_cache, ids):
    data = None
    for chunk in chunks:
        payload = {"content": chunk}
        if isinstance(mentions, dict):
            payload["allowed_mentions"] = mentions
        data, error = await post_paced(session, url, reader, "Discord API error",
                                       headers=headers, json=payload, **request_kwargs)
        if error:
            return data, error
        ids.append(str(data.get("id")))
        if redirect_cache is not None:
            redirect_cache["prefix_sent"] = True
    return data, None


async def post_rest_file(aiohttp, session, url, path, caption, mentions, headers, request_kwargs, reader):
    form = aiohttp.FormData()
    payload = {"allowed_mentions": mentions} if isinstance(mentions, dict) else {}
    if caption:
        payload["content"] = caption
    if payload:
        form.add_field("payload_json", json.dumps(payload), content_type="application/json")
    file = await asyncio.to_thread(open, path, "rb")
    try:
        form.add_field("files[0]", file, filename=os.path.basename(path))
        async with session.post(url, headers=headers, data=form, **request_kwargs) as response:
            return await reader(response, "Discord API error")
    finally:
        await asyncio.to_thread(file.close)


async def post_missing_caption(session, url, caption, headers, request_kwargs, reader, limit, ids, redirect_cache, mentions=None):
    try:
        async with session.post(url, headers=headers, json={"content": caption, **({"allowed_mentions": mentions} if isinstance(mentions, dict) else {})}, **request_kwargs) as response:
            if response.status not in (200, 201):
                return None
            data = await reader(response, limit)
            ids.append(str(data.get("id")))
            if redirect_cache is not None:
                redirect_cache["prefix_sent"] = True
            return data
    except Exception:
        logger.warning("Discord caption-fallback send failed for missing media", exc_info=True)
        return None
