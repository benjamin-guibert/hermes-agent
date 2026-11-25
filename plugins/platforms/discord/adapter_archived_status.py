"""A status notice never reopens an archived Discord thread.

Discord unarchives an unlocked thread when a message is sent to it. A late non-conversational send
(the background review summary, tool progress, "Working — N min", restart and update notices) would
reopen a thread the owner has just closed, so ``discord.skip_status_in_archived_threads`` (default
on) skips it. Conversational sends still go out and reopen the thread: a reply means work continues.

The check reads ``archived`` from the thread object ``_resolve_channel`` returned (discord.py keeps
it current from THREAD_UPDATE events) and never adds a REST call; an unknown state sends as before.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

CONFIG_KEY = "skip_status_in_archived_threads"


def archived_status_extra(discord_cfg: dict) -> dict:
    """``config.extra`` seed for the ``discord.skip_status_in_archived_threads`` YAML key (no env bridge)."""
    return {CONFIG_KEY: discord_cfg[CONFIG_KEY]} if CONFIG_KEY in discord_cfg else {}


def _skip_enabled(extra: Any) -> bool:
    value = extra.get(CONFIG_KEY) if isinstance(extra, dict) else None
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def skip_status_in_archived_thread(extra: Any, channel: Any, nonconversational: bool) -> Optional[SendResult]:
    """A delivered no-op for a status send to a known-archived thread, else ``None`` (send as usual).

    The result is ``success=True`` with no message id and is not written to the recovery ledger, so no
    caller retries it, logs it as an error, replays it or falls back to another channel."""
    from plugins.platforms.discord import adapter as facade  # the facade rebinds ``discord`` on a late install
    thread_type = getattr(facade.discord, "Thread", None)
    if not (nonconversational and thread_type is not None and isinstance(channel, thread_type)):
        return None
    if getattr(channel, "archived", None) is not True or not _skip_enabled(extra):
        return None
    logger.info("[Discord] Skipped status notice to archived thread %s", channel.id)
    return SendResult(success=True, raw_response={"message_ids": [], "skipped": "archived_thread"})
