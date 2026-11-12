"""Discord bot-tag handoffs: one tag plus its unmentioned continuation chunks is one handoff.

A bot that tags this one (``<@self> brief``) often sends the rest of the handoff as unmentioned
messages: a long text split at Discord's 2000-char cap, or one file per message. A short window
per (channel, author) admits those chunks. When the tag opens an auto-thread, its chunks run in
that thread too (one thread, one session), and while the tag's text batch is pending they merge
into it, files included, so the whole handoff arrives as one turn."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)


class BotTagWindow:
    """One open continuation window and, once its tag is handled, the auto-thread the tag ran in.

    The thread lives on the window, so it expires with it."""

    __slots__ = ("handled", "tag_id", "thread", "thread_failed", "until")

    def __init__(self, until: float, tag_id: str) -> None:
        self.until = until
        self.tag_id = tag_id
        self.handled = asyncio.Event()
        self.thread: Any = None
        self.thread_failed = False


class DiscordBotTagMixin:
    name: str
    _text_batch_delay_seconds: float
    _text_batch_split_delay_seconds: float
    _bot_tag_windows: dict[str, BotTagWindow]
    _pending_text_batches: dict
    _self_is_explicitly_mentioned: Any
    _text_batch_key: Any
    _handle_message: Any

    @staticmethod
    def _bot_tag_debounce_key(message: Any) -> str:
        return (
            f"{getattr(getattr(message, 'channel', None), 'id', '')}:"
            f"{getattr(getattr(message, 'author', None), 'id', '')}"
        )

    def _bot_tag_window_seconds(self) -> float:
        return max(self._text_batch_delay_seconds, self._text_batch_split_delay_seconds)

    def _record_bot_tag_debounce(self, message: Any) -> Optional[BotTagWindow]:
        """Open a short continuation window after a bot-authored tag; return it, else None."""
        if (
            self._text_batch_delay_seconds <= 0
            or not getattr(message.author, "bot", False)
            or not self._self_is_explicitly_mentioned(message)
        ):
            return None
        now = time.monotonic()
        for key in [key for key, window in self._bot_tag_windows.items() if window.until <= now]:
            del self._bot_tag_windows[key]
        window = BotTagWindow(now + self._bot_tag_window_seconds(), str(message.id))
        self._bot_tag_windows[self._bot_tag_debounce_key(message)] = window
        return window

    def _live_bot_tag_window(self, message: Any) -> Optional[BotTagWindow]:
        """The open window of ``message``'s author in its channel, re-armed; None once it closed.

        A hit re-arms the window: Discord paces a bot's sends at roughly one per
        second, so chunk N of a long handoff lands well after the tag itself; each
        admitted chunk therefore vouches for the next one. The gateway bot loop
        guard bounds a bot that never stops talking."""
        if self._text_batch_delay_seconds <= 0 or not getattr(message.author, "bot", False):
            return None
        key = self._bot_tag_debounce_key(message)
        window = self._bot_tag_windows.get(key)
        now = time.monotonic()
        if window is None or window.until <= now:
            self._bot_tag_windows.pop(key, None)
            return None
        window.until = now + self._bot_tag_window_seconds()
        return window

    def _is_bot_tag_debounce_continuation(self, message: Any) -> bool:
        """Return whether an unmentioned chunk belongs to a recent bot tag (re-arms the window)."""
        return self._live_bot_tag_window(message) is not None

    async def _handle_live_message(self, message: Any, *, role_authorized: bool) -> bool:
        """``_handle_message`` for a live event. A bot tag marks its window handled only after its
        own event is queued, so its chunks route to its thread and merge behind it, not ahead."""
        window = self._record_bot_tag_debounce(message)
        try:
            return await self._handle_message(message, role_authorized=role_authorized)
        finally:
            if window is not None:
                window.handled.set()

    async def _bot_tag_chunk_thread(self, message: Any) -> tuple[bool, Any]:
        """For a top-level message about to auto-thread: ``(True, thread)`` when it is a chunk of a
        bot tag that auto-threaded; ``thread`` is the tag's thread, or None when creating it failed
        (the tag already said so). ``(False, None)`` when the message opens its own thread."""
        if self._self_is_explicitly_mentioned(message):
            return False, None
        window = self._live_bot_tag_window(message)
        if window is None or window.tag_id == str(message.id):
            return False, None
        await window.handled.wait()
        if window.thread is None and not window.thread_failed:
            return False, None  # the tag did not auto-thread (dropped, reply): routing as before
        if window.thread is None:
            logger.info("[%s] Dropping bot-tag chunk %s: its tag's auto-thread failed",
                        self.name, getattr(message, "id", "?"))
        return True, window.thread

    def _note_bot_tag_thread(self, message: Any, thread: Any) -> None:
        """Record the auto-thread a bot tag opened (or failed to open) for its window's chunks."""
        window = self._bot_tag_windows.get(self._bot_tag_debounce_key(message))
        if window is not None and window.tag_id == str(message.id):
            window.thread = thread
            window.thread_failed = thread is None

    def _joins_bot_tag_batch(self, event: MessageEvent) -> bool:
        """A bot-tag chunk that is not plain text (a file) still merges into its tag's pending batch."""
        return (
            getattr(event, "_bot_tag_debounce", False)
            and event.message_type != MessageType.COMMAND
            and self._text_batch_key(event) in self._pending_text_batches
        )

    def _text_batch_delay_for(self, pending: Optional[MessageEvent]) -> float:
        """A bot handoff's continuation chunks arrive at Discord's send rate (~1/s), so a
        batch opened by a bot tag waits the split delay regardless of chunk length."""
        if getattr(pending, "_bot_tag_debounce", False):
            return self._text_batch_split_delay_seconds
        return super()._text_batch_delay_for(pending)  # type: ignore[misc]
