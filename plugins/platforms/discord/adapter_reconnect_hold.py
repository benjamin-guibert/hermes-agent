"""Discord inbound across a runner reconnect: the Discord side of the #132829 hand-over.

When the WebSocket health check fails, the runner retires this adapter (it leaves ``runner.adapters``
and becomes the reconnect queue's ``predecessor``) and builds a new one. A text batch still in its
debounce, or an attachment message already on its way, then reaches ``handle_message`` on the retired
instance, where admission finds no Discord adapter and the message is lost. The retired instance
holds it instead; ``adopt_held_inbound`` on the published replacement (``hand_over_held_inbound``)
dispatches it once, in arrival order, and later inbound on the retired instance forwards there. A
failed candidate hands back to the predecessor through the same call, which keeps holding. A planned
stop is not a reconnect: its path (the stop replay) is unchanged.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import weakref
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_HELD = 64
_MAX_AGE_SECONDS = 1800.0  # from the retire: an answer this late is worse than none
_FALSE = {"false", "0", "no", "off"}


class DiscordReconnectHoldMixin:
    """Hold / hand-over of inbound for a ``DiscordAdapter`` the runner retired for a reconnect."""

    def _reconnect_hold_enabled(self) -> bool:
        """``discord.reconnect_hold_inbound`` (default on)."""
        value = self.config.extra.get("reconnect_hold_inbound", True)
        return value if isinstance(value, bool) else str(value).strip().lower() not in _FALSE

    def _reconnect_entry(self) -> Optional[dict]:
        """The runner's reconnect-queue entry when it retired this instance for a reconnect, else None."""
        runner = getattr(self, "gateway_runner", None)
        if runner is None or getattr(runner, "_draining", False):
            return None  # a stop drains: the stop replay owns what arrives now
        info = getattr(runner, "_failed_platforms", {}).get(self.platform)
        return info if info is not None and info.get("predecessor") is self else None

    async def handle_message(self, event) -> None:
        successor = getattr(self, "_reconnect_successor", None)
        target = successor() if successor is not None else None
        if target is not None:
            target._take_held_inbound([event], retired_at=None)
            return
        entry = self._reconnect_entry() if self._reconnect_hold_enabled() else None
        if entry is None:
            await super().handle_message(event)
            return
        if getattr(self, "_reconnect_retired_at", None) is None:
            self._reconnect_retired_at = entry.get("queued_at", time.monotonic())
        held = self.__dict__.setdefault("_reconnect_held", [])
        if len(held) >= _MAX_HELD:
            held.pop(0)
            logger.warning("[%s] Held-inbound queue full (%d); dropping the oldest", self.name, _MAX_HELD)
        held.append(event)
        logger.info("[%s] Holding inbound for the replacement adapter (%d chars)", self.name, len(event.text or ""))

    def adopt_held_inbound(self, predecessor: Any) -> None:
        """Take what ``predecessor`` holds; later inbound there forwards here. The runner calls it on
        the published replacement, and on the predecessor after a failed candidate."""
        if not self._reconnect_hold_enabled():
            return
        self._reconnect_successor = None  # we own the queue now: a reverse link would forward in a cycle
        predecessor._reconnect_successor = weakref.ref(self)
        held = predecessor.__dict__.pop("_reconnect_held", [])
        self._take_held_inbound(held, retired_at=getattr(predecessor, "_reconnect_retired_at", None))

    def _take_held_inbound(self, events: list, *, retired_at: Optional[float]) -> None:
        if not events:
            return
        if self._reconnect_entry() is not None:  # retired too (a failed candidate handed back): keep holding
            held = self.__dict__.setdefault("_reconnect_held", [])
            held.extend(events)
            del held[:-_MAX_HELD]
            if retired_at is not None and getattr(self, "_reconnect_retired_at", None) is None:
                self._reconnect_retired_at = retired_at
            return
        if retired_at is not None and time.monotonic() - retired_at > _MAX_AGE_SECONDS:
            logger.warning("[%s] Dropping %d held inbound message(s): the reconnect took over %.0fs",
                           self.name, len(events), _MAX_AGE_SECONDS)
            return
        prior = getattr(self, "_reconnect_redispatch_task", None)
        self._reconnect_redispatch_task = asyncio.get_running_loop().create_task(
            self._redispatch_held_inbound(list(events), prior))

    async def _redispatch_held_inbound(self, events: list, prior: Optional[asyncio.Task]) -> None:
        if prior is not None:
            await asyncio.wait([prior])  # an earlier hand-over's messages arrived first
        for event in events:
            # The identity a retired instance pinned names a transport the runner no longer lists, and
            # admission refuses it (profile_mismatch). A plain replace drops that wire-invisible
            # provenance, so the source routes again from here, as a fresh inbound would.
            event.source = dataclasses.replace(event.source, profile=None, profile_route_rejected=False)
            event.source._transport_adapter_ref = weakref.ref(self)
            await self.handle_message(event)
        logger.info("[%s] Redispatched %d held inbound message(s) after reconnect", self.name, len(events))
