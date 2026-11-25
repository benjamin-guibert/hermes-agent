"""A final reply in flight is active work until its delivery-ledger row is marked.

Carry of NousResearch/hermes-agent#92589. A turn frees its work slot when it settles, before its
adapter sends the reply. A restart that stopped the gateway in that window tore the send down after
the platform had the message but before ``mark_delivered()``; the row stayed ``attempting`` and the
next boot sent the reply again as a recovered duplicate.

A claim covers the window. The session drain claims the reply for the adapter event that waits for
it while the turn still counts (``claim_owed_reply``); the delivery-ledger bracket takes that claim
over, or takes its own for any other ledgered final, and releases it in a ``finally`` once the row
is marked; the adapter lifecycle releases whatever it still holds when it ends
(``BasePlatformAdapter._release_turn_marker``). The restart wait, ``_active_work_count`` and the
``active-work`` listing count every live claim. Like a wedged chat turn, a claim older than
``agent.gateway_timeout`` (``HERMES_AGENT_TIMEOUT``) no longer holds the restart wait: a hung send
holds the stop at most that long.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

_ATTR = "_final_delivery_claim"


class FinalDeliveryClaim:
    """One reply this process is sending (or about to send) whose ledger row is not marked yet."""

    __slots__ = ("_live", "platform", "session", "started")

    def __init__(self, live: set, session: Optional[str], platform: Any) -> None:
        self._live, self.session, self.started = live, session, time.monotonic()
        self.platform = getattr(platform, "value", platform)
        live.add(self)

    def release(self) -> None:
        self._live.discard(self)


def _live(runner: Any) -> set:
    live = getattr(runner, "_final_delivery_claims", None)
    if live is None:
        live = runner._final_delivery_claims = set()
    return live


def claim(runner: Any, source: Any, session_key: Optional[str]) -> Optional[FinalDeliveryClaim]:
    """A new claim on *runner* (None without a runner: no restart wait reads it)."""
    if runner is None:
        return None
    return FinalDeliveryClaim(_live(runner), session_key, getattr(source, "platform", None))


def take_claim(event: Any) -> Optional[FinalDeliveryClaim]:
    """Remove and return the claim *event* holds."""
    held = getattr(event, _ATTR, None)
    if held is not None:
        setattr(event, _ATTR, None)
    return held


def move_claim(source: Any, target: Any) -> None:
    """The reply moves to another event object (a busy follow-up's delivery copy)."""
    held = take_claim(source)
    if held is not None:
        setattr(target, _ATTR, held)


def release_final_delivery(event: Any) -> None:
    """The lifecycle that would send *event*'s reply is done: drop a claim no send took over."""
    held = take_claim(event)
    if held is not None:
        held.release()


def claim_owed_reply(authority: Any, admission_id: str, session_key: str) -> None:
    """The settling turn owes a reply to the adapter event waiting for it: claim it now, while the
    turn still counts, so no restart wait sees a gap before the send."""
    from gateway.session_ingress import _marker_adopters
    adopter = _marker_adopters(authority).get(admission_id)
    if adopter is not None and getattr(adopter, _ATTR, None) is None:
        setattr(adopter, _ATTR, claim(authority.runner, adopter.source, session_key))


def _wedged(held: FinalDeliveryClaim, now: float) -> bool:
    from gateway.run import _float_env
    timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
    return timeout > 0 and now - held.started >= timeout


def final_delivery_count(runner: Any, *, awaitable: bool = False) -> int:
    """Live claims; ``awaitable`` leaves out the ones past ``agent.gateway_timeout``."""
    now = time.monotonic()
    return sum(1 for held in list(_live(runner)) if not (awaitable and _wedged(held, now)))


def final_delivery_units(runner: Any) -> list[dict[str, Any]]:
    """One ``active_work`` entry per live claim."""
    now = time.monotonic()
    return [{"kind": "final_delivery", "session": held.session, "platform": held.platform,
             "pid": os.getpid(), "elapsed_s": round(now - held.started, 1), "wedged": _wedged(held, now)}
            for held in list(_live(runner))]
