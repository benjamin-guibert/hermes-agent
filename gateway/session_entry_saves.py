"""One routing entry's save, kept off the event loop.

``SessionStore._save_entry`` may wait the whole write patience (20 s) for the state.db writer
mutex. A caller on an event loop thread (a ``pre_gateway_dispatch`` plugin that stamps session
metadata on every inbound message) would freeze every socket, heartbeat and timer of that loop for
as long, so there the save runs on the loop's default executor, in the caller's context (profile
scope). Pending saves coalesce per key: the worker serializes the entry's latest state when it
starts, under the shared routing generation, so a late worker never regresses a newer row.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging

logger = logging.getLogger(__name__)


def save_entry_off_loop(store, session_key: str) -> None:
    """Persist ONE routing entry (``_save_entry``): inline on a plain thread, on a worker when the
    calling thread runs an event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        store._save_entry(session_key)
        return
    with store._lock:
        pending = store._lazy("_off_loop_saves", set)
        if session_key in pending:
            return
        pending.add(session_key)

    def save() -> None:
        with store._lock:
            pending.discard(session_key)
        store._save_entry(session_key)

    future = loop.run_in_executor(None, contextvars.copy_context().run, save)
    future.add_done_callback(lambda done: _report(done, session_key))


def _report(done: asyncio.Future, session_key: str) -> None:
    if not done.cancelled() and done.exception() is not None:
        logger.warning("gateway.session: routing entry save for %r failed: %s", session_key, done.exception())
