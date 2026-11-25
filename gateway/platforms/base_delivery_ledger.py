"""The delivery-ledger bracket around a final reply (``BasePlatformAdapter.send_final_ledgered`` and
``ledger_attempted_final``): record the obligation before the send, settle it from the result, so a
refused or unconfirmed final leaves a row the boot sweep / timed redelivery can act on.

Calls go through the adapter's ``_record_delivery_obligation`` / ``_finalize_delivery_obligation``
methods so a subclass (the local session adapter) can opt out of ledgering.
"""
import asyncio
from contextlib import contextmanager

from gateway.platforms.event import current_reply_trigger
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from gateway.final_delivery_claims import claim, take_claim
from gateway.delivery_ledger import current_delivery_obligation, ledger_enabled

logger = logging.getLogger("gateway.platforms.base")


@contextmanager
def final_reply_scope(adapter, event):
    existing = current_reply_trigger.get()
    if isinstance(existing, tuple) and len(existing) == 4 and existing[1] is asyncio.current_task():
        yield
        return
    token = current_reply_trigger.set((getattr(event, "reply_trigger", None), asyncio.current_task(),
                                       adapter._final_delivery_adapter(event.source), {}))
    try:
        yield
    finally:
        current_reply_trigger.reset(token)


async def ledger_bracket(adapter: Any, event: Any, session_key: str, text_content: str,
                         send: Callable[[Any, Optional[str]], Awaitable[Any]], *,
                         is_ephemeral_response: bool = False) -> "tuple[Any, Any]":
    """Resolve the CURRENT delivery adapter, record, ``send(delivery_adapter, obligation_id)``, finalize.
    Returns the send result with the adapter that owns it. A ledgered send is active work until its
    row is marked (``gateway.final_delivery_claims``), so a restart cannot stop it mid-send."""
    delivery_adapter = adapter._final_delivery_adapter(event.source)
    held = take_claim(event) or claim(getattr(adapter, "gateway_runner", None), event.source, session_key)
    try:
        obligation_id = await adapter._record_delivery_obligation(
            event, session_key, text_content, delivery_adapter, is_ephemeral_response)
        if obligation_id is None and held is not None:
            held.release()  # not ledgered (ledger off, a command reply): no boot resends it
            held = None
        disabled = obligation_id is None and not await asyncio.to_thread(ledger_enabled)
        progress_scope = current_delivery_obligation.set((obligation_id, asyncio.current_task(), disabled))
        try:
            result = await send(delivery_adapter, obligation_id)
        finally:
            current_delivery_obligation.reset(progress_scope)
        if obligation_id is not None:
            await adapter._finalize_delivery_obligation(obligation_id, result, event, delivery_adapter)
    finally:
        if held is not None:
            held.release()
    return result, delivery_adapter


async def ledger_attempted(adapter: Any, event: Any, session_key: str, text: str, error: str) -> bool:
    """``BasePlatformAdapter.ledger_attempted_final``: ledger ``text`` on the CURRENT transport as a failed
    row instead of sending it again, so the timed redelivery sends it once more with the duplicate marker
    and a lost send is not silence. False when nothing was saved (ledger off, a refused reply)."""
    saved: list[str] = []

    async def _attempted(_delivery_adapter: Any, obligation_id: Optional[str]) -> Any:
        from gateway.platforms.base import SendResult
        if obligation_id:
            saved.append(obligation_id)
            logger.warning("Final for session %s may not have landed; the delivery ledger "
                           "redelivers it with the duplicate marker", session_key)
        return SendResult(success=False, error=error)
    await ledger_bracket(adapter, event, session_key, text, _attempted)
    return bool(saved)


async def record_final_obligation(adapter: Any, event: Any, session_key: str, text_content: str,
                            delivery_adapter: Any, is_ephemeral_response: bool) -> Optional[str]:
    """Ledger the final response BEFORE the send so a crash before platform ACK redelivers on
    next boot; best-effort, skips slash-command and ephemeral replies. Returns the obligation id
    or None."""
    if is_ephemeral_response or str(event.text or "").lstrip().startswith(
        ("/", adapter.typed_command_prefix or "!")):
        return None
    try:
        from gateway.delivery_ledger import (
            compute_obligation_id, ledger_enabled, mark_attempting, record_obligation)
        if not await asyncio.to_thread(ledger_enabled):
            return None
        source = event.source
        # ``ledger_message_id`` wins when set: a queued chain's final answers the last message
        # of the chain, not the event that opened it (see ``MessageEvent.ledger_message_id``).
        _ledger_id = getattr(event, "ledger_message_id", None)
        if _ledger_id is None:
            _ledger_id = getattr(event, "message_id", "")
        obligation_id = compute_obligation_id(
            session_key, str(_ledger_id or ""), text_content)
        await asyncio.to_thread(
            record_obligation, obligation_id=obligation_id, session_key=session_key,
            platform=str(getattr(source.platform, "value", source.platform)),
            chat_id=source.chat_id, thread_id=getattr(source, "thread_id", None),
            content=text_content,
            adapter_profile=getattr(delivery_adapter, "_owner_profile", None))
        await asyncio.to_thread(mark_attempting, obligation_id)
        return obligation_id
    except Exception:
        logger.debug("delivery ledger record failed", exc_info=True)
        return None


async def finalize_final_obligation(adapter: Any, obligation_id: str, result: Any, event: Any,
                              delivery_adapter: Any) -> None:
    """Mark the ledger row delivered/failed (best-effort). On ``send_path_degraded`` with a
    replacement adapter live, trigger another redelivery sweep (the watcher's may have run
    before this failure landed; atomic claiming keeps it idempotent). On any other rejection arm
    the runner's timed redelivery, so the reply goes out once the flood penalty or the retry
    backoff has passed instead of waiting for the next restart (#91653)."""
    try:
        from gateway.dead_targets import classify_dead_error
        from gateway.delivery_ledger import is_reconnect_only, mark_delivered, mark_failed, receipt_json
        if getattr(result, "success", False):
            raw = getattr(result, "raw_response", None)
            if isinstance(raw, dict) and "chat_id" in raw:
                receipt = {key: raw[key] for key in ("chat_id", "thread_id", "message_ids") if key in raw}
                await asyncio.to_thread(mark_delivered, obligation_id, json.dumps(receipt))
            else:
                await asyncio.to_thread(mark_delivered, obligation_id)
            return
        error = str(getattr(result, "error", "") or "")
        raw = getattr(result, "raw_response", None) or {}
        receipt = receipt_json(result) if raw.get("progress_contract") or raw.get("prevent_fallback") else None
        if receipt is None:
            await asyncio.to_thread(mark_failed, obligation_id, error)
        else:
            await asyncio.to_thread(mark_failed, obligation_id, error, receipt)
        if is_reconnect_only(error):
            redeliver = getattr(
                adapter.gateway_runner, "_redeliver_failed_obligations_for_platform", None)
            live = adapter._registry_delivery_adapter(event.source)
            if live is not delivery_adapter and callable(redeliver):
                await redeliver(event.source.platform,
                                profile=getattr(delivery_adapter, "_owner_profile", None))
        elif classify_dead_error(error) is None:  # a dead chat is never retried: no timer to wake
            schedule = getattr(adapter.gateway_runner, "_schedule_flood_redelivery", None)
            if callable(schedule):
                schedule(event.source.platform,
                         profile=getattr(delivery_adapter, "_owner_profile", None))
    except Exception:
        logger.debug("delivery ledger update failed", exc_info=True)
