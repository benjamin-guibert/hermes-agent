"""A person's turn lost in a messaging chat settles like ``/discard`` (carry for v0.23.3).

A turn lost across an owner restart (or with its managed worker) is fenced ``unknown`` and pauses
its FIFO, because only its sender knows whether to resend it. Desktop, TUI and the API offer that
choice. A messaging chat does not: its only ways out are ``/discard`` (drops the lost turn, never
re-runs it) and ``/reset``, and the incident chat sat silent behind one pause notice until an
operator discarded the turn. So in a chat the automatic outcome is what ``/discard`` does: the same
fenced resolution (``resolve_unknown``, terminal ``interrupted``, never requeued or replayed), with
a transcript notice for the next turn, and ONE visible line in the chat once its adapter is
connected. The line is recorded in the delivery ledger before it is sent, so a second restart
before the send delivers it once from the boot sweep; a sent line is never sent again.

A chat turn is a person's message a messaging adapter committed: a ``messaging:`` principal with
the native envelope of a messaging platform. A viewer's message into a chat session (Desktop, TUI)
and every non-chat surface keep the pause. ``gateway.messaging_auto_discard_lost_turns: false``
turns this off.

The same switch covers a chat's head refused ``not_found`` because its adapter was absent or
replaced (a reconnect, a platform not connected yet): the FIFO waits, the notice says the message
is answered when the connection is back, and the adapter's connect drains it again. The head is
never dropped: it can still run.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# The transcript boundary: an assistant row (only its display type is stripped before the wire),
# so the next turn reads it as its own last word.
LOST_CHAT_TURN_NOTICE = (
    "My previous turn was cut off by a gateway restart (or a lost worker) and was not run again. "
    "Some of its steps may already have run: check the real state (files, sent messages, jobs) "
    "before continuing or repeating any step."
)
LOST_CHAT_TURN_LINE = (
    "⚠️ A restart cut my last turn, and it was not re-run. Newer messages are answered next; "
    "send the lost one again if you still need it."
)
_FALSE = {'false', '0', 'no', 'off'}


def auto_discard_enabled(store) -> bool:
    """``gateway.messaging_auto_discard_lost_turns`` of the profile that owns *store* (default on)."""
    from gateway.run import _load_gateway_config
    from gateway.session_policy import present_sections
    gateway = present_sections(_load_gateway_config(store.scope.profile_dir / 'config.yaml')).get('gateway') or {}
    value = gateway.get('messaging_auto_discard_lost_turns', True)
    return value.strip().lower() not in _FALSE if isinstance(value, str) else bool(value)


def is_chat_turn(row) -> bool:
    """A person's message from a messaging chat (not automation, not a viewer, not an API caller)."""
    from gateway.session_context import NON_MESSAGING_SESSION_SURFACES
    from gateway.session_unattended_unknown import is_unattended
    envelope = (row.get('payload') or {}).get('native_text_v1') or {}
    platform = str((envelope.get('source') or {}).get('platform') or '').strip().lower()
    return ((row.get('principal_id') or '').startswith('messaging:') and not is_unattended(row)
            and platform not in NON_MESSAGING_SESSION_SURFACES)


def _lost_chat_rows(records):
    from gateway.session_authority import admission_row
    return [row for row in map(admission_row, records) if row['status'] == 'unknown' and is_chat_turn(row)]


def owe_line(row):
    """Record the visible line as owed to the chat (ledger ``pending``) and return what the send
    needs. With the ledger off nothing is recorded, so no later boot can resend it."""
    from gateway.delivery_ledger import compute_obligation_id, ledger_enabled, record_obligation
    from gateway.session_envelope import restore_native_source
    envelope = row['payload']['native_text_v1']
    source = restore_native_source(row['payload'])
    owed = {'payload': row['payload'], 'session_key': envelope['route'],
            'message_ref': 'lost-turn:' + row['admission_id']}
    if ledger_enabled():
        record_obligation(
            obligation_id=compute_obligation_id(owed['session_key'], owed['message_ref'], LOST_CHAT_TURN_LINE),
            session_key=owed['session_key'], platform=source.platform.value, chat_id=source.chat_id,
            thread_id=source.thread_id, content=LOST_CHAT_TURN_LINE)
    return owed


def settle_recovered_chat_turns(store, epoch) -> list:
    """Boot recovery, beside ``settle_recovered_unattended``: settle every lost chat turn before any
    FIFO is scheduled. Returns the lines owed to the chats (sent once adapters connect)."""
    from gateway.session_ingress_media import release_admission_media
    from persistence.core.errors import AuthorityRefusal
    rows = _lost_chat_rows(store.admissions.list_unknown())
    if not rows or not auto_discard_enabled(store):
        return []
    owed = []
    for row in rows:
        try:
            store.admissions.resolve_unknown(epoch=epoch, admission_id=row['admission_id'],
                                             generation=row['generation'], notice=LOST_CHAT_TURN_NOTICE)
        except AuthorityRefusal as exc:
            logger.warning('Lost chat turn %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        release_admission_media(store, row['admission_id'])
        owed.append(owe_line(row))
        logger.info('Lost chat turn %s on session %s settled interrupted (not re-run), like /discard',
                    row['admission_id'], row['target_session_id'])
    return owed


async def settle_chat_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` chat head fenced in this owner: settle it through the owner's
    resolution and post the line before the queued messages run. True when any settled."""
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal
    store = authority_persistence(authority)
    rows = _lost_chat_rows(await asyncio.to_thread(store.admissions.list, session_id=ref.session_id))
    if not rows or not await asyncio.to_thread(auto_discard_enabled, store):
        return False
    owed, settled = [], False
    for row in rows:
        # A turn whose answer this owner still holds commits that answer instead (no line is owed).
        answered = row['admission_id'] in authority.pending_results
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'],
                                                     notice=LOST_CHAT_TURN_NOTICE)
        except AuthorityRefusal as exc:
            logger.warning('Lost chat turn %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
        if not answered:
            owed.append(await asyncio.to_thread(owe_line, row))
    await send_lines(authority.runner, owed)
    return settled


async def settle_lost_unknown(authority, ref) -> bool:
    """The drain's ``unknown`` head: automation rows (GW-10), then a person's chat turn."""
    from gateway.session_unattended_unknown import settle_unattended_unknown
    automation = await settle_unattended_unknown(authority, ref)
    return await settle_chat_unknown(authority, ref) or automation


async def send_lines(runner, owed) -> None:
    """Post each owed line through the adapter's ledgered final send (the CURRENT transport), in the
    chat and thread of the lost turn. No adapter now: the ledger row stays for the next boot."""
    from gateway.platforms.base import _thread_metadata_for_event
    from gateway.platforms.event import MessageEvent
    from gateway.session_envelope import restore_native_source
    for entry in owed:
        source = restore_native_source(entry['payload'], runner)
        adapter = runner._adapter_for_source(source)
        if adapter is None:
            logger.warning('No adapter for %s: the lost-turn line waits in the delivery ledger', source.platform)
            continue
        # Synthetic (``internal``): no reply anchor on a platform message id it does not have.
        event = MessageEvent(text='', source=source, internal=True, ledger_message_id=entry['message_ref'])
        try:
            await adapter.send_final_ledgered(event, entry['session_key'], LOST_CHAT_TURN_LINE,
                                              _thread_metadata_for_event(event), reply_to=None)
        except Exception:  # health: allow BLE001 -- logged; the ledger row is retried by the sweeps
            logger.exception('Lost-turn line for %s failed', entry['message_ref'])


async def send_owed_lines(runner, authority) -> None:
    """After *authority*'s adapters connected: the lines its boot recovery owes."""
    owed, authority.lost_chat_lines = getattr(authority, 'lost_chat_lines', []), []
    await send_lines(runner, owed)


async def adapter_pause_heals(authority, live) -> bool:
    """A messaging chat's head refused ``not_found`` (its adapter absent, or replaced during a
    reconnect): the pause heals by itself once the adapter connects, so it waits for that."""
    from gateway.session_context import NON_MESSAGING_SESSION_SURFACES
    from gateway.session_persistence import authority_persistence
    platform = str(getattr(getattr(live.source, 'platform', None), 'value', '') or '').lower()
    if platform in NON_MESSAGING_SESSION_SURFACES:
        return False
    return await asyncio.to_thread(auto_discard_enabled, authority_persistence(authority))


def redrive_adapter_pauses(runner, platform) -> None:
    """*platform*'s adapter connected (startup, a reconnect, a profile's reconnect): drain again
    every chat paused waiting for it, so its saved head runs without a new message."""
    from gateway.session_authorities import owner_scope, serving_authorities
    from gateway.session_contract import SessionRef
    for authority in serving_authorities(runner):
        waiting = [sid for sid, live in list(authority.sessions.items())
                   if live.awaits_adapter and getattr(live.source, 'platform', None) == platform]
        with owner_scope(authority):
            for sid in waiting:
                authority.sessions[sid].awaits_adapter = False  # a second not_found pause sets it again
                authority._schedule(SessionRef(authority.profile_id, sid))
        if waiting:
            logger.info('%s connected: %d paused chat(s) drain again', platform.value, len(waiting))
