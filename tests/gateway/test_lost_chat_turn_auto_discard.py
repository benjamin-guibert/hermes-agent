"""A person's turn lost in a messaging chat settles like ``/discard`` (carry for v0.23.3).

Incident: two gateway restarts cut a person's running turn in a Discord thread. Boot recovery fenced
the admission ``unknown`` and the FIFO paused; the chat got one pause notice, and every later message
got only a "saved" reaction until an operator ran ``hermes sessions discard``. In a messaging chat the
only ways out are ``/discard`` (drops the lost turn) and ``/reset``, so the automatic outcome is what
``/discard`` does: the same fenced resolution, terminal ``interrupted``, never re-run, a transcript
notice for the next turn, one visible line in the chat, and the queued messages run in order.
``gateway.messaging_auto_discard_lost_turns: false`` keeps the base behaviour. Desktop, TUI and API
sessions keep their pause and Discard control.

Real ``SessionStore`` SQLite, real owner epochs, a real ``BasePlatformAdapter`` double.
"""
import asyncio
import json
import queue
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.waiting import async_wait_until

PAUSE_MARKER = 'This conversation is paused'
LINE_MARKER = 'was not re-run'


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


class ChatAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token='***'), platform)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, (metadata or {}).get('thread_id')))
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {'id': chat_id}


def _source(platform=Platform.DISCORD):
    return SessionSource(platform=platform, chat_id='chan', chat_type='thread', thread_id='thr', user_id='person')


def _runner(tmp_path, monkeypatch, executed, adapter, switch=None):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    if switch is not None:
        (tmp_path / 'config.yaml').write_text(
            json.dumps({'gateway': {'messaging_auto_discard_lost_turns': switch}}), encoding='utf-8')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store.persistence, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: adapter)
    return store, runner


def _admit_chat_person(authority, source, text, message_id):
    """A person's message as a messaging adapter commits it: a ``messaging:`` principal and the
    native envelope that carries the chat (platform, channel, thread)."""
    encoded = {**source.to_dict(), 'is_bot': False}
    envelope = {'source': encoded, 'route': 'agent:main:discord:thread:chan:thr',
                'event': {'message_id': message_id}, 'timestamp': '2026-11-21T10:00:00+00:00'}
    row, _created = authority.persistence.admissions.admit(
        epoch=authority.epoch, principal_id='messaging:["discord","person"]', session_id='s',
        request_id=message_id, payload={'text': text, 'native_text_v1': envelope})
    return row.admission_id


async def _queue(authority, texts):
    """Messages queued behind the head (their own kind does not matter to the FIFO)."""
    actor = AuthorityConnection(authority, Peer(), {'user_id': 'human'}).actor
    ref = SessionRef('p', 's')
    return [(await authority.submit(actor, Submission(t, ref, {'text': t}, 'queue'))).admission_id for t in texts]


def _statuses(db):
    return {r.admission_id: (r.status, r.outcome) for r in db.admissions.list(session_id='s', pending_only=False)}


async def _owner(runner, name, source):
    authority = await initialize_session_authority(runner, profile_id='p', instance_id=name)
    authority.sessions['s'] = LiveSession(source, 's')
    return authority


async def _lost_chat_head_then_restart(tmp_path, monkeypatch, executed, adapter, *, switch=None,
                                       platform=Platform.DISCORD, chat_head=True):
    """Owner 1 commits a person's message (the head) and two queued followers, claims the head and
    dies mid-turn. Owner 2 boots on the same store. Returns (store, runner, owner2, head_id, followers)."""
    store, runner = _runner(tmp_path, monkeypatch, executed, adapter, switch)
    db = store.persistence
    source = _source(platform)
    seed_session(db, 's', platform.value)
    first = await _owner(runner, 'owner-1', source)
    first._schedule = lambda ref: None
    ref = SessionRef('p', 's')
    if chat_head:
        head_id = _admit_chat_person(first, source, 'HEAD', 'm-head')
    else:
        actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
        head_id = (await first.submit(actor, Submission('head', ref, {'text': 'HEAD'}, 'queue'))).admission_id
    followers = await _queue(first, ('TWO', 'THREE'))
    assert db.admissions.claim(epoch=first.epoch, session_id='s').admission_id == head_id
    seed_messages(db, 's', {'role': 'user', 'content': 'HEAD'})  # the lost turn reached its user row
    second = await _owner(runner, 'owner-2', source)
    return store, runner, second, head_id, followers


async def _connect_and_drain(runner, authority):
    """What boot does once the adapters connected: native-session recovery, then the FIFO runs."""
    from gateway.run_runtime import recover_profile_native_sessions
    await recover_profile_native_sessions(runner, authority)
    authority._schedule(SessionRef('p', 's'))
    live = authority.sessions['s']
    await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')


def _lines(adapter):
    return [sent for sent in adapter.sent if LINE_MARKER in sent[1]]


@pytest.mark.asyncio
async def test_boot_settles_a_discord_persons_lost_turn_and_the_queue_runs_once(tmp_path, monkeypatch):
    """Step 1 (boot recovery path) with R1, R2, R3 and R6: one restart gives one settle, one visible
    line in the thread, each queued message answered once; a second restart repeats nothing."""
    executed, adapter = [], ChatAdapter()
    store, runner, authority, head_id, followers = await _lost_chat_head_then_restart(
        tmp_path, monkeypatch, executed, adapter)
    try:
        db = authority.persistence
        assert _statuses(db)[head_id] == ('terminal', 'interrupted')
        await _connect_and_drain(runner, authority)
        assert executed == ['TWO', 'THREE']
        assert all(_statuses(db)[f] == ('terminal', 'completed') for f in followers)
        # R3: one short line, in the chat's own thread, before the answers.
        assert _lines(adapter) == [('chan', _lines(adapter)[0][1], 'thr')], adapter.sent
        assert adapter.sent.index(_lines(adapter)[0]) == 0, adapter.sent
        # R2: the next turn reads why the lost turn stopped and what to check first.
        notices = [m['content'] for m in transcript_rows(db, 's') if m.get('display_kind') == 'failed_turn']
        assert len(notices) == 1, notices
        assert 'restart' in notices[0] and 'not run again' in notices[0] and 'files' in notices[0], notices
        # A second restart: nothing is settled, sent or run again.
        third = await _owner(runner, 'owner-3', _source())
        await _connect_and_drain(runner, third)
        assert executed == ['TWO', 'THREE'] and len(_lines(adapter)) == 1, adapter.sent
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_messages_after_the_restart_are_answered_not_parked_behind_one_notice(tmp_path, monkeypatch):
    """Step 1 (the incident at the ingress): the chat's next messages arrive after the restart. On
    the base the first gets the pause notice, the second nothing, and neither gets a turn."""
    from gateway.session_authority import SessionAuthority
    from gateway.platforms.event import MessageEvent
    from gateway.session_contract import SessionActor
    from gateway.session_ingress import admit_message
    executed, adapter = [], ChatAdapter()
    store, _runner, authority, _head_id, _followers = await _lost_chat_head_then_restart(
        tmp_path, monkeypatch, executed, adapter)
    try:
        actor = SessionActor('human', 'p', frozenset({'session:submit'}), 'fixture')
        ref = SessionRef('p', 's')

        async def admit_native(event):
            return await authority.submit(actor, Submission(event.text, ref, {'text': event.text}, 'queue'))
        monkeypatch.setattr(authority, 'admit_native', admit_native)
        authority._schedule = lambda ref: SessionAuthority._schedule(authority, ref)
        source = authority.sessions['s'].source
        replies = await asyncio.wait_for(asyncio.gather(
            admit_message(authority, MessageEvent(text='FOUR', source=source)),
            admit_message(authority, MessageEvent(text='FIVE', source=source))), 10)
        assert not any(r and PAUSE_MARKER in r for r in replies), replies
        await async_wait_until(lambda: authority.sessions['s'].task.done(), what='drain finished')
        assert executed == ['TWO', 'THREE', 'FOUR', 'FIVE'], (executed, replies)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_lost_managed_worker_settles_a_chat_turn_on_its_own_exit(tmp_path, monkeypatch):
    """Step 1 (managed-worker exit path): a worker lost mid-turn fences the person's chat turn
    ``unknown`` and raises ManagedExecutionUnknown; the same drain settles it, posts the line and
    runs the queued messages."""
    from gateway.session_managed_worker import ManagedExecutionUnknown
    from gateway.session_settlement_recovery import recover_failed_settlement
    executed, adapter = [], ChatAdapter()
    store, runner = _runner(tmp_path, monkeypatch, executed, adapter)
    seed_session(store.persistence, 's', 'discord')
    authority = await _owner(runner, 'owner-1', _source())
    try:
        authority._schedule = lambda ref: None
        head_id = _admit_chat_person(authority, _source(), 'HEAD', 'm-head')
        followers = await _queue(authority, ('TWO', 'THREE'))

        async def execute(owner, ref, row):
            if row['admission_id'] == head_id:
                await recover_failed_settlement(owner, row)
                raise ManagedExecutionUnknown('managed_worker_unknown')
            executed.append(row['payload']['text'])
            return 'ACK'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)

        async def route_ok(_runner, payload, *_args):  # the chat's route preflight is not under test here
            from gateway.session_envelope import restore_native_source
            return restore_native_source(payload), payload['native_text_v1']['route']
        monkeypatch.setattr('gateway.session_envelope.check_native_route', route_ok)
        del authority._schedule
        authority._schedule(SessionRef('p', 's'))
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        statuses = _statuses(authority.persistence)
        assert statuses[head_id] == ('terminal', 'interrupted')
        assert executed == ['TWO', 'THREE'] and all(statuses[f][0] == 'terminal' for f in followers)
        assert len(_lines(adapter)) == 1, adapter.sent
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_chat_turn_fenced_unknown_in_this_owner_settles_on_the_claim(tmp_path, monkeypatch):
    """The claim path: a head this owner committed ``unknown`` settles on the next claim."""
    executed, adapter = [], ChatAdapter()
    store, runner = _runner(tmp_path, monkeypatch, executed, adapter)
    db = store.persistence
    seed_session(db, 's', 'discord')
    authority = await _owner(runner, 'owner-1', _source())
    try:
        authority._schedule = lambda ref: None
        head_id = _admit_chat_person(authority, _source(), 'HEAD', 'm-head')
        await _queue(authority, ('TWO',))
        claimed = db.admissions.claim(epoch=authority.epoch, session_id='s')
        assert db.admissions.mark_unknown(epoch=authority.epoch, admission_id=head_id,
                                          generation=claimed.generation) == 'unknown'
        del authority._schedule
        authority._schedule(SessionRef('p', 's'))
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        assert _statuses(db)[head_id] == ('terminal', 'interrupted')
        assert executed == ['TWO'] and len(_lines(adapter)) == 1, adapter.sent
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('platform,chat_head', [(Platform.API_SERVER, True), (Platform.DISCORD, False)],
                         ids=['api_server', 'desktop-or-tui-viewer'])
async def test_desktop_api_and_viewer_turns_still_pause(tmp_path, monkeypatch, platform, chat_head):
    """R4: a turn not sent from a messaging chat (Desktop/TUI, the API server, a local viewer's
    message into a chat session) keeps today's pause and Discard control. A Desktop/TUI submit has
    the same shape in any session: the viewer's principal and no native chat envelope."""
    executed, adapter = [], ChatAdapter()
    store, runner, authority, head_id, followers = await _lost_chat_head_then_restart(
        tmp_path, monkeypatch, executed, adapter, platform=platform, chat_head=chat_head)
    try:
        assert _statuses(authority.persistence)[head_id] == ('unknown', None)
        await _connect_and_drain(runner, authority)
        assert executed == [] and adapter.sent == []
        assert all(_statuses(authority.persistence)[f] == ('queued', None) for f in followers)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_switch_off_keeps_the_pause(tmp_path, monkeypatch):
    """R7: ``gateway.messaging_auto_discard_lost_turns: false`` is the base behaviour."""
    executed, adapter = [], ChatAdapter()
    store, runner, authority, head_id, followers = await _lost_chat_head_then_restart(
        tmp_path, monkeypatch, executed, adapter, switch=False)
    try:
        assert _statuses(authority.persistence)[head_id] == ('unknown', None)
        await _connect_and_drain(runner, authority)
        assert executed == [] and adapter.sent == []
        assert all(_statuses(authority.persistence)[f] == ('queued', None) for f in followers)
    finally:
        store.close_all_db_handles()


async def _head_paused_on_a_dropped_adapter(tmp_path, monkeypatch, executed, adapter, switch=None):
    """A chat's saved head meets no Discord adapter (retired by a reconnect, or not connected yet):
    the preflight refuses it ``not_found`` and the FIFO pauses. Returns (store, runner, owner, head)."""
    from gateway.session_envelope import restore_native_source
    from persistence.core.errors import AuthorityRefusal
    store, runner = _runner(tmp_path, monkeypatch, executed, adapter, switch)
    seed_session(store.persistence, 's', 'discord')
    authority = await _owner(runner, 'owner-1', _source())
    runner.session_authority = authority
    runner.adapters = {}
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._sync_voice_mode_state_to_adapter = runner._bind_voice_input_callback = lambda adapter: None

    async def route(_runner, payload, _target, _available, current):  # the adapter check of check_native_route
        if current is None:
            raise AuthorityRefusal('not_found')
        return restore_native_source(payload), payload['native_text_v1']['route']
    monkeypatch.setattr('gateway.session_envelope.check_native_route', route)

    async def execute(owner, ref, row):
        executed.append(row['payload']['text'])
        return 'ACK'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    head_id = _admit_chat_person(authority, _source(), 'HEAD', 'm-head')
    authority._schedule(SessionRef('p', 's'))
    live = authority.sessions['s']
    await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain paused')
    assert executed == [] and _statuses(store.persistence)[head_id] == ('queued', None)
    return store, runner, authority, head_id


@pytest.mark.asyncio
async def test_adapter_reconnect_runs_a_head_paused_on_not_found_without_a_new_message(tmp_path, monkeypatch):
    """R5: the chat never stays silent. The saved head runs once the adapter is published again
    (startup connect and every primary reconnect go through ``_publish_primary_adapter``), with no
    new inbound, and is answered once; the notice says it waits for the connection."""
    from gateway.run import GatewayRunner
    from gateway.session_ingress import pause_notice
    executed, adapter = [], ChatAdapter()
    store, runner, authority, head_id = await _head_paused_on_a_dropped_adapter(
        tmp_path, monkeypatch, executed, adapter)
    try:
        notice = pause_notice(authority, SessionRef('p', 's'), 'not_found')
        assert 'saved' in notice and 'when it is back' in notice and 'will not be answered' not in notice, notice
        GatewayRunner._publish_primary_adapter(runner, Platform.DISCORD, adapter)
        await async_wait_until(lambda: _statuses(store.persistence)[head_id][0] == 'terminal', what='head answered')
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task.done(), what='drain finished')
        GatewayRunner._publish_primary_adapter(runner, Platform.DISCORD, adapter)  # a later reconnect: no rerun
        await async_wait_until(lambda: live.task.done(), what='no second drain left running')
        assert executed == ['HEAD'] and _statuses(store.persistence)[head_id] == ('terminal', 'completed')
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_switch_off_keeps_the_not_found_pause(tmp_path, monkeypatch):
    """R7 for R5: with the switch off the reconnect does not drain the chat and the notice is the base text."""
    from gateway.run import GatewayRunner
    from gateway.session_ingress import pause_notice
    executed, adapter = [], ChatAdapter()
    store, runner, authority, head_id = await _head_paused_on_a_dropped_adapter(
        tmp_path, monkeypatch, executed, adapter, switch=False)
    try:
        assert 'will not be answered here' in pause_notice(authority, SessionRef('p', 's'), 'not_found')
        paused = authority.sessions['s'].task
        GatewayRunner._publish_primary_adapter(runner, Platform.DISCORD, adapter)
        assert authority.sessions['s'].task is paused and executed == []
        assert _statuses(store.persistence)[head_id] == ('queued', None)
    finally:
        store.close_all_db_handles()
