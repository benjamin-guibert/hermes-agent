"""Shared harness for the final-delivery claim tests (``gateway.final_delivery_claims``): a platform
whose ACK the test releases, a restart-ready runner, the delivery-ledger rows, the next boot's sweep,
and a real session authority whose turns the adapter waits on."""

import asyncio
import itertools
from unittest.mock import AsyncMock

from gateway import delivery_ledger
from gateway.platforms.base import MessageEvent, SendResult
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner, make_restart_source

REPLY = "the final answer"


class SlowAckAdapter(RestartTestAdapter):
    """The platform takes the message at once; its ACK reaches the gateway only on ``ack``."""

    def __init__(self):
        super().__init__()
        self.on_platform = asyncio.Event()
        self.ack = asyncio.Event()
        self.acks: dict[str, asyncio.Event] = {}  # per chat; ``ack`` for any other
        self.result = SendResult(success=True, message_id="m1")
        self.raises: BaseException | None = None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.on_platform.set()
        await self.acks.get(chat_id, self.ack).wait()
        if self.raises is not None:
            raise self.raises
        return self.result


def make_event(message_id="in-1", text="please answer", chat_id="123456"):
    return MessageEvent(text=text, source=make_restart_source(chat_id=chat_id), message_id=message_id)


def make_gateway(adapter=None):
    adapter = adapter or SlowAckAdapter()
    runner, _ = make_restart_runner(adapter)
    adapter.gateway_runner = runner
    adapter.set_message_handler(AsyncMock(return_value=REPLY))  # the turn is done: its slot is free
    return runner, adapter


def process(runner, adapter, event):
    """The adapter's processing task, tracked like ``handle_message`` tracks it (stop() tears it down)."""
    task = asyncio.create_task(
        adapter._process_message_background(event, runner._session_key_for_source(event.source)))
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    return task


def ledger_rows():
    from persistence_assembly import open_persistence
    from tests.fixtures.storage_holder import raw_holder
    store = open_persistence(delivery_ledger._scope())
    try:
        with raw_holder(store)._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'delivery_obligations'").fetchone():
                return []  # the ledger creates its table on its first call: nothing was ledgered
            return [tuple(row) for row in conn.execute("SELECT state, content FROM delivery_obligations")]
    finally:
        store.close()


def boot_sweep_of_the_next_process(monkeypatch):
    monkeypatch.setattr(delivery_ledger, "_owner_alive", lambda *_: False)  # the old process is gone
    return delivery_ledger.sweep_recoverable()


async def held_after_a_while(task) -> bool:
    """True when *task* is still waiting after 2 s (an absence proof: no early stop)."""
    done, _ = await asyncio.wait({task}, timeout=2.0)
    return not done


def quiet_stop_teardown(monkeypatch):
    """The host-wide parts of a real ``stop()`` (tool processes, environments, browsers, PID file)."""
    from tools import browser_session_lifecycle, process_registry, terminal_tool, terminal_tool_lifecycle
    monkeypatch.setattr(process_registry.process_registry, "kill_all", lambda *a, **k: 0)
    monkeypatch.setattr(terminal_tool, "cleanup_all_environments", lambda: None)
    monkeypatch.setattr(terminal_tool_lifecycle, "cleanup_all_environments", lambda: None)
    monkeypatch.setattr(browser_session_lifecycle, "cleanup_all_browsers", lambda: None)
    from gateway import status
    monkeypatch.setattr(status, "remove_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(status, "publish_runtime_status", lambda *a, **k: None)


async def authority_gateway(tmp_path, monkeypatch, run_turn):
    """A runner whose chat turns run in a real session authority: the adapter's handler is
    ``admit_message``, the turn is ``run_turn(event)``. Returns ``(runner, adapter, authority, source)``."""
    from gateway import session_ingress
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from persistence.conversation.sessions import NewSession

    runner, adapter = make_gateway()
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner._session_db, runner.session_store, runner._handle_message = store.persistence, store, run_turn
    runner._adapter_for_source = lambda _source: adapter
    authority = await initialize_session_authority(runner, profile_id="p", instance_id="owner")
    runner.session_authority = authority
    store.persistence.sessions.create(NewSession(session_id="s", source="telegram"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c")
    authority.sessions["s"] = LiveSession(source, "s")
    actor = SessionActor("human", "p", frozenset({"session:submit"}), "t")
    request_ids = itertools.count(1)

    async def admit_native(event):
        payload = {"text": event.text if isinstance(event.text, str) else "hi"}
        return await authority.submit(actor, Submission(f"m{next(request_ids)}", SessionRef("p", "s"), payload, "queue"))
    monkeypatch.setattr(authority, "admit_native", admit_native)
    adapter.set_message_handler(lambda event: session_ingress.admit_message(authority, event))
    return runner, adapter, authority, source


async def settle(adapter, authority):
    """Every adapter task (follow-up drains included), the session's own drain and the authority's
    tracked mutations (a turn's settle-and-deliver) are done; a follow-up starts new ones, so all
    are re-read until none is left."""
    from gateway.session_runtime_workers import mutation_tasks
    while True:
        session_task = authority.sessions["s"].task
        pending = list(adapter._background_tasks) + mutation_tasks(authority) + (
            [session_task] if session_task is not None and not session_task.done() else [])
        if not pending:
            return
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 10)
