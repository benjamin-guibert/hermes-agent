"""A background-process notice owed at a gateway restart reaches its session after the restart.

Incident shape (a SIGUSR1 drain-first restart): the stop's kill sweep killed a
``notify_on_complete`` watcher process; its completion reached admission while the gateway
drained and was refused (``runtime_draining``); the retry lived only in memory, the checkpoint
holds live PIDs only, so the new gateway sent nothing and the owning session slept for hours.
Carry of #146256 (replay owed completions from their receipt) and #154776 (death notice for a
lifecycle-killed process).
"""

import asyncio
import contextlib
import json
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms("posix")  # the watched job is a real `sleep` child

_OWNER = "sess-owner"


def _runner(monkeypatch, tmp_path, *, deliver) -> GatewayRunner:
    import gateway.run as gateway_run
    (tmp_path / "config.yaml").write_text("display:\n  background_process_notifications: concise\n",
                                          encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.DISCORD] = SimpleNamespace(send=AsyncMock(), _active_sessions={})
    runner._enqueue_process_completion_notification = AsyncMock(side_effect=deliver)
    return runner


def _receipt(session_id: str) -> dict:
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "logs" / "process-results" / f"{session_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fresh_registry(monkeypatch):
    import tools.process_registry as pr_module
    from tools.process_registry import ProcessRegistry
    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _watched_job(registry, session_id: str, command=("sleep", "600")):
    """A ``terminal(background=true, notify_on_complete=true)`` job with its gateway watcher."""
    from tools.process_registry import ProcessSession
    proc = subprocess.Popen(list(command))
    session = ProcessSession(
        id=session_id, command="poll-long-job.sh", task_id="task-1", owner_task_id="task-1",
        session_key="agent:main:discord:thread:111:222", parent_session_id=_OWNER,
        started_at=time.time(), notify_on_complete=True, watcher_interval=5,
        watcher_platform="discord", watcher_chat_id="111", watcher_thread_id="222",
        process=proc, pid=proc.pid, output_buffer="still waiting\n",
    )
    with registry._lock:
        registry._running[session.id] = session
    return session, {"session_id": session_id, "check_interval": 0, "session_key": session.session_key,
                     "platform": "discord", "chat_id": "111", "thread_id": "222",
                     "notify_on_complete": True, "parent_session_id": _OWNER}


@contextlib.contextmanager
def _instant_sleep(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_kw: real_sleep(0))
    try:
        yield
    finally:
        monkeypatch.setattr(asyncio, "sleep", real_sleep)


async def _stop_old_gateway(monkeypatch, tmp_path, session_id: str, *, cancel_watcher=True):
    """The old gateway drains, its shutdown sweep kills the job, and admission refuses the
    notice with ``runtime_draining``; then the stop cancels the watcher task (or, with
    ``cancel_watcher=False``, the watcher runs to its own end first)."""
    registry = _fresh_registry(monkeypatch)
    _session, watcher = _watched_job(registry, session_id)
    runner = _runner(monkeypatch, tmp_path, deliver=lambda *_a: False)  # not admitted: runtime_draining
    runner._draining = True
    task = asyncio.create_task(runner._run_process_watcher(watcher))
    assert await asyncio.to_thread(registry.kill_all, source="gateway_shutdown") == 1
    if cancel_watcher:
        await async_wait_until(lambda: runner._enqueue_process_completion_notification.await_count >= 1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    else:
        with _instant_sleep(monkeypatch):
            await asyncio.wait_for(task, timeout=30)
    return runner


async def _start_new_gateway(monkeypatch, tmp_path):
    """A fresh process on the same HERMES_HOME: checkpoint recovery, then every queued watcher."""
    registry = _fresh_registry(monkeypatch)
    registry.recover_from_checkpoint(restore_notices=True)
    runner = _runner(monkeypatch, tmp_path, deliver=lambda *_a: True)
    watchers, registry.pending_watchers = registry.pending_watchers, []
    with _instant_sleep(monkeypatch):
        for watcher in watchers:
            await runner._run_process_watcher(watcher)
    return runner


def _wakes(runner) -> list[tuple[str, dict]]:
    return [call.args for call in runner._enqueue_process_completion_notification.await_args_list]


@pytest.mark.asyncio
async def test_watcher_killed_while_draining_wakes_its_session_once_after_restart(monkeypatch, tmp_path):
    await _stop_old_gateway(monkeypatch, tmp_path, "proc_incident0001")
    # The kill left a receipt; nothing else of the notice survives the process.
    receipt = _receipt("proc_incident0001")
    assert (receipt["completion_reason"], receipt["termination_source"]) == ("killed", "gateway_shutdown")

    after = await _start_new_gateway(monkeypatch, tmp_path)

    [(synth_text, event)] = _wakes(after)  # one death notice, never also a "killed" completion
    assert (event["session_id"], event["parent_session_id"]) == ("proc_incident0001", _OWNER)
    assert (event["platform"], event["chat_id"], event["thread_id"]) == ("discord", "111", "222")
    assert "task interrupted" in synth_text and "gateway_shutdown" in synth_text
    assert "gateway restart" in synth_text and "SIGTERM" in synth_text
    # Delivered: the next restart owes nothing.
    assert _wakes(await _start_new_gateway(monkeypatch, tmp_path)) == []


@pytest.mark.asyncio
async def test_refusal_while_stopping_stays_owed_instead_of_falling_back(monkeypatch, tmp_path):
    """Retries and the plain chat notice are not spent on a gateway that is going away."""
    before = await _stop_old_gateway(monkeypatch, tmp_path, "proc_refused0001", cancel_watcher=False)

    before._enqueue_process_completion_notification.assert_awaited_once()
    before.adapters[Platform.DISCORD].send.assert_not_awaited()
    receipt = _receipt("proc_refused0001")
    assert receipt["notify_pending"] is True and receipt["interrupt_notice_state"] == "pending"
    assert len(_wakes(await _start_new_gateway(monkeypatch, tmp_path))) == 1


@pytest.mark.asyncio
async def test_job_that_exits_while_the_gateway_stops_is_delivered_after_restart(monkeypatch, tmp_path):
    """Its own exit, not a kill: an ordinary completion notice, replayed once."""
    registry = _fresh_registry(monkeypatch)
    session, _watcher = _watched_job(registry, "proc_exited00001", command=("true",))
    session.process.wait(timeout=10)
    registry._finish_exited(session, 0)  # the reader saw the exit; the watcher was already cancelled

    [(synth_text, event)] = _wakes(await _start_new_gateway(monkeypatch, tmp_path))

    assert (event["session_id"], event["parent_session_id"]) == ("proc_exited00001", _OWNER)
    assert "completed normally" in synth_text and "task interrupted" not in synth_text


@pytest.mark.asyncio
async def test_agent_close_kill_wakes_the_session_live_and_is_not_replayed(monkeypatch, tmp_path):
    """#154776: the agent_close sweep forces ``consume_output``; the death notice still goes out,
    once, while the gateway runs, and a later start does not send it again."""
    registry = _fresh_registry(monkeypatch)
    _session, watcher = _watched_job(registry, "proc_closed00001")
    live = _runner(monkeypatch, tmp_path, deliver=lambda *_a: True)
    registry.kill_process("proc_closed00001", source="agent_close", consume_output=True)
    assert registry.is_completion_consumed("proc_closed00001")

    await live._run_process_watcher(watcher)

    [(synth_text, _event)] = _wakes(live)
    assert "task interrupted" in synth_text and "agent_close" in synth_text
    assert _wakes(await _start_new_gateway(monkeypatch, tmp_path)) == []


@pytest.mark.asyncio
async def test_switch_off_keeps_the_base_behaviour(monkeypatch, tmp_path):
    """``gateway.process_notice_replay: false``: no replay after the restart and no death notice."""
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text("gateway:\n  process_notice_replay: false\n", encoding="utf-8")

    before = await _stop_old_gateway(monkeypatch, tmp_path, "proc_switchoff01", cancel_watcher=False)
    assert before._enqueue_process_completion_notification.await_count > 1  # base retries, then falls back
    assert _wakes(await _start_new_gateway(monkeypatch, tmp_path)) == []

    registry = _fresh_registry(monkeypatch)
    _session, watcher = _watched_job(registry, "proc_switchoff02")
    live = _runner(monkeypatch, tmp_path, deliver=lambda *_a: True)
    registry.kill_process("proc_switchoff02", source="agent_close", consume_output=True)
    await live._run_process_watcher(watcher)
    assert _wakes(live) == []  # consumed by the sweep, as on the base
