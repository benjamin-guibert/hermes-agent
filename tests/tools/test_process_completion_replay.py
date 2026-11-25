"""A gateway ``notify_on_complete`` completion survives a gateway restart (carry of #146256).

The completion queue and the per-process watcher are in-memory and the checkpoint holds live
PIDs only, so a process that exited while the gateway was down or stopping left a receipt and
no notification, permanently. Startup now re-arms a watcher from the receipt until a watcher
settles it.
"""

import time

from hermes_constants import get_hermes_home
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import mark_result_notified


def _finish(registry, session_id, *, watcher_interval=5):
    """Exit a tracked process the way the reader thread does; the gateway armed its watcher."""
    session = ProcessSession(
        id=session_id, command="echo review", task_id="task-1", owner_task_id="task-1",
        session_key="agent:main:telegram:group:-100999:7", parent_session_id="sess-owner",
        started_at=time.time(), notify_on_complete=True, watcher_interval=watcher_interval,
        watcher_platform="telegram", watcher_chat_id="-100999", watcher_thread_id="7",
        output_buffer="REVIEW DONE\n",
    )
    with registry._lock:
        registry._running[session.id] = session
    session.mark_exited(0)
    registry._move_to_finished(session)


def test_completion_finished_while_gateway_down_is_rearmed_until_settled():
    _finish(ProcessRegistry(), "proc_replay000001")

    restarted = ProcessRegistry()  # the queued completion died with the old process
    assert restarted.completion_queue.empty()
    assert restarted.restore_pending_notifications() == 1
    assert restarted.restore_pending_notifications() == 0  # never armed twice in one process
    [watcher] = restarted.pending_watchers
    assert watcher["session_id"] == "proc_replay000001"
    assert watcher["notify_on_complete"] is True
    assert (watcher["session_key"], watcher["platform"], watcher["chat_id"], watcher["thread_id"]) == (
        "agent:main:telegram:group:-100999:7", "telegram", "-100999", "7")
    assert watcher["parent_session_id"] == "sess-owner"
    # An exited snapshot: never re-run, never adopted as a live PID.
    assert "proc_replay000001" not in restarted._running
    restored = restarted.get("proc_replay000001")
    assert restored.exited and restored.pid is None and restored.process is None
    assert "REVIEW DONE" in restored.output_buffer

    mark_result_notified("proc_replay000001")
    assert ProcessRegistry().restore_pending_notifications() == 0


def test_completion_without_a_gateway_watcher_is_not_rearmed():
    """A CLI process has no watcher route; replaying it from a later gateway would invent one."""
    _finish(ProcessRegistry(), "proc_replay000002", watcher_interval=0)
    assert (get_hermes_home() / "logs" / "process-results" / "proc_replay000002.json").exists()

    restarted = ProcessRegistry()
    assert restarted.restore_pending_notifications() == 0
    assert restarted.pending_watchers == []


def _age_receipt(session_id, hours):
    import json
    path = get_hermes_home() / "logs" / "process-results" / f"{session_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["completed_at"] = time.time() - hours * 3600
    path.write_text(json.dumps(record), encoding="utf-8")


def test_receipts_past_48_hours_are_not_replayed_but_stay_readable(monkeypatch):
    """Past the cap a replay would start a turn nobody waits on; process_manage still reads it."""
    from tools.process_registry_results import load_completed_results

    registry = ProcessRegistry()
    _finish(registry, "proc_replay0fresh")
    _finish(registry, "proc_replay0stale")
    killed = registry.get("proc_replay0stale")
    killed.completion_reason, killed.termination_source = "killed", "gateway_shutdown"
    from tools.process_registry_results import save_completed_result
    save_completed_result(killed)  # a death notice is capped the same way
    _age_receipt("proc_replay0fresh", 47)
    _age_receipt("proc_replay0stale", 49)

    restarted = ProcessRegistry()
    assert restarted.restore_pending_notifications() == 1
    assert [w["session_id"] for w in restarted.pending_watchers] == ["proc_replay0fresh"]
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-owner")
    assert "proc_replay0stale" in load_completed_results("proc_replay0stale")


def test_managed_worker_recovery_never_arms_an_owed_notice():
    """A managed worker adopts the checkpoint's processes; only the gateway replays notices, or
    one completion would wake its session twice."""
    _finish(ProcessRegistry(), "proc_replay0worker")

    worker = ProcessRegistry()
    worker.recover_from_checkpoint()  # agent/managed_worker.py
    assert worker.pending_watchers == []
    gateway = ProcessRegistry()
    gateway.recover_from_checkpoint(restore_notices=True)  # gateway startup
    assert [w["session_id"] for w in gateway.pending_watchers] == ["proc_replay0worker"]
