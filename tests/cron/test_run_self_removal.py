"""A cron run's own child process (agent terminal tool or job script) must not remove its job.

The removal used to cancel the live run: its final response was never delivered and no incident
was booked, because the record the failure watcher reads was gone. Covers both execution modes:
the restart-safe external worker (``_HERMES_CRON_EXTERNAL_WORKER``, carry of #143995) and the
in-process ticker. Each fire runs the real ``hermes cron`` CLI in a real child process whose env
comes from the production builder (the terminal tool, or the cron script runner); only the model
turn and the owner's RPC hop are stood in for.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
FINAL = "Final report"
MODES = ["in-process", "worker"]
CHILDREN = ["agent-terminal", "job-script"]
IDENTITY_VARS = ("HERMES_CRON_EXECUTION_ID", "_HERMES_CRON_EXTERNAL_WORKER")


def _refusal(job_id):
    return (f"Refused: job {job_id} is the job of the run that calls this command. A run must not "
            f"remove its own job: pause it (hermes cron pause {job_id}), or let a --repeat 1 job "
            "finish by itself.")


@pytest.fixture
def home(tmp_path, monkeypatch):
    from cron import delivery_queue, executions

    home = tmp_path / "home"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    for name in IDENTITY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    return home


def _snapshot(job_id, execution_id):
    """What a refused removal must leave alone (heartbeat timestamps aside)."""
    from cron import cancellation, executions, jobs, notepad

    job = jobs.get_job(job_id)
    row = executions.get_execution(execution_id)
    return {
        "job": job and {k: v for k, v in job.items() if k != "fire_claim"},
        "claim_owner": job and (job.get("fire_claim") or {}).get("by"),
        "outputs": sorted(p.name for p in jobs._job_output_dir(job_id).glob("*")),
        "notes": [(n["key"], n["value"]) for n in notepad.list_notes(job_id)],
        "ledger": row and (row["job_id"], row["status"], row["error"]),
        "cancel_requested": cancellation.cancel_requested(execution_id),
    }


@pytest.fixture
def run(home, monkeypatch):
    """A runner that fires one job whose child runs ``hermes cron <argv>``."""
    from cron import delivery_queue, executions, jobs, notepad, scheduler, scheduler_authority
    from gateway import session_cron
    from tools.terminal_tool_lifecycle import cleanup_all_environments

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    delivered = []

    def deliver(job, content, *args, **kwargs):
        delivered.append(content)

    # A script job sends in-process; an agent run's result comes back through the durable queue.
    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    state = SimpleNamespace(home=home, delivered=delivered)

    def cli(job):
        out = home / "child.out"
        unset = "".join(f" -u {name}" for name in IDENTITY_VARS) if state.independent else ""
        argv = state.argv.format(id=job["id"], other=state.other)
        return (f"cd {shlex.quote(str(REPO))} && env{unset} {shlex.quote(sys.executable)} "
                f"-m hermes_cli.main cron {argv} >{shlex.quote(str(out))} 2>&1; "
                f"echo $? >{shlex.quote(str(out))}.rc")

    def agent_turn(job, execution_id=None, **kwargs):
        """The model's one tool call; the session owner (the gateway) runs it, so in worker
        mode its process env does not carry the worker's marker."""
        from tools.terminal_tool import terminal_tool

        state.before = _snapshot(job["id"], execution_id)
        with monkeypatch.context() as owner_env:
            owner_env.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
            result = json.loads(terminal_tool(
                command=cli(job), task_id=f"cron:{job['id']}:{execution_id}"))
        state.after = _snapshot(job["id"], execution_id)
        assert result.get("exit_code") == 0, result
        return True, "Run archive", FINAL, None

    def owner(job, *, extra_prompt=None, cancel_event=None, execution_id=None):
        # The owner side of run_canonical_job's transport (gateway.session_cron.execute), minus RPC.
        return session_cron._run_identified(agent_turn, job, execution_id)

    monkeypatch.setattr(scheduler_authority, "run_canonical_job", owner)
    cleanup_all_environments()  # a backend cached by an earlier test holds that test's home

    def fire(mode, child, argv, *, config="", independent=False):
        state.argv, state.independent = argv, independent
        if config:
            (home / "config.yaml").write_text(config, encoding="utf-8")
        with jobs.use_cron_store(home):
            state.other = jobs.create_job(prompt="Another job", schedule="every 5m")["id"]
            if child == "job-script":
                script = home / "scripts" / "child.sh"
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                job = jobs.create_job(prompt=None, schedule="every 5m", script="child.sh",
                                      no_agent=True, deliver="telegram")
                script.write_text(f"#!/usr/bin/env bash\n{cli(job)}\necho {FINAL}\n", encoding="utf-8")
            else:
                job = jobs.create_job(prompt="Produce a report", schedule="every 5m",
                                      deliver="telegram")
            assert jobs.claim_job_for_fire(job["id"])
            job = jobs.get_job(job["id"])
            output = jobs.save_job_output(job["id"], "Earlier output")
            notepad.set_note(job["id"], "seen", "42")
            with monkeypatch.context() as worker_env:
                if mode == "worker":  # what _run_external_cron_worker does around run_one_job
                    job["execution_id"] = executions.create_execution(job["id"], source="test")["id"]
                    worker_env.setenv("_HERMES_CRON_EXTERNAL_WORKER", job["execution_id"])
                assert scheduler.run_one_job(job)
            delivery_queue.drain(deliver)
            out = home / "child.out"
            state.job = job
            state.execution = executions.get_execution(job["execution_id"])
            assert out.exists(), state.execution
            rc = Path(f"{out}.rc")  # absent: the run was cancelled and its script tree-killed
            state.child = SimpleNamespace(
                returncode=int(rc.read_text(encoding="utf-8")) if rc.exists() else None,
                output=out.read_text(encoding="utf-8"))
            state.output = output
            state.stored = jobs.get_job(job["id"])
            state.other_stored = jobs.get_job(state.other)
            state.notes = [(n["key"], n["value"]) for n in notepad.list_notes(job["id"])]
        return state

    yield fire
    cleanup_all_environments()


def _assert_delivered(state, status="scheduled"):
    assert state.delivered == [FINAL]
    assert state.execution["status"] == "completed"
    assert state.stored["last_status"] == "ok"
    assert state.stored["state"] == status


def _assert_base_cancellation(state):
    """Today's outside removal: the record goes and the live run stops without delivering."""
    assert state.child.returncode in (0, None), state.child.output  # None: script tree-killed
    assert "Removed job" in state.child.output
    assert state.stored is None
    assert state.delivered == []
    assert state.execution["status"] == "cancelled"
    assert state.execution["error"] == "job removed while running"


@pytest.mark.parametrize("child", CHILDREN)
@pytest.mark.parametrize("mode", MODES)
def test_run_cannot_remove_its_own_job_and_still_delivers(run, mode, child):
    state = run(mode, child, "remove {id}")
    assert state.child.returncode not in (0, None), f"the run's own child removed its live job: {state}"
    lines = [line for line in state.child.output.splitlines() if "Refused" in line]
    assert len(lines) == 1, state.child.output
    assert _refusal(state.job["id"]) in lines[0]
    assert state.stored is not None
    assert state.output.read_text(encoding="utf-8") == "Earlier output"
    assert state.notes == [("seen", "42")]
    _assert_delivered(state)


@pytest.mark.parametrize("mode", MODES)
def test_refused_removal_mutates_nothing_while_the_run_is_live(run, mode):
    state = run(mode, "agent-terminal", "remove {id}")
    assert state.child.returncode == 1, state.child.output
    assert state.before["job"] is not None and state.before["ledger"][1] in {"claimed", "running"}
    assert state.after == state.before


@pytest.mark.parametrize("child", CHILDREN)
@pytest.mark.parametrize("mode", MODES)
def test_pause_from_the_run_keeps_the_job_and_delivers(run, mode, child):
    state = run(mode, child, "pause {id}")
    assert state.child.returncode == 0, state.child.output
    _assert_delivered(state, status="paused")


@pytest.mark.parametrize("child", CHILDREN)
@pytest.mark.parametrize("mode", MODES)
def test_run_may_remove_another_job(run, mode, child):
    state = run(mode, child, "remove {other}")
    assert state.child.returncode == 0, state.child.output
    assert state.other_stored is None
    _assert_delivered(state)


@pytest.mark.parametrize("mode", MODES)
def test_removal_without_run_identity_still_stops_the_live_run(run, mode):
    _assert_base_cancellation(run(mode, "agent-terminal", "remove {id}", independent=True))


@pytest.mark.parametrize("mode", MODES)
def test_off_switch_restores_the_base_behaviour(run, mode):
    state = run(mode, "agent-terminal", "remove {id}",
                config="cron:\n  refuse_self_removal: false\n")
    _assert_base_cancellation(state)


def _cli_remove(home, job_id, identity):
    """A ``hermes cron remove`` child holding *identity* as its inherited run identity."""
    from tools.environments.local import build_subprocess_env

    env = build_subprocess_env(dict(os.environ))
    for name in IDENTITY_VARS:
        env.pop(name, None)
    env["HERMES_CRON_EXECUTION_ID"] = identity
    env["HERMES_HOME"] = str(home)
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "cron", "remove", job_id],
                          cwd=str(REPO), env=env, text=True, capture_output=True, timeout=120,
                          check=False)


@pytest.mark.parametrize("identity", ["ended-run", "other-job", "other-profile"])
def test_stale_or_foreign_run_identity_does_not_refuse(home, tmp_path, monkeypatch, identity):
    from cron import executions, jobs

    with jobs.use_cron_store(home):
        job = jobs.create_job(prompt="Produce a report", schedule="every 5m")
        other = jobs.create_job(prompt="Another job", schedule="every 5m")
        target = other["id"] if identity == "other-job" else job["id"]
        with monkeypatch.context() as scoped:
            if identity == "other-profile":  # same job id, live in a sibling profile's ledger
                scoped.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
            execution = executions.create_execution(job["id"], source="test")
            assert executions.mark_execution_running(execution["id"])
            if identity == "ended-run":
                assert executions.finish_execution(execution["id"], success=True)
        child = _cli_remove(home, target, execution["id"])
        assert child.returncode == 0, child.stdout + child.stderr
        assert jobs.get_job(target) is None


def test_in_process_self_removal_keeps_its_run_local_permission(home, monkeypatch):
    """The cronjob tool inside the run's own context removes the job and keeps its delivery."""
    from cron import executions, jobs
    from cron.jobs_removal import run_identity_scope
    from tools.cronjob_tools import cronjob

    with jobs.use_cron_store(home):
        job = jobs.create_job(prompt="Produce a report", schedule="every 5m")
        execution = executions.create_execution(job["id"], source="test")
        assert executions.mark_execution_running(execution["id"])
        monkeypatch.setenv("HERMES_CRON_EXECUTION_ID", execution["id"])
        with jobs.self_removal_delivery_scope(job["id"]), run_identity_scope(execution["id"]):
            result = json.loads(cronjob(action="remove", job_id=job["id"]))
            assert result["success"], result
            assert jobs.self_removal_delivery_allowed(job["id"])
        assert jobs.get_job(job["id"]) is None
