"""Job removal, and the refusal of a removal that comes from inside the job's own live run.

A run's in-process self-removal is permitted through the run-local ``_self_removal_delivery``
marker, but a child process (an agent's terminal command, a job script, a ``hermes`` CLI they
start) cannot share that ContextVar. Without the refusal such a child deletes the record, which
cancels the live run: its final response is never delivered and no incident is booked.
"""

import contextlib
import logging
import os
import shutil
import sqlite3

logger = logging.getLogger(__name__)

# Bridged into the run's tool and script children (``gateway.session_context._VAR_MAP``).
RUN_EXECUTION_ENV = "HERMES_CRON_EXECUTION_ID"


@contextlib.contextmanager
def run_identity_scope(execution_id):
    """Give the children this scope spawns the ledger id of the cron run they belong to."""
    if not execution_id:
        yield
        return
    from gateway.session_context import _VAR_MAP

    var = _VAR_MAP[RUN_EXECUTION_ENV]
    token = var.set(str(execution_id))
    try:
        yield
    finally:
        var.reset(token)


def remove_job(job_id: str) -> bool:
    """Remove a job by ID or name; refuse a removal from a child of the job's own live run."""
    from cron import jobs as store

    job = store.resolve_job_ref(job_id)
    if not job:
        return False
    canonical_id = job["id"]
    with store._jobs_lock():
        jobs = store.load_jobs()
        original_len = len(jobs)
        jobs = [j for j in jobs if j["id"] != canonical_id]
        if len(jobs) == original_len:
            return False
        marker = store._self_removal_delivery.get()
        if marker is None or marker.job_id != canonical_id:
            _refuse_own_run_removal(canonical_id)
        # Resolve BEFORE saving so a legacy unsafe ID fails closed without a half-applied removal.
        job_output_dir = store._job_output_dir(canonical_id)
        store.save_jobs(jobs, removed_ids={canonical_id})
        if marker is not None and marker.job_id == canonical_id:
            marker.removed = True
        else:  # someone else removed it: its live run must stop, not burn tokens to a discard
            store.cancel_job_run(canonical_id, reason="job removed while running")
        if job_output_dir.exists():
            shutil.rmtree(job_output_dir)
        try:
            from cron.notepad import clear_notepad
            clear_notepad(canonical_id)
        except Exception:
            logger.debug("Failed to clear notepad for removed job %s", canonical_id, exc_info=True)
        # Prune the fire-fence lock entry so the registry doesn't grow monotonically.
        fence_key = f"{store._current_cron_store().cron_dir.resolve()}::{canonical_id}"
        with store._fire_fence_locks_guard:
            store._fire_fence_locks.pop(fence_key, None)
        return True


def _own_run_refusal(job_id: str) -> str:
    return (f"Refused: job {job_id} is the job of the run that calls this command. A run must not "
            f"remove its own job: pause it (hermes cron pause {job_id}), or let a --repeat 1 job "
            "finish by itself.")


def _refuse_own_run_removal(job_id: str) -> None:
    """Raise ``ValueError`` before any mutation when this process belongs to *job_id*'s live run."""
    worker_execution = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER")
    run_execution = os.environ.get(RUN_EXECUTION_ENV)
    if worker_execution is None and not run_execution:
        return
    from hermes_cli.config import load_config_readonly

    if not (load_config_readonly().get("cron") or {}).get("refuse_self_removal", True):
        return
    if worker_execution is not None:
        _check_worker_removal(job_id, worker_execution)
    if run_execution:
        from cron.executions import cancellable_execution

        # Only this profile's live attempt of this job counts: a stale id (run ended) or a
        # foreign one (another job or profile) never matches, so that removal goes ahead.
        live = cancellable_execution(job_id)
        if live is not None and live["id"] == run_execution:
            raise ValueError(_own_run_refusal(job_id))


def _check_worker_removal(job_id: str, execution_id: str) -> None:
    """A restart-safe worker's child inherits the worker's execution id: verify it in this
    profile's ledger, and fail closed when it cannot be verified."""
    from cron import executions, jobs
    from hermes_constants import get_hermes_home

    unverifiable = (
        f"Cannot remove job '{job_id}': cannot verify the inherited cron worker execution "
        "in this profile's ledger. Pause it instead, or remove it from an independent "
        "session after the run finishes.")
    cron_dir = get_hermes_home().resolve() / "cron"
    ledger = executions.EXECUTIONS_FILE or (cron_dir / "executions.db")
    if (not execution_id or jobs._current_cron_store().cron_dir.resolve() != cron_dir
            or ledger.resolve() != cron_dir / "executions.db" or not ledger.is_file()):
        raise ValueError(unverifiable)
    try:
        execution = executions.get_execution(execution_id)
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(unverifiable) from exc
    if execution is None:
        raise ValueError(unverifiable)
    if execution["job_id"] == job_id and execution["status"] in {"claimed", "running"}:
        raise ValueError(_own_run_refusal(job_id))
