"""Bounded, profile-local receipts for completed terminal processes.

Receipts are read through process_manage and never adopted as live PIDs. Each
producer writes its own file so independent one-shot parents cannot overwrite each
other's results in the running-PID checkpoint.

A receipt is also the only durable trace of a gateway notice it still owes: the
completion queue and the watcher live in memory, and the checkpoint holds live PIDs
only. ``notify_pending`` marks a completion a gateway watcher has not settled;
``interrupt_notice_state="pending"`` marks a lifecycle kill (or a death found at
boot) of a ``notify_on_complete`` / ``persist_on_release`` process. Gateway startup
re-arms one watcher per owed receipt (``restore_pending_notifications``) and the
watcher settles it. Gate: ``gateway.process_notice_replay``.
"""

import json
import logging
import re
import time
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home
from persistence import STORE_FAILURES
from utils import atomic_json_write

logger = logging.getLogger("tools.process_registry")

RESULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_RETAINED_RESULTS = 64
_RESULT_FIELDS = (
    "id", "command", "cwd", "task_id", "owner_task_id", "session_key",
    "parent_session_id", "started_at", "exit_code", "completion_reason",
    "termination_source", "notify_on_complete",
)
# Restored when present: receipts written before they existed keep loading.
_REPLAY_FIELDS = (
    "watcher_platform", "watcher_chat_id", "watcher_user_id", "watcher_user_name",
    "watcher_thread_id", "watcher_message_id", "watcher_interval", "exited_at", "persist_on_release",
)
# Past this age a replayed notice would start a turn nobody waits on; the receipt stays
# readable through process_manage. Matches the async delegation replay cap.
MAX_NOTIFICATION_REPLAY_AGE_SECONDS = 48 * 60 * 60
# Lifecycle kills that owe the owner a death notice. ``gateway_turn_timeout`` and an explicit
# ``process.kill`` are deliberate stops and stay silent.
INTERRUPT_KILL_SOURCES = frozenset({"agent_close", "kill_all", "gateway_shutdown"})
_INTERRUPT_REASONS = {"killed": INTERRUPT_KILL_SOURCES, "lost": frozenset({"gateway_downtime"})}


def process_notice_replay_enabled() -> bool:
    """The ``gateway.process_notice_replay`` gate (default on) of the active profile."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("gateway") or {}).get("process_notice_replay", True)
    except Exception:
        # An unreadable config keeps the default, as gateway.delivery_ledger does.
        logger.debug("Could not read gateway.process_notice_replay", exc_info=True)
        return True
    return value.strip().lower() not in {"false", "0", "no", "off"} if isinstance(value, str) else bool(value)


def needs_interrupt_notice(session) -> bool:
    """A lifecycle kill or a boot-found death of a process that carried a completion contract."""
    if not (getattr(session, "notify_on_complete", False) or getattr(session, "persist_on_release", False)):
        return False
    reason = getattr(session, "completion_reason", "") or ""
    return getattr(session, "termination_source", "") in _INTERRUPT_REASONS.get(reason, ())


def _result_paths():
    """Prune by completion time, not start time (jobs can take days)."""
    directory = get_hermes_home() / "logs" / "process-results"
    cutoff = time.time() - RESULT_RETENTION_SECONDS
    retained = []
    for path in directory.glob("proc_*.json"):
        try:
            modified = path.stat().st_mtime
            if modified < cutoff:
                path.unlink(missing_ok=True)
            else:
                retained.append((modified, path))
        except FileNotFoundError:
            continue  # Another producer pruned it.
    retained.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    for _, path in retained[MAX_RETAINED_RESULTS:]:
        path.unlink(missing_ok=True)
    return [path for _, path in retained[:MAX_RETAINED_RESULTS]]


def save_completed_result(session) -> None:
    """Retain *session*'s completion receipt under this profile's logs. A hosted agent's process
    has none on this host: its command and output are the tenant's, kept in the live registry only."""
    if session.sandbox_key:
        return
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        record = {key: getattr(session, key) for key in (*_RESULT_FIELDS, *_REPLAY_FIELDS)}
        record["output"] = session.output_buffer[-MAX_OUTPUT_CHARS:]
    # A gateway watcher owes this completion a notice until it settles.
    record["notify_pending"] = bool(record["notify_on_complete"] and record["watcher_interval"])
    record["interrupt_notice_state"] = "pending" if needs_interrupt_notice(session) else ""
    record["completed_at"] = time.time()
    # Live-output opt-out must not persist raw credentials in durable receipts.
    record["output"] = redact_terminal_output(record["output"], record["command"], force=True)
    record["command"] = redact_sensitive_text(record["command"], code_file=True, force=True)
    directory = get_hermes_home() / "logs" / "process-results"
    try:
        from hermes_constants import assert_named_profile_home_live
        assert_named_profile_home_live(directory)
        make_user_dir(directory, mode=0o700, parents=True, exist_ok=True)
        atomic_json_write(directory / f"{session.id}.json", record, mode=0o600)
        _result_paths()
    except OSError:
        # Preserve live delivery on disk failure, but never silently claim durability.
        logger.warning("Could not retain completed process result %s", session.id, exc_info=True)


def _owns_result(owner: str, parent: str | None) -> bool:
    if not parent:
        return False
    if owner == parent:
        return True
    from persistence_assembly import open_persistence_reader, profile_scope

    # Pure lineage read on the hot path of every retained-result load; a writable open here
    # was one more writer handle per call inside the gateway (#100896).
    with open_persistence_reader(profile_scope().profile_dir) as persistence:
        return persistence.lineage.tip(parent) == owner


def _session_from_record(record: dict):
    """Read-only exited snapshot: no process handle or PID, so nothing can be re-run or signalled."""
    from tools.process_registry import ProcessSession

    session = ProcessSession(
        **{key: record[key] for key in _RESULT_FIELDS},
        **{key: record[key] for key in _REPLAY_FIELDS if key in record},
        # Receipts written before processes carried a sandbox are a local activity's.
        sandbox_key=record.get("sandbox_key", ""), exited=True, output_buffer=record["output"],
    )
    session._completion_event.set()
    return session


def load_completed_results(prefix: str = "") -> dict:
    """Restore read-only snapshots; no process handles, watchers, or queue events."""
    from gateway.session_context import get_session_env
    from tools.environments.sandbox_scope import hosted_scope

    owner = get_session_env("HERMES_SESSION_ID", "")
    # A hosted agent's processes leave no receipt here and it sees no local one: never read or prune them.
    if not owner or hosted_scope() is not None:
        return {}
    results = {}
    try:
        paths = _result_paths()
    except OSError:
        logger.warning("Could not read retained process results", exc_info=True)
        return results
    for path in paths:
        if not path.stem.startswith(prefix):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            if not _owns_result(owner, record.get("parent_session_id")):
                continue
            session = _session_from_record(record)
            results[session.id] = session
        except (OSError, ValueError, KeyError, TypeError, *STORE_FAILURES):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return results


def _owes_notice(record: dict, cutoff: float) -> bool:
    owed = record.get("notify_pending") or record.get("interrupt_notice_state") == "pending"
    # A notice needs a gateway route: a CLI receipt replayed by a gateway would invent one.
    return bool(owed and record.get("watcher_platform") and record["completed_at"] >= cutoff)


def pending_notification_results() -> list:
    """This profile's exited sessions that still owe a gateway notice, oldest first.

    Startup-only, so not ownership-filtered: no session owns anything yet, and the receipt
    carries its own route. Over-age receipts are skipped, not replayed.
    """
    cutoff = time.time() - MAX_NOTIFICATION_REPLAY_AGE_SECONDS
    pending = []
    try:
        paths = _result_paths()
    except OSError:
        logger.warning("Could not read retained process results", exc_info=True)
        return pending
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            if _owes_notice(record, cutoff):
                pending.append((record["completed_at"], record["id"], _session_from_record(record)))
        except (OSError, ValueError, KeyError, TypeError):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return [session for *_, session in sorted(pending, key=lambda item: item[:2])]


def mark_result_notified(session_id: str) -> bool:
    """Settle every notice a receipt owes, so no later startup replays it. True iff one was owed."""
    path = get_hermes_home() / "logs" / "process-results" / f"{session_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if not (record.get("notify_pending") or record.get("interrupt_notice_state") == "pending"):
            return False
        record["notify_pending"] = False
        if record.get("interrupt_notice_state") == "pending":
            record["interrupt_notice_state"] = "delivered"
        atomic_json_write(path, record, mode=0o600)
        return True
    except FileNotFoundError:
        return False  # No durable receipt (disk failure at exit, or a receipt-less test session).
    except (OSError, ValueError):
        # The next startup re-arms it: one duplicate notice, never a lost one.
        logger.warning("Could not settle the notice on process result %s", session_id, exc_info=True)
        return False
