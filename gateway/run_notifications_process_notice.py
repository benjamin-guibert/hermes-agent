"""Background-process notices a gateway restart must not lose (``gateway.process_notice_replay``).

The completion receipt in ``logs/process-results/`` is the durable carrier of a notice the
gateway still owes (``tools.process_registry_results``); the process watcher settles it once it
is done. A lifecycle kill of a completion-contract holder owes a death notice that the kill
sweep's forced ``consume_output`` mark must not suppress: the killer took the output, the agent
learned nothing.
"""

import logging

logger = logging.getLogger("gateway.run")


def owed_death_notice(watcher: dict, session) -> bool:
    """The exited *session* owes its owner a death notice (a re-armed one, or a lifecycle kill)."""
    from tools.process_registry_results import needs_interrupt_notice, process_notice_replay_enabled
    owed = bool(watcher.get("interrupted_death")) or needs_interrupt_notice(session)
    return owed and process_notice_replay_enabled()


def mark_death_notice(event: dict, session) -> None:
    """Make the completion event read as a death notice: cause and time of the kill."""
    event["interrupted"] = True
    event["exited_at"] = getattr(session, "exited_at", None) or None


def refused_while_stopping(runner, session_id: str) -> bool:
    """A notice refused while this gateway stops stays owed on disk for the next start, instead of
    spending its retries and its plain-notice fallback on a gateway that is going away."""
    from tools.process_registry_results import process_notice_replay_enabled
    if not getattr(runner, "_draining", False) or not process_notice_replay_enabled():
        return False
    logger.info("Process notice for %s refused while the gateway stops; the next start replays it", session_id)
    return True


def settle_owed_notice(watcher: dict, session) -> None:
    """The watcher is done with the exited process (delivered, consumed, presented, or nothing to
    send): stop its receipt from re-arming a watcher at the next start. Runs in the owning
    profile's scope (``_run_process_watcher``), where its receipt lives."""
    if session is None:
        return  # Untracked: a pending receipt keeps the notice owed.
    from tools.process_registry_results import mark_result_notified, process_notice_replay_enabled
    if process_notice_replay_enabled():
        mark_result_notified(watcher["session_id"])


async def completion_committed(runner, completion_evt: dict, session_id: str) -> bool:
    """After the last refusal: was it a lost ACK of a turn the authority did commit?"""
    if (getattr(runner, "session_authority", None) is None
            or runner._completion_delivery_identity(completion_evt) is None):
        return False
    from gateway.session_automation import completion_admission
    try:
        # The owning profile's authority, as on the delivery path.
        async with runner._completion_event_scope(completion_evt):
            return completion_admission(runner, completion_evt) is not None
    except Exception:
        # Failing open sends the notice; a dead watcher would send nothing.
        logger.debug("Process watcher: admission recheck failed for %s", session_id, exc_info=True)
        return False
