"""The attach verdict of a start: exit 0, or 75 under a supervisor (``gateway.supervised_attach_retry``).

A start that finds the host gateway already serving its profile has nothing to run. Under a
supervisor that verdict is a runtime observation: during a restart the owner is often the OLD
process, still draining. launchd ``KeepAlive {SuccessfulExit=false}`` parks a job that exits 0, so
the replacement must exit EX_TEMPFAIL (75) and be retried; once the owner is gone the retry starts.
The CLI guard (``hermes_cli.gateway._attach_to_host_gateway_or_guard``) answers the same way; this is
the second ``decide()``, in ``start_gateway``, which returned success and exited 0.
"""
from __future__ import annotations

import logging

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE, is_gateway_supervisor_process
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def attach_or_retry(decision) -> bool:
    """True (exit 0, nothing spawned) for an attach; raise SystemExit(75) when a supervisor must retry."""
    owner = decision.owner.describe() if decision.owner else "unknown"
    logger.info("Attaching to the host gateway instead of starting a second one: %s", owner)
    print(decision.message)
    if decision.transient and is_gateway_supervisor_process() and _retry_enabled():
        logger.warning("Supervised start found the host gateway still alive (%s); exit %d so the supervisor "
                       "retries once it is gone", owner, GATEWAY_SERVICE_RESTART_EXIT_CODE)
        raise SystemExit(GATEWAY_SERVICE_RESTART_EXIT_CODE)
    return True


def _retry_enabled() -> bool:
    """``gateway.supervised_attach_retry`` of the home ``decide()`` judged (default on)."""
    from gateway.run import _load_gateway_config  # gateway.run imports this module

    config = _load_gateway_config(get_hermes_home() / "config.yaml")
    return cfg_get(config, "gateway", "supervised_attach_retry", default=True) is not False
