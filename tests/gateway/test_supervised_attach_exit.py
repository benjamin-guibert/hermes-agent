"""A supervised start that finds the host gateway still alive never exits 0.

Field report (macOS, launchd ``KeepAlive {SuccessfulExit=false}``): a restart's old gateway was
still draining when its replacement started. The replacement printed "✓ The host gateway already
serves profile 'default' — nothing to start." and exited 0, so launchd parked the job; the old
process died a minute later and no gateway ran until a manual start.

``decide()`` runs twice per start. The CLI guard already exits 75 (retry) for a supervised attach;
the second answer, in ``start_gateway``, returned True and the run exited 0. Both now exit 75 under
a supervisor (launchd or systemd), a shell start still exits 0, and
``gateway.supervised_attach_retry: false`` keeps the base exit 0.
"""
import logging
import os
from pathlib import Path

import pytest

from gateway.host_attach import ATTACH, HostAttachDecision, HostGateway, attach_message
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

SUPERVISORS = {"launchd": {"XPC_SERVICE_NAME": "ai.hermes.gateway"}, "systemd": {"INVOCATION_ID": "f00d"}}
_MARKERS = ("XPC_SERVICE_NAME", "INVOCATION_ID", "HERMES_S6_SUPERVISED_CHILD", "HERMES_LAUNCHD_LABEL",
            "HERMES_GATEWAY_EXTERNAL_SUPERVISOR")


@pytest.fixture
def stopping_owner(monkeypatch, tmp_path):
    """The host record names a live owner (the draining old gateway) that serves 'default'."""
    from gateway import host_attach

    owner = HostGateway(os.getpid() + 1, tmp_path, ("default",), standalone=False)
    decision = HostAttachDecision(ATTACH, attach_message(owner, "default"), owner, transient=True)
    monkeypatch.setattr(host_attach, "decide", lambda *_a, **_k: decision)
    for name in _MARKERS:
        monkeypatch.delenv(name, raising=False)
    return decision


def _supervise(monkeypatch, kind):
    for name, value in SUPERVISORS.get(kind, {}).items():
        monkeypatch.setenv(name, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(SUPERVISORS))
async def test_a_supervised_start_beside_a_live_owner_exits_for_a_retry(stopping_owner, monkeypatch, caplog, kind):
    from gateway.run import _host_attach_or_none

    _supervise(monkeypatch, kind)
    with caplog.at_level(logging.WARNING, logger="gateway.supervised_attach"), pytest.raises(SystemExit) as raised:
        await _host_attach_or_none(replace=False)
    assert raised.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and str(GATEWAY_SERVICE_RESTART_EXIT_CODE) in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_a_shell_start_beside_a_live_owner_still_attaches(stopping_owner):
    from gateway.run import _host_attach_or_none

    assert await _host_attach_or_none(replace=False) is True  # nothing to start: exit 0


@pytest.mark.asyncio
async def test_the_off_switch_keeps_the_clean_exit(stopping_owner, monkeypatch):
    from gateway.run import _host_attach_or_none

    _supervise(monkeypatch, "launchd")
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text("gateway:\n  supervised_attach_retry: false\n")
    assert await _host_attach_or_none(replace=False) is True


@pytest.mark.parametrize("kind", sorted(SUPERVISORS))
def test_the_cli_guard_already_exits_for_a_retry(stopping_owner, monkeypatch, kind):
    from hermes_cli.gateway import _attach_to_host_gateway_or_guard

    _supervise(monkeypatch, kind)
    with pytest.raises(SystemExit) as raised:
        _attach_to_host_gateway_or_guard()
    assert raised.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
