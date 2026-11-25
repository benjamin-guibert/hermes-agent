"""The launchd reload helper waits for the old gateway as long as launchd lets it live.

Field report: a plist reload's helper booted the job out, waited its 30 s drain budget for the
old gateway, logged "still alive after 30s drain wait — bootstrapping anyway" and bootstrapped
beside it. launchd only SIGKILLs the old process ``ExitTimeOut`` (60 s) after the bootout's SIGTERM,
so the new start found the owner still alive. The wait, measured from the bootout, now covers the
``ExitTimeOut`` plus a margin.

The REAL helper script runs in bash with ``launchctl``, ``date`` and ``sleep`` faked on a virtual
clock (no real waiting); the old gateway is a real process the fake ``sleep`` kills at a chosen
virtual second after the bootout.
"""
import os
import subprocess
import textwrap

import pytest

EXIT_TIMEOUT = 60  # ExitTimeOut: the generated plist's value and the gui domain clamp
DRAIN = 30  # the field host's drain budget: the old wait
MARGIN = 10


def _fake_bin(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "date": 'c=$(cat "$CLOCK"); case "$1" in +%s) echo "$c";; *) echo "t=$c";; esac',
        "sleep": textwrap.dedent('''\
            c=$(( $(cat "$CLOCK") + ${1%.*} )); echo "$c" > "$CLOCK"
            if [ "$c" -ge "$DIE_AT" ]; then kill "$OLD_PID" 2>/dev/null; fi'''),
        "launchctl": textwrap.dedent('''\
            echo "$1 $(cat "$CLOCK")" >> "$CALLS"
            case "$1" in
              bootstrap) touch "$REG";;
              list) [ -f "$REG" ] && echo '"PID" = 4242;';;
            esac
            exit 0'''),
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    return bin_dir


def _old_gateway() -> int:
    """A real process, reparented so a kill reaps it (``kill -0`` stays true on an unreaped zombie)."""
    out = subprocess.run(["/bin/sh", "-c", "/bin/sleep 600 >/dev/null 2>&1 & echo $!"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def _helper_script(monkeypatch, old_pid: int) -> str:
    import hermes_cli.gateway as gw
    from hermes_cli import gateway_launchd

    monkeypatch.setattr(gw, "_get_restart_drain_timeout", lambda: DRAIN)
    spawned = []
    with monkeypatch.context() as m:
        m.setattr(gateway_launchd.subprocess, "Popen", lambda argv, **_k: spawned.append(argv))
        assert gateway_launchd._spawn_deferred_launchd_reload(
            domain="gui/501", label="ai.hermes.gateway", target="gui/501/ai.hermes.gateway",
            plist_path=gateway_launchd.Path("/tmp/ai.hermes.gateway.plist"), gateway_pid=old_pid)
    return spawned[0][-1]  # launchctl submit ... -- /bin/bash -c <script>


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("dies_after_bootout, waited", [
    (EXIT_TIMEOUT + MARGIN - 1, True),  # just under the wait: bootstrap once it is gone
    (EXIT_TIMEOUT + MARGIN + 1, False),  # just over: bootstrap anyway, logged
], ids=["just-under", "just-over"])
def test_the_helper_waits_out_exit_timeout_from_the_bootout(tmp_path, monkeypatch, dies_after_bootout, waited):
    from hermes_cli.gateway_launchd import _launchd_reload_log_path

    old_pid = _old_gateway()
    try:
        script = _helper_script(monkeypatch, old_pid)
        clock, calls = tmp_path / "clock", tmp_path / "calls"
        clock.write_text("1000\n")
        bootout_at = 1002  # the script's leading `sleep 2`
        env = {**os.environ, "PATH": f"{_fake_bin(tmp_path)}:{os.environ['PATH']}", "CLOCK": str(clock),
               "CALLS": str(calls), "REG": str(tmp_path / "registered"), "OLD_PID": str(old_pid),
               "DIE_AT": str(bootout_at + dies_after_bootout)}
        subprocess.run(["/bin/bash", "-c", script], env=env, check=True, timeout=60)
    finally:
        subprocess.run(["kill", str(old_pid)], capture_output=True, check=False)
    events = [line.split() for line in calls.read_text().splitlines()]
    assert ["bootout", str(bootout_at)] in events
    first_bootstrap = next(int(at) for name, at in events if name == "bootstrap")
    log = _launchd_reload_log_path().read_text()
    if waited:
        assert "bootstrapping anyway" not in log
        assert first_bootstrap - bootout_at >= dies_after_bootout  # after the old gateway was gone
    else:
        assert f"still alive after {EXIT_TIMEOUT + MARGIN}s drain wait — bootstrapping anyway" in log
        assert first_bootstrap - bootout_at < dies_after_bootout + 2


def test_the_in_process_fallback_waits_out_exit_timeout(monkeypatch):
    import hermes_cli.gateway as gw
    from hermes_cli import gateway_launchd

    monkeypatch.setattr(gw, "_get_restart_drain_timeout", lambda: DRAIN)
    waits = []
    monkeypatch.setattr(gateway_launchd.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(gw, "_wait_for_pid_exit", lambda pid, timeout, **_k: waits.append(timeout) or True)
    monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered", lambda *_a, **_k: True)

    assert gateway_launchd._bootout_drain_and_rebootstrap("gui/501", "/tmp/p.plist", "ai.hermes.gateway", 4242, DRAIN)
    assert waits == [EXIT_TIMEOUT + MARGIN]
