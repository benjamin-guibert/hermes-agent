"""Final review: uncertain delivery cannot copy output into an invoking session."""
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platform_registry import platform_registry
from plugins.platforms.discord import adapter as discord_platform
from tests.gateway.test_discord_free_response import FakeThread, FakeTextChannel, FakeForumChannel
from tests.gateway.test_discord_outbound_operation import configured as configured_fixture

configured = configured_fixture


@pytest.mark.parametrize("lane", ["sync", "async"])
@pytest.mark.parametrize("outcome", ["unknown", "partial", "none", "local"])
def test_final_review_1_unknown_not_reached_none(configured, monkeypatch, lane, outcome):
    from cron import jobs, scheduler
    from tools import cronjob_tools as tool
    from tools.approval_context import _approval_session_key
    from tools.process_registry import process_registry
    _api, load = configured
    _config, pconfig = load()
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel)
    parent = FakeTextChannel(channel_id=4000)
    class Refused(Exception):
        status = 403
    async def post(**kwargs):
        if outcome == "none":
            raise Refused("definitive rejection")
        raise TimeoutError("acknowledgement lost")
    parent.send = AsyncMock(side_effect=post)
    if outcome == "partial":
        parent.send.side_effect = [SimpleNamespace(id=9001), TimeoutError("tail acknowledgement lost")]
    thread = FakeThread(channel_id=4001, parent=parent)
    thread.archived = True
    live = platform_registry.get("discord").adapter_factory(pconfig)
    live._client = SimpleNamespace(get_channel=lambda value: thread if value == 4001 else parent,
                                   fetch_channel=AsyncMock(return_value=thread))
    # External execution and transport boundaries only; delivery, persistence,
    # claim/final settlement and completion generation remain real.
    payload = "PRIVATE JOB OUTPUT " + "x" * (4100 if outcome == "partial" else 20)
    monkeypatch.setattr(scheduler, "run_job", lambda *args, **kwargs: (True, payload, payload, None))
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda *args, **kwargs: False)
    monkeypatch.setattr("gateway.outbound.send._send_to_platform", AsyncMock(return_value={
        "success": False, "error": "definitively rejected", "delivery": "none", "nothing_sent": True}))
    deliver = "local" if outcome == "local" else "discord:5000:4001"
    from hermes_constants import get_hermes_home
    script = get_hermes_home() / "scripts" / "review.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('synthetic output')")
    job = jobs.create_job(None, "every 1h", name="review proof", deliver=deliver,
                          no_agent=True, script=str(script))
    loop = asyncio.new_event_loop()
    worker = threading.Thread(target=loop.run_forever)
    worker.start()
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: SimpleNamespace(
        adapters={Platform.DISCORD: live}, _gateway_loop=loop), raising=False)
    token = _approval_session_key.set("agent:main:discord:channel:8000")
    try:
        if lane == "sync":
            claimed, error = tool._claim_for_manual_run(job["id"], "review proof")
            assert error is None
            result = tool._run_claimed_job(claimed)
            completion = tool._manual_run_completion(result, job["id"], job["name"], deliver, time.time())
        else:
            result = tool._try_dispatch_background_run(job, session_id="invoking-session")
            assert result["dispatched"]
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                completion = process_registry.completion_queue.get(timeout=20)
                if completion.get("delegation_id") == result["delegation_id"]:
                    break
            else:
                pytest.fail("manual completion did not arrive")
            assert completion["session_key"] == "agent:main:discord:channel:8000"
        saved = jobs.get_job(job["id"])
        assert parent.send.await_count == (0 if outcome == "local" else 2 if outcome == "partial" else 1)
        if outcome in ("unknown", "partial"):
            assert not saved.get("last_delivery_reached_none")
            assert "PRIVATE JOB OUTPUT" not in completion["summary"]
        else:
            assert "PRIVATE JOB OUTPUT" in completion["summary"]
        assert "PRIVATE JOB OUTPUT" in tool._latest_job_output_excerpt(job["id"])
    finally:
        _approval_session_key.reset(token)
        loop.call_soon_threadsafe(loop.stop)
        worker.join(timeout=5)
        loop.close()
