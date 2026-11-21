"""A multi-MB tool result never stalls the gateway event loop.

Field premise to prove or disprove: on a loaded macOS host ``computer_use`` returned results of
0.7 to 6 MB (``tool computer_use completed (2.37s, 876867 chars)``, then 5966736 chars) during a
turn whose gateway loop looked blocked. Here a REAL agent turn runs on a worker thread, as the
gateway runs it: the fake model calls a tool that returns 6 MB, the agent spills/persists the
result and flushes it to a real state.db, and every result-bearing callback hops onto the loop and
builds the viewer's ``tool.complete`` payload there (the heaviest result handling a gateway loop
does). A 50 ms timer chain measures how late the loop runs while that happens.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.fixtures.storage_read import transcript_rows
from tests.gateway.fixtures.loop_lag import LoopLag
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.waiting import async_wait_until

RESULT_CHARS = 5_966_736  # the biggest result in the field logs


@pytest.fixture
def big_tool():
    from tools.registry import registry

    name = "big_screen_probe"
    registry.register(name=name, toolset="probe", schema={"name": name, "description": name, "parameters": {
        "type": "object", "properties": {}}}, handler=lambda args, **kw: "x" * RESULT_CHARS)
    yield name
    registry.deregister(name)


@pytest.mark.asyncio
async def test_a_6mb_tool_result_keeps_the_loop_under_one_second(tmp_path, big_tool):
    from gateway.run_turn_progress import _tool_complete_payload
    from hermes_cli import plugins
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    db = storage_handle(tmp_path)
    loop = asyncio.get_running_loop()
    built: list[int] = []

    def on_loop(result):  # the viewer payload a gateway builds from a finished tool
        built.append(len(_tool_complete_payload("call-1", big_tool, {}, result)["result"]))

    def progress(event_type, *_args, **kwargs):
        if event_type == "tool.completed":
            loop.call_soon_threadsafe(on_loop, kwargs["result"])

    with FakeLLMServer([ToolCall(big_tool, {}), Text("done")]) as srv:
        write_hermes_home(home, srv.base_url)
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = AIAgent(base_url=srv.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                            session_db=db, persistence=db, session_id="s-big", quiet_mode=True,
                            skip_context_files=True, skip_memory=True, enabled_toolsets=["probe"])
            agent.tool_progress_callback = progress
            agent.tool_complete_callback = lambda _id, _name, _args, result: loop.call_soon_threadsafe(on_loop, result)
            lag = LoopLag()
            await async_wait_until(lambda: lag.ticks >= 2, timeout=5)
            started = loop.time()
            result = await asyncio.to_thread(agent.run_conversation, "take a screenshot")
            elapsed = loop.time() - started
            after = lag.ticks
            await async_wait_until(lambda: lag.ticks >= after + 3, timeout=5)
            worst = lag.stop()
            agent.close()
        finally:
            plugins._reset_plugin_managers_for_tests()
    stored = max(len(str(row.get("content") or "")) for row in transcript_rows(db, "s-big") if row["role"] == "tool")
    print(f"6 MB tool turn: {elapsed:.2f}s on the worker, stored tool row {stored} chars, worst loop lag {worst * 1000:.0f} ms")
    assert result["final_response"] == "done"
    assert len(built) == 2 and all(size < RESULT_CHARS for size in built)  # the loop got bounded copies
    assert any(row["role"] == "tool" for row in transcript_rows(db, "s-big"))  # persisted by the worker
    assert worst < 1.0, f"the event loop stalled {worst:.2f}s while a 6 MB tool result was handled"
