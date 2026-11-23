"""Archived thread delivery must use the parent channel."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_archived_thread_status import FakeThread, make_adapter as adapter_fixture

make_adapter = adapter_fixture


@pytest.mark.asyncio
async def test_bot_origin_send_does_not_reopen_archived_thread(make_adapter):
    parent = SimpleNamespace(id=4000, send=AsyncMock(return_value=SimpleNamespace(id=9002)))
    thread = FakeThread("4001", archived=True, parent=parent, parent_id=4000)
    adapter = make_adapter(thread)

    result = await adapter.send("5000", "scheduled output", metadata={"thread_id": "4001"})

    thread.send.assert_not_awaited()
    parent.send.assert_awaited_once()
    assert result.success
    assert result.message_id == "9002"
    assert result.raw_response["chat_id"] == "4000"
    assert result.raw_response["thread_id"] is None
    assert parent.send.await_args.kwargs["reference"] is None
    assert "<#4001>" in parent.send.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_redirect_keeps_original_inbound_ledger_anchor(make_adapter):
    parent = SimpleNamespace(id=4000, send=AsyncMock(return_value=SimpleNamespace(id=9002)))
    thread = FakeThread("4001", archived=True, parent=parent, parent_id=4000)
    adapter = make_adapter(thread)
    adapter._record_response_async = AsyncMock(side_effect=lambda *args: args[1])

    await adapter.send("5000", "output", reply_to="7001", metadata={"thread_id": "4001", "notify": True})

    recorded = adapter._record_response_async.await_args.args
    assert recorded[4]["reply_to_message_id"] == "7001"
    assert recorded[3] is True
    assert parent.send.await_args.kwargs["reference"] is None
