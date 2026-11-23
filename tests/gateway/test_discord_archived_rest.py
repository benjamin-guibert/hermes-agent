"""REST sends must not reopen an archived thread."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord.adapter import _standalone_send
from tests.tools.test_discord_send_message_caption import _pconfig, _resp, _session_with


def test_rest_archived_send_uses_parent():
    response_post = _resp(200, {"id": "9002"})
    response_post.headers = {}
    context, posts = _session_with([response_post])
    session = context.__aenter__.return_value
    response = _resp(200, {"id": "4001", "type": 11, "parent_id": "4000", "thread_metadata": {"archived": True}})
    get_context = MagicMock()
    get_context.__aenter__ = AsyncMock(return_value=response)
    get_context.__aexit__ = AsyncMock(return_value=False)
    parent_context = MagicMock()
    parent_context.__aenter__ = AsyncMock(return_value=_resp(200, {"id": "4000", "type": 0}))
    parent_context.__aexit__ = AsyncMock(return_value=False)
    session.get.side_effect = [get_context, parent_context]
    with patch("aiohttp.ClientSession", return_value=context):
        result = asyncio.run(_standalone_send(_pconfig(), "5000", "original <@6000>", thread_id="4001"))
    assert len(posts) == 1
    assert posts[0][0].endswith("/channels/4000/messages")
    assert posts[0][1]["content"] == "<#4001>\noriginal <@6000>"
    assert result["chat_id"] == "4000"
    assert session.get.call_count == 2
    assert sum(call.args[0].endswith("/4001") for call in session.get.call_args_list) == 1


def test_rest_get_failure_sends_nothing():
    response_post = _resp(200, {"id": "9002"})
    response_post.headers = {}
    context, posts = _session_with([response_post])
    session = context.__aenter__.return_value
    get_context = MagicMock()
    get_context.__aenter__ = AsyncMock(return_value=_resp(403, {}))
    get_context.__aexit__ = AsyncMock(return_value=False)
    session.get.return_value = get_context
    with patch("aiohttp.ClientSession", return_value=context):
        result = asyncio.run(_standalone_send(_pconfig(), "5000", "original", thread_id="4001"))
    assert result.get("success") is not True
    assert "delivery refused" in result["error"]
    assert posts == []


def test_redirect_yaml_reaches_adapter_extra():
    from plugins.platforms.discord.adapter import _apply_yaml_config
    assert _apply_yaml_config({}, {"redirect_archived_thread_messages": False})["redirect_archived_thread_messages"] is False
