"""A top-level Discord post that mentions the owner opens its own thread.

The owner gets pinged by a cron delivery or an agent post in the home channel and wants to answer
in place; without a thread a top-level answer opens an unrelated session with no context of the
post. The adapter opens a public thread on the delivered message (the first chunk of a split
send), marks it as a bot thread and seeds its session with the post text.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.conftest import _ensure_discord_mock

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter

OWNER = "111111111111111111"
OTHER_BOT = "222222222222222222"
SELF = "333333333333333333"
CHANNEL = 4000
CRON_POST = (
    "Cronjob Response: Morning review\n(job_id: abc123)\n-------------\n\n"
    f"<@{OWNER}> two decisions wait for you.\n\n"
    'To stop or manage this job, send me a new message (e.g. "stop reminder Morning review").'
)


class FakeTextChannel:
    """Stands in for ``discord.TextChannel``: each send returns a message with a fresh id."""

    def __init__(self, channel_id=CHANNEL, create_thread=None):
        self.id = channel_id
        self.type = discord_platform.discord.ChannelType.text
        self.name = "home"
        self.guild = SimpleNamespace(id=1, name="guild")
        self._next = 9000
        self.sent = []
        self.send = AsyncMock(side_effect=self._send)
        self.create_thread = create_thread or AsyncMock(side_effect=lambda **kw: SimpleNamespace(
            id=int(kw["_message_id"]), name=kw["name"]))
        self.thread_calls = []

    async def _send(self, content=None, reference=None):
        self._next += 1
        self.sent.append((self._next, content))
        return SimpleNamespace(id=self._next)

    def get_partial_message(self, message_id):
        async def create_thread(**kwargs):
            self.thread_calls.append((message_id, kwargs))
            return await self.create_thread(_message_id=message_id, **kwargs)
        return SimpleNamespace(id=message_id, create_thread=create_thread)


class FakeHTTPException(Exception):
    """``discord.HTTPException`` with the two fields the sender reads."""

    def __init__(self, status, code):
        super().__init__(f"{status} (error code: {code})")
        self.status = status
        self.code = code


class FakeThread:
    def __init__(self, channel_id, **attrs):
        self.id = int(channel_id)
        self.send = AsyncMock(return_value=SimpleNamespace(id=9500))
        for name, value in attrs.items():
            setattr(self, name, value)


@pytest.fixture
def make_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "TextChannel", FakeTextChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "HTTPException", FakeHTTPException, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{OWNER},{OTHER_BOT}")

    def _make(channel, **extra):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
        users = {int(OWNER): SimpleNamespace(id=int(OWNER), bot=False),
                 int(OTHER_BOT): SimpleNamespace(id=int(OTHER_BOT), bot=True)}
        adapter._client = SimpleNamespace(
            get_channel=MagicMock(return_value=channel), fetch_channel=AsyncMock(return_value=channel),
            user=SimpleNamespace(id=int(SELF), bot=True), get_user=MagicMock(side_effect=users.get),
            fetch_user=AsyncMock(side_effect=lambda uid: users[uid]),
        )
        adapter._allowed_user_ids = {OWNER, OTHER_BOT}
        return adapter

    return _make


@pytest.mark.asyncio
async def test_cron_delivery_with_owner_mention_opens_a_thread(make_adapter):
    """The incident: a cron delivery that pings the owner in the home channel had no thread."""
    channel = FakeTextChannel()
    adapter = make_adapter(channel)

    result = await adapter.send(str(CHANNEL), CRON_POST)

    assert result.success
    assert len(channel.thread_calls) == 1
    message_id, kwargs = channel.thread_calls[0]
    assert str(message_id) == result.message_id
    assert kwargs["name"] == "Morning review"


def _http_error(code, status=400):
    return FakeHTTPException(status, code)


# R1: every send path; a split message threads on its FIRST chunk.

@pytest.mark.asyncio
async def test_live_split_post_threads_on_its_first_chunk(make_adapter):
    channel = FakeTextChannel()
    adapter = make_adapter(channel)

    result = await adapter.send(str(CHANNEL), f"<@{OWNER}> report\n" + "x" * 2500)

    assert len(channel.sent) > 1
    assert [call[0] for call in channel.thread_calls] == [channel.sent[0][0]]
    assert result.raw_response["owner_ping_thread_id"] == str(channel.sent[0][0])


def _rest_send(text):
    """``hermes send`` / the send_message tool: the REST sender; returns ``(result, posts)``."""
    import asyncio
    from unittest.mock import patch
    from plugins.platforms.discord.adapter import _standalone_send
    from tests.tools.test_discord_send_message_caption import _pconfig, _resp, _session_with

    context, posts = _session_with([])
    session = context.__aenter__.return_value
    ids = iter(range(9001, 9100))

    def _post(url, **kwargs):
        posts.append((url, kwargs.get("json")))
        body = {"id": "9001", "type": 11} if url.endswith("/threads") else {"id": str(next(ids))}
        response = _resp(201 if url.endswith("/threads") else 200, body)
        response.headers = {}
        post = MagicMock()
        post.__aenter__ = AsyncMock(return_value=response)
        post.__aexit__ = AsyncMock(return_value=False)
        return post

    session.post = MagicMock(side_effect=_post)
    with patch("aiohttp.ClientSession", return_value=context):
        return asyncio.run(_standalone_send(_pconfig(), str(CHANNEL), text)), posts


def test_rest_send_threads_on_its_first_chunk(make_adapter):
    result, posts = _rest_send(f"<@{OWNER}> report\n" + "x" * 2500)

    assert result["success"] and len(result["message_ids"]) > 1 and result["message_ids"][0] == "9001"
    thread_posts = [(url, body) for url, body in posts if url.endswith("/threads")]
    assert thread_posts == [(thread_posts[0][0], {"name": "report", "auto_archive_duration": 1440})]
    assert thread_posts[0][0].endswith(f"channels/{CHANNEL}/messages/9001/threads")
    assert result["owner_ping_thread_id"] == "9001"


# R2: thread name.

@pytest.mark.parametrize("text, name", [
    (CRON_POST, "Morning review"),
    (f"\n\n🔔 ⚠️ **<@{OWNER}> Build `main`** is _red_\nmore", "Build main is red"),
    (f"## <@{OWNER}> Decision needed", "Decision needed"),
    (f"<@{OWNER}>\n-------------\n> Pick a slot", "Pick a slot"),
    (f"<@{OWNER}> 🎉", "Reply here"),
    (f"<@{OWNER}> " + "word " * 40, ("word " * 40).strip()[:77] + "..."),
])
@pytest.mark.asyncio
async def test_thread_name_is_the_cleaned_first_line(make_adapter, text, name):
    channel = FakeTextChannel()
    await make_adapter(channel).send(str(CHANNEL), text)

    assert channel.thread_calls[0][1]["name"] == name
    assert len(name) <= 80


# R3: the owner's unmentioned reply in the thread reaches the agent with the post as context.

class InboundThread(FakeThread):
    def history(self, **_):
        async def _empty():
            return
            yield
        return _empty()


@pytest.mark.parametrize("sender", ["live", "rest"])
@pytest.mark.asyncio
async def test_owner_reply_in_the_thread_is_admitted_and_sees_the_post(make_adapter, tmp_path, monkeypatch, sender):
    import asyncio
    from datetime import UTC, datetime
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    for var in ("DISCORD_REQUIRE_MENTION", "DISCORD_THREAD_REQUIRE_MENTION", "DISCORD_FREE_RESPONSE_CHANNELS",
                "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS", "DISCORD_HISTORY_BACKFILL"):
        monkeypatch.delenv(var, raising=False)
    channel = FakeTextChannel()
    if sender == "live":
        result = await make_adapter(channel).send(str(CHANNEL), CRON_POST)
        thread_id = result.raw_response["owner_ping_thread_id"]
    else:  # another process: only the hand-off record reaches the gateway
        result, _posts = await asyncio.to_thread(_rest_send, CRON_POST)
        thread_id = result["owner_ping_thread_id"]

    gateway = make_adapter(channel)  # a new adapter: the gateway restarted since the post
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    gateway.set_session_store(store)
    gateway._ready_event.set()
    gateway._text_batch_delay_seconds = 0
    gateway.handle_message = AsyncMock()
    thread = InboundThread(thread_id, name="Morning review", parent=channel, parent_id=CHANNEL,
                           guild=channel.guild, topic=None)
    reply = SimpleNamespace(
        id=9700, content="approve the first one", mentions=[], role_mentions=[], attachments=[],
        author=SimpleNamespace(id=int(OWNER), bot=False, display_name="owner", name="owner"),
        channel=thread, guild=channel.guild, type=discord_platform.discord.MessageType.default,
        reference=None, webhook_id=None, created_at=datetime.now(UTC), flags=SimpleNamespace(),
        message_snapshots=[],
    )

    assert await gateway._dispatch_discord_message(reply) is True

    event = gateway.handle_message.await_args.args[0]
    assert event.source.thread_id == thread_id and event.source.chat_type == "thread"
    transcript = store.load_transcript(store.get_or_create_session(event.source).session_id)
    assert any("two decisions wait for you" in (m.get("content") or "") for m in transcript)


# R4: no thread for a thread or DM target, a post without a human owner mention, an existing thread,
# or a failed send.

def _thread_target(channel, monkeypatch):
    thread = FakeThread(4100, archived=False, send=channel.send, get_partial_message=channel.get_partial_message)
    return thread, f"<@{OWNER}> look", {"thread_id": "4100"}


def _dm_target(channel, monkeypatch):
    class FakeDM:
        id = 4200
        send = channel.send
        get_partial_message = channel.get_partial_message
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDM, raising=False)
    return FakeDM(), f"<@{OWNER}> look", None


def _failed_send(channel, monkeypatch):
    channel.send.side_effect = _http_error(50013, status=403)
    return channel, f"<@{OWNER}> look", None


_NOT_AN_OWNER_PING = {
    "thread_target": _thread_target,
    "dm_target": _dm_target,
    "no_owner": lambda channel, _mp: (channel, "nothing for the owner <@444444444444444444>", None),
    "only_bot": lambda channel, _mp: (channel, f"<@{OTHER_BOT}> your turn", None),
    "failed_send": _failed_send,
}


@pytest.mark.parametrize("case", list(_NOT_AN_OWNER_PING))
@pytest.mark.asyncio
async def test_no_thread_when_not_a_top_level_owner_ping(make_adapter, monkeypatch, case):
    channel = FakeTextChannel()
    target, text, metadata = _NOT_AN_OWNER_PING[case](channel, monkeypatch)
    adapter = make_adapter(target)

    result = await adapter.send(str(CHANNEL), text, metadata=metadata)

    assert result.success is (case != "failed_send")
    assert channel.thread_calls == []


@pytest.mark.asyncio
async def test_an_existing_thread_counts_as_opened(make_adapter, caplog):
    channel = FakeTextChannel(create_thread=AsyncMock(side_effect=_http_error(160004)))
    adapter = make_adapter(channel)

    with caplog.at_level("WARNING"):
        result = await adapter.send(str(CHANNEL), CRON_POST)

    assert result.success and len(channel.thread_calls) == 1 and len(channel.sent) == 1
    assert result.raw_response["owner_ping_thread_id"] == result.message_id
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# R5: a thread-create failure never fails, delays or repeats the send.

@pytest.mark.parametrize("failure", ["forbidden", "hang"])
@pytest.mark.asyncio
async def test_thread_create_failure_keeps_the_send_result(make_adapter, monkeypatch, caplog, failure):
    import asyncio
    from plugins.platforms.discord import owner_ping_thread

    never = asyncio.Event()

    async def _hang(**_):
        await never.wait()

    create = AsyncMock(side_effect=_http_error(50001, status=403)) if failure == "forbidden" else AsyncMock(side_effect=_hang)
    monkeypatch.setattr(owner_ping_thread, "CREATE_TIMEOUT_SECONDS", 0.2)
    channel = FakeTextChannel(create_thread=create)
    adapter = make_adapter(channel)

    with caplog.at_level("WARNING"):
        result = await adapter.send(str(CHANNEL), CRON_POST)

    assert result.success and result.message_id == str(channel.sent[0][0]) and len(channel.sent) == 1
    assert result.raw_response["message_ids"] == [result.message_id]
    assert "owner_ping_thread_id" not in result.raw_response
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


# R6: a post redirected from an archived thread to its parent threads once, on the parent message.

@pytest.mark.asyncio
async def test_archived_redirect_threads_the_parent_message_once(make_adapter):
    parent = FakeTextChannel()
    archived = FakeThread(4001, archived=True, parent=parent, parent_id=CHANNEL)
    adapter = make_adapter(archived)
    adapter._resolve_channel = AsyncMock(side_effect=lambda cid: parent if str(cid) == str(CHANNEL) else archived)

    result = await adapter.send(str(CHANNEL), f"<@{OWNER}> follow-up\n" + "y" * 2500, metadata={"thread_id": "4001"})
    resumed = await adapter._resume_partial_send(str(CHANNEL), SimpleNamespace(raw_response={
        **result.raw_response, "resume_safe": True, "remaining_chunks": [f"<@{OWNER}> tail"]}))

    assert archived.send.await_count == 0 and resumed.success
    assert len(parent.thread_calls) == 1
    assert parent.thread_calls[0][0] == parent.sent[0][0]
    assert parent.thread_calls[0][1]["name"] == "follow-up"


# R7: discord.owner_ping_thread: false keeps the base behaviour.

@pytest.mark.asyncio
async def test_off_switch_keeps_the_base_send(make_adapter):
    from plugins.platforms.discord.adapter import _apply_yaml_config

    channel = FakeTextChannel()
    result = await make_adapter(channel, owner_ping_thread=False).send(str(CHANNEL), CRON_POST)

    assert channel.thread_calls == []
    assert result.raw_response == {"message_ids": [result.message_id], "chat_id": str(CHANNEL), "thread_id": None}
    assert _apply_yaml_config({}, {"owner_ping_thread": False})["owner_ping_thread"] is False
