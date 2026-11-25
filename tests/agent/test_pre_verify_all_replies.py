"""``agent.pre_verify_all_replies``: ``pre_verify`` hooks also judge a final reply on a turn that
edited no files, and a draft such a hook sends back is not shown to the user.

Through a real ``AIAgent`` + store against the loopback model, with a user plugin whose
``pre_verify`` hook asks for one more try on attempt 0. The draft goes back to the model (it sees
what it wrote, then the nudge), never through the interim callback the gateway posts as
commentary, and stays out of the displayed history unless the budget runs out and it is the reply.
A draft that was already streamed to the user cannot be taken back: core keeps showing it.
"""

import json
import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.fixtures.storage_read import transcript_rows

DRAFT = "Please run the migration yourself, then tell me."
FINAL = "I ran the migration; all 12 rows moved."
NUDGE = "You can run that yourself. Do it, then answer."

_PLUGIN = (
    "import json, os\n"
    "def verify(**kw):\n"
    "    with open(os.path.join(os.environ['HERMES_HOME'], 'calls.jsonl'), 'a') as f:\n"
    "        f.write(json.dumps({'attempt': kw.get('attempt'), 'changed_paths': kw.get('changed_paths'),\n"
    "                            'final_response': kw.get('final_response')}) + '\\n')\n"
    f"    return None if kw.get('attempt') else {{'action': 'continue', 'message': {NUDGE!r}}}\n"
    "def register(ctx):\n"
    "    ctx.register_hook('pre_verify', verify)\n"
)

_CONFIG = (
    "updates:\n  check: false\n"
    "models_dev:\n  url: http://127.0.0.1:9/api.json\n"
    "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"
    'tools:\n  tool_search:\n    enabled: "off"\n'
    "plugins:\n  enabled:\n    - verifier\n"
)


@pytest.fixture
def turn(tmp_path):
    """``run(script, all_replies=True|False|None, stream=False, **agent_kwargs)``
    -> ``(result, interim, streamed, hook calls, displayed history, server)`` for one turn."""
    from gateway.session_controls import display_projection
    from hermes_cli import plugins
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    opened = []

    def run(script, all_replies=True, stream=False, **kwargs):
        server = FakeLLMServer(script).__enter__()
        opened.append(lambda: server.__exit__(None, None, None))
        plugin = home / "plugins" / "verifier"
        plugin.mkdir(parents=True, exist_ok=True)
        (plugin / "plugin.yaml").write_text("name: verifier\nversion: 0.1.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
        config = write_hermes_home(home, server.base_url, extra_config=_CONFIG) / "config.yaml"
        if all_replies is not None:
            key = f"agent:\n  pre_verify_all_replies: {str(all_replies).lower()}\n"
            config.write_text(config.read_text(encoding="utf-8").replace("agent:\n", key, 1), encoding="utf-8")
        plugins._reset_plugin_managers_for_tests()
        db = open_persistence(profile_scope(home=tmp_path))
        opened.append(db.close)
        kwargs.setdefault("enabled_toolsets", ["todo"])
        agent = AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        session_id="reply-check", persistence=db, **kwargs)
        interim, streamed = [], []
        agent.interim_assistant_callback = lambda text, **kw: interim.append((text, kw.get("already_streamed")))
        if stream:
            agent.stream_delta_callback = lambda delta: streamed.append(delta) if delta else None
        result = agent.run_conversation("move the rows")
        log = home / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        displayed = display_projection(transcript_rows(db, "reply-check"), home)
        return result, interim, "".join(streamed), calls, displayed, server

    yield run
    for close in opened:
        close()
    plugins._reset_plugin_managers_for_tests()


def _assistant_texts(displayed) -> list:
    return [row["text"] for row in displayed if row["role"] == "assistant"]


def test_a_no_edit_reply_is_checked_and_its_draft_is_not_shown(turn):
    result, interim, _, calls, displayed, server = turn([Text(DRAFT), Text(FINAL)])

    assert [(c["attempt"], c["changed_paths"]) for c in calls] == [(0, []), (1, [])]
    assert calls[0]["final_response"] == DRAFT
    assert interim == []
    # The model sees the draft it wrote, then the nudge.
    second = server.main_requests()[1]["messages"]
    assert [m["content"] for m in second[-2:]] == [DRAFT, NUDGE]
    assert result["final_response"] == FINAL
    assert _assistant_texts(displayed) == [FINAL]


def test_a_draft_held_back_when_the_budget_runs_out_is_the_reply(turn):
    result, interim, _, calls, displayed, _ = turn([Text(DRAFT)], max_iterations=1)

    assert [c["attempt"] for c in calls] == [0]
    assert interim == []
    assert result["final_response"] == DRAFT
    assert _assistant_texts(displayed) == [DRAFT]


def test_a_draft_already_streamed_to_the_user_stays_shown(turn):
    # Hiding cannot take back streamed text: core's interim path runs as before.
    result, interim, streamed, calls, _, _ = turn([Text(DRAFT), Text(FINAL)], stream=True)

    assert [c["attempt"] for c in calls] == [0, 1]
    assert DRAFT in streamed
    assert interim == [(DRAFT, True)]
    assert result["final_response"] == FINAL


@pytest.mark.parametrize("all_replies", [False, None], ids=["off", "unset"])
def test_without_the_key_a_no_edit_reply_is_not_checked(turn, all_replies):
    result, _interim, _, calls, displayed, server = turn([Text(DRAFT), Text(FINAL)], all_replies=all_replies)

    assert calls == []
    assert len(server.main_requests()) == 1
    assert result["final_response"] == DRAFT
    assert _assistant_texts(displayed) == [DRAFT]


def test_after_an_edit_the_draft_is_still_shown_with_the_key_on(turn, tmp_path):
    edit = ToolCall("write_file", {"path": str(tmp_path / "app.py"), "content": "x = 1\n"})
    result, interim, _, calls, _, _ = turn([edit, Text(DRAFT), Text(FINAL)], enabled_toolsets=["file"])

    assert [(c["attempt"], c["changed_paths"]) for c in calls] == [(0, [str(tmp_path / "app.py")]), (1, [str(tmp_path / "app.py")])]
    assert [text for text, _ in interim] == [DRAFT]
    assert result["final_response"] == FINAL
