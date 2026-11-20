"""``agent.pre_verify_all_replies`` on a turn that edited files: a draft a ``pre_verify`` hook
sends back is hidden there too, so a messaging user never sees two versions of one reply.

Same real ``AIAgent`` + store + loopback model harness as
``test_pre_verify_all_replies.py``; the turn first writes a file, then drafts a reply the hook
sends back once. With the key off, an edited turn keeps upstream's code-verify behaviour (the
draft is shown as interim commentary).
"""

import pytest

from tests.agent import test_pre_verify_all_replies as base
from tests.fakes.fake_llm_provider import Text, ToolCall

DRAFT, FINAL, NUDGE = base.DRAFT, base.FINAL, base.NUDGE
turn = base.turn


def _edit(tmp_path):
    return ToolCall("write_file", {"path": str(tmp_path / "app.py"), "content": "x = 1\n"})


def test_a_draft_after_an_edit_is_not_shown_with_the_key_on(turn, tmp_path):
    result, interim, _, calls, displayed, server = turn(
        [_edit(tmp_path), Text(DRAFT), Text(FINAL)], enabled_toolsets=["file"])

    assert [(c["attempt"], c["changed_paths"]) for c in calls] == [
        (0, [str(tmp_path / "app.py")]), (1, [str(tmp_path / "app.py")])]
    assert interim == []
    # The model sees the draft it wrote, then the nudge.
    assert [m["content"] for m in server.main_requests()[2]["messages"][-2:]] == [DRAFT, NUDGE]
    assert result["final_response"] == FINAL
    texts = base._assistant_texts(displayed)
    assert DRAFT not in texts and texts[-1] == FINAL


def test_a_draft_after_an_edit_held_back_when_the_budget_runs_out_is_the_reply(turn, tmp_path):
    # Iteration 1 writes the file, iteration 2 drafts: the budget is spent when the hook holds it.
    result, interim, _, calls, displayed, _ = turn(
        [_edit(tmp_path), Text(DRAFT)], enabled_toolsets=["file"], max_iterations=2)

    assert [c["attempt"] for c in calls] == [0]
    assert interim == []
    assert result["final_response"] == DRAFT
    assert base._assistant_texts(displayed)[-1] == DRAFT


def test_a_draft_after_an_edit_already_streamed_stays_shown(turn, tmp_path):
    result, interim, streamed, calls, _, _ = turn(
        [_edit(tmp_path), Text(DRAFT), Text(FINAL)], enabled_toolsets=["file"], stream=True)

    assert [c["attempt"] for c in calls] == [0, 1]
    assert DRAFT in streamed
    assert interim == [(DRAFT, True)]
    assert result["final_response"] == FINAL


@pytest.mark.parametrize("all_replies", [False, None], ids=["off", "unset"])
def test_without_the_key_a_draft_after_an_edit_is_shown(turn, tmp_path, all_replies):
    result, interim, _, calls, displayed, _ = turn(
        [_edit(tmp_path), Text(DRAFT), Text(FINAL)], enabled_toolsets=["file"], all_replies=all_replies)

    assert [c["attempt"] for c in calls] == [0, 1]
    assert interim == [(DRAFT, False)]
    assert result["final_response"] == FINAL
    assert base._assistant_texts(displayed)[-1] == FINAL
