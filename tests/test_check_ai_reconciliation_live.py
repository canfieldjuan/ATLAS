from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_ai_reconciliation_live.py"


def load_check():
    spec = importlib.util.spec_from_file_location("check_ai_reconciliation_live", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thread(*, resolved=False, outdated=False, author="copilot-pull-request-reviewer[bot]",
           path="atlas_brain/x.py", line=12, body="use the typed config field"):
    return {
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "line": line,
        "comments": {"nodes": [{"author": {"login": author}, "bodyText": body}]},
    }


BODY_CLEAR = "## AI reconciliation\n- All fixed or waived: Yes\n"
BODY_OPEN = "## AI reconciliation\n- fixed or waived: No\n"
BODY_ABSENT = "## Summary\njust a normal PR body\n"
BOTS = ["copilot", "codex"]


# --- open_bot_threads filtering -------------------------------------------

def test_unresolved_bot_thread_counts():
    c = load_check()
    assert len(c.open_bot_threads([thread()], BOTS)) == 1


def test_resolved_outdated_and_nonbot_excluded():
    c = load_check()
    nodes = [
        thread(resolved=True),
        thread(outdated=True),
        thread(author="alice"),  # human reviewer, not a bot
    ]
    assert c.open_bot_threads(nodes, BOTS) == []


def test_bot_login_substring_match():
    c = load_check()
    assert c.open_bot_threads([thread(author="chatgpt-codex-connector[bot]")], BOTS)


# --- body classification (reuses Phase-2 parser) --------------------------

def test_classify_body():
    c = load_check()
    assert c.classify_body(BODY_CLEAR) == "claims_clear"
    assert c.classify_body(BODY_OPEN) == "acknowledges_open"
    assert c.classify_body(BODY_ABSENT) == "absent"


# --- evaluate: the failure branch (the lie) MUST fire ---------------------

def test_open_thread_plus_clear_body_fails():
    c = load_check()
    code, msgs = c.evaluate([thread()], BODY_CLEAR, BOTS)
    assert code == 1
    assert any("contradicts reality" in m for m in msgs)
    assert any("atlas_brain/x.py:12" in m for m in msgs)


def test_open_thread_plus_absent_body_fails():
    c = load_check()
    code, msgs = c.evaluate([thread()], BODY_ABSENT, BOTS)
    assert code == 1
    assert any("no AI reconciliation record" in m for m in msgs)


# --- evaluate: the pass branches ------------------------------------------

def test_open_thread_but_body_acknowledges_open_passes():
    # The body is honest about open findings; the local audit owns blocking it,
    # so the live check does not double-flag.
    c = load_check()
    code, _ = c.evaluate([thread()], BODY_OPEN, BOTS)
    assert code == 0


def test_no_open_threads_passes_even_with_clear_body():
    c = load_check()
    code, _ = c.evaluate([thread(resolved=True)], BODY_CLEAR, BOTS)
    assert code == 0


# --- main() via injection (no live GitHub) --------------------------------

def test_main_contradiction_exit_1(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread()]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")
    assert c.main(["--threads-file", str(tf), "--body-file", str(bf)]) == 1


def test_main_clean_exit_0(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")
    assert c.main(["--threads-file", str(tf), "--body-file", str(bf)]) == 0


def test_main_missing_pr_repo_exit_2():
    c = load_check()
    # No --threads-file and no --pr/--repo -> usage error, never a silent pass.
    assert c.main(["--repo", "", "--gh", "false"]) == 2


def test_main_empty_bots_exit_2(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text("[]", encoding="utf-8")
    assert c.main(["--threads-file", str(tf), "--bots", "  "]) == 2


# --- pagination: a thread past the first page must not be missed -----------

def _page(nodes, *, has_next, cursor=None):
    return json.dumps(
        {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
            "nodes": nodes,
        }}}}}
    )


def test_fetch_threads_paginates(monkeypatch):
    c = load_check()
    pages = [
        _page([thread(path="page1.py")], has_next=True, cursor="C1"),
        _page([thread(path="page2.py")], has_next=False),
    ]
    seen = {"n": 0, "cursors": []}

    def fake_gh(args, gh):
        # capture whether the cursor was forwarded on the second call
        seen["cursors"].append("C1" in " ".join(args))
        out = pages[seen["n"]]
        seen["n"] += 1
        return out

    monkeypatch.setattr(c, "_gh", fake_gh)
    nodes = c.fetch_threads(1431, "owner", "name", "gh")
    assert len(nodes) == 2  # both pages collected, not just the first 100
    assert seen["n"] == 2 and seen["cursors"] == [False, True]
