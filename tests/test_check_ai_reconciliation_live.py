from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_ai_reconciliation_live.py"
ROOT = Path(__file__).resolve().parents[1]
LIVE_WORKFLOW = ROOT / ".github" / "workflows" / "ai_reconciliation_live.yml"
RETRIGGER_WORKFLOW = ROOT / ".github" / "workflows" / "ai_reconciliation_review_retrigger.yml"


def load_check():
    spec = importlib.util.spec_from_file_location("check_ai_reconciliation_live", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thread(*, resolved=False, outdated=False, author="chatgpt-codex-connector[bot]",
           path="atlas_brain/x.py", line=12, body="use the typed config field"):
    return {
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "line": line,
        "comments": {"nodes": [{"author": {"login": author}, "bodyText": body}]},
    }


def review(*, author="chatgpt-codex-connector", commit="head-a", state="COMMENTED"):
    return {
        "author": {"login": author},
        "commit": {"oid": commit},
        "state": state,
    }


def pr_comment(
    *,
    author="chatgpt-codex-connector",
    commit="head-a",
    body="Codex Review: Didn't find any major issues.",
):
    text = f"{body}\n\n**Reviewed commit:** `{commit}`"
    return {
        "author": {"login": author},
        "body": text,
        "bodyText": text,
    }


def changed_file(
    filename,
    *,
    status="modified",
    previous_filename=None,
    head_mode="100644",
    head_type="blob",
    base_mode=None,
    base_type=None,
):
    item = {
        "filename": filename,
        "status": status,
        "previous_filename": previous_filename,
    }
    if status != "removed":
        item["head_mode"] = head_mode
        item["head_type"] = head_type
    if status == "removed" or previous_filename is not None:
        item["base_mode"] = base_mode or "100644"
        item["base_type"] = base_type or "blob"
    return item


def changed_file_proof(c, files=None, *, base="base-a", head="head-a", merge_base=None, expected_count=None):
    file_list = list(files if files is not None else [changed_file("docs/a.md")])
    return c.ChangedFileProof(
        base_sha=base,
        head_sha=head,
        merge_base_sha=merge_base or "a" * 40,
        expected_count=len(file_list) if expected_count is None else expected_count,
        files=file_list,
    )


BODY_CLEAR = "## AI reconciliation\n- All fixed or waived: Yes\n"
BODY_OPEN = "## AI reconciliation\n- fixed or waived: No\n"
BODY_ABSENT = "## Summary\njust a normal PR body\n"
BODY_DOCS_ONLY = "Docs-only: true\n\nArchive merged plans.\n"
BOTS = ["chatgpt-codex-connector", "chatgpt-codex-connector[bot]"]


def test_live_reconciliation_uses_supported_review_events_not_review_threads():
    text = LIVE_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request_review:" in text
    assert "pull_request_review_comment:" in text
    assert "pull_request_review_thread:" not in text
    assert "AI Reconciliation PR #" in text
    assert "github.event.pull_request.base.sha" in text
    assert "github.event.repository.default_branch" not in text
    assert "--pr \"${{ github.event.issue.number }}\"" not in text


def test_supported_review_events_retrigger_required_live_context():
    text = RETRIGGER_WORKFLOW.read_text(encoding="utf-8")

    assert "github.event.workflow_run.event == 'pull_request_review'" in text
    assert "github.event.workflow_run.event == 'pull_request_review_comment'" in text
    assert "github.event.workflow_run.event == 'issue_comment'" not in text
    assert "github.event.workflow_run.event == 'pull_request_review_thread'" not in text
    assert "pull-requests: read" in text
    assert "WORKFLOW_RUN_TITLE" not in text
    assert "pulls/${pr_number}" not in text
    assert "head_sha=$(gh api" not in text
    assert "head_sha=${head_sha}" in text
    assert "actions/runs/${run_id}/rerun" in text


# --- open_bot_threads filtering -------------------------------------------

def test_unresolved_bot_thread_counts():
    c = load_check()
    assert len(c.open_bot_threads([thread()], BOTS)) == 1


def test_resolved_and_nonbot_excluded_but_outdated_unresolved_blocks():
    c = load_check()
    nodes = [
        thread(resolved=True),
        thread(outdated=True),
        thread(author="alice"),  # human reviewer, not a bot
    ]
    assert c.open_bot_threads(nodes, BOTS) == [
        {
            "path": "atlas_brain/x.py",
            "line": 12,
            "author": "chatgpt-codex-connector[bot]",
            "snippet": "use the typed config field",
        }
    ]


def test_exact_codex_connector_login_matches():
    c = load_check()
    assert c.open_bot_threads([thread(author="chatgpt-codex-connector[bot]")], BOTS)


def test_codex_substring_helper_account_does_not_match():
    c = load_check()
    assert c.open_bot_threads([thread(author="codex-helper")], BOTS) == []


def test_copilot_thread_is_not_a_codex_gate_by_default():
    c = load_check()
    nodes = [thread(author="copilot-pull-request-reviewer[bot]")]
    assert c.open_bot_threads(nodes, BOTS) == []


def test_current_head_bot_review_requires_exact_author_and_head():
    c = load_check()
    reviews = [
        review(author="codex-helper", commit="head-a"),
        review(author="chatgpt-codex-connector", commit="old-head"),
        review(author="chatgpt-codex-connector", commit="head-a"),
    ]

    assert c.current_head_bot_reviews(reviews, head_sha="head-a", bot_logins=BOTS) == [reviews[2]]


def test_current_head_changes_requested_review_is_not_satisfactory_attestation():
    c = load_check()
    reviews = [
        review(author="chatgpt-codex-connector", commit="head-a", state="CHANGES_REQUESTED"),
    ]

    assert c.current_head_bot_reviews(reviews, head_sha="head-a", bot_logins=BOTS) == []
    assert c.current_head_change_requests(reviews, head_sha="head-a", bot_logins=BOTS) == reviews


def test_current_head_clean_review_comment_requires_exact_author_and_head_prefix():
    c = load_check()
    comments = [
        pr_comment(author="codex-helper", commit="head-a"),
        pr_comment(commit="old-head"),
        pr_comment(commit="abc1234567"),
    ]

    assert c.current_head_clean_review_comments(
        comments,
        head_sha="abc1234567890",
        bot_logins=BOTS,
    ) == [comments[2]]


def test_clean_review_comment_without_clean_phrase_is_not_satisfactory_attestation():
    c = load_check()
    comments = [pr_comment(commit="abc1234567", body="Codex Review: found issues.")]

    assert c.current_head_clean_review_comments(
        comments,
        head_sha="abc1234567890",
        bot_logins=BOTS,
    ) == []


def test_review_thread_generation_sorts_file_level_and_inline_threads():
    c = load_check()
    nodes = [
        thread(path="atlas_brain/x.py", line=None),
        thread(path="atlas_brain/x.py", line=12),
    ]

    assert len(c.review_thread_generation(nodes)) == 2


def test_parse_bot_logins_accepts_exact_defaults():
    c = load_check()
    assert c.parse_bot_logins("chatgpt-codex-connector,chatgpt-codex-connector[bot]") == BOTS


def test_parse_bot_logins_rejects_legacy_codex_alias():
    c = load_check()
    try:
        c.parse_bot_logins("codex")
    except ValueError as exc:
        assert "exact GitHub logins" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("legacy codex alias must fail instead of matching nothing")


# --- body classification (reuses Phase-2 parser) --------------------------

def test_classify_body():
    c = load_check()
    assert c.classify_body(BODY_CLEAR) == "claims_clear"
    assert c.classify_body(BODY_OPEN) == "acknowledges_open"
    assert c.classify_body(BODY_ABSENT) == "absent"


def test_is_docs_only_body_requires_first_nonblank_marker():
    c = load_check()

    assert c.is_docs_only_body(BODY_DOCS_ONLY) is True
    assert c.is_docs_only_body("\nDocs-only: true\n") is True
    assert c.is_docs_only_body("\ndocs-only: true\n") is True
    assert c.is_docs_only_body("\nDocs-only:   TRUE\n") is True
    assert c.is_docs_only_body("Summary\n\nDocs-only: true\n") is False
    assert c.is_docs_only_body("Docs-only: false\n") is False


def test_changed_files_are_docs_only_requires_markdown_only_paths():
    c = load_check()

    assert c.changed_files_are_docs_only([changed_file("docs/guide.md")]) is True
    assert c.changed_files_are_docs_only(
        [
            changed_file(
                "plans/archive/PR-Finished.md",
                status="renamed",
                previous_filename="plans/PR-Finished.md",
            )
        ]
    ) is True
    assert c.changed_files_are_docs_only([]) is False
    assert c.changed_files_are_docs_only([changed_file("scripts/check.py")]) is False
    assert c.changed_files_are_docs_only([changed_file("docs/guide.sh.md")]) is False
    assert c.changed_files_are_docs_only(
        [changed_file("docs/guide.md", head_mode="120000", head_type="blob")]
    ) is False
    assert c.changed_files_are_docs_only(
        [changed_file("plans/archive/PR-Finished.md", previous_filename="scripts/old.py")]
    ) is False
    assert c.changed_files_are_docs_only(
        [changed_file("docs/removed.md", status="removed", base_mode="100644", base_type="blob")]
    ) is True
    assert c.changed_files_are_docs_only(
        [changed_file("docs/removed.md", status="removed", base_mode="120000", base_type="blob")]
    ) is False
    assert c.changed_files_are_docs_only(
        [{"filename": "docs/guide.md", "previous_filename": None, "head_mode": "100644", "head_type": "blob"}]
    ) is False
    assert c.changed_files_are_docs_only(
        [changed_file("docs/guide.md", status="mystery")]
    ) is False
    assert c.changed_files_are_docs_only(
        [changed_file("docs/new.md", status="renamed", previous_filename=None)]
    ) is False


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


# --- evaluate: the remaining failure branches ------------------------------

def test_open_thread_plus_acknowledges_open_still_fails():
    c = load_check()
    code, msgs = c.evaluate([thread()], BODY_OPEN, BOTS)
    assert code == 1
    assert any("acknowledges open findings" in m for m in msgs)


# --- evaluate: the pass branches ------------------------------------------


def test_no_open_threads_passes_even_with_clear_body():
    c = load_check()
    code, _ = c.evaluate([thread(resolved=True)], BODY_CLEAR, BOTS)
    assert code == 0


def test_missing_current_head_codex_review_fails_even_without_open_threads():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_CLEAR,
        BOTS,
        reviews=[review(commit="old-head")],
        head_sha="head-a",
    )

    assert code == 1
    assert any("missing current-head Codex connector review" in msg for msg in msgs)


def test_docs_only_no_open_threads_passes_without_current_head_review():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_DOCS_ONLY,
        BOTS,
        reviews=[],
        comments=[],
        changed_files=[changed_file("plans/archive/PR-Finished.md")],
        head_sha="head-a",
    )

    assert code == 0
    assert any("docs-only PR diff has no open scoped Codex review threads" in msg for msg in msgs)


def test_docs_only_current_head_change_request_still_fails_with_valid_file_proof():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_DOCS_ONLY,
        BOTS,
        reviews=[review(commit="head-a", state="CHANGES_REQUESTED")],
        comments=[],
        changed_files=[changed_file("plans/archive/PR-Finished.md")],
        head_sha="head-a",
    )

    assert code == 1
    assert any("requested changes" in msg for msg in msgs)
    assert not any("docs-only PR diff" in msg for msg in msgs)


def test_docs_only_non_markdown_diff_still_requires_current_head_review():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_DOCS_ONLY,
        BOTS,
        reviews=[],
        comments=[],
        changed_files=[changed_file("scripts/check_ai_reconciliation_live.py")],
        head_sha="head-a",
    )

    assert code == 1
    assert any("missing current-head Codex connector review" in msg for msg in msgs)


def test_docs_only_open_codex_thread_still_fails_without_ai_record():
    c = load_check()
    code, msgs = c.evaluate(
        [thread()],
        BODY_DOCS_ONLY,
        BOTS,
        reviews=[],
        comments=[],
        head_sha="head-a",
    )

    assert code == 1
    assert any("no AI reconciliation record" in msg for msg in msgs)
    assert any("atlas_brain/x.py:12" in msg for msg in msgs)


def test_current_head_changes_requested_review_fails_even_without_open_threads():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_CLEAR,
        BOTS,
        reviews=[review(commit="head-a", state="CHANGES_REQUESTED")],
        head_sha="head-a",
    )

    assert code == 1
    assert any("requested changes" in msg for msg in msgs)
    assert not any("missing current-head Codex connector review" in msg for msg in msgs)


def test_current_head_codex_review_plus_no_open_threads_passes():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_CLEAR,
        BOTS,
        reviews=[review(commit="head-a")],
        head_sha="head-a",
    )

    assert code == 0
    assert any("current-head Codex review attestation is present" in msg for msg in msgs)


def test_current_head_clean_review_comment_plus_no_open_threads_passes():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_CLEAR,
        BOTS,
        comments=[pr_comment(commit="abc1234567")],
        head_sha="abc1234567890",
    )

    assert code == 0
    assert any("current-head Codex review attestation is present" in msg for msg in msgs)


def test_docs_only_exemption_skips_file_proof_when_current_head_review_already_present():
    c = load_check()

    assert (
        c.docs_only_exemption_needs_file_proof(
            [thread(resolved=True)],
            BODY_DOCS_ONLY,
            BOTS,
            reviews=[review(commit="head-a")],
            comments=[],
            head_sha="head-a",
        )
        is False
    )


def test_docs_only_exemption_needs_file_proof_only_for_missing_attestation_candidate():
    c = load_check()

    assert (
        c.docs_only_exemption_needs_file_proof(
            [thread(resolved=True)],
            BODY_DOCS_ONLY,
            BOTS,
            reviews=[],
            comments=[],
            head_sha="head-a",
        )
        is True
    )


def test_changes_requested_review_overrides_clean_review_comment():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_CLEAR,
        BOTS,
        reviews=[review(commit="abc1234567890", state="CHANGES_REQUESTED")],
        comments=[pr_comment(commit="abc1234567")],
        head_sha="abc1234567890",
    )

    assert code == 1
    assert any("requested changes" in msg for msg in msgs)


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


def test_main_requires_current_head_review_when_head_sha_supplied(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")
    rf = tmp_path / "reviews.json"
    rf.write_text(json.dumps([review(commit="old-head")]), encoding="utf-8")

    assert c.main(
        [
            "--threads-file",
            str(tf),
            "--body-file",
            str(bf),
            "--reviews-file",
            str(rf),
            "--head-sha",
            "head-a",
        ]
    ) == 1


def test_main_accepts_current_head_clean_review_comment_when_head_sha_supplied(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")
    cf = tmp_path / "comments.json"
    cf.write_text(json.dumps([pr_comment(commit="abc1234567")]), encoding="utf-8")

    assert c.main(
        [
            "--threads-file",
            str(tf),
            "--body-file",
            str(bf),
            "--comments-file",
            str(cf),
            "--head-sha",
            "abc1234567890",
        ]
    ) == 0


def test_main_accepts_docs_only_without_current_head_review_when_threads_clear(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_DOCS_ONLY, encoding="utf-8")
    ff = tmp_path / "changed-files.json"
    ff.write_text(json.dumps([changed_file("plans/archive/PR-Finished.md")]), encoding="utf-8")

    assert c.main(
        [
            "--threads-file",
            str(tf),
            "--body-file",
            str(bf),
            "--changed-files-file",
            str(ff),
            "--head-sha",
            "head-a",
        ]
    ) == 0


def test_main_live_fetch_fails_when_review_generation_changes(monkeypatch, tmp_path):
    c = load_check()
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")
    calls = {"reviews": 0}

    def fake_gh(args, gh):
        query = " ".join(args)
        if "reviews(first:100" in query:
            calls["reviews"] += 1
            if calls["reviews"] == 1:
                return _review_page([], has_next=False)
            return _review_page([review()], has_next=False)
        if "comments(first:100" in query:
            return _comment_page([], has_next=False)
        if "reviewThreads" in query:
            return _page([], has_next=False)
        if "/pulls/1431/files" in query:
            return json.dumps(changed_file("atlas_brain/x.py"))
        raise AssertionError(f"unexpected gh call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--body-file", str(bf), "--gh", "gh"]) == 2
    assert calls["reviews"] == 3


def test_main_live_fetch_fails_when_review_generation_changes_after_file_proof(monkeypatch):
    c = load_check()
    calls = {"snapshots": 0}

    def fake_snapshot(pr, owner, name, gh, bot_logins):
        calls["snapshots"] += 1
        if calls["snapshots"] == 1:
            return [thread(resolved=True)], "head-a", [], []
        return [thread(resolved=True)], "head-a", [review(commit="head-a", state="CHANGES_REQUESTED")], []

    monkeypatch.setattr(c, "fetch_consistent_review_thread_snapshot", fake_snapshot)
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: BODY_DOCS_ONLY)
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda pr, repo, gh, head_sha=None: changed_file_proof(c))
    monkeypatch.setattr(c, "fetch_pr_refs", lambda pr, repo, gh: ("base-a", "head-a", 1))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--gh", "gh"]) == 2
    assert calls["snapshots"] == 2


def test_main_live_fetch_fails_when_body_changes_after_file_proof(monkeypatch):
    c = load_check()
    bodies = [BODY_DOCS_ONLY, BODY_DOCS_ONLY + "\nchanged\n", BODY_DOCS_ONLY + "\nchanged\n"]

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [], []),
    )
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: bodies.pop(0))
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda pr, repo, gh, head_sha=None: changed_file_proof(c))
    monkeypatch.setattr(c, "fetch_pr_refs", lambda pr, repo, gh: ("base-a", "head-a", 1))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--gh", "gh"]) == 2


def test_main_live_fetch_fails_when_body_changes_after_final_snapshot(monkeypatch):
    c = load_check()
    bodies = [BODY_DOCS_ONLY, BODY_DOCS_ONLY, BODY_DOCS_ONLY + "\nchanged\n"]

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [], []),
    )
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: bodies.pop(0))
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda pr, repo, gh, head_sha=None: changed_file_proof(c))
    monkeypatch.setattr(c, "fetch_pr_refs", lambda pr, repo, gh: ("base-a", "head-a", 1))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--gh", "gh"]) == 2


def test_main_live_fetch_fails_when_base_changes_after_file_proof(monkeypatch):
    c = load_check()

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [], []),
    )
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: BODY_DOCS_ONLY)
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda pr, repo, gh, head_sha=None: changed_file_proof(c))
    monkeypatch.setattr(c, "fetch_pr_refs", lambda pr, repo, gh: ("base-b", "head-a", 1))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--gh", "gh"]) == 2


def test_main_live_fetch_fails_when_changed_file_count_changes_after_file_proof(monkeypatch):
    c = load_check()

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [], []),
    )
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: BODY_DOCS_ONLY)
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda pr, repo, gh, head_sha=None: changed_file_proof(c))
    monkeypatch.setattr(c, "fetch_pr_refs", lambda pr, repo, gh: ("base-a", "head-a", 2))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--gh", "gh"]) == 2


def test_main_does_not_fetch_file_proof_for_non_docs_body(monkeypatch, tmp_path):
    c = load_check()
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [review(commit="head-a")], []),
    )
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--body-file", str(bf), "--gh", "gh"]) == 0


def test_fetch_changed_files_parses_paginated_rows(monkeypatch):
    c = load_check()

    def fake_gh(args, gh):
        query = " ".join(args)
        if "pr view" in query:
            return json.dumps(
                {
                    "baseRefName": "main",
                    "baseRefOid": "base-a",
                    "changedFiles": 2,
                    "headRefOid": "head-a",
                }
            )
        if "compare/base-a...head-a" in query:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        raise AssertionError(f"unexpected gh call: {args}")

    def fake_git(args):
        if args[:4] == ["fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"]:
            return ""
        if args[:2] == ["cat-file", "-e"]:
            return ""
        if args[:3] == ["rev-parse", "--verify", "refs/remotes/origin/pr-1431^{commit}"]:
            return "head-a\n"
        if args[:4] == ["diff", "--name-status", "--no-renames", "-z"]:
            return "M\0docs/a.md\0A\0plans/archive/PR-Finished.md\0"
        if args[:2] == ["ls-tree", "head-a"]:
            return f"100644 blob {'b' * 40}\t{args[-1]}\n"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)
    monkeypatch.setattr(c, "_git_stdout", fake_git)

    assert c.fetch_changed_files(1431, "owner/name", "gh") == [
        changed_file("docs/a.md"),
        changed_file("plans/archive/PR-Finished.md", status="added"),
    ]


def test_fetch_changed_files_head_movement_fails_closed(monkeypatch):
    c = load_check()

    monkeypatch.setattr(
        c,
        "_gh",
        lambda args, gh: json.dumps(
            {
                "baseRefName": "main",
                "baseRefOid": "base-a",
                "changedFiles": 0,
                "headRefOid": "head-b",
            }
        ),
    )

    try:
        c.fetch_changed_files(1431, "owner/name", "gh", head_sha="head-a")
    except RuntimeError as exc:
        assert "head changed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("head movement before file fetch must fail closed")


def test_fetch_changed_files_fetched_head_mismatch_fails_closed(monkeypatch):
    c = load_check()

    def fake_gh(args, gh):
        query = " ".join(args)
        if "pr view" in query:
            return json.dumps(
                {
                    "baseRefName": "main",
                    "baseRefOid": "base-a",
                    "changedFiles": 1,
                    "headRefOid": "head-a",
                }
            )
        if "compare/base-a...head-a" in query:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        raise AssertionError(f"unexpected gh call: {args}")

    def fake_git(args):
        if args[:4] == ["fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"]:
            return ""
        if args[:2] == ["cat-file", "-e"]:
            return ""
        if args[:3] == ["rev-parse", "--verify", "refs/remotes/origin/pr-1431^{commit}"]:
            return "head-b\n"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)
    monkeypatch.setattr(c, "_git_stdout", fake_git)

    try:
        c.fetch_changed_files(1431, "owner/name", "gh", head_sha="head-a")
    except RuntimeError as exc:
        assert "fetched PR ref does not match observed head SHA" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("PR ref mismatch must fail closed")


def test_fetch_changed_files_uses_git_diff_instead_of_pull_file_listing(monkeypatch):
    c = load_check()
    seen_gh = []

    def fake_gh(args, gh):
        seen_gh.append(args)
        query = " ".join(args)
        if "pr view" in query:
            return json.dumps(
                {
                    "baseRefName": "main",
                    "baseRefOid": "base-a",
                    "changedFiles": 3001,
                    "headRefOid": "head-a",
                }
            )
        if "compare/base-a...head-a" in query:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        raise AssertionError(f"unexpected gh call: {args}")

    def fake_git(args):
        if args[:4] == ["fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"]:
            return ""
        if args[:2] == ["cat-file", "-e"]:
            return ""
        if args[:3] == ["rev-parse", "--verify", "refs/remotes/origin/pr-1431^{commit}"]:
            return "head-a\n"
        if args[:4] == ["diff", "--name-status", "--no-renames", "-z"]:
            return "M\0docs/a.md\0"
        if args[:2] == ["ls-tree", "head-a"]:
            return f"100644 blob {'b' * 40}\tdocs/a.md\n"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)
    monkeypatch.setattr(c, "_git_stdout", fake_git)

    assert c.fetch_changed_files(1431, "owner/name", "gh", head_sha="head-a") == [
        changed_file("docs/a.md")
    ]
    assert not any("/pulls/1431/files" in " ".join(args) for args in seen_gh)


def test_fetch_changed_files_rejects_malformed_status_rows(monkeypatch):
    c = load_check()

    def fake_gh(args, gh):
        query = " ".join(args)
        if "pr view" in query:
            return json.dumps(
                {
                    "baseRefName": "main",
                    "baseRefOid": "base-a",
                    "changedFiles": 1,
                    "headRefOid": "head-a",
                }
            )
        if "compare/base-a...head-a" in query:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        raise AssertionError(f"unexpected gh call: {args}")

    def fake_git(args):
        if args[:4] == ["fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"]:
            return ""
        if args[:2] == ["cat-file", "-e"]:
            return ""
        if args[:3] == ["rev-parse", "--verify", "refs/remotes/origin/pr-1431^{commit}"]:
            return "head-a\n"
        if args[:4] == ["diff", "--name-status", "--no-renames", "-z"]:
            return "T\0docs/a.md\0"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)
    monkeypatch.setattr(c, "_git_stdout", fake_git)

    assert c.changed_files_are_docs_only(c.fetch_changed_files(1431, "owner/name", "gh", head_sha="head-a")) is False


def test_fetch_changed_files_uses_merge_base_tree_for_removed_paths(monkeypatch):
    c = load_check()
    seen_tree_refs = []

    def fake_gh(args, gh):
        query = " ".join(args)
        if "pr view" in query:
            return json.dumps(
                {
                    "baseRefName": "main",
                    "baseRefOid": "base-tip",
                    "changedFiles": 1,
                    "headRefOid": "head-a",
                }
            )
        if "compare/base-tip...head-a" in query:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        raise AssertionError(f"unexpected gh call: {args}")

    def fake_git(args):
        if args[:4] == ["fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"]:
            return ""
        if args[:2] == ["cat-file", "-e"]:
            return ""
        if args[:3] == ["rev-parse", "--verify", "refs/remotes/origin/pr-1431^{commit}"]:
            return "head-a\n"
        if args[:4] == ["diff", "--name-status", "--no-renames", "-z"]:
            return "D\0docs/removed.md\0"
        if args[:2] == ["ls-tree", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]:
            seen_tree_refs.append(args[1])
            return f"120000 blob {'b' * 40}\tdocs/removed.md\n"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)
    monkeypatch.setattr(c, "_git_stdout", fake_git)

    files = c.fetch_changed_files(1431, "owner/name", "gh", head_sha="head-a")

    assert seen_tree_refs == ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
    assert files[0]["base_mode"] == "120000"


def test_main_missing_pr_repo_exit_2():
    c = load_check()
    # No --threads-file and no --pr/--repo -> usage error, never a silent pass.
    assert c.main(["--repo", "", "--gh", "false"]) == 2


def test_main_empty_bots_exit_2(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text("[]", encoding="utf-8")
    assert c.main(["--threads-file", str(tf), "--bots", "  "]) == 2


def test_main_legacy_bot_alias_exit_2(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text("[]", encoding="utf-8")
    assert c.main(["--threads-file", str(tf), "--bots", "codex"]) == 2


# --- pagination: a thread past the first page must not be missed -----------

def _page(nodes, *, has_next, cursor=None):
    return json.dumps(
        {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
            "nodes": nodes,
        }}}}}
    )


def _review_page(nodes, *, head="head-a", has_next=False, cursor=None):
    return json.dumps(
        {"data": {"repository": {"pullRequest": {
            "headRefOid": head,
            "reviews": {
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                "nodes": nodes,
            },
        }}}}
    )


def _comment_page(nodes, *, head="head-a", has_next=False, cursor=None):
    return json.dumps(
        {"data": {"repository": {"pullRequest": {
            "headRefOid": head,
            "comments": {
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                "nodes": nodes,
            },
        }}}}
    )


def _tree(entries, *, truncated=False):
    return json.dumps({"truncated": truncated, "tree": entries})


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


def test_fetch_threads_graphql_errors_fail_closed(monkeypatch):
    c = load_check()

    monkeypatch.setattr(
        c,
        "_gh",
        lambda args, gh: json.dumps({"errors": [{"message": "denied"}], "data": {"repository": None}}),
    )

    try:
        c.fetch_threads(1431, "owner", "name", "gh")
    except RuntimeError as exc:
        assert "GraphQL returned errors" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("GraphQL errors must not look like an empty thread list")


def test_fetch_threads_malformed_page_info_fails_closed(monkeypatch):
    c = load_check()

    monkeypatch.setattr(
        c,
        "_gh",
        lambda args, gh: json.dumps(
            {"data": {"repository": {"pullRequest": {"reviewThreads": {
                "pageInfo": {},
                "nodes": [],
            }}}}}
        ),
    )

    try:
        c.fetch_threads(1431, "owner", "name", "gh")
    except RuntimeError as exc:
        assert "hasNextPage" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("malformed pageInfo must not look like an empty thread list")


def test_fetch_threads_page_cap_exhaustion_fails_closed(monkeypatch):
    c = load_check()
    monkeypatch.setattr(c, "_MAX_THREAD_PAGES", 1)
    monkeypatch.setattr(c, "_gh", lambda args, gh: _page([], has_next=True, cursor="C1"))

    try:
        c.fetch_threads(1431, "owner", "name", "gh")
    except RuntimeError as exc:
        assert "pagination exceeded 1 pages" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("pagination cap exhaustion must fail closed")


def test_fetch_review_attestation_paginates(monkeypatch):
    c = load_check()
    pages = [
        _review_page([review(author="human")], has_next=True, cursor="C1"),
        _review_page([review()], has_next=False),
    ]
    seen = {"n": 0}

    def fake_gh(args, gh):
        out = pages[seen["n"]]
        seen["n"] += 1
        return out

    monkeypatch.setattr(c, "_gh", fake_gh)
    head, reviews = c.fetch_review_attestation(1431, "owner", "name", "gh")

    assert head == "head-a"
    assert len(reviews) == 2
    assert seen["n"] == 2


def test_fetch_comment_attestation_paginates(monkeypatch):
    c = load_check()
    pages = [
        _comment_page([pr_comment(commit="abc1234567")], has_next=True, cursor="C1"),
        _comment_page([pr_comment(commit="def1234567")], has_next=False),
    ]
    seen = {"n": 0, "cursors": []}

    def fake_gh(args, gh):
        seen["cursors"].append("C1" in " ".join(args))
        out = pages[seen["n"]]
        seen["n"] += 1
        return out

    monkeypatch.setattr(c, "_gh", fake_gh)
    head, fetched_comments = c.fetch_comment_attestation(1431, "owner", "name", "gh")

    assert head == "head-a"
    assert len(fetched_comments) == 2
    assert seen == {"n": 2, "cursors": [False, True]}


def test_fetch_review_attestation_head_change_fails_closed(monkeypatch):
    c = load_check()
    pages = [
        _review_page([], head="head-a", has_next=True, cursor="C1"),
        _review_page([], head="head-b", has_next=False),
    ]
    seen = {"n": 0}

    def fake_gh(args, gh):
        out = pages[seen["n"]]
        seen["n"] += 1
        return out

    monkeypatch.setattr(c, "_gh", fake_gh)

    try:
        c.fetch_review_attestation(1431, "owner", "name", "gh")
    except RuntimeError as exc:
        assert "changed PR head" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("head changes during review pagination must fail closed")


def test_consistent_snapshot_refetches_threads_after_attestation(monkeypatch):
    c = load_check()
    calls = {"reviews": 0, "threads": 0}

    def fake_fetch_reviews(pr, owner, name, gh):
        calls["reviews"] += 1
        return "head-a", [review(commit="head-a")]

    def fake_fetch_comments(pr, owner, name, gh):
        return "head-a", []

    def fake_fetch_threads(pr, owner, name, gh):
        calls["threads"] += 1
        return [thread(path=f"snapshot-{calls['threads']}.py")]

    monkeypatch.setattr(c, "fetch_review_attestation", fake_fetch_reviews)
    monkeypatch.setattr(c, "fetch_comment_attestation", fake_fetch_comments)
    monkeypatch.setattr(c, "fetch_threads", fake_fetch_threads)

    try:
        c.fetch_consistent_review_thread_snapshot(1431, "owner", "name", "gh", BOTS)
    except RuntimeError as exc:
        assert "review thread generation changed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("thread movement during snapshot fetch must fail closed")
    assert calls == {"reviews": 3, "threads": 2}


def test_consistent_snapshot_fails_when_review_generation_changes(monkeypatch):
    c = load_check()
    seen = {"reviews": 0}

    def fake_fetch_reviews(pr, owner, name, gh):
        seen["reviews"] += 1
        state = "COMMENTED" if seen["reviews"] == 1 else "CHANGES_REQUESTED"
        return "head-a", [review(commit="head-a", state=state)]

    monkeypatch.setattr(c, "fetch_review_attestation", fake_fetch_reviews)
    monkeypatch.setattr(c, "fetch_comment_attestation", lambda pr, owner, name, gh: ("head-a", []))
    monkeypatch.setattr(c, "fetch_threads", lambda pr, owner, name, gh: [thread()])

    try:
        c.fetch_consistent_review_thread_snapshot(1431, "owner", "name", "gh", BOTS)
    except RuntimeError as exc:
        assert "Codex review generation changed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("review movement during snapshot fetch must fail closed")


def test_consistent_snapshot_fails_when_clean_comment_generation_changes(monkeypatch):
    c = load_check()
    seen = {"comments": 0}

    def fake_fetch_comments(pr, owner, name, gh):
        seen["comments"] += 1
        comments = [] if seen["comments"] == 1 else [pr_comment(commit="abc1234567")]
        return "abc1234567890", comments

    monkeypatch.setattr(c, "fetch_review_attestation", lambda pr, owner, name, gh: ("abc1234567890", []))
    monkeypatch.setattr(c, "fetch_comment_attestation", fake_fetch_comments)
    monkeypatch.setattr(c, "fetch_threads", lambda pr, owner, name, gh: [thread()])

    try:
        c.fetch_consistent_review_thread_snapshot(1431, "owner", "name", "gh", BOTS)
    except RuntimeError as exc:
        assert "Codex review generation changed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("comment movement during snapshot fetch must fail closed")
