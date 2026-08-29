from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
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
           path="atlas_brain/x.py", line=12,
           body="use the typed config field\nR2 (BLOCKER) details"):
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


BODY_CLEAR = "## AI reconciliation\n- All fixed or waived: Yes\n"
BODY_NO_FINDINGS = "## AI reconciliation\n- no-findings\n"
BODY_OPEN = "## AI reconciliation\n- fixed or waived: No\n"
BODY_ABSENT = "## Summary\njust a normal PR body\n"
BODY_DOCS_ONLY = "Docs-only: true\n\nArchive merged plans.\n"
BOTS = ["chatgpt-codex-connector", "chatgpt-codex-connector[bot]"]


def body_with_dispositions(*decisions: str) -> str:
    lines = ["## AI reconciliation", "- AI findings reviewed: Yes", "- All fixed or waived: Yes"]
    lines.extend(
        f"- {decision} -- fixed-in: tests/test_check_ai_reconciliation_live.py"
        for decision in decisions
    )
    return "\n".join(lines) + "\n"


BODY_COVERS_DEFAULT_THREAD = body_with_dispositions("use the typed config field")


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
    assert "timeout-minutes: 15" in text
    assert "target run ${run_id} is ${status:-unknown}" in text
    assert "actions/runs/${run_id}\" --jq '.status // empty'" in text
    assert "rerun request for ${run_id} was rejected" in text


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
            "title": "use the typed config field",
            "decision": "use the typed config field",
            "snippet": "use the typed config field R2 (BLOCKER) details",
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
    assert c.classify_body(BODY_NO_FINDINGS) == "claims_clear"
    assert c.body_uses_no_findings(BODY_NO_FINDINGS) is True
    assert c.body_uses_no_findings(BODY_CLEAR) is False
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
    code, _ = c.evaluate([], BODY_CLEAR, BOTS)
    assert code == 0


def test_no_findings_fails_when_resolved_codex_thread_history_exists():
    c = load_check()
    code, msgs = c.evaluate([thread(resolved=True)], BODY_NO_FINDINGS, BOTS)
    assert code == 1
    assert any("records no-findings" in msg for msg in msgs)
    assert any("atlas_brain/x.py:12" in msg for msg in msgs)


def test_clear_body_requires_disposition_for_each_resolved_codex_thread():
    c = load_check()
    nodes = [
        thread(resolved=True, body="First parser issue needs history coverage R2 (BLOCKER) details"),
        thread(
            resolved=True,
            path="scripts/y.py",
            body="Second wrapper issue needs history coverage R13 (BLOCKER) details",
        ),
    ]

    code, msgs = c.evaluate(nodes, body_with_dispositions("Unrelated issue"), BOTS)

    assert code == 1
    assert any("missing dispositions" in msg for msg in msgs)
    assert any("First parser issue" in msg for msg in msgs)
    assert any("Second wrapper issue" in msg for msg in msgs)


def test_structured_disposition_only_body_rejects_missing_thread_history():
    c = load_check()
    nodes = [
        thread(
            resolved=True,
            body="Run history correlation for disposition-only ledgers R2/R13 (BLOCKER) details",
        )
    ]
    body = "## AI reconciliation\n- unrelated decision -- fixed-in: fake.py\n"

    code, msgs = c.evaluate(nodes, body, BOTS)

    assert code == 1
    assert any("missing dispositions" in msg for msg in msgs)
    assert any("Run history correlation for disposition-only ledgers" in msg for msg in msgs)


def test_thread_dispositions_reject_tiny_substring_roots():
    c = load_check()
    nodes = [
        thread(
            resolved=True,
            body="Require a disposition for every resolved thread R2/R13 (BLOCKER) details",
        )
    ]
    body = "## AI reconciliation\n- a -- fixed-in: fake.py\n"

    code, msgs = c.evaluate(nodes, body, BOTS)

    assert code == 1
    assert any("missing dispositions" in msg for msg in msgs)
    assert any("Require a disposition for every resolved thread" in msg for msg in msgs)


def test_root_decision_matching_rejects_tiny_exact_roots():
    c = load_check()

    assert c.root_decision_matches_thread("x", "x") is False
    assert c.root_decision_matches_thread(
        "Require a disposition for every resolved thread",
        "Require a disposition for every resolved thread",
    ) is True


def test_clear_body_passes_when_each_resolved_codex_thread_is_named():
    c = load_check()
    nodes = [
        thread(resolved=True, body="First parser issue needs history coverage R2 (BLOCKER) details"),
        thread(
            resolved=True,
            path="scripts/y.py",
            body="Second wrapper issue needs history coverage R13 (BLOCKER) details",
        ),
    ]

    code, msgs = c.evaluate(
        nodes,
        body_with_dispositions(
            "First parser issue needs history coverage",
            "Second wrapper issue needs history coverage",
        ),
        BOTS,
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_multiline_bodytext_uses_title_paired_with_rule_evidence_for_resolved_thread_dispositions():
    c = load_check()
    cases = (
        (
            "Reject generated columns in the attested signature",
            "R1/R4/R13 — detail from a current connector review",
        ),
        (
            "Update callers for the lazy suppression import",
            "R2/R5/R12 — detail from a current connector review",
        ),
        (
            "Keep missing review findings fail closed",
            "R1/R2 — detail with a distinct rule combination",
        ),
        (
            "Preserve trusted-base review evidence",
            "R10/R14 — detail with another root decision",
        ),
        (
            "Correlate multiline review titles structurally",
            "R2/R13 — detail with a held-out title",
        ),
    )

    for title, detail in cases:
        node = thread(resolved=True, body=f"{title}\n{detail}")

        code, messages = c.evaluate([node], body_with_dispositions(title), BOTS)

        assert code == 0
        assert any("no open scoped Codex review threads remain" in message for message in messages)


def test_multiline_bodytext_rejects_an_unrelated_resolved_thread_disposition():
    c = load_check()
    title = "Keep missing review findings fail closed"
    node = thread(
        resolved=True,
        body=f"{title}\nR1/R2 — detail from a current connector review",
    )

    code, messages = c.evaluate(
        [node],
        body_with_dispositions("An unrelated root decision"),
        BOTS,
    )

    assert code == 1
    assert any("missing dispositions" in message for message in messages)
    assert any(title in message for message in messages)


def test_multiline_bodytext_skips_nonsemantic_prefixes_before_a_real_title():
    c = load_check()
    title = "Preserve unparseable review history evidence"

    for prefix in ("---", "***", "___", "###", "⚠️", "..."):
        node = thread(
            resolved=True,
            body=f"{prefix}\n{title}\nR1/R2 — current connector detail",
        )

        code, messages = c.evaluate([node], body_with_dispositions(title), BOTS)

        assert code == 0
        assert any("no open scoped Codex review threads remain" in message for message in messages)


def test_unparseable_bodytext_cannot_be_waived_by_a_generic_disposition():
    c = load_check()
    body = body_with_dispositions(
        "An unrelated root decision",
        "unparseable trusted bot review title",
    )

    for malformed_body in ("---", "***\n___", "###\n⚠️", "", " \n\t"):
        code, messages = c.evaluate(
            [thread(resolved=True, body=malformed_body)],
            body,
            BOTS,
        )

        assert code == 1
        assert any("missing dispositions" in message for message in messages)
        assert any("unparseable trusted-bot review title" in message for message in messages)
        assert any("cannot be reconciled by a generic disposition" in message for message in messages)


def test_ambiguous_multiline_prefix_cannot_supply_a_reconciled_title():
    c = load_check()
    body = body_with_dispositions("x", "Real decision")
    node = thread(
        resolved=True,
        body="x\nReal decision R2 (BLOCKER) details",
    )

    code, messages = c.evaluate([node], body, BOTS)

    assert code == 1
    assert any("missing dispositions" in message for message in messages)
    assert any("unparseable trusted-bot review title" in message for message in messages)
    assert any("cannot be reconciled by a generic disposition" in message for message in messages)


def test_short_inline_legacy_title_cannot_supply_a_reconciled_title():
    c = load_check()
    body = body_with_dispositions("x")
    node = thread(resolved=True, body="x R2 (BLOCKER) details")

    code, messages = c.evaluate([node], body, BOTS)

    assert code == 1
    assert any("missing dispositions" in message for message in messages)
    assert any("unparseable trusted-bot review title" in message for message in messages)
    assert any("cannot be reconciled by a generic disposition" in message for message in messages)


def test_complete_rule_label_evidence_rejects_incomplete_separator_forms():
    c = load_check()
    prefix = "This is a sufficiently long ambiguous prefix"
    actual = "Real decision needs an explicit disposition"

    labels = [
        (f"{reference}{suffix}complete rule detail", 0)
        for reference in ("R1", "R1/R2")
        for suffix in (
            " — ",
            " - ",
            ": ",
            " (BLOCKER) ",
            " (BLOCKER) — ",
            " (BLOCKER): ",
        )
    ]
    labels.extend(
        (f"{reference}{suffix}", 1)
        for reference in ("R1", "R1/R2")
        for suffix in (
            "-",
            "(",
            " (",
            " (BLOCKER",
            " (BLOCKER)- detail",
            " (BLOCKER):",
            " (BLOCKER): ",
            ":",
            ": ",
            " —",
        )
    )
    for label, expected_code in labels:
        node = thread(resolved=True, body=f"{prefix}\n{label}\n{actual} R2 (BLOCKER) details")
        code, messages = c.evaluate([node], body_with_dispositions(prefix), BOTS)

        assert code == expected_code
        if expected_code:
            assert any(actual in message for message in messages)

    node = thread(resolved=True, body=f"{prefix} R1(\nR2 — complete rule detail")
    code, messages = c.evaluate([node], body_with_dispositions(prefix), BOTS)

    assert code == 1
    assert any("unparseable trusted-bot review title" in message for message in messages)


def test_severity_colon_rule_labels_correlate_multiline_and_inline_titles():
    c = load_check()
    cases = (
        (
            "Compare recovered trigger columns in catalog order",
            "Compare recovered trigger columns in catalog order\n"
            "R4 (BLOCKER): trigger update columns use physical catalog order",
        ),
        (
            "Require a trusted owner for the security definer function",
            "Require a trusted owner for the security definer function "
            "R3 (BLOCKER): direct writers can replace an unguarded function",
        ),
    )

    for title, source_body in cases:
        code, messages = c.evaluate(
            [thread(resolved=True, body=source_body)],
            body_with_dispositions(title),
            BOTS,
        )

        assert code == 0
        assert any("no open scoped Codex review threads remain" in message for message in messages)


def test_severity_less_colon_rule_labels_correlate_multiline_and_inline_titles():
    c = load_check()
    cases = (
        (
            "Exercise the deployed manifest entrypoint",
            "Exercise the deployed manifest entrypoint\n"
            "R2/R14: this test must call the deployed application",
        ),
        (
            "Correlate the trusted producer colon delimiter",
            "Correlate the trusted producer colon delimiter "
            "R4: detail from the connector review",
        ),
    )

    for title, source_body in cases:
        code, messages = c.evaluate(
            [thread(resolved=True, body=source_body)],
            body_with_dispositions(title),
            BOTS,
        )

        assert code == 0
        assert any(
            "no open scoped Codex review threads remain" in message
            for message in messages
        )


def test_severity_less_colon_rule_labels_still_reject_incomplete_or_malformed_evidence():
    c = load_check()
    title = "Require complete colon evidence for a review title"
    malformed_bodies = (
        f"{title}\nR4:",
        f"{title}\nR4: ",
        f"{title}\nR4 : detail",
        f"{title}\nR4   : detail",
        f"{title}\nR4foo: detail",
        "x\nR4: detail",
        f"{title} R4:\nR2 — separate complete evidence",
    )

    for source_body in malformed_bodies:
        code, messages = c.evaluate(
            [thread(resolved=True, body=source_body)],
            body_with_dispositions(title),
            BOTS,
        )

        assert code == 1
        assert any(
            "unparseable trusted-bot review title" in message for message in messages
        )


def test_legacy_inline_rule_label_remains_a_resolved_thread_disposition_title():
    c = load_check()
    node = thread(
        resolved=True,
        body="Keep legacy inline root extraction R2/R13 (BLOCKER) detail",
    )

    code, messages = c.evaluate(
        [node],
        body_with_dispositions("Keep legacy inline root extraction"),
        BOTS,
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in message for message in messages)


def test_thread_dispositions_use_canonical_pr_body_section():
    c = load_check()
    body = "\n".join(
        [
            "    ## AI reconciliation",
            "- Correlate only the canonical reconciliation ledger -- fixed-in: fake.py",
            "## AI reconciliation",
            "- AI findings reviewed: Yes",
            "- All fixed or waived: Yes",
            "- Some other finding -- fixed-in: tests/test_check_ai_reconciliation_live.py",
        ]
    )
    nodes = [
        thread(
            resolved=True,
            body="Correlate only the canonical reconciliation ledger R2/R13 (BLOCKER) details",
        )
    ]

    code, msgs = c.evaluate(nodes, body, BOTS)

    assert code == 1
    assert any("missing dispositions" in msg for msg in msgs)
    assert any("Correlate only the canonical reconciliation ledger" in msg for msg in msgs)


def test_missing_current_head_codex_review_passes_when_threads_are_clear():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        reviews=[review(commit="old-head")],
        head_sha="head-a",
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_missing_current_head_codex_review_passes_inside_fresh_update_window():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        reviews=[review(commit="old-head")],
        head_sha="head-a",
        pr_updated_at="2026-07-30T18:00:00Z",
        review_grace_seconds=300,
        now=datetime(2026, 7, 30, 18, 2, tzinfo=UTC),
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_missing_current_head_codex_review_passes_after_fresh_update_window():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        reviews=[review(commit="old-head")],
        head_sha="head-a",
        pr_updated_at="2026-07-30T18:00:00Z",
        review_grace_seconds=300,
        now=datetime(2026, 7, 30, 18, 6, tzinfo=UTC),
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


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
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_docs_only_current_head_change_request_does_not_block_without_open_threads():
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

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_docs_only_non_markdown_diff_passes_when_threads_are_clear():
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

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


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


def test_current_head_changes_requested_review_does_not_block_without_open_threads():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        reviews=[review(commit="head-a", state="CHANGES_REQUESTED")],
        head_sha="head-a",
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_current_head_codex_review_plus_no_open_threads_passes():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        reviews=[review(commit="head-a")],
        head_sha="head-a",
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_current_head_clean_review_comment_plus_no_open_threads_passes():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        comments=[pr_comment(commit="abc1234567")],
        head_sha="abc1234567890",
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


def test_changes_requested_review_does_not_override_clean_review_comment_without_open_threads():
    c = load_check()
    code, msgs = c.evaluate(
        [thread(resolved=True)],
        BODY_COVERS_DEFAULT_THREAD,
        BOTS,
        reviews=[review(commit="abc1234567890", state="CHANGES_REQUESTED")],
        comments=[pr_comment(commit="abc1234567")],
        head_sha="abc1234567890",
    )

    assert code == 0
    assert any("no open scoped Codex review threads remain" in msg for msg in msgs)


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
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")
    assert c.main(["--threads-file", str(tf), "--body-file", str(bf)]) == 0


def test_main_accepts_missing_current_head_review_when_threads_clear(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")
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
    ) == 0


def test_main_accepts_current_head_clean_review_comment_when_head_sha_supplied(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")
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


def test_main_ignores_malformed_review_window_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_CODEX_REVIEW_GRACE_SECONDS", "not-an-int")
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")

    assert c.main(["--threads-file", str(tf), "--body-file", str(bf)]) == 0


def test_main_malformed_pr_updated_at_exit_2(tmp_path):
    c = load_check()
    tf = tmp_path / "threads.json"
    tf.write_text(json.dumps([thread(resolved=True)]), encoding="utf-8")
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")

    assert c.main(
        [
            "--threads-file",
            str(tf),
            "--body-file",
            str(bf),
            "--head-sha",
            "head-a",
            "--pr-updated-at",
            "not-a-timestamp",
        ]
    ) == 2


def test_main_live_default_does_not_refetch_inside_review_window(monkeypatch, tmp_path):
    c = load_check()
    snapshots = [
        ([thread(resolved=True)], "head-a", [], []),
        ([thread(resolved=True)], "head-a", [], []),
    ]
    bf = tmp_path / "body.md"
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")

    def fake_snapshot(pr, owner, name, gh, bot_logins):
        return snapshots.pop(0)

    monkeypatch.setattr(c, "fetch_consistent_review_thread_snapshot", fake_snapshot)
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: BODY_COVERS_DEFAULT_THREAD)
    monkeypatch.setattr(c, "fetch_pr_updated_at", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--body-file", str(bf), "--gh", "gh"]) == 0
    assert len(snapshots) == 1


def test_main_live_wait_flag_is_noop_inside_review_window(monkeypatch, tmp_path):
    c = load_check()
    snapshots = [
        ([thread(resolved=True)], "head-a", [], []),
        ([thread(resolved=True)], "head-a", [], []),
    ]
    bf = tmp_path / "body.md"
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")

    def fake_snapshot(pr, owner, name, gh, bot_logins):
        return snapshots.pop(0)

    monkeypatch.setattr(c, "fetch_consistent_review_thread_snapshot", fake_snapshot)
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: BODY_COVERS_DEFAULT_THREAD)
    monkeypatch.setattr(c, "fetch_pr_updated_at", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    assert c.main(
        [
            "--pr",
            "1431",
            "--repo",
            "owner/name",
            "--body-file",
            str(bf),
            "--gh",
            "gh",
            "--wait-for-review-window",
        ]
    ) == 0
    assert len(snapshots) == 1


def test_main_live_fetch_does_not_call_review_or_comment_attestation(monkeypatch, tmp_path):
    c = load_check()
    bf = tmp_path / "body.md"
    bf.write_text(BODY_CLEAR, encoding="utf-8")

    def fake_gh(args, gh):
        query = " ".join(args)
        if "reviews(first:100" in query:
            raise AssertionError("thread-only live fetch must not call reviews")
        if "comments(first:100" in query:
            raise AssertionError("thread-only live fetch must not call comments")
        if "reviewThreads" in query:
            return _page([], has_next=False)
        if "/pulls/1431/files" in query:
            return json.dumps(changed_file("atlas_brain/x.py"))
        raise AssertionError(f"unexpected gh call: {args}")

    monkeypatch.setattr(c, "_gh", fake_gh)

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--body-file", str(bf), "--gh", "gh"]) == 0


def test_main_does_not_fetch_file_proof_for_docs_only_no_thread_signal(monkeypatch, tmp_path):
    c = load_check()
    bf = tmp_path / "body.md"
    bf.write_text(BODY_DOCS_ONLY, encoding="utf-8")

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [], []),
    )
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_pr_updated_at", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_body", lambda pr, repo, gh: BODY_DOCS_ONLY)

    assert c.main(["--pr", "1431", "--repo", "owner/name", "--body-file", str(bf), "--gh", "gh"]) == 0


def test_main_does_not_fetch_file_proof_for_non_docs_body(monkeypatch, tmp_path):
    c = load_check()
    bf = tmp_path / "body.md"
    bf.write_text(BODY_COVERS_DEFAULT_THREAD, encoding="utf-8")

    monkeypatch.setattr(
        c,
        "fetch_consistent_review_thread_snapshot",
        lambda pr, owner, name, gh, bot_logins: ([thread(resolved=True)], "head-a", [review(commit="head-a")], []),
    )
    monkeypatch.setattr(c, "fetch_changed_file_proof", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_pr_updated_at", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

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
    seen_fetch = []

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
            seen_fetch.append(args)
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
    assert seen_fetch == [
        [
            "fetch",
            "--no-tags",
            "origin",
            "+refs/heads/main:refs/remotes/origin/main",
            "+pull/1431/head:refs/remotes/origin/pr-1431",
        ]
    ]


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

def _page(nodes, *, head="head-a", has_next, cursor=None):
    return json.dumps(
        {"data": {"repository": {"pullRequest": {
            "headRefOid": head,
            "reviewThreads": {
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                "nodes": nodes,
            },
        }}}}
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
            {"data": {"repository": {"pullRequest": {
                "headRefOid": "head-a",
                "reviewThreads": {
                    "pageInfo": {},
                    "nodes": [],
                },
            }}}}
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


def test_fetch_thread_snapshot_head_change_fails_closed(monkeypatch):
    c = load_check()
    pages = [
        _page([], head="head-a", has_next=True, cursor="C1"),
        _page([], head="head-b", has_next=False),
    ]
    seen = {"n": 0}

    def fake_gh(args, gh):
        out = pages[seen["n"]]
        seen["n"] += 1
        return out

    monkeypatch.setattr(c, "_gh", fake_gh)

    try:
        c.fetch_thread_snapshot(1431, "owner", "name", "gh")
    except RuntimeError as exc:
        assert "changed PR head" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("head changes during thread pagination must fail closed")


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


def test_consistent_snapshot_refetches_threads(monkeypatch):
    c = load_check()
    calls = {"snapshots": 0}

    def fake_fetch_thread_snapshot(pr, owner, name, gh):
        calls["snapshots"] += 1
        return "head-a", [thread(path=f"snapshot-{calls['snapshots']}.py")]

    monkeypatch.setattr(c, "fetch_thread_snapshot", fake_fetch_thread_snapshot)

    try:
        c.fetch_consistent_review_thread_snapshot(1431, "owner", "name", "gh", BOTS)
    except RuntimeError as exc:
        assert "review thread generation changed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("thread movement during snapshot fetch must fail closed")
    assert calls == {"snapshots": 2}


def test_consistent_snapshot_does_not_fetch_reviews(monkeypatch):
    c = load_check()

    monkeypatch.setattr(c, "fetch_review_attestation", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_comment_attestation", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_thread_snapshot", lambda pr, owner, name, gh: ("head-a", [thread()]))

    nodes, head, reviews, comments = c.fetch_consistent_review_thread_snapshot(
        1431,
        "owner",
        "name",
        "gh",
        BOTS,
    )

    assert nodes == [thread()]
    assert head == "head-a"
    assert reviews == []
    assert comments == []


def test_consistent_snapshot_does_not_fetch_comments(monkeypatch):
    c = load_check()

    monkeypatch.setattr(c, "fetch_review_attestation", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_comment_attestation", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(c, "fetch_thread_snapshot", lambda pr, owner, name, gh: ("abc1234567890", [thread()]))

    nodes, head, reviews, comments = c.fetch_consistent_review_thread_snapshot(
        1431,
        "owner",
        "name",
        "gh",
        BOTS,
    )

    assert nodes == [thread()]
    assert head == "abc1234567890"
    assert reviews == []
    assert comments == []
