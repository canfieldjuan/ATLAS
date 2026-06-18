"""Drift guard: the free snapshot must stay a faithful, non-leaking projection
of the paid deflection report.

The snapshot and the structured report model are sibling projections of the
same FAQ result; neither derives from the other today, so a report-shape change
can silently desync the snapshot. This test pins the snapshot to the report so
that drift turns red, and reuses the submit smoke's canonical forbidden-field
detector so the paywall-leak half cannot fork its own denylist.

The boundary policy under test (see docs/frontend/content_ops_faq_report_contract.md):
- snapshot ranked coverage equals the report's ranked questions, no more/fewer;
- top_questions carry only index-safe fields: question/ticket_count are pinned to
  the report ranked rows, and weighted_frequency/customer_wording to the source
  FAQ items they project from (the report model does not re-expose those two);
- locked_questions expose rank + ticket_count only (never question text);
- the single teaser full answer is a genuinely scoped resolution_evidence row;
- no paid body/evidence field appears outside $.teaser.full_answer.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from extracted_content_pipeline.faq_deflection_report import (
    build_deflection_report_artifact,
    build_deflection_snapshot,
)
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownResult


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_content_ops_deflection_submit_handoff.py"
_SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_deflection_submit_handoff",
    SMOKE_SCRIPT,
)
assert _SPEC is not None
assert _SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(_SPEC)
# Register before exec so the module's frozen dataclasses can resolve their own
# __module__ during class creation (importlib does not auto-register).
sys.modules[_SPEC.name] = SMOKE
_SPEC.loader.exec_module(SMOKE)


def _drift_fixture_result() -> TicketFAQMarkdownResult:
    """Two scoped proven items + two review items so the snapshot exercises
    top_questions, locked_questions, the teaser full answer, and a preview."""

    return TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=14,
        ticket_source_count=14,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "customer_wording": "export attribution reports",
                "topic": "exports",
                "weighted_frequency": 9,
                "ticket_count": 6,
                "opportunity_score": 24,
                "answer": "Open Analytics, choose Attribution, then Download report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "steps": ["Open Analytics and select Download report."],
                "source_ids": ("ticket-export-1", "ticket-export-2", "ticket-export-3"),
                "evidence_quotes": ("`ticket-export-1` - Export attribution",),
                "term_mappings": (
                    {
                        "customer_term": "report download",
                        "documentation_term": "Download report",
                        "suggestion": "Use report download in the FAQ title.",
                        "source_id_count": 3,
                    },
                ),
            },
            {
                "question": "Can I turn on SSO for all users?",
                "customer_wording": "turn on SSO for all users",
                "topic": "sso",
                "weighted_frequency": 5,
                "ticket_count": 4,
                "opportunity_score": 17,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-sso-1", "ticket-sso-2"),
                "evidence_quotes": ("`ticket-sso-1` - SSO setup",),
            },
            {
                "question": "How do I update my billing card?",
                "customer_wording": "update billing card",
                "topic": "billing",
                "weighted_frequency": 4,
                "ticket_count": 3,
                "opportunity_score": 12,
                "answer": "Open Billing, choose Payment method, then Update card.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "steps": ["Open Billing and update the payment method."],
                "source_ids": ("ticket-billing-1", "ticket-billing-2"),
                "evidence_quotes": ("`ticket-billing-1` - Billing card",),
            },
            {
                "question": "Why does the API return 429 errors?",
                "customer_wording": "API returns 429",
                "topic": "api",
                "weighted_frequency": 3,
                "ticket_count": 2,
                "opportunity_score": 8,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-api-1", "ticket-api-2"),
                "evidence_quotes": ("`ticket-api-1` - rate limit",),
            },
        ),
    )


def _sections_by_id(report_model: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sections: dict[str, Mapping[str, Any]] = {}
    for section in report_model.get("sections", ()):
        if isinstance(section, Mapping) and isinstance(section.get("id"), str):
            sections[section["id"]] = section
    return sections


def _rows_by_rank(section: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    if not isinstance(section, Mapping):
        return rows
    data = section.get("data")
    if not isinstance(data, Mapping):
        return rows
    for row in data.get("rows", ()):
        if isinstance(row, Mapping) and isinstance(row.get("rank"), int):
            rows[row["rank"]] = row
    return rows


@pytest.fixture()
def artifact_snapshot() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact = build_deflection_report_artifact(_drift_fixture_result())
    # top_n=2 forces ranks 3-4 into the locked/teaser region.
    snapshot = build_deflection_snapshot(artifact, top_n=2).as_dict()
    report_model = artifact.report_model.as_dict()
    sections = _sections_by_id(report_model)
    return artifact, snapshot, report_model, sections


def test_snapshot_ranked_coverage_matches_report_ranked_questions(artifact_snapshot) -> None:
    _, snapshot, _, sections = artifact_snapshot
    ranked_ranks = set(_rows_by_rank(sections.get("ranked_questions")))

    snapshot_ranks = {q["rank"] for q in snapshot["top_questions"]}
    snapshot_ranks |= {q["rank"] for q in snapshot["locked_questions"]}
    full_answer = snapshot["teaser"]["full_answer"]
    if isinstance(full_answer, Mapping):
        snapshot_ranks.add(full_answer["rank"])

    # The snapshot must cover exactly the report's ranked questions: a report
    # that adds or drops a ranked question must change the snapshot in lockstep.
    assert snapshot_ranks == ranked_ranks


def test_snapshot_summary_is_derived_from_report(artifact_snapshot) -> None:
    artifact, snapshot, _, sections = artifact_snapshot
    summary = artifact.summary
    support_tax = sections["support_tax"]["data"]

    assert snapshot["summary"]["generated"] == summary["generated"]
    assert snapshot["summary"]["drafted_answer_count"] == summary["drafted_answer_count"]
    assert snapshot["summary"]["no_proven_answer_count"] == summary["no_proven_answer_count"]
    assert snapshot["summary"]["repeat_ticket_count"] == support_tax["repeat_ticket_count"]
    assert (
        snapshot["summary"]["non_repeat_ticket_count"]
        == support_tax["non_repeat_ticket_count"]
    )


def test_snapshot_top_questions_match_report_ranked_rows(artifact_snapshot) -> None:
    _, snapshot, _, sections = artifact_snapshot
    ranked = _rows_by_rank(sections.get("ranked_questions"))
    items_by_rank = {
        rank: item for rank, item in enumerate(_drift_fixture_result().items, start=1)
    }

    assert snapshot["top_questions"], "expected top questions in the snapshot"
    for question in snapshot["top_questions"]:
        # Pin the full contracted top-question shape, every field to its source,
        # so corrupting weighted_frequency or customer_wording also fails drift.
        assert set(question) == {
            "rank",
            "question",
            "ticket_count",
            "weighted_frequency",
            "customer_wording",
        }
        row = ranked[question["rank"]]
        item = items_by_rank[question["rank"]]
        assert question["question"] == row["question"]
        assert question["ticket_count"] == row["ticket_count"]
        assert question["weighted_frequency"] == item["weighted_frequency"]
        assert question["customer_wording"] == item["customer_wording"]


def test_snapshot_locked_questions_expose_rank_and_count_only(artifact_snapshot) -> None:
    _, snapshot, _, sections = artifact_snapshot
    ranked = _rows_by_rank(sections.get("ranked_questions"))

    assert snapshot["locked_questions"], "expected locked questions at top_n=2"
    for locked in snapshot["locked_questions"]:
        assert set(locked) == {"rank", "ticket_count"}
        assert locked["ticket_count"] == ranked[locked["rank"]]["ticket_count"]


def test_snapshot_teaser_full_answer_derives_from_scoped_report_detail(
    artifact_snapshot,
) -> None:
    _, snapshot, _, sections = artifact_snapshot
    full_answer = snapshot["teaser"]["full_answer"]
    assert isinstance(full_answer, Mapping), "fixture has scoped proven items"

    detail = _rows_by_rank(sections.get("question_details"))[full_answer["rank"]]
    # The one allowed free body must be a genuinely scoped, proven report row.
    assert detail["answer_evidence_status"] == "resolution_evidence"
    assert detail["resolution_evidence_scope"] == "scoped"
    assert full_answer["question"] == detail["question"]
    assert full_answer["answer"] == detail["answer"]
    assert list(full_answer["steps"]) == list(detail["steps"])


def test_snapshot_teaser_previews_withhold_body(artifact_snapshot) -> None:
    _, snapshot, _, _ = artifact_snapshot
    for preview in snapshot["teaser"]["previews"]:
        assert preview["body_withheld"] is True
        assert "answer" not in preview
        assert "steps" not in preview


def test_snapshot_carries_no_report_only_surfaces(artifact_snapshot) -> None:
    _, snapshot, _, _ = artifact_snapshot
    assert "markdown" not in snapshot
    assert "faq_result" not in snapshot
    assert "report_model" not in snapshot
    assert "evidence_export" not in snapshot


def test_snapshot_has_no_forbidden_paid_fields(artifact_snapshot) -> None:
    _, snapshot, _, _ = artifact_snapshot
    # Reuse the submit smoke's canonical detector so this guard cannot fork its
    # own denylist away from the paywall boundary the smoke enforces.
    assert SMOKE._forbidden_key_paths(snapshot) == []
    assert SMOKE._validate_snapshot(snapshot, label="snapshot") == []


def test_drift_detector_flags_injected_top_question_leak(artifact_snapshot) -> None:
    _, snapshot, _, _ = artifact_snapshot
    leaked = dict(snapshot)
    leaked["top_questions"] = [
        {**dict(snapshot["top_questions"][0]), "source_ids": ["ticket-export-1"]},
        *snapshot["top_questions"][1:],
    ]
    paths = SMOKE._forbidden_key_paths(leaked)
    assert any("source_ids" in path for path in paths)


def test_drift_detector_flags_answer_leak_outside_teaser_full_answer(
    artifact_snapshot,
) -> None:
    _, snapshot, _, _ = artifact_snapshot
    leaked = dict(snapshot)
    teaser = dict(snapshot["teaser"])
    # An answer body in a preview (not the full_answer) must still be a leak.
    previews = [dict(p) for p in teaser.get("previews", ())]
    # Fail closed: the fixture is built to produce a preview, so the guarantee
    # that a preview answer-body leak is detected must not silently vanish if a
    # future change stops producing previews.
    assert previews, "fixture must produce a teaser preview to exercise the leak guard"
    previews[0]["answer"] = "leaked body text"
    teaser["previews"] = previews
    leaked["teaser"] = teaser
    paths = SMOKE._forbidden_key_paths(leaked)
    assert any(path.endswith("answer") for path in paths)
