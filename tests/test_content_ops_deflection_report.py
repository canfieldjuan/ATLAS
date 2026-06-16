from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.faq_deflection_report import (
    DEFAULT_DEFLECTION_SEO_TARGET_LIMIT,
    DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
    DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION,
    DEFLECTION_REPORT_SCHEMA_VERSION,
    DEFLECTION_REPORT_SECTION_REGISTRY,
    build_deflection_evidence_export,
    build_deflection_full_report_qa_scorecard,
    build_deflection_report_model,
    build_deflection_snapshot,
    build_deflection_report_artifact,
    deflection_snapshot_content_opportunities,
    render_deflection_report_model,
    _report_section,
)
from extracted_content_pipeline.deflection_report_access import (
    InMemoryDeflectionReportArtifactStore,
    PostgresDeflectionReportArtifactStore,
    stored_deflection_report_model,
)
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)
from extracted_content_pipeline.ticket_faq_markdown import (
    TicketFAQMarkdownResult,
    build_ticket_faq_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_content_ops_deflection_report.py"
SAAS_DEMO = ROOT / "extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv"
SPEC = importlib.util.spec_from_file_location(
    "build_content_ops_deflection_report",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report_access_snapshot(question: str, *, generated: int = 1) -> dict[str, object]:
    return {
        "summary": {
            "generated": generated,
            "drafted_answer_count": 0,
            "no_proven_answer_count": generated,
        },
        "top_questions": [
            {
                "rank": 1,
                "question": question,
                "weighted_frequency": 7,
                "customer_wording": question,
                "answer": "Open Analytics and export the report.",
                "source_ids": ["ticket-1"],
                "evidence_quotes": ["ticket-1 says export is blocked"],
            }
        ],
    }


def _money(value: object) -> str:
    return f"${int(float(value) + 0.5):,}"


def _structured_report_fixture_result() -> TicketFAQMarkdownResult:
    return TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=8,
        ticket_source_count=8,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "customer_wording": "export attribution reports",
                "topic": "exports",
                "weighted_frequency": 8,
                "ticket_count": 5,
                "opportunity_score": 21,
                "answer": "To resolve this, open Analytics, choose Attribution, then Download report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "steps": ["Open Analytics and select Download report."],
                "source_ids": (
                    "ticket-export-1",
                    "ticket-export-2",
                    "ticket-export-3",
                    "ticket-export-4",
                    "ticket-export-5",
                ),
                "source_date_span": {
                    "start": "2026-05-01",
                    "end": "2026-05-05",
                    "missing_source_count": 0,
                },
                "evidence_quotes": ("`ticket-export-1` - Export attribution",),
                "term_mappings": (
                    {
                        "customer_term": "report download",
                        "documentation_term": "Download report",
                        "suggestion": "Use report download in the FAQ title.",
                        "source_id_count": 5,
                    },
                ),
            },
            {
                "question": "Can I turn on SSO for all users?",
                "customer_wording": "turn on SSO for all users",
                "topic": "sso",
                "weighted_frequency": 4,
                "ticket_count": 3,
                "opportunity_score": 13,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (
                    "ticket-sso-1",
                    "ticket-sso-2",
                    "ticket-sso-3",
                ),
                "source_date_span": {
                    "start": "2026-05-10",
                    "end": "2026-05-15",
                    "missing_source_count": 0,
                },
                "evidence_quotes": ("`ticket-sso-1` - SSO setup",),
                "outcome_diagnostics": {
                    "ticket_status_summary": {"resolved": 2, "reopened": 1},
                    "diagnostic_ticket_count": 3,
                    "outcome_risk_ticket_count": 1,
                    "reopened_ticket_count": 1,
                    "negative_csat_ticket_count": 0,
                },
            },
        ),
    )


_STRUCTURED_REPORT_GOLDEN_MARKDOWN = """# Support Ticket Deflection Report

## Support Tax Confirmation

This report found 8 question-level repeat tickets across 2 ranked questions. At the Gartner $13.50 assisted-contact benchmark, that repeated-question work sizes to about $108 of assisted-contact handling.

The source window is 2026-05-01 to 2026-05-15 (15 days). At the same measured daily pace, that is about $2,628 over 12 months.

Estimate only. This is not a savings guarantee; adjust the $13.50 benchmark to your own loaded support cost.

The full unlocked report below gives you every ranked question, the estimated support cost by question, publishable help-center copy where your uploaded resolutions prove the answer, the no-proven-answer roadmap, and a complete evidence export for audit/detail review.

- Publishable answers drafted from proven resolutions: 1
- Questions still needing an approved resolution: 1
- Ticket sources represented: 8

## Your Help-Desk SEO Targeting List

Use these source-backed phrases as help-center headings, internal-search synonyms, and FAQ wording. These were mined from the tickets you uploaded; this report does not claim keyword volume, search rank, or traffic.

1. export attribution reports
2. report download
3. turn on SSO for all users

## Ranked Question Opportunities

| Rank | Customer question | Tickets | Estimated support cost | Opportunity | Answer status | Source proof |
|---:|---|---:|---:|---:|---|---|
| 1 | How do I export attribution reports? | 5 | $68 | 21 | drafted from resolution evidence | 5 source tickets |
| 2 | Can I turn on SSO for all users? | 3 | $41 | 13 | no proven answer yet | 3 source tickets |

## Resolution Outcome Diagnostics

These status and CSAT signals flag answers that may need review. They do not prove a publishable answer; only uploaded resolution evidence can do that.

- Tickets with outcome diagnostics: 3
- Tickets with reopened or negative-CSAT risk: 1
- Reopened tickets: 1
- Negative CSAT tickets: 0

| Customer question | Status mix | Reopened | Negative CSAT | Guidance |
|---|---|---:|---:|---|
| Can I turn on SSO for all users? | reopened: 1, resolved: 2 | 1 | 0 | Review the answer before publishing because at least one ticket reopened. |

## Question Details and Evidence

Each ranked question appears once below with its answer status, publishable copy or review guidance, vocabulary gaps, and complete source evidence.

Questions without uploaded resolution evidence stay in review: outcome/status signals can prioritize them, but only resolution evidence can make an answer publishable.

### 1. How do I export attribution reports?

**Customer wording:** export attribution reports

**Answer status:** drafted from resolution evidence

**Ticket/support-cost context:** 5 tickets, estimated at $68 of assisted-contact handling.

**Publishable answer draft:**

To resolve this, open Analytics, choose Attribution, then Download report.

**Draft answer steps:**

1. Open Analytics and select Download report.

**Evidence backing:** Backed by 5 resolved tickets (ticket-export-1, ticket-export-2, ticket-export-3, +2 more). Complete source IDs are in this question detail block.

**Vocabulary gaps:**

- report download -> Download report: Use report download in the FAQ title. (5 sources)

**Complete evidence:**

**Source IDs (full list):** ticket-export-1, ticket-export-2, ticket-export-3, ticket-export-4, ticket-export-5

- `ticket-export-1` - Export attribution

### 2. Can I turn on SSO for all users?

**Customer wording:** turn on SSO for all users

**Answer status:** no proven answer yet

**Ticket/support-cost context:** 3 tickets, estimated at $41 of assisted-contact handling.

**No proven answer yet:**

No uploaded resolution evidence was present for this question.

**Ticket backing:** Seen in 3 repeated tickets (ticket-sso-1, ticket-sso-2, ticket-sso-3). Complete source IDs are in this question detail block.

**Complete evidence:**

**Source IDs (full list):** ticket-sso-1, ticket-sso-2, ticket-sso-3

- `ticket-sso-1` - SSO setup
"""


def test_deflection_report_model_keeps_current_markdown_golden_snapshot() -> None:
    result = _structured_report_fixture_result()

    artifact = build_deflection_report_artifact(result)
    rebuilt_model = build_deflection_report_model(result)

    assert artifact.markdown == _STRUCTURED_REPORT_GOLDEN_MARKDOWN
    assert render_deflection_report_model(artifact.report_model) == (
        _STRUCTURED_REPORT_GOLDEN_MARKDOWN
    )
    assert render_deflection_report_model(rebuilt_model) == (
        _STRUCTURED_REPORT_GOLDEN_MARKDOWN
    )


def test_deflection_report_artifact_exposes_structured_model_sections() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())

    payload = artifact.as_dict()["report_model"]
    section_by_id = {
        section["id"]: section
        for section in payload["sections"]
    }

    assert payload["schema_version"] == DEFLECTION_REPORT_SCHEMA_VERSION
    assert payload["title"] == "Support Ticket Deflection Report"
    assert payload["summary"]["generated"] == 2
    assert [section["id"] for section in payload["sections"]] == [
        "support_tax",
        "seo_targets",
        "ranked_questions",
        "outcome_diagnostics",
        "question_details",
        "complete_evidence",
    ]
    assert [section["priority"] for section in payload["sections"]] == [
        10,
        20,
        30,
        40,
        50,
        90,
    ]

    support_tax = section_by_id["support_tax"]
    assert support_tax["surfaces"] == ["web", "pdf", "email_summary", "markdown"]
    assert support_tax["required_data"] == [
        "repeat_ticket_count",
        "non_repeat_ticket_count",
        "generated_question_count",
        "assisted_contact_cost",
        "estimated_support_cost",
        "source_date_window",
        "drafted_answer_count",
        "no_proven_answer_count",
        "ticket_source_count",
    ]
    assert support_tax["data"]["repeat_ticket_count"] == 8
    assert support_tax["data"]["annualized_support_cost"] == 2628.0
    assert support_tax["data"]["source_date_window"] == {
        "source_date_start": "2026-05-01",
        "source_date_end": "2026-05-15",
        "source_window_days": 15,
    }
    assert "markdown_lines" not in support_tax

    seo_targets = section_by_id["seo_targets"]
    assert seo_targets["default_limit"] == DEFAULT_DEFLECTION_SEO_TARGET_LIMIT
    assert seo_targets["required_data"] == [
        "phrases",
        "total_phrase_count",
        "displayed_phrase_count",
        "omitted_phrase_count",
        "limit",
    ]
    assert seo_targets["data"]["phrases"] == [
        "export attribution reports",
        "report download",
        "turn on SSO for all users",
    ]
    assert seo_targets["data"]["omitted_phrase_count"] == 0

    ranked = section_by_id["ranked_questions"]["data"]["rows"]
    assert ranked[0] == {
        "rank": 1,
        "question": "How do I export attribution reports?",
        "ticket_count": 5,
        "estimated_support_cost": 67.5,
        "opportunity_score": 21,
        "answer_status": "drafted from resolution evidence",
        "source_proof": "5 source tickets",
    }

    diagnostics = section_by_id["outcome_diagnostics"]["data"]
    assert diagnostics["outcome_risk_ticket_count"] == 1
    assert diagnostics["rows"] == [
        {
            "question": "Can I turn on SSO for all users?",
            "status_mix": "reopened: 1, resolved: 2",
            "reopened_ticket_count": 1,
            "negative_csat_ticket_count": 0,
            "guidance": (
                "Review the answer before publishing because at least one "
                "ticket reopened."
            ),
        }
    ]

    details = section_by_id["question_details"]["data"]["rows"]
    assert details[0]["answer_linkage"] == "publishable_answer"
    assert details[0]["source_ids"] == [
        "ticket-export-1",
        "ticket-export-2",
        "ticket-export-3",
        "ticket-export-4",
        "ticket-export-5",
    ]
    assert details[1]["answer_linkage"] == "needs_review"
    assert details[1]["evidence_quotes"] == ["`ticket-sso-1` - SSO setup"]

    complete_evidence = section_by_id["complete_evidence"]
    assert complete_evidence["surfaces"] == ["export"]
    assert complete_evidence["required_data"] == [
        "question_count",
        "evidence_row_count",
        "source_id_count",
        "surfaces",
    ]
    assert complete_evidence["data"] == {
        "question_count": 2,
        "evidence_row_count": 8,
        "source_id_count": 8,
        "surfaces": ["export"],
    }


def test_deflection_report_section_registry_drives_section_metadata() -> None:
    artifact = build_deflection_report_artifact(
        _structured_report_fixture_result(),
        source_label="zendesk-trial-export.json",
    )
    section_by_id = {
        section.id: section
        for section in artifact.report_model.sections
    }

    assert set(section_by_id) == set(DEFLECTION_REPORT_SECTION_REGISTRY)
    assert len({
        definition.priority
        for definition in DEFLECTION_REPORT_SECTION_REGISTRY.values()
    }) == len(DEFLECTION_REPORT_SECTION_REGISTRY)

    for section_id, definition in DEFLECTION_REPORT_SECTION_REGISTRY.items():
        section = section_by_id[section_id]
        assert section.title == definition.title
        assert section.priority == definition.priority
        assert section.surfaces == definition.surfaces
        assert section.default_limit == definition.default_limit
        assert section.required_data == definition.required_data
        assert set(section.required_data) <= set(section.data)


def test_deflection_report_section_registry_fails_closed_on_missing_data() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "section 'seo_targets' missing required data keys: "
            "total_phrase_count, displayed_phrase_count, omitted_phrase_count, limit"
        ),
    ):
        _report_section(
            section_id="seo_targets",
            data={"phrases": ["export attribution reports"]},
            markdown_lines=["## Your Help-Desk SEO Targeting List"],
        )


def test_deflection_report_section_registry_rejects_unknown_section_id() -> None:
    with pytest.raises(
        ValueError,
        match="unknown deflection report section: surprise_appendix",
    ):
        _report_section(
            section_id="surprise_appendix",
            data={},
            markdown_lines=["## Surprise Appendix"],
        )


def test_deflection_report_model_support_tax_data_matches_markdown() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    section_by_id = {
        section.id: section
        for section in artifact.report_model.sections
    }
    support_tax = section_by_id["support_tax"]
    support_tax_markdown = "\n".join(support_tax.markdown_lines)

    assert str(support_tax.data["repeat_ticket_count"]) in support_tax_markdown
    assert str(support_tax.data["generated_question_count"]) in support_tax_markdown
    assert _money(support_tax.data["estimated_support_cost"]) in support_tax_markdown
    assert _money(support_tax.data["annualized_support_cost"]) in support_tax_markdown
    assert (
        f"{support_tax.data['source_date_window']['source_date_start']} to "
        f"{support_tax.data['source_date_window']['source_date_end']}"
    ) in support_tax_markdown


def test_deflection_report_partitions_proven_and_unproven_answers() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={
            "uses_user_vocabulary": True,
            "condensed": True,
            "has_action_items": True,
        },
        items=(
            {
                "question": "How do I export attribution reports?",
                "summary": "Customers ask how to export attribution reports.",
                "weighted_frequency": 8,
                "opportunity_score": 14,
                "answer": (
                    "To resolve this, open Analytics, choose Attribution, then "
                    "select Download report."
                ),
                "answer_evidence_status": "resolution_evidence",
                "steps": [
                    "Open Analytics, choose Attribution, then select Download report.",
                    "If it still does not work, contact support and include the cited ticket details.",
                ],
                "source_ids": ("ticket-1", "ticket-2"),
                "evidence_quotes": (
                    "`ticket-1` - Export question: How do I export attribution reports?",
                ),
                "term_mappings": (
                    {
                        "customer_term": "export",
                        "documentation_term": "Download report",
                        "suggestion": "Add export as an alternate heading.",
                        "source_id_count": 2,
                    },
                ),
            },
            {
                "question": "Why is the dashboard stale?",
                "summary": "Customers ask why dashboard totals are not refreshed.",
                "frequency": 3,
                "opportunity_score": 6,
                "answer_evidence_status": "draft_needs_review",
                "steps": [
                    "Review the cited ticket evidence and confirm the policy-approved answer before publishing.",
                ],
                "source_ids": ("ticket-3",),
                "evidence_quotes": (
                    "`ticket-3` - Stale dashboard: Why is the dashboard stale?",
                ),
            },
        ),
    )

    artifact = build_deflection_report_artifact(result)

    assert artifact.summary["drafted_answer_count"] == 1
    assert artifact.summary["no_proven_answer_count"] == 1
    assert "## Support Tax Confirmation" in artifact.markdown
    assert "Gartner $13.50 assisted-contact benchmark" in artifact.markdown
    assert "## Your Help-Desk SEO Targeting List" in artifact.markdown
    assert "## Ranked Question Opportunities" in artifact.markdown
    assert "## Question Details and Evidence" in artifact.markdown
    assert "## Publishable Help-Center Copy From Proven Resolutions" not in artifact.markdown
    assert "How do I export attribution reports?" in artifact.markdown
    assert "To resolve this, open Analytics" in artifact.markdown
    assert "Uploaded resolution evidence supports this draft answer" not in artifact.markdown
    assert "Open Analytics, choose Attribution" in artifact.markdown
    assert "Use the uploaded resolution evidence" not in artifact.markdown
    assert "**Sources:**" not in artifact.markdown
    assert "## No Proven Answer Yet" not in artifact.markdown
    assert "Why is the dashboard stale?" in artifact.markdown
    assert "No uploaded resolution evidence was present for this question." in artifact.markdown
    assert artifact.markdown.count("only resolution evidence can make an answer publishable") == 1
    assert "Customers repeatedly asked this question" not in artifact.markdown
    assert "No verified support resolution was present" not in artifact.markdown
    assert "export" in artifact.markdown
    assert "Download report" in artifact.markdown


def test_deflection_report_artifact_includes_complete_evidence_export() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "customer_wording": "export attribution reports",
                "topic": "exports",
                "weighted_frequency": 8,
                "ticket_count": 2,
                "opportunity_score": 14,
                "answer": "Open Analytics, choose Attribution, then select Download report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "steps": ["Open Analytics and select Download report."],
                "source_ids": ("ticket-1", "ticket-2"),
                "evidence_quotes": (
                    "`ticket-1` - Export question: How do I export reports?",
                ),
                "term_mappings": (
                    {
                        "customer_term": "export",
                        "documentation_term": "Download report",
                    },
                ),
                "outcome_diagnostics": {
                    "ticket_status_summary": {"resolved": 2},
                    "diagnostic_ticket_count": 2,
                },
            },
            {
                "question": "Why is the dashboard stale?",
                "weighted_frequency": 3,
                "ticket_count": 2,
                "opportunity_score": 6,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-3", "ticket-4"),
                "evidence_quotes": (
                    "`ticket-3` - Stale dashboard: why is the dashboard stale?",
                ),
            },
        ),
    )

    artifact = build_deflection_report_artifact(result)
    export = build_deflection_evidence_export(artifact)

    assert artifact.as_dict()["evidence_export"] == export
    assert export["schema_version"] == DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION
    assert export["summary"] == {
        "question_count": 2,
        "evidence_row_count": 4,
        "source_id_count": 4,
        "drafted_answer_count": 1,
        "no_proven_answer_count": 1,
    }
    assert export["questions"][0]["question_id"] == "q001"
    assert export["questions"][0]["answer_linkage"] == "publishable_answer"
    assert export["questions"][0]["term_mappings"] == [
        {"customer_term": "export", "documentation_term": "Download report"}
    ]
    assert export["questions"][0]["outcome_diagnostics"] == {
        "ticket_status_summary": {"resolved": 2},
        "diagnostic_ticket_count": 2,
    }
    assert export["questions"][1]["answer_linkage"] == "needs_review"
    assert export["evidence_rows"][0] == {
        "row_id": "q001-e001",
        "question_id": "q001",
        "rank": 1,
        "question": "How do I export attribution reports?",
        "source_id": "ticket-1",
        "source_field": "evidence_quote",
        "evidence_quote": "`ticket-1` - Export question: How do I export reports?",
        "answer_evidence_status": "resolution_evidence",
        "resolution_evidence_scope": "scoped",
        "answer_linkage": "publishable_answer",
    }
    assert export["evidence_rows"][1]["source_id"] == "ticket-2"
    assert export["evidence_rows"][1]["source_field"] == "source_id"
    assert export["evidence_rows"][1]["evidence_quote"] == ""
    assert export["evidence_rows"][2]["answer_linkage"] == "needs_review"


def test_deflection_full_report_qa_scorecard_anchors_counts_without_leaking_evidence() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "result_page": {
                "counts": {
                    "repeat_ticket_count": 8,
                    "generated_question_count": 2,
                    "drafted_answer_count": 1,
                    "no_proven_answer_count": 1,
                    "ticket_source_count": 8,
                    "evidence_row_count": 8,
                },
                "displayed_rows": {
                    "ranked_questions": 2,
                    "question_details": 2,
                    "seo_targets": 3,
                    "outcome_diagnostics": 1,
                },
            }
        },
    )

    assert scorecard["schema_version"] == (
        DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION
    )
    assert scorecard["ok"] is True
    assert scorecard["counts"]["repeat_ticket_count"] == 8
    assert scorecard["counts"]["evidence_row_count"] == 8
    assert scorecard["counts"]["question_detail_count"] == 2
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-export-1" not in encoded
    assert "Export attribution" not in encoded
    assert "ticket-sso-1" not in encoded
    assert "SSO setup" not in encoded


def test_deflection_full_report_qa_scorecard_fails_on_export_model_mismatch() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)
    bad_export = json.loads(json.dumps(export))
    bad_export["summary"]["evidence_row_count"] = 7
    bad_export["evidence_rows"] = bad_export["evidence_rows"][:-1]

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=bad_export,
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "evidence_export.summary.evidence_row_count" in failed
    assert "evidence_export.evidence_rows.length" in failed


def test_deflection_full_report_qa_scorecard_checks_surface_caps_behaviorally() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    model = json.loads(json.dumps(artifact.report_model.as_dict()))
    sections = {section["id"]: section for section in model["sections"]}
    support_tax = sections["support_tax"]["data"]
    ranked_rows = sections["ranked_questions"]["data"]["rows"]
    detail_rows = sections["question_details"]["data"]["rows"]
    diagnostic_rows = sections["outcome_diagnostics"]["data"]["rows"]
    seo_targets = sections["seo_targets"]["data"]
    complete_evidence = sections["complete_evidence"]["data"]

    support_tax["generated_question_count"] = 30
    ranked_rows[:] = [dict(ranked_rows[0], rank=index) for index in range(1, 31)]
    detail_rows[:] = [dict(detail_rows[0], rank=index) for index in range(1, 13)]
    diagnostic_rows[:] = [dict(diagnostic_rows[0]) for _ in range(30)]
    seo_targets["phrases"] = [f"phrase {index}" for index in range(1, 31)]
    seo_targets["total_phrase_count"] = 30
    seo_targets["displayed_phrase_count"] = 30
    complete_evidence["question_count"] = 30
    complete_evidence["evidence_row_count"] = 60
    complete_evidence["source_id_count"] = 60
    evidence_export = {
        "schema_version": DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
        "summary": {
            "question_count": 30,
            "evidence_row_count": 60,
            "source_id_count": 60,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 1,
        },
        "questions": [{} for _ in range(30)],
        "evidence_rows": [{} for _ in range(60)],
    }

    scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=evidence_export,
        surface_observations={
            "result_page": {
                "displayed_rows": {
                    "ranked_questions": 25,
                    "question_details": 10,
                    "seo_targets": 20,
                    "outcome_diagnostics": 25,
                },
            }
        },
    )
    bad_scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=evidence_export,
        surface_observations={
            "result_page": {"displayed_rows": {"ranked_questions": 26}},
        },
    )

    assert scorecard["ok"] is True
    assert bad_scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in bad_scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.result_page.displayed_rows.ranked_questions" in failed


def test_deflection_full_report_qa_scorecard_redacts_bad_observed_strings() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "result_page": {
                "counts": {"repeat_ticket_count": "ticket-export-1"},
            }
        },
    )

    assert scorecard["ok"] is False
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-export-1" not in encoded
    assert "<redacted-string>" in encoded


def test_deflection_snapshot_excludes_complete_evidence_export() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=2,
        ticket_source_count=2,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "weighted_frequency": 4,
                "ticket_count": 2,
                "answer": "Open Analytics, choose Attribution, then select Download report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "steps": ["Open Analytics and select Download report."],
                "source_ids": ("ticket-export-1", "ticket-export-2"),
                "evidence_quotes": (
                    "`ticket-export-1` - Export question: How do I export reports?",
                ),
            },
        ),
    )

    artifact = build_deflection_report_artifact(result)
    snapshot = build_deflection_snapshot(artifact).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert "evidence_export" in artifact.as_dict()
    assert "evidence_export" not in snapshot
    assert "evidence_export" not in encoded
    assert "report_model" in artifact.as_dict()
    assert "report_model" not in snapshot
    assert "report_model" not in encoded
    assert "evidence_quotes" not in encoded
    assert "source_ids" not in encoded
    assert "ticket-export-1" not in encoded


def test_deflection_report_reframes_paid_artifact_with_cost_and_seo_sections() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=8,
        ticket_source_count=8,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "customer_wording": "export attribution reports",
                "weighted_frequency": 8,
                "ticket_count": 5,
                "opportunity_score": 21,
                "answer": "To resolve this, open Analytics, choose Attribution, then Download report.",
                "answer_evidence_status": "resolution_evidence",
                "steps": ["Open Analytics and select Download report."],
                "source_ids": (
                    "ticket-export-1",
                    "ticket-export-2",
                    "ticket-export-3",
                    "ticket-export-4",
                    "ticket-export-5",
                ),
                "source_date_span": {
                    "start": "2026-05-01",
                    "end": "2026-05-05",
                    "missing_source_count": 0,
                },
                "evidence_quotes": ("`ticket-export-1` - Export attribution",),
                "term_mappings": (
                    {
                        "customer_term": "report download",
                        "documentation_term": "Download report",
                        "suggestion": "Use report download in the FAQ title.",
                        "source_id_count": 5,
                    },
                ),
            },
            {
                "question": "Can I turn on SSO for all users?",
                "customer_wording": "turn on SSO for all users",
                "weighted_frequency": 4,
                "ticket_count": 3,
                "opportunity_score": 13,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (
                    "ticket-sso-1",
                    "ticket-sso-2",
                    "ticket-sso-3",
                ),
                "source_date_span": {
                    "start": "2026-05-10",
                    "end": "2026-05-15",
                    "missing_source_count": 0,
                },
                "evidence_quotes": ("`ticket-sso-1` - SSO setup",),
            },
        ),
    )

    markdown = build_deflection_report_artifact(result).markdown
    details = markdown.split("## Question Details and Evidence", 1)[1]
    export_block = details.split("### 1. How do I export attribution reports?", 1)[1].split(
        "### 2. Can I turn on SSO for all users?",
        1,
    )[0]
    sso_block = details.split("### 2. Can I turn on SSO for all users?", 1)[1]

    assert markdown.startswith("# Support Ticket Deflection Report\n\n## Support Tax Confirmation")
    assert "8 question-level repeat tickets across 2 ranked questions" in markdown
    assert "repeat-ticket hits" not in markdown
    assert "ranked repeat questions" not in markdown
    assert "every ranked repeat question" not in markdown
    assert "Gartner $13.50 assisted-contact benchmark" in markdown
    assert "about $108 of assisted-contact handling" in markdown
    assert "2026-05-01 to 2026-05-15 (15 days)" in markdown
    assert "about $2,628 over 12 months" in markdown
    assert "Estimate only. This is not a savings guarantee" in markdown
    assert "## Your Help-Desk SEO Targeting List" in markdown
    assert "export attribution reports" in markdown
    assert "report download" in markdown
    assert "turn on SSO for all users" in markdown
    assert "does not claim keyword volume, search rank, or traffic" in markdown
    assert (
        "| Rank | Customer question | Tickets | Estimated support cost | Opportunity |"
        in markdown
    )
    assert "| 1 | How do I export attribution reports? | 5 | $68 | 21 |" in markdown
    assert "| 2 | Can I turn on SSO for all users? | 3 | $41 | 13 |" in markdown
    assert (
        "Backed by 5 resolved tickets (ticket-export-1, ticket-export-2, "
        "ticket-export-3, +2 more)."
    ) in export_block
    assert "Seen in 3 repeated tickets (ticket-sso-1, ticket-sso-2, ticket-sso-3)." in sso_block
    assert "**Sources:**" not in markdown
    assert "## Publishable Help-Center Copy From Proven Resolutions" not in markdown
    assert "## No Proven Answer Yet" not in markdown
    assert "## Vocabulary Gaps" not in markdown
    assert "## Evidence Appendix" not in markdown
    assert (
        "**Source IDs (full list):** ticket-export-1, ticket-export-2, "
        "ticket-export-3, ticket-export-4, ticket-export-5"
    ) in export_block
    assert "**Source IDs (full list):** ticket-sso-1, ticket-sso-2, ticket-sso-3" in sso_block
    assert "`ticket-export-1` - Export attribution" in export_block
    assert "`ticket-sso-1` - SSO setup" in sso_block
    assert "will rank" not in markdown
    assert "search volume" not in markdown
    assert "guaranteed traffic" not in markdown


def test_deflection_report_caps_seo_index_without_capping_details_or_evidence() -> None:
    source_ids = tuple(f"ticket-overflow-{index}" for index in range(1, 61))
    evidence_quotes = tuple(
        f"`ticket-overflow-{index}` - Overflow evidence {index}"
        for index in range(1, 8)
    )
    items = tuple(
        {
            "question": f"Canonical question {index}",
            "customer_wording": f"Customer phrase {index}",
            "ticket_count": 1,
            "answer_evidence_status": "resolution_evidence",
            "answer": "Use the documented workflow.",
            "source_ids": source_ids if index == 1 else (f"ticket-{index}",),
            "evidence_quotes": (
                evidence_quotes if index == 1 else (f"`ticket-{index}` - Evidence",)
            ),
        }
        for index in range(1, DEFAULT_DEFLECTION_SEO_TARGET_LIMIT + 3)
    )
    markdown = build_deflection_report_artifact(
        TicketFAQMarkdownResult(
            markdown="# FAQ",
            source_count=len(items),
            ticket_source_count=len(items),
            output_checks={"condensed": True},
            items=items,
        )
    ).markdown

    seo_section = markdown.split("## Your Help-Desk SEO Targeting List", 1)[1].split(
        "## Ranked Question Opportunities",
        1,
    )[0]
    detail_section = markdown.split("## Question Details and Evidence", 1)[1]
    seo_lines = [
        line for line in seo_section.splitlines()
        if line.split(".", 1)[0].isdigit()
    ]

    assert len(seo_lines) == DEFAULT_DEFLECTION_SEO_TARGET_LIMIT
    assert (
        f"{DEFAULT_DEFLECTION_SEO_TARGET_LIMIT}. "
        f"Customer phrase {DEFAULT_DEFLECTION_SEO_TARGET_LIMIT}"
    ) in seo_section
    assert f"Customer phrase {DEFAULT_DEFLECTION_SEO_TARGET_LIMIT + 1}" not in seo_section
    assert "2 additional source-backed phrases remain represented" in seo_section
    assert (
        f"### {DEFAULT_DEFLECTION_SEO_TARGET_LIMIT + 1}. "
        f"Canonical question {DEFAULT_DEFLECTION_SEO_TARGET_LIMIT + 1}"
    ) in detail_section
    assert (
        f"**Customer wording:** "
        f"Customer phrase {DEFAULT_DEFLECTION_SEO_TARGET_LIMIT + 1}"
    ) in detail_section
    assert "ticket-overflow-60" in detail_section
    assert "`ticket-overflow-7` - Overflow evidence 7" in detail_section


def test_deflection_report_does_not_show_seo_cap_note_at_exact_limit() -> None:
    items = tuple(
        {
            "question": f"Exact phrase {index}",
            "customer_wording": f"Exact phrase {index}",
            "ticket_count": 1,
            "answer_evidence_status": "draft_needs_review",
            "source_ids": (f"ticket-{index}",),
            "evidence_quotes": (f"`ticket-{index}` - Evidence",),
        }
        for index in range(1, DEFAULT_DEFLECTION_SEO_TARGET_LIMIT + 1)
    )
    markdown = build_deflection_report_artifact(
        TicketFAQMarkdownResult(
            markdown="# FAQ",
            source_count=len(items),
            ticket_source_count=len(items),
            output_checks={"condensed": True},
            items=items,
        )
    ).markdown

    seo_section = markdown.split("## Your Help-Desk SEO Targeting List", 1)[1].split(
        "## Ranked Question Opportunities",
        1,
    )[0]

    assert (
        f"{DEFAULT_DEFLECTION_SEO_TARGET_LIMIT}. "
        f"Exact phrase {DEFAULT_DEFLECTION_SEO_TARGET_LIMIT}"
    ) in seo_section
    assert "SEO phrase index capped" not in seo_section
    assert "**Customer wording:** Exact phrase 1" not in markdown


def test_deflection_report_uses_unknown_window_cost_fallback_without_inference() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=2,
        ticket_source_count=2,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export reports?",
                "ticket_count": 2,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-1", "ticket-2"),
            },
        ),
    )

    artifact = build_deflection_report_artifact(result)
    markdown = artifact.markdown
    support_tax = next(
        section
        for section in artifact.report_model.sections
        if section.id == "support_tax"
    )
    support_tax_markdown = "\n".join(support_tax.markdown_lines)

    assert "2 question-level repeat tickets across 1 ranked questions" in markdown
    assert "repeat-ticket hits" not in markdown
    assert "ranked repeat questions" not in markdown
    assert "about $27 of assisted-contact handling" in markdown
    assert "did not receive a complete source-date window" in markdown
    assert "does not infer a monthly or annual reporting period" in markdown
    assert "If this uploaded batch is monthly pace" in markdown
    assert "the 12-month run-rate would be about $324" in markdown
    assert "same measured daily pace" not in markdown
    assert "monthly_pace_support_cost" not in support_tax.data
    assert support_tax.data["annualized_run_rate_support_cost"] == 324.0
    assert _money(support_tax.data["estimated_support_cost"]) in support_tax_markdown
    assert (
        _money(support_tax.data["annualized_run_rate_support_cost"])
        in support_tax_markdown
    )


def test_deflection_report_does_not_put_review_needed_steps_in_drafted_section() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=1,
        ticket_source_count=1,
        output_checks={
            "uses_user_vocabulary": True,
            "condensed": True,
            "has_action_items": True,
        },
        items=(
            {
                "question": "Can I update permissions?",
                "summary": "Customers ask about permissions.",
                "frequency": 1,
                "opportunity_score": 1,
                "answer_evidence_status": "draft_needs_review",
                "steps": [
                    "Review the cited ticket evidence and confirm the policy-approved answer before publishing.",
                ],
                "source_ids": ("ticket-1",),
            },
        ),
    )

    markdown = build_deflection_report_artifact(result).markdown
    details = markdown.split("## Question Details and Evidence", 1)[1]

    assert "No uploaded resolution evidence was present for this question." in details
    assert "Review the cited ticket evidence" not in markdown
    assert "Can I update permissions?" in details


def test_deflection_snapshot_strips_answers_evidence_and_sources() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "question_source": "customer_wording",
                "weighted_frequency": 8,
                "ticket_count": 4,
                "answer": "Customers mention export trouble.",
                "steps": ["Open Analytics and download the report."],
                "evidence_quotes": ("ticket-export-1 said export failed",),
                "source_ids": ("ticket-export-1",),
                "answer_evidence_status": "resolution_evidence",
            },
            {
                "question": "Can I enable SSO?",
                "question_source": "customer_wording",
                "weighted_frequency": 6,
                "steps": ["Review before publishing."],
                "evidence_quotes": ("ticket-sso-1 asked about SSO",),
                "source_ids": ("ticket-sso-1",),
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    )
    artifact = build_deflection_report_artifact(result)

    snapshot = build_deflection_snapshot(artifact, top_n=1).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot == {
        "summary": {
            "generated": 2,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 1,
            "support_ticket_resolution_evidence_count": 1,
            "support_ticket_resolution_evidence_present": True,
            # #1460: only questions asked at least twice count as repeats; the
            # single-ticket SSO item is reported separately.
            "repeat_ticket_count": 4,
            "non_repeat_ticket_count": 1,
        },
        "top_questions": [
            {
                "rank": 1,
                "question": "How do I export attribution reports?",
                "ticket_count": 4,
                "weighted_frequency": 8,
                "customer_wording": "How do I export attribution reports?",
            }
        ],
        "locked_questions": [
            {
                "rank": 2,
                "ticket_count": 1,
            }
        ],
        "teaser": {"full_answer": None, "previews": []},
    }
    assert "Open Analytics" not in encoded
    assert "ticket-export-1" not in encoded
    assert "evidence_quotes" not in encoded
    assert "source_ids" not in encoded


def test_deflection_snapshot_marks_question_only_exports_absent_resolution_evidence() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=2,
        ticket_source_count=2,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I reset my password?",
                "question_source": "customer_wording",
                "weighted_frequency": 5,
                "ticket_count": 2,
                "steps": ["Review before publishing."],
                "evidence_quotes": ("ticket-login-1 asked how to reset a password",),
                "source_ids": ("ticket-login-1",),
                "answer_evidence_status": "draft_needs_review",
            },
            {
                "question": "Where do I update my email?",
                "question_source": "customer_wording",
                "weighted_frequency": 4,
                "ticket_count": 2,
                "steps": ["Review before publishing."],
                "evidence_quotes": ("ticket-email-1 asked where to update email",),
                "source_ids": ("ticket-email-1",),
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    )

    snapshot = build_deflection_snapshot(build_deflection_report_artifact(result)).as_dict()

    assert snapshot["summary"]["support_ticket_resolution_evidence_count"] == 0
    assert snapshot["summary"]["support_ticket_resolution_evidence_present"] is False
    assert snapshot["summary"]["drafted_answer_count"] == 0
    assert snapshot["summary"]["no_proven_answer_count"] == 2


def test_deflection_snapshot_counts_are_raw_and_locked_rows_hide_questions() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=10,
        ticket_source_count=10,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export attribution reports?",
                "weighted_frequency": 99,
                "ticket_count": 7,
                "source_ids": ("ticket-export-1",),
                "answer_evidence_status": "draft_needs_review",
            },
            {
                "question": "Can I enable SSO?",
                "weighted_frequency": 88,
                "source_ids": ("ticket-sso-1", "ticket-sso-2"),
                "answer_evidence_status": "draft_needs_review",
            },
            {
                "question": "Weighted score is not a ticket count",
                "weighted_frequency": 77,
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    )

    snapshot = build_deflection_snapshot(
        build_deflection_report_artifact(result),
        top_n=1,
    ).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["summary"]["repeat_ticket_count"] == 9
    assert snapshot["top_questions"] == [
        {
            "rank": 1,
            "question": "How do I export attribution reports?",
            "ticket_count": 7,
            "weighted_frequency": 99,
            "customer_wording": "",
        }
    ]
    assert snapshot["locked_questions"] == [
        {"rank": 2, "ticket_count": 2},
        {"rank": 3, "ticket_count": 0},
    ]
    assert "Can I enable SSO?" not in encoded
    assert "Weighted score is not a ticket count" not in encoded
    assert "ticket-sso-1" not in encoded


def test_deflection_snapshot_exposes_complete_source_date_window() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "ticket-export-1",
                "source_type": "support_ticket",
                "source_title": "Export report",
                "text": "How do I export attribution reports?",
                "created_at": "2026-05-01T13:00:00Z",
                "resolution_text": "Open Analytics and download the report.",
            },
            {
                "source_id": "ticket-sso-1",
                "source_type": "support_ticket",
                "source_title": "SSO setup",
                "text": "Can I enable SSO for my workspace?",
                "created_at": "2026-05-15",
                "resolution_text": "Open Settings, choose Security, then enable SSO.",
            },
        ],
        max_items=2,
    )

    artifact = build_deflection_report_artifact(result)
    snapshot = build_deflection_snapshot(artifact).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert artifact.summary["source_date_start"] == "2026-05-01"
    assert artifact.summary["source_date_end"] == "2026-05-15"
    assert artifact.summary["source_window_days"] == 15
    assert snapshot["summary"]["source_date_start"] == "2026-05-01"
    assert snapshot["summary"]["source_date_end"] == "2026-05-15"
    assert snapshot["summary"]["source_window_days"] == 15
    assert "ticket-export-1" not in encoded
    assert "source_date_span" not in encoded


def test_deflection_snapshot_omits_date_window_when_source_dates_are_partial() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "ticket-export-1",
                "source_type": "support_ticket",
                "source_title": "Export report",
                "text": "How do I export attribution reports?",
                "created_at": "2026-05-01",
            },
            {
                "source_id": "ticket-sso-1",
                "source_type": "support_ticket",
                "source_title": "SSO setup",
                "text": "Can I enable SSO for my workspace?",
            },
        ],
        max_items=2,
    )

    artifact = build_deflection_report_artifact(result)
    snapshot = build_deflection_snapshot(artifact).as_dict()

    assert "source_date_start" not in artifact.summary
    assert "source_date_end" not in artifact.summary
    assert "source_window_days" not in artifact.summary
    assert "source_date_start" not in snapshot["summary"]
    assert "source_date_end" not in snapshot["summary"]
    assert "source_window_days" not in snapshot["summary"]


def test_deflection_report_surfaces_outcome_risk_diagnostics() -> None:
    package = build_support_ticket_input_package(
        [
            {
                "source_id": "zd-refund-1",
                "source_type": "support_ticket",
                "source_title": "Duplicate charge refund",
                "description": "How do I get a duplicate charge refunded?",
                "resolution_text": (
                    "Open Billing > Payments, select the duplicate charge, "
                    "choose Refund, enter the customer note, and confirm the refund."
                ),
                "ticket_status": "resolved",
                "csat": "good",
            },
            {
                "source_id": "zd-refund-2",
                "source_type": "support_ticket",
                "source_title": "Duplicate charge reopened",
                "description": "How do I get a duplicate charge refunded?",
                "resolution_text": (
                    "Open Billing > Payments, select the duplicate charge, "
                    "choose Refund, enter the customer note, and confirm the refund."
                ),
                "ticket_status": "reopened",
                "csat": "bad",
            },
            {
                "source_id": "zd-refund-3",
                "source_type": "support_ticket",
                "source_title": "Duplicate charge low CSAT",
                "description": "How do I get a duplicate charge refunded?",
                "resolution_text": (
                    "Open Billing > Payments, select the duplicate charge, "
                    "choose Refund, enter the customer note, and confirm the refund."
                ),
                "ticket_status": "resolved",
                "csat": "2",
            },
        ],
    )
    result = build_ticket_faq_markdown(
        package.inputs["source_material"],
        max_items=1,
    )

    artifact = build_deflection_report_artifact(result)
    item = artifact.faq_result.items[0]
    diagnostics = item["outcome_diagnostics"]
    snapshot = build_deflection_snapshot(artifact).as_dict()
    encoded_snapshot = json.dumps(snapshot, sort_keys=True)

    assert item["answer_evidence_status"] == "resolution_evidence"
    assert diagnostics == {
        "csat_present_count": 3,
        "csat_score_average": 2.0,
        "diagnostic_ticket_count": 3,
        "negative_csat_ticket_count": 2,
        "outcome_risk_ticket_count": 2,
        "reopened_ticket_count": 1,
        "ticket_status_summary": {"reopened": 1, "resolved": 2},
    }
    assert artifact.summary["outcome_diagnostics_present"] is True
    assert artifact.summary["outcome_diagnostic_ticket_count"] == 3
    assert artifact.summary["outcome_risk_ticket_count"] == 2
    assert artifact.summary["reopened_ticket_count"] == 1
    assert artifact.summary["negative_csat_ticket_count"] == 2
    assert artifact.summary["csat_present_count"] == 3
    assert artifact.summary["ticket_status_summary"] == {
        "reopened": 1,
        "resolved": 2,
    }
    assert "## Resolution Outcome Diagnostics" in artifact.markdown
    assert "They do not prove a publishable answer" in artifact.markdown
    assert "Tickets with reopened or negative-CSAT risk: 2" in artifact.markdown
    assert "reopened: 1, resolved: 2" in artifact.markdown
    assert "Review the answer before publishing because at least one ticket reopened." in (
        artifact.markdown
    )
    assert "outcome" not in encoded_snapshot
    assert "csat" not in encoded_snapshot.lower()


def test_outcome_diagnostics_do_not_create_publishable_answers_without_resolution() -> None:
    package = build_support_ticket_input_package(
        [
            {
                "source_id": "zd-export-1",
                "source_type": "support_ticket",
                "source_title": "Export permission reopened",
                "description": "What permission do I need for account exports?",
                "ticket_status": "reopened",
                "csat": "bad",
            },
            {
                "source_id": "zd-export-2",
                "source_type": "support_ticket",
                "source_title": "Export permission low CSAT",
                "description": "What permission do I need for account exports?",
                "ticket_status": "resolved",
                "csat": "1",
            },
        ],
    )
    result = build_ticket_faq_markdown(
        package.inputs["source_material"],
        max_items=1,
    )

    artifact = build_deflection_report_artifact(result)
    item = artifact.faq_result.items[0]
    details = artifact.markdown.split("## Question Details and Evidence", 1)[1]

    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["outcome_diagnostics"]["outcome_risk_ticket_count"] == 2
    assert artifact.summary["drafted_answer_count"] == 0
    assert artifact.summary["no_proven_answer_count"] == 1
    assert "**Publishable answer draft:**" not in details
    assert "What permission do I need for account exports?" in details
    assert "No uploaded resolution evidence was present for this question." in details
    assert "No verified support resolution was present" not in details
    assert "## Resolution Outcome Diagnostics" in artifact.markdown
    assert "They do not prove a publishable answer" in artifact.markdown


def test_outcome_diagnostics_count_duplicate_evidence_by_source_ticket() -> None:
    resolution_text = (
        "Open Billing > Payments, select the duplicate charge, choose Refund, "
        "enter the customer note, and confirm the refund."
    )
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "zd-duplicate-1",
                "source_type": "support_ticket",
                "source_title": "Duplicate charge thread",
                "support_ticket_cluster": "duplicate charge refund",
                "evidence": [
                    {
                        "source_id": "zd-duplicate-1",
                        "source_type": "support_ticket",
                        "source_title": "Duplicate charge thread",
                        "text": "How do I get the duplicate charge refunded?",
                        "resolution_text": resolution_text,
                        "ticket_status_state": "reopened",
                        "csat": "bad",
                        "csat_score": 1,
                    },
                    {
                        "source_id": "zd-duplicate-1",
                        "source_type": "support_ticket",
                        "source_title": "Duplicate charge thread",
                        "text": "The same duplicate charge still needs a refund.",
                        "resolution_text": resolution_text,
                        "ticket_status_state": "reopened",
                        "csat": "bad",
                        "csat_score": 1,
                    },
                ],
            },
        ],
        max_items=1,
    )

    artifact = build_deflection_report_artifact(result)
    diagnostics = artifact.faq_result.items[0]["outcome_diagnostics"]

    assert diagnostics == {
        "csat_present_count": 1,
        "csat_score_average": 1.0,
        "diagnostic_ticket_count": 1,
        "negative_csat_ticket_count": 1,
        "outcome_risk_ticket_count": 1,
        "reopened_ticket_count": 1,
        "ticket_status_summary": {"reopened": 1},
    }
    assert artifact.summary["outcome_diagnostic_ticket_count"] == 1
    assert artifact.summary["reopened_ticket_count"] == 1
    assert artifact.summary["negative_csat_ticket_count"] == 1
    assert "Tickets with outcome diagnostics: 1" in artifact.markdown
    assert "Reopened tickets: 1" in artifact.markdown


def test_outcome_diagnostics_normalize_direct_raw_status_and_csat() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "direct-1",
                "source_type": "support_ticket",
                "source_title": "Refund reopened",
                "text": "How do I get a duplicate charge refunded?",
                "support_ticket_cluster": "duplicate charge refund",
                "ticket_status": "reopened",
                "csat": "bad",
            },
            {
                "source_id": "direct-2",
                "source_type": "support_ticket",
                "source_title": "Refund resolved",
                "text": "How do I get a duplicate charge refunded?",
                "support_ticket_cluster": "duplicate charge refund",
                "status": "Closed",
                "satisfaction_rating": "3",
            },
        ],
        max_items=1,
    )

    artifact = build_deflection_report_artifact(result)
    diagnostics = artifact.faq_result.items[0]["outcome_diagnostics"]

    assert artifact.faq_result.items[0]["answer_evidence_status"] == "draft_needs_review"
    assert diagnostics == {
        "csat_present_count": 2,
        "csat_score_average": 3.0,
        "diagnostic_ticket_count": 2,
        "negative_csat_ticket_count": 1,
        "outcome_risk_ticket_count": 1,
        "reopened_ticket_count": 1,
        "ticket_status_summary": {"reopened": 1, "resolved": 1},
    }
    assert artifact.summary["outcome_diagnostic_ticket_count"] == 2
    assert artifact.summary["outcome_risk_ticket_count"] == 1
    assert "Tickets with reopened or negative-CSAT risk: 1" in artifact.markdown
    assert "reopened: 1, resolved: 1" in artifact.markdown


def test_deflection_snapshot_omits_contradictory_summary_date_window() -> None:
    snapshot = build_deflection_snapshot(
        {
            "summary": {
                "generated": 1,
                "drafted_answer_count": 0,
                "no_proven_answer_count": 1,
                "source_date_start": "2026-05-01",
                "source_date_end": "2026-05-15",
                "source_window_days": 30,
            },
            "faq_result": {
                "items": [
                    {
                        "question": "Can I enable SSO?",
                        "source_ids": ["ticket-sso-1"],
                    }
                ],
            },
        }
    ).as_dict()

    assert "source_date_start" not in snapshot["summary"]
    assert "source_date_end" not in snapshot["summary"]
    assert "source_window_days" not in snapshot["summary"]


def test_deflection_snapshot_includes_bounded_fail_closed_teaser() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=6,
        ticket_source_count=6,
        output_checks={"condensed": True},
        items=(
            _scoped_answer_item(1, "How do I export reports?", "Locked top answer one"),
            _scoped_answer_item(2, "How do I update billing?", "Locked top answer two"),
            _scoped_answer_item(3, "How do I enable SSO?", "Locked top answer three"),
            _scoped_answer_item(
                4,
                "How do we rotate API tokens?",
                "Create the replacement token, deploy it, then revoke the old token.",
                step="Create the replacement token before revoking the old one.",
            ),
            {
                **_scoped_answer_item(
                    5,
                    "Why is the dashboard stale?",
                    "Mismatched answer must stay locked.",
                ),
                "resolution_evidence_scope": "scope_mismatch",
            },
            _scoped_answer_item(6, "How do I add a teammate?", "Tail answer locked"),
        ),
    )

    snapshot = build_deflection_snapshot(
        build_deflection_report_artifact(result),
        top_n=3,
        teaser_preview_count=2,
    ).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["summary"]["generated"] == 6
    assert len(snapshot["top_questions"]) == 3
    assert snapshot["teaser"]["full_answer"] == {
        "rank": 1,
        "question": "How do I export reports?",
        "answer": "Locked top answer one",
        "steps": ["Step for How do I export reports?"],
        "answer_evidence_status": "resolution_evidence",
        "resolution_evidence_scope": "scoped",
        "weighted_frequency": 1,
        "source_count": 1,
    }
    assert [preview["rank"] for preview in snapshot["teaser"]["previews"]] == [2, 3]
    for preview in snapshot["teaser"]["previews"]:
        assert preview["body_withheld"] is True
        assert preview["answer_evidence_status"] == "resolution_evidence"
        assert preview["resolution_evidence_scope"] == "scoped"
        assert "answer" not in preview
        assert "steps" not in preview

    assert "Locked top answer one" in encoded
    assert "Locked top answer two" not in encoded
    assert "Locked top answer three" not in encoded
    assert "Mismatched answer must stay locked" not in encoded
    assert "Tail answer locked" not in encoded
    assert "ticket-" not in encoded
    assert "evidence_quotes" not in encoded
    assert "source_ids" not in encoded


def test_deflection_snapshot_teaser_falls_through_to_next_eligible_rank() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=3,
        ticket_source_count=3,
        output_checks={"condensed": True},
        items=(
            {
                **_scoped_answer_item(
                    1,
                    "How do I export reports?",
                    "Blocked top answer must not leak.",
                ),
                "answer_evidence_status": "draft_needs_review",
            },
            _scoped_answer_item(2, "How do I update billing?", "Second answer"),
            _scoped_answer_item(3, "How do I enable SSO?", "Third answer"),
        ),
    )

    snapshot = build_deflection_snapshot(
        build_deflection_report_artifact(result),
        top_n=2,
        teaser_preview_count=1,
    ).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["teaser"]["full_answer"] == {
        "rank": 2,
        "question": "How do I update billing?",
        "answer": "Second answer",
        "steps": ["Step for How do I update billing?"],
        "answer_evidence_status": "resolution_evidence",
        "resolution_evidence_scope": "scoped",
        "weighted_frequency": 2,
        "source_count": 1,
    }
    assert [preview["rank"] for preview in snapshot["teaser"]["previews"]] == [3]
    assert "Blocked top answer must not leak" not in encoded
    assert "Third answer" not in encoded


def test_deflection_snapshot_omits_full_teaser_rank_from_locked_rows() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={"condensed": True},
        items=(
            {
                **_scoped_answer_item(
                    1,
                    "How do I export reports?",
                    "Blocked top answer must not leak.",
                ),
                "answer_evidence_status": "draft_needs_review",
            },
            {
                **_scoped_answer_item(
                    2,
                    "How do I update billing?",
                    "Blocked second answer must not leak.",
                ),
                "answer_evidence_status": "draft_needs_review",
            },
            _scoped_answer_item(3, "How do I enable SSO?", "Third answer"),
            _scoped_answer_item(4, "How do I rotate tokens?", "Fourth answer"),
        ),
    )

    snapshot = build_deflection_snapshot(
        build_deflection_report_artifact(result),
        top_n=1,
        teaser_preview_count=1,
    ).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["teaser"]["full_answer"]["rank"] == 3
    assert snapshot["locked_questions"] == [
        {"rank": 2, "ticket_count": 1},
        {"rank": 4, "ticket_count": 1},
    ]
    assert "Third answer" in encoded
    assert "Blocked top answer must not leak" not in encoded
    assert "Blocked second answer must not leak" not in encoded
    assert "Fourth answer" not in encoded


def test_deflection_snapshot_teaser_empty_when_no_scoped_resolution_evidence() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=2,
        ticket_source_count=2,
        output_checks={"condensed": True},
        items=(
            {
                **_scoped_answer_item(1, "How do I export reports?", "Locked answer"),
                "resolution_evidence_scope": "scope_mismatch",
            },
            {
                "question": "Scoped status without answer body stays hidden",
                "weighted_frequency": 1,
                "answer": "",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": ("ticket-2",),
            },
        ),
    )

    snapshot = build_deflection_snapshot(build_deflection_report_artifact(result)).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["teaser"] == {"full_answer": None, "previews": []}
    assert "Locked answer" not in encoded
    assert "ticket-1" not in encoded


def test_deflection_snapshot_rejects_non_positive_top_n() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=0,
        ticket_source_count=0,
        output_checks={},
        items=(),
    )

    with pytest.raises(ValueError, match="top_n must be positive"):
        build_deflection_snapshot(build_deflection_report_artifact(result), top_n=0)


def test_deflection_snapshot_rejects_negative_teaser_preview_count() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=0,
        ticket_source_count=0,
        output_checks={},
        items=(),
    )

    with pytest.raises(ValueError, match="teaser_preview_count must be non-negative"):
        build_deflection_snapshot(
            build_deflection_report_artifact(result),
            teaser_preview_count=-1,
        )


def _scoped_answer_item(
    rank: int,
    question: str,
    answer: str,
    *,
    step: str | None = None,
) -> dict[str, object]:
    return {
        "question": question,
        "weighted_frequency": rank,
        "answer": answer,
        "steps": [step or f"Step for {question}"],
        "answer_evidence_status": "resolution_evidence",
        "resolution_evidence_scope": "scoped",
        "source_ids": (f"ticket-{rank}",),
        "evidence_quotes": (f"`ticket-{rank}` - {question}",),
    }


@pytest.mark.asyncio
async def test_postgres_deflection_report_store_round_trips_paid_gate() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.rows: dict[tuple[str, str], dict[str, object]] = {}

        async def execute(self, query: str, *args: object) -> str:
            if "INSERT INTO content_ops_deflection_reports" in query:
                account_id, request_id, snapshot, artifact, delivery_email = args
                key = (str(account_id), str(request_id))
                existing = self.rows.get(key, {})
                self.rows[key] = {
                    "account_id": account_id,
                    "request_id": request_id,
                    "snapshot": snapshot,
                    "artifact": artifact,
                    "paid": bool(existing.get("paid")),
                    "payment_reference": existing.get("payment_reference"),
                    "delivery_email": delivery_email or existing.get("delivery_email"),
                }
                return "INSERT 0 1"
            if "UPDATE content_ops_deflection_reports" in query:
                account_id, request_id, payment_reference = args
                key = (str(account_id), str(request_id))
                if key not in self.rows:
                    return "UPDATE 0"
                if "SET paid = false" in query:
                    stored_reference = self.rows[key].get("payment_reference")
                    if payment_reference and stored_reference not in {
                        None,
                        payment_reference,
                    }:
                        return "UPDATE 0"
                    self.rows[key]["paid"] = False
                    return "UPDATE 1"
                self.rows[key]["paid"] = True
                self.rows[key]["payment_reference"] = payment_reference
                return "UPDATE 1"
            raise AssertionError(query)

        async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
            account_id, request_id = args
            row = self.rows.get((str(account_id), str(request_id)))
            if row is None:
                return None
            if "SELECT snapshot" in query:
                return {"snapshot": row["snapshot"]}
            return dict(row)

    store = PostgresDeflectionReportArtifactStore(pool=_Pool())
    snapshot = {
        "summary": {
            "generated": 1,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 0,
        },
        "top_questions": [
            {
                "rank": 1,
                "question": "How do I export reports?",
                "weighted_frequency": 3,
                "customer_wording": "How do I export reports?",
            }
        ],
    }
    artifact = build_deflection_report_artifact(
        _structured_report_fixture_result()
    ).as_dict()
    postgres_artifact = json.loads(json.dumps(artifact))

    await store.save_report(
        account_id="acct-1",
        request_id="request-1",
        snapshot=snapshot,
        artifact=artifact,
        delivery_email=" buyer@example.com ",
    )

    assert await store.get_snapshot(
        account_id="acct-1",
        request_id="request-1",
    ) == snapshot
    assert "buyer@example.com" not in str(snapshot)
    locked = await store.get_artifact_record(
        account_id="acct-1",
        request_id="request-1",
    )
    assert locked is not None
    assert locked.paid is False
    assert locked.artifact == postgres_artifact
    assert locked.report_model() == artifact["report_model"]
    assert locked.delivery_email == "buyer@example.com"
    assert await store.mark_paid(
        account_id="acct-1",
        request_id="request-1",
        payment_reference="checkout-session:test",
    ) is True
    unlocked = await store.get_artifact_record(
        account_id="acct-1",
        request_id="request-1",
    )
    assert unlocked is not None
    assert unlocked.paid is True
    assert unlocked.payment_reference == "checkout-session:test"
    assert unlocked.report_model() == artifact["report_model"]
    assert unlocked.delivery_email == "buyer@example.com"
    await store.save_report(
        account_id="acct-1",
        request_id="request-1",
        snapshot=snapshot,
        artifact=artifact,
    )
    preserved = await store.get_artifact_record(
        account_id="acct-1",
        request_id="request-1",
    )
    assert preserved is not None
    assert preserved.delivery_email == "buyer@example.com"
    assert preserved.paid is True
    assert preserved.payment_reference == "checkout-session:test"
    assert await store.mark_unpaid(
        account_id="acct-1",
        request_id="request-1",
        payment_reference="other-checkout-session",
    ) is False
    assert await store.mark_unpaid(
        account_id="acct-1",
        request_id="request-1",
        payment_reference="checkout-session:test",
    ) is True
    relocked = await store.get_artifact_record(
        account_id="acct-1",
        request_id="request-1",
    )
    assert relocked is not None
    assert relocked.paid is False
    assert relocked.payment_reference == "checkout-session:test"
    assert await store.mark_paid(
        account_id="acct-1",
        request_id="missing",
    ) is False
    assert await store.mark_unpaid(
        account_id="acct-1",
        request_id="missing",
    ) is False


@pytest.mark.asyncio
async def test_in_memory_deflection_report_store_round_trips_report_model() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    artifact = build_deflection_report_artifact(
        _structured_report_fixture_result()
    ).as_dict()

    await store.save_report(
        account_id="acct-model",
        request_id="request-model",
        snapshot=build_deflection_snapshot(artifact).as_dict(),
        artifact=artifact,
    )

    record = await store.get_artifact_record(
        account_id="acct-model",
        request_id="request-model",
    )

    assert record is not None
    assert record.artifact is not None
    assert record.artifact["report_model"] == artifact["report_model"]
    assert record.report_model() == artifact["report_model"]


def test_stored_deflection_report_model_tolerates_legacy_and_schema_drift() -> None:
    assert stored_deflection_report_model(None) is None
    assert stored_deflection_report_model({}) is None
    assert stored_deflection_report_model({"report_model": "not-json"}) is None
    assert (
        stored_deflection_report_model({
            "report_model": {
                "schema_version": "deflection.v2",
                "sections": [],
            }
        })
        is None
    )

    payload = stored_deflection_report_model({
        "report_model": {
            "schema_version": DEFLECTION_REPORT_SCHEMA_VERSION,
            "title": "Stored report",
            "summary": {"generated": 2},
            "sections": [
                "not-a-section",
                {"title": "Missing id", "priority": 5, "data": {"ignored": True}},
                {
                    "id": "missing_priority",
                    "data": {"ignored": True},
                },
                {
                    "id": "invalid_priority",
                    "priority": "not-a-number",
                    "data": {"ignored": True},
                },
                {
                    "id": "missing_required_data",
                    "priority": 40,
                    "required_data": ["rows"],
                    "data": {},
                },
                {
                    "id": "non_mapping_required_data_source",
                    "priority": 50,
                    "required_data": ["rows"],
                    "data": "not-json",
                },
                {
                    "id": "future_section",
                    "title": "Future section",
                    "priority": "70",
                    "surfaces": ["web", "pdf"],
                    "required_data": ["future_rows"],
                    "data": {"future_rows": [{"id": "future-1"}]},
                },
                {
                    "id": "legacy_section",
                    "priority": 20,
                    "data": {"legacy_count": 3},
                },
            ],
        }
    })

    assert payload == {
        "schema_version": DEFLECTION_REPORT_SCHEMA_VERSION,
        "title": "Stored report",
        "summary": {"generated": 2},
        "sections": [
            {
                "id": "legacy_section",
                "title": "Legacy Section",
                "priority": 20,
                "surfaces": [],
                "default_limit": None,
                "required_data": [],
                "data": {"legacy_count": 3},
            },
            {
                "id": "future_section",
                "title": "Future section",
                "priority": 70,
                "surfaces": ["web", "pdf"],
                "default_limit": None,
                "required_data": ["future_rows"],
                "data": {"future_rows": [{"id": "future-1"}]},
            },
        ],
    }


@pytest.mark.asyncio
async def test_in_memory_list_reports_filters_tenant_paid_state_and_orders_newest() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await store.save_report(
        account_id="acct-1",
        request_id="request-old",
        snapshot=_report_access_snapshot("Old question"),
        artifact={},
    )
    await store.save_report(
        account_id="acct-1",
        request_id="request-new",
        snapshot=_report_access_snapshot("New question"),
        artifact={},
    )
    await store.save_report(
        account_id="acct-2",
        request_id="request-other",
        snapshot=_report_access_snapshot("Other question"),
        artifact={},
    )
    await store.mark_paid(account_id="acct-1", request_id="request-old")
    await store.mark_unpaid(account_id="acct-1", request_id="request-old")

    all_rows = await store.list_reports(account_id="acct-1", limit=10)
    paid_rows = await store.list_reports(account_id="acct-1", limit=10, paid=True)
    unpaid_rows = await store.list_reports(account_id="acct-1", limit=10, paid=False)
    unbounded_rows = await store.list_reports(account_id="acct-1", limit=None)

    assert [row.request_id for row in all_rows] == ["request-new", "request-old"]
    assert [row.request_id for row in paid_rows] == []
    assert [row.request_id for row in unpaid_rows] == ["request-new", "request-old"]
    assert [row.request_id for row in unbounded_rows] == ["request-new", "request-old"]


@pytest.mark.asyncio
async def test_postgres_list_reports_uses_account_scope_optional_paid_filter_and_unbounded_limit() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
            self.calls.append((query, args))
            return [
                {
                    "account_id": "acct-1",
                    "request_id": "request-1",
                    "snapshot": json.dumps(
                        _report_access_snapshot("How do I export reports?")
                    ),
                    "paid": False,
                    "delivery_email": None,
                    "created_at": "2026-01-02T00:00:00Z",
                    "updated_at": "2026-01-02T00:00:00Z",
                }
            ]

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    rows = await store.list_reports(account_id=" acct-1 ", limit=7)
    paid_rows = await store.list_reports(account_id="acct-1", limit=7, paid=True)
    unbounded_rows = await store.list_reports(account_id="acct-1", limit=None)

    assert rows[0].request_id == "request-1"
    assert rows[0].snapshot["top_questions"][0]["question"] == "How do I export reports?"
    assert pool.calls[0][1] == ("acct-1", 7)
    assert "WHERE account_id = $1" in pool.calls[0][0]
    assert "LIMIT $2" in pool.calls[0][0]
    assert paid_rows[0].request_id == "request-1"
    assert pool.calls[1][1] == ("acct-1", True, 7)
    assert "AND paid = $2" in pool.calls[1][0]
    assert "LIMIT $3" in pool.calls[1][0]
    assert unbounded_rows[0].request_id == "request-1"
    assert pool.calls[2][1] == ("acct-1",)
    assert "ORDER BY created_at DESC" in pool.calls[2][0]
    assert "LIMIT $" not in pool.calls[2][0]


def test_deflection_snapshot_content_opportunities_are_unpaid_safe() -> None:
    opportunities = deflection_snapshot_content_opportunities(
        {
            "top_questions": [
                {
                    "rank": 1,
                    "question": "How do I export reports?",
                    "ticket_count": 3,
                    "weighted_frequency": 5,
                    "customer_wording": "export reports",
                    "answer": "Open Analytics.",
                    "source_ids": ["ticket-1"],
                    "evidence_quotes": ["ticket quote"],
                    "markdown": "# Full report",
                },
                "not a mapping",
                {"question": ""},
            ]
        }
    )
    encoded = json.dumps(opportunities, sort_keys=True)

    assert opportunities == (
        {
            "rank": 1,
            "question": "How do I export reports?",
            "ticket_count": 3,
            "weighted_frequency": 5,
            "customer_wording": "export reports",
            "opportunity_score": 5,
            "coverage_status": "locked_snapshot",
            "recommended_content_action": (
                "Create or improve an FAQ entry for this repeated customer question."
            ),
            "unlock_hint": "Unlock the full report for detailed source-backed guidance.",
        },
    )
    assert "Open Analytics" not in encoded
    assert "ticket-1" not in encoded
    assert "markdown" not in encoded


def test_deflection_report_cli_builds_saas_demo_artifact(tmp_path: Path) -> None:
    output = tmp_path / "deflection-report.md"
    summary_output = tmp_path / "deflection-report-summary.json"
    result_output = tmp_path / "deflection-report-result.json"

    exit_code = MODULE.main([
        str(SAAS_DEMO),
        "--source-format",
        "csv",
        "--require-output-checks",
        "--output",
        str(output),
        "--summary-output",
        str(summary_output),
        "--result-output",
        str(result_output),
    ])

    assert exit_code == 0
    markdown = output.read_text()
    summary = json.loads(summary_output.read_text())
    result = json.loads(result_output.read_text())
    assert summary["generated"] >= 7
    assert summary["source_count"] == 36
    assert summary["no_proven_answer_count"] >= 1
    assert result["status"] == "ok"
    assert result["failed_output_checks"] == []
    assert result["summary"] == summary
    assert result["config"]["require_output_checks"] is True
    assert result["diagnostics"]["item_count"] == result["generated"]
    assert "## Ranked Question Opportunities" in markdown
    assert "## Question Details and Evidence" in markdown
    assert "## No Proven Answer Yet" not in markdown
    assert "support_ticket_saas_demo_sources.csv" in markdown
    assert "tell users exactly what to try next" not in markdown


def test_deflection_report_cli_writes_complete_evidence_export(tmp_path: Path) -> None:
    source = tmp_path / "mixed-support-tickets.json"
    evidence_output = tmp_path / "deflection-evidence.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-export-1",
                "source_type": "support_ticket",
                "subject": "Export attribution report",
                "message": "How do I export attribution reports?",
                "pain_category": "exports",
                "resolution_text": (
                    "Open Analytics then Attribution then click Download report."
                ),
            },
            {
                "ticket_id": "ticket-export-2",
                "source_type": "support_ticket",
                "subject": "Export attribution report",
                "message": "How do I export attribution reports?",
                "pain_category": "exports",
                "resolution_text": (
                    "Open Analytics then Attribution then click Download report."
                ),
            },
        ]),
        encoding="utf-8",
    )

    exit_code = MODULE.main([
        str(source),
        "--source-format",
        "json",
        "--evidence-output",
        str(evidence_output),
    ])

    assert exit_code == 0
    export = json.loads(evidence_output.read_text(encoding="utf-8"))
    assert export["schema_version"] == DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION
    assert export["summary"]["question_count"] == 1
    assert export["summary"]["source_id_count"] == 2
    assert export["questions"][0]["answer_linkage"] == "publishable_answer"
    assert export["questions"][0]["source_ids"] == [
        "ticket-export-1",
        "ticket-export-2",
    ]
    assert {
        row["source_id"] for row in export["evidence_rows"]
    } == {"ticket-export-1", "ticket-export-2"}


def test_deflection_report_cli_ignores_legacy_max_items_cap(tmp_path: Path) -> None:
    source = tmp_path / "uncapped-support-tickets.json"
    result_output = tmp_path / "deflection-report-result.json"
    rows = [
        {
            "ticket_id": "ticket-sso-1",
            "source_type": "support_ticket",
            "subject": "SSO setup",
            "message": "Can we sync users before enforcing SSO?",
            "pain_category": "single sign-on rollout",
            "resolution_text": "Enable SSO, verify emails, then enable SCIM.",
        },
        {
            "ticket_id": "ticket-dashboard-1",
            "source_type": "support_ticket",
            "subject": "Dashboard refresh",
            "message": "Why is the executive dashboard not refreshing?",
            "pain_category": "warehouse refresh status",
            "resolution_text": "Rerun the failed warehouse job, then refresh the model.",
        },
        {
            "ticket_id": "ticket-billing-1",
            "source_type": "support_ticket",
            "subject": "Renewal invoice",
            "message": "How do I confirm my renewal invoice before payment?",
            "pain_category": "renewal billing review",
            "resolution_text": "Open Billing > Invoices, then download the PDF.",
        },
        {
            "ticket_id": "ticket-token-1",
            "source_type": "support_ticket",
            "subject": "API token rotation",
            "message": "How do we rotate API tokens without downtime?",
            "pain_category": "api token rotation",
            "resolution_text": "Deploy the replacement token, then revoke the old token.",
        },
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")

    exit_code = MODULE.main([
        str(source),
        "--source-format",
        "json",
        "--max-items",
        "2",
        "--result-output",
        str(result_output),
        "--json",
    ])

    assert exit_code == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["generated"] == 4
    assert result["summary"]["generated"] == 4
    assert result["config"]["max_items"] == 0
    assert result["config"]["requested_max_items"] == 2
    assert result["diagnostics"]["item_count"] == 4
    assert "other support issues" not in {
        item["topic"] for item in result["diagnostics"]["items"]
    }
    assert {
        item["first_source_id"] for item in result["diagnostics"]["items"]
    } == {
        "ticket-sso-1",
        "ticket-dashboard-1",
        "ticket-billing-1",
        "ticket-token-1",
    }


def test_deflection_report_cli_splits_backed_and_unproven_answers(tmp_path: Path) -> None:
    source = tmp_path / "mixed-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    summary_output = tmp_path / "deflection-report-summary.json"
    rows = [
        {
            "ticket_id": f"ticket-export-{index}",
            "source_type": "support_ticket",
            "subject": "Export attribution report",
            "message": "How do I export attribution reports?",
            "pain_category": "exports",
            "resolution_text": (
                "Open Analytics then Attribution then click Download report."
            ),
        }
        for index in range(4)
    ]
    rows.extend(
        {
            "ticket_id": f"ticket-sso-{index}",
            "source_type": "support_ticket",
            "subject": "SSO setup",
            "message": "How do I enable SSO for my team?",
            "pain_category": "authentication",
        }
        for index in range(3)
    )
    source.write_text(json.dumps(rows), encoding="utf-8")

    exit_code = MODULE.main([
        str(source),
        "--source-format",
        "json",
        "--output",
        str(output),
        "--summary-output",
        str(summary_output),
    ])

    assert exit_code == 0
    markdown = output.read_text()
    summary = json.loads(summary_output.read_text())
    details = markdown.split("## Question Details and Evidence", 1)[1]
    export_block = details.split("### 1. How do I export attribution reports?", 1)[1].split(
        "### 2. How do I enable SSO for my team?",
        1,
    )[0]
    sso_block = details.split("### 2. How do I enable SSO for my team?", 1)[1]

    assert summary["generated"] == 2
    assert summary["source_count"] == 7
    assert summary["drafted_answer_count"] == 1
    assert summary["no_proven_answer_count"] == 1
    assert "Open Analytics then Attribution then click Download report" in export_block
    assert "How do I enable SSO for my team?" in sso_block
    assert "Open Analytics then Attribution then click Download report" not in sso_block
    assert "tell users exactly what to try next" not in export_block


def test_deflection_report_cli_fails_required_output_checks_for_weak_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weak-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    summary_output = tmp_path / "deflection-report-summary.json"
    result_output = tmp_path / "deflection-report-result.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-1",
                "source_type": "support_ticket",
                "subject": "Unique export issue",
                "message": "The export button moved.",
                "pain_category": "exports",
            },
            {
                "ticket_id": "ticket-2",
                "source_type": "support_ticket",
                "subject": "Unique billing issue",
                "message": "Billing receipt is missing.",
                "pain_category": "billing",
            },
        ]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        MODULE.main([
            str(source),
            "--source-format",
            "json",
            "--require-output-checks",
            "--output",
            str(output),
            "--summary-output",
            str(summary_output),
            "--result-output",
            str(result_output),
        ])

    # #1460: two one-off questions no longer count as repeat FAQ items, so no
    # items are generated and every required output check fails.
    assert str(exc_info.value) == (
        "Deflection report output checks failed: condensed, has_action_items, "
        "resolution_evidence_scoped, uses_user_vocabulary"
    )
    assert not output.exists()
    assert not summary_output.exists()
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["status"] == "failed_output_checks"
    assert result["failed_output_checks"] == [
        "condensed",
        "has_action_items",
        "resolution_evidence_scoped",
        "uses_user_vocabulary",
    ]
    assert result["generated"] == 0
    assert result["summary"]["source_count"] == 2
    assert result["summary"]["ticket_source_count"] == 2
    assert result["summary"]["non_repeat_ticket_count"] == 2
    assert result["summary"]["non_repeat_question_count"] == 2
    assert result["output"]["markdown_path"] == str(output)
    assert result["output"]["summary_path"] == str(summary_output)
    assert result["output"]["result_path"] == str(result_output)
    assert result["output"]["evidence_path"] is None
    assert result["diagnostics"]["item_count"] == 0
    assert result["diagnostics"]["items"] == []
    assert "markdown" not in result


def test_deflection_report_cli_applies_custom_intent_rules(tmp_path: Path) -> None:
    source = tmp_path / "intent-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    result_output = tmp_path / "deflection-report-result.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-access-1",
                "source_type": "support_ticket",
                "subject": "Invite links",
                "message": "The invite link for the new teammate login email expired.",
            },
            {
                "ticket_id": "ticket-access-2",
                "source_type": "support_ticket",
                "subject": "Login email",
                "message": "How do I get the expired invite link for the new teammate login email?",
            },
        ]),
        encoding="utf-8",
    )

    exit_code = MODULE.main([
        str(source),
        "--source-format",
        "json",
        "--intent-rule",
        "Account access=invite link,login email,LOGIN EMAIL",
        "--require-output-checks",
        "--output",
        str(output),
        "--result-output",
        str(result_output),
    ])

    assert exit_code == 0
    markdown = output.read_text(encoding="utf-8")
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["generated"] == 1
    assert result["output_checks"]["condensed"] is True
    assert result["failed_output_checks"] == []
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "Account access", "keywords": ["invite link", "login email"]}
    ]
    assert result["diagnostics"]["items"][0]["topic"] == "account access"
    assert result["diagnostics"]["items"][0]["source_id_count"] == 2
    assert "| 1 |" in markdown
    assert "ticket-access-1, ticket-access-2" in markdown


def test_deflection_report_cli_accepts_json_rule_file(tmp_path: Path) -> None:
    source = tmp_path / "rule-file-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    result_output = tmp_path / "deflection-report-result.json"
    rule_file = tmp_path / "faq-rules.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-rule-1",
                "source_type": "support_ticket",
                "subject": "SSO sync",
                "message": "How do I enable SSO after warehouse sync?",
            },
            {
                "ticket_id": "ticket-rule-2",
                "source_type": "support_ticket",
                "subject": "Connector lag",
                "message": "Can we enable SSO after the warehouse sync?",
            },
        ]),
        encoding="utf-8",
    )
    rule_file.write_text(
        json.dumps({
            "intent_rules": [
                {"topic": "file topic", "keywords": ["warehouse sync", "connector lag"]}
            ],
            "vocabulary_gap_rules": [["SSO", "single sign-on"]],
        }),
        encoding="utf-8",
    )

    exit_code = MODULE.main([
        str(source),
        "--source-format",
        "json",
        "--documentation-term",
        "Single sign-on setup",
        "--rule-file",
        str(rule_file),
        "--intent-rule",
        "cli topic=warehouse sync,connector lag",
        "--require-output-checks",
        "--output",
        str(output),
        "--result-output",
        str(result_output),
    ])

    assert exit_code == 0
    markdown = output.read_text(encoding="utf-8")
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["rule_files"] == [str(rule_file)]
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "cli topic", "keywords": ["warehouse sync", "connector lag"]},
        {"topic": "file topic", "keywords": ["warehouse sync", "connector lag"]},
    ]
    assert result["config"]["vocabulary_gap_rules"] == [["SSO", "single sign-on"]]
    assert result["generated"] == 1
    assert result["diagnostics"]["items"][0]["topic"] == "cli topic"
    assert result["diagnostics"]["items"][0]["source_id_count"] == 2
    assert result["diagnostics"]["items"][0]["term_mapping_count"] >= 1
    assert "SSO" in markdown
    assert "Single sign-on setup" in markdown


def test_deflection_report_cli_accepts_documentation_term_file(tmp_path: Path) -> None:
    source = tmp_path / "term-file-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    result_output = tmp_path / "deflection-report-result.json"
    term_file = tmp_path / "documentation-terms.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-export-1",
                "source_type": "support_ticket",
                "subject": "Export attribution report",
                "message": "How do I export attribution reports?",
                "pain_category": "exports",
            },
            {
                "ticket_id": "ticket-export-2",
                "source_type": "support_ticket",
                "subject": "Export report",
                "message": "Can I export the attribution report?",
                "pain_category": "exports",
            },
        ]),
        encoding="utf-8",
    )
    term_file.write_text(
        json.dumps({
            "documentation_terms": ["Single sign-on setup"],
            "documents": [{"title": "Download report"}],
        }),
        encoding="utf-8",
    )

    exit_code = MODULE.main([
        str(source),
        "--source-format",
        "json",
        "--documentation-term",
        "Billing center",
        "--documentation-term-file",
        str(term_file),
        "--require-output-checks",
        "--output",
        str(output),
        "--result-output",
        str(result_output),
    ])

    assert exit_code == 0
    markdown = output.read_text(encoding="utf-8")
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_term_files"] == [str(term_file)]
    assert result["config"]["documentation_term_format"] == "auto"
    assert result["config"]["documentation_terms"] == [
        "Billing center",
        "Single sign-on setup",
        "Download report",
    ]
    assert result["diagnostics"]["items"][0]["term_mapping_count"] >= 1
    assert "Download report" in markdown
    assert "export" in markdown


def test_deflection_report_cli_rejects_bad_documentation_term_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad-term-file-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    term_file = tmp_path / "documentation-terms.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-export-1",
                "source_type": "support_ticket",
                "subject": "Export attribution report",
                "message": "How do I export attribution reports?",
            },
        ]),
        encoding="utf-8",
    )
    term_file.write_text(
        json.dumps({"documents": [{"url": "https://help.example/export"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        MODULE.main([
            str(source),
            "--source-format",
            "json",
            "--documentation-term-file",
            str(term_file),
            "--output",
            str(output),
        ])

    assert "--documentation-term-file has no recognized term fields:" in str(
        exc_info.value
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "--rule-file must contain a JSON object"),
        ({"unknown": []}, "--rule-file contains unsupported key(s): unknown"),
        ({"intent_rules": "bad"}, "--rule-file intent_rules must be an array"),
        (
            {"intent_rules": [{"topic": "data freshness", "keywords": []}]},
            "--rule-file intent_rules[1] is invalid",
        ),
        (
            {"vocabulary_gap_rules": [["SSO"]]},
            "--rule-file vocabulary_gap_rules[1] is invalid",
        ),
    ],
)
def test_deflection_report_cli_rejects_invalid_rule_file(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    source = tmp_path / "bad-rule-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    rule_file = tmp_path / "faq-rules.json"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-rule-1",
                "source_type": "support_ticket",
                "subject": "Sync lag",
                "message": "The warehouse sync is delayed.",
            },
        ]),
        encoding="utf-8",
    )
    rule_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        MODULE.main([
            str(source),
            "--source-format",
            "json",
            "--rule-file",
            str(rule_file),
            "--output",
            str(output),
        ])

    assert message in str(exc_info.value)
    assert not output.exists()


@pytest.mark.parametrize(
    "rule",
    [
        "Account access",
        "=invite link",
        "Account access=",
        "Account access=,",
    ],
)
def test_deflection_report_cli_rejects_malformed_intent_rule(
    rule: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / "intent-support-tickets.json"
    output = tmp_path / "deflection-report.md"
    source.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-access-1",
                "source_type": "support_ticket",
                "subject": "Invite links",
                "message": "The invite link expired before a contractor could join.",
            },
        ]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        MODULE.main([
            str(source),
            "--source-format",
            "json",
            "--intent-rule",
            rule,
            "--output",
            str(output),
        ])

    assert str(exc_info.value) == (
        "--intent-rule must use topic=keyword,keyword with at least one keyword"
    )
    assert not output.exists()
