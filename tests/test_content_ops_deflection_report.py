from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
    build_deflection_full_report_qa_deterministic_harness,
    build_deflection_full_report_qa_scorecard,
    build_deflection_report_model,
    build_deflection_snapshot,
    build_deflection_report_artifact,
    deflection_report_email_action_rows,
    deflection_report_model_contract_shape,
    deflection_snapshot_content_opportunities,
    render_deflection_report_model,
    scrub_deflection_report_payload,
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
RETENTION_SCRIPT = ROOT / "scripts" / "purge_content_ops_deflection_reports.py"
SAAS_DEMO = ROOT / "extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv"
SPEC = importlib.util.spec_from_file_location(
    "build_content_ops_deflection_report",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RETENTION_SPEC = importlib.util.spec_from_file_location(
    "purge_content_ops_deflection_reports",
    RETENTION_SCRIPT,
)
assert RETENTION_SPEC is not None
assert RETENTION_SPEC.loader is not None
RETENTION_MODULE = importlib.util.module_from_spec(RETENTION_SPEC)
RETENTION_SPEC.loader.exec_module(RETENTION_MODULE)


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


def _report_projection_runtime_fixture_result() -> TicketFAQMarkdownResult:
    base = _structured_report_fixture_result()
    sso_item = {
        **base.items[1],
        "term_mappings": (
            {
                "customer_term": "SSO for all",
                "documentation_term": "single sign-on rollout",
                "suggestion": "Mirror SSO rollout wording in setup docs.",
                "source_id_count": 3,
            },
        ),
    }
    recurring_item = {
        "question": "How do I fix a stale dashboard?",
        "customer_wording": "dashboard is stale again",
        "topic": "analytics",
        "weighted_frequency": 8,
        "ticket_count": 4,
        "opportunity_score": 19,
        "answer": "Refresh the dashboard cache from Analytics settings.",
        "answer_evidence_status": "resolution_evidence",
        "resolution_evidence_scope": "scoped",
        "source_ids": (
            "ticket-dashboard-1",
            "ticket-dashboard-2",
            "ticket-dashboard-3",
            "ticket-dashboard-4",
        ),
        "source_date_span": {
            "start": "2026-05-16",
            "end": "2026-05-19",
            "missing_source_count": 0,
        },
        "evidence_quotes": ("`ticket-dashboard-1` - Dashboard stale again",),
        "term_mappings": (
            {
                "customer_term": "stale dashboard",
                "documentation_term": "dashboard cache refresh",
                "suggestion": "Use stale dashboard wording in the troubleshooting guide.",
                "source_id_count": 4,
            },
        ),
        "outcome_diagnostics": {
            "csat_present_count": 4,
            "csat_score_average": 2.0,
            "diagnostic_ticket_count": 4,
            "negative_csat_ticket_count": 2,
            "outcome_risk_ticket_count": 2,
            "reopened_ticket_count": 1,
            "ticket_status_summary": {"reopened": 1, "resolved": 3},
        },
    }
    sparse_item = {
        "question": "Why did my imported contacts duplicate?",
        "customer_wording": "contacts imported twice",
        "topic": "imports",
        "weighted_frequency": 5,
        "ticket_count": 3,
        "opportunity_score": 20,
        "answer_evidence_status": "draft_needs_review",
        "source_count": 1,
        "source_ids": ("ticket-sparse-only",),
        "source_date_span": {
            "start": "2026-05-20",
            "end": "2026-05-20",
            "missing_source_count": 0,
        },
        "term_mappings": (
            {
                "customer_term": "contacts duplicated",
                "documentation_term": "deduplicate imported contacts",
                "suggestion": "Add duplicate-import wording to the import docs.",
                "source_id_count": 1,
            },
        ),
    }
    return replace(
        base,
        source_count=base.source_count + 5,
        ticket_source_count=base.ticket_source_count + 5,
        items=(base.items[0], sso_item, recurring_item, sparse_item),
    )


def _report_projection_no_conditionals_fixture_result() -> TicketFAQMarkdownResult:
    base = _structured_report_fixture_result()
    return replace(
        base,
        items=tuple(
            {
                key: value
                for key, value in item.items()
                if key != "outcome_diagnostics"
            }
            for item in base.items
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
        "priority_fix_queue",
        "top_unresolved_repeats",
        "drafted_resolutions",
        "already_covered_still_recurring",
        "backlog_table",
        "outcome_diagnostics",
        "suppressed_repeat_review_queue",
        "question_details",
        "complete_evidence",
    ]
    assert [section["priority"] for section in payload["sections"]] == [
        10,
        20,
        30,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
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
    assert support_tax["snapshot_safe_fields"] == [
        "repeat_ticket_count",
        "non_repeat_ticket_count",
        "generated_question_count",
        "drafted_answer_count",
        "no_proven_answer_count",
        "ticket_source_count",
        "source_date_window",
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
        "weighted_frequency": 8,
        "customer_wording": "export attribution reports",
        "estimated_support_cost": 67.5,
        "opportunity_score": 21,
        "answer_status": "drafted from resolution evidence",
        "source_proof": "5 source tickets",
    }

    priority_queue = section_by_id["priority_fix_queue"]
    assert priority_queue["snapshot_safe_fields"] == []
    assert priority_queue["data"]["support_cost_basis"] == {
        "status": "benchmark_only",
        "assisted_contact_cost": 13.5,
        "formula": "ticket_count * assisted_contact_cost",
        "source": "default_assisted_contact_benchmark",
    }
    assert priority_queue["data"]["status_counts"] == {
        "Draft ready": 1,
        "Needs answer": 1,
    }
    assert [item["status"] for item in priority_queue["data"]["items"]] == [
        "Draft ready",
        "Needs answer",
    ]
    assert priority_queue["data"]["items"][0]["fix_type"] == "publish_help_center_answer"
    assert priority_queue["data"]["items"][0]["owner_lane"] == "Reporting"
    assert priority_queue["data"]["items"][0]["owner_category"] == (
        "Content / Support Enablement"
    )
    assert priority_queue["data"]["items"][0]["estimated_support_cost"] == 67.5
    assert priority_queue["data"]["items"][0]["csat_signal"] == {
        "status": "insufficient_data",
        "csat_present_count": 0,
        "negative_csat_ticket_count": 0,
        "numeric_average": None,
    }
    assert priority_queue["data"]["items"][1]["fix_type"] == "create_missing_answer"
    assert priority_queue["data"]["items"][1]["owner_category"] == (
        "Content / Support Enablement"
    )

    unresolved = section_by_id["top_unresolved_repeats"]["data"]
    assert unresolved["top_item_count"] == 1
    assert unresolved["result_page_limit"] == 3
    assert unresolved["pdf_limit"] == 10
    assert unresolved["items"][0]["question"] == "Can I turn on SSO for all users?"
    assert unresolved["items"][0]["estimated_support_cost"] == 40.5

    drafted = section_by_id["drafted_resolutions"]["data"]
    assert drafted["top_item_count"] == 1
    assert drafted["result_page_limit"] == 3
    assert drafted["pdf_limit"] == 10
    assert drafted["items"][0]["question"] == "How do I export attribution reports?"

    recurring = section_by_id["already_covered_still_recurring"]["data"]
    assert recurring["top_item_count"] == 0
    assert recurring["result_page_limit"] == 3
    assert recurring["pdf_limit"] == 10
    assert recurring["items"] == []

    backlog = section_by_id["backlog_table"]["data"]
    assert backlog["total_item_count"] == 2
    assert backlog["default_limit"] == 25

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

    suppressed = section_by_id["suppressed_repeat_review_queue"]["data"]
    assert suppressed == {
        "items": [],
        "total_item_count": 0,
        "default_limit": 25,
        "reason_counts": {},
    }

    details = section_by_id["question_details"]["data"]["rows"]
    assert details[0]["answer_linkage"] == "publishable_answer"
    assert details[0]["evidence_tier"] == "csv_full_thread_resolution_evidence"
    assert details[0]["source_ids"] == [
        "ticket-export-1",
        "ticket-export-2",
        "ticket-export-3",
        "ticket-export-4",
        "ticket-export-5",
    ]
    assert details[1]["answer_linkage"] == "needs_review"
    assert details[1]["evidence_tier"] == "csv_index_metadata_only"
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


def test_deflection_action_sections_classify_recurring_covered_answers() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I fix a stale dashboard?",
                "customer_wording": "dashboard is stale again",
                "topic": "analytics",
                "weighted_frequency": 8,
                "ticket_count": 4,
                "opportunity_score": 19,
                "answer": "Refresh the dashboard cache from Analytics settings.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": (
                    "ticket-dashboard-1",
                    "ticket-dashboard-2",
                    "ticket-dashboard-3",
                    "ticket-dashboard-4",
                ),
                "outcome_diagnostics": {
                    "csat_present_count": 4,
                    "csat_score_average": 2.0,
                    "diagnostic_ticket_count": 4,
                    "negative_csat_ticket_count": 2,
                    "outcome_risk_ticket_count": 2,
                    "reopened_ticket_count": 1,
                    "ticket_status_summary": {"reopened": 1, "resolved": 3},
                },
            },
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    recurring = sections["already_covered_still_recurring"]["data"]
    priority = sections["priority_fix_queue"]["data"]

    assert recurring["top_item_count"] == 1
    assert recurring["items"][0]["status"] == "Already covered but still recurring"
    assert recurring["items"][0]["fix_type"] == (
        "improve_discoverability_or_answer_quality"
    )
    assert recurring["items"][0]["owner_lane"] == "Analytics"
    assert recurring["items"][0]["owner_category"] == (
        "Product / Support Experience"
    )
    assert recurring["items"][0]["csat_signal"] == {
        "status": "present",
        "csat_present_count": 4,
        "negative_csat_ticket_count": 2,
        "numeric_average": 2.0,
    }
    assert "negative_csat" in recurring["items"][0]["priority_drivers"]
    assert "reopened_after_answer" in recurring["items"][0]["priority_drivers"]
    assert priority["status_counts"] == {"Already covered but still recurring": 1}


def test_csv_product_gap_owner_lane_vertical_routes_login_gap() -> None:
    rows = [
        {
            "Ticket ID": f"zd-login-{index}",
            "Subject": "Where is the login button?",
            "Requester Comment": "Where is the login button?",
            "Created At": f"2026-05-0{index}T09:00:00Z",
            "Group": "Billing Support",
            "Tags": "navigation",
            "Organization": "Billing Team LLC",
            "Assignee": "Export Agent",
            "Brand": "Admin Co",
        }
        for index in range(1, 5)
    ]
    rows.append({
        "Ticket ID": "zd-export-1",
        "Subject": "Where is the CSV export?",
        "Requester Comment": "Where is the CSV export?",
        "Created At": "2026-05-05T09:00:00Z",
        "Group": "Reporting",
        "Tags": "export",
    })
    package = build_support_ticket_input_package(rows)
    faq_result = build_ticket_faq_markdown(package.inputs["source_material"], max_items=4)

    artifact = build_deflection_report_artifact(faq_result)
    model = artifact.report_model.as_dict()
    sections = {section["id"]: section for section in model["sections"]}
    priority_item = sections["priority_fix_queue"]["data"]["items"][0]

    assert priority_item["question"] == "Where is the login button?"
    assert priority_item["owner_lane"] == "Auth / Product UX"
    assert priority_item["owner_category"] == "Content / Support Enablement"
    assert priority_item["ticket_count"] == 4
    assert priority_item["estimated_support_cost"] == 54.0
    assert priority_item["evidence_tier"] == "csv_customer_text"
    assert priority_item["routing_signals"]["group"] == ["Billing Support"]
    assert priority_item["routing_signals"]["tags"] == ["navigation"]
    assert priority_item["routing_signals"]["organization"] == ["Billing Team LLC"]
    assert priority_item["routing_signals"]["assignee"] == ["Export Agent"]
    assert priority_item["routing_signals"]["brand"] == ["Admin Co"]
    assert priority_item["product_gap_summary"] == (
        "Repeated support friction routes to Auth / Product UX. "
        "4 support tickets in this upload; estimated assisted-contact cost "
        "is $54 based on CSV customer text."
    )
    assert priority_item["customer_vocabulary"] == [
        "Where is the login button?",
        "button login",
    ]
    assert priority_item["cost_period"] == "batch_upload"
    assert priority_item["cost_confidence"] == "benchmark_with_customer_text"
    assert priority_item["jira_template"] == {
        "recommended_title": "Where is the login button?",
        "question": "Where is the login button?",
        "owner_lane": "Auth / Product UX",
        "owner_category": "Content / Support Enablement",
        "product_gap_summary": priority_item["product_gap_summary"],
        "ticket_count": 4,
        "estimated_support_cost": 54.0,
        "cost_period": "batch_upload",
        "cost_confidence": "benchmark_with_customer_text",
        "evidence_tier": "csv_customer_text",
        "customer_vocabulary": [
            "Where is the login button?",
            "button login",
        ],
        "recommended_action": (
            "Write and approve the missing answer for this repeated customer question."
        ),
    }
    assert "Account Settings" not in priority_item["recommended_action"]
    assert "buried" not in priority_item["recommended_action"].lower()
    assert "Account Settings" not in priority_item["product_gap_summary"]
    assert "buried" not in priority_item["product_gap_summary"].lower()

    evidence_export = build_deflection_evidence_export(artifact)
    assert evidence_export["summary"] == {
        "question_count": 1,
        "evidence_row_count": 4,
        "source_id_count": 4,
        "drafted_answer_count": 0,
        "no_proven_answer_count": 1,
    }
    assert evidence_export["questions"][0]["question"] == "Where is the login button?"
    assert evidence_export["questions"][0]["ticket_count"] == 4
    assert evidence_export["questions"][0]["answer_linkage"] == "needs_review"
    assert evidence_export["questions"][0]["source_ids"] == [
        "zd-login-1",
        "zd-login-2",
        "zd-login-3",
        "zd-login-4",
    ]
    login_rows = [
        row
        for row in evidence_export["evidence_rows"]
        if row["question"] == "Where is the login button?"
    ]
    assert len(login_rows) == 4
    assert {row["source_id"] for row in login_rows} == {
        "zd-login-1",
        "zd-login-2",
        "zd-login-3",
        "zd-login-4",
    }
    assert {row["answer_linkage"] for row in login_rows} == {"needs_review"}


def test_product_gap_summary_does_not_copy_root_cause_or_screen_path_question() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=3,
        ticket_source_count=3,
        output_checks={"condensed": True},
        items=(
            {
                "question": "Why is the login button buried under Account Settings?",
                "customer_wording": "Why is the login button buried under Account Settings?",
                "topic": "login",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 10,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-login-1", "ticket-login-2", "ticket-login-3"),
            },
        ),
    )

    model = build_deflection_report_model(result).as_dict()
    priority_item = next(
        section
        for section in model["sections"]
        if section["id"] == "priority_fix_queue"
    )["data"]["items"][0]

    assert priority_item["owner_lane"] == "Auth / Product UX"
    assert priority_item["product_gap_summary"] == (
        "Repeated support friction routes to Auth / Product UX. "
        "3 support tickets in this upload; estimated assisted-contact cost "
        "is $41 based on CSV index metadata only."
    )
    assert "Account Settings" not in priority_item["product_gap_summary"]
    assert "buried" not in priority_item["product_gap_summary"].lower()
    assert "Why is the login button" not in priority_item["product_gap_summary"]


def test_product_gap_summary_is_omitted_for_non_repeated_low_confidence_rows() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=1,
        ticket_source_count=1,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I rename a workspace?",
                "customer_wording": "How do I rename a workspace?",
                "topic": "workspace",
                "weighted_frequency": 1,
                "ticket_count": 1,
                "opportunity_score": 3,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-single-1",),
            },
            {
                "question": "",
                "customer_wording": "",
                "topic": "",
                "weighted_frequency": 0,
                "ticket_count": 0,
                "opportunity_score": 0,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (),
            },
        ),
    )

    model = build_deflection_report_model(result).as_dict()
    sections = {section["id"]: section for section in model["sections"]}
    action_items = sections["priority_fix_queue"]["data"]["items"]

    assert [item["ticket_count"] for item in action_items] == [1, 0]
    assert all(item["status"] == "Low confidence" for item in action_items)
    assert all(item["product_gap_summary"] == "" for item in action_items)
    assert "repeated across 1" not in str(action_items)
    assert "0 support tickets" not in str(action_items)


def test_owner_lane_keyword_matching_uses_tokens_not_substrings() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=2,
        ticket_source_count=2,
        output_checks={"condensed": True},
        items=(
            {
                "question": "Who authored this guide?",
                "customer_wording": "Who authored this guide?",
                "topic": "publishing",
                "weighted_frequency": 2,
                "ticket_count": 2,
                "opportunity_score": 10,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-author-1", "ticket-author-2"),
            },
        ),
    )

    model = build_deflection_report_model(result).as_dict()
    priority_item = next(
        section
        for section in model["sections"]
        if section["id"] == "priority_fix_queue"
    )["data"]["items"][0]

    assert priority_item["owner_lane"] == "Publishing"


def test_deflection_priority_queue_scores_status_and_csat_signals() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=22,
        ticket_source_count=22,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I find the saved analytics dashboard?",
                "customer_wording": "dashboard answer still not helping",
                "topic": "analytics",
                "weighted_frequency": 12,
                "ticket_count": 6,
                "opportunity_score": 12,
                "answer": "Open Analytics and choose Saved dashboards.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": tuple(f"ticket-risk-{index}" for index in range(6)),
                "outcome_diagnostics": {
                    "csat_present_count": 4,
                    "csat_score_average": 1.5,
                    "negative_csat_ticket_count": 3,
                    "reopened_ticket_count": 1,
                    "ticket_status_summary": {"reopened": 1, "resolved": 5},
                },
            },
            {
                "question": "How do I export attribution reports?",
                "customer_wording": "export attribution reports",
                "topic": "exports",
                "weighted_frequency": 11,
                "ticket_count": 6,
                "opportunity_score": 12,
                "answer": "Open Attribution and select Download report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": tuple(f"ticket-good-{index}" for index in range(6)),
                "outcome_diagnostics": {
                    "csat_present_count": 4,
                    "csat_score_average": 4.5,
                    "negative_csat_ticket_count": 0,
                    "reopened_ticket_count": 0,
                    "ticket_status_summary": {"resolved": 6},
                },
            },
            {
                "question": "How do I configure SSO for contractors?",
                "customer_wording": "contractor SSO setup",
                "topic": "identity",
                "weighted_frequency": 7,
                "ticket_count": 4,
                "opportunity_score": 9,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": tuple(f"ticket-missing-{index}" for index in range(4)),
            },
            {
                "question": "How do I change invoice contacts?",
                "customer_wording": "invoice contact change",
                "topic": "billing",
                "weighted_frequency": 6,
                "ticket_count": 4,
                "opportunity_score": 9,
                "answer": "Open Billing, choose Contacts, and update the recipient.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": tuple(f"ticket-draft-{index}" for index in range(4)),
            },
            {
                "question": "How do I review warehouse sync failures?",
                "customer_wording": "warehouse sync failed",
                "topic": "integrations",
                "weighted_frequency": 5,
                "ticket_count": 3,
                "opportunity_score": 8,
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": tuple(f"ticket-review-{index}" for index in range(3)),
            },
            {
                "question": "Why did my imported contacts duplicate?",
                "customer_wording": "contacts imported twice",
                "topic": "imports",
                "weighted_frequency": 5,
                "ticket_count": 3,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_count": 1,
                "source_ids": ("ticket-sparse-only",),
            },
            {
                "question": "Can I rename one workspace?",
                "customer_wording": "rename one workspace",
                "topic": "workspace",
                "weighted_frequency": 1,
                "ticket_count": 1,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-one-off",),
            },
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    priority = sections["priority_fix_queue"]["data"]
    items = {item["question"]: item for item in priority["items"]}

    assert priority["items"][0]["question"] == (
        "How do I find the saved analytics dashboard?"
    )
    assert items["How do I find the saved analytics dashboard?"]["status"] == (
        "Already covered but still recurring"
    )
    assert items["How do I export attribution reports?"]["status"] == "Draft ready"
    assert (
        items["How do I find the saved analytics dashboard?"]["priority_score"]
        > items["How do I export attribution reports?"]["priority_score"]
    )
    assert items["How do I configure SSO for contractors?"]["status"] == (
        "Needs answer"
    )
    assert "missing_answer" in items[
        "How do I configure SSO for contractors?"
    ]["priority_drivers"]
    assert items["How do I change invoice contacts?"]["status"] == "Draft ready"
    assert (
        items["How do I configure SSO for contractors?"]["priority_score"]
        > items["How do I change invoice contacts?"]["priority_score"]
    )
    assert items["How do I review warehouse sync failures?"]["status"] == (
        "Needs review"
    )
    sparse = items["Why did my imported contacts duplicate?"]
    assert sparse["status"] == "Low confidence"
    assert sparse["confidence"] == "low"
    assert sparse["priority_drivers"].count("low_confidence") == 1
    sparse_score_without_low_penalty = (
        int(round(sparse["estimated_support_cost"] * 3))
        + min(sparse["opportunity_score"], 50)
        + 5
    )
    assert sparse["priority_score"] == sparse_score_without_low_penalty - 25
    assert items["Can I rename one workspace?"]["status"] == "Low confidence"
    assert "low_confidence" in items["Can I rename one workspace?"][
        "priority_drivers"
    ]
    assert priority["status_counts"] == {
        "Already covered but still recurring": 1,
        "Draft ready": 2,
        "Low confidence": 2,
        "Needs answer": 1,
        "Needs review": 1,
    }

    unresolved = sections["top_unresolved_repeats"]["data"]
    unresolved_questions = {item["question"] for item in unresolved["items"]}
    assert "How do I configure SSO for contractors?" in unresolved_questions
    assert "How do I review warehouse sync failures?" in unresolved_questions
    assert "Why did my imported contacts duplicate?" not in unresolved_questions
    assert "Can I rename one workspace?" not in unresolved_questions

    suppressed = sections["suppressed_repeat_review_queue"]["data"]
    suppressed_by_question = {
        item["question"]: item
        for item in suppressed["items"]
    }
    assert suppressed["total_item_count"] == 2
    assert suppressed["default_limit"] == 25
    assert suppressed["reason_counts"] == {
        "insufficient_source_support": 1,
        "too_low_volume": 1,
    }
    assert suppressed_by_question[
        "Why did my imported contacts duplicate?"
    ]["suppression_reason"] == "insufficient_source_support"
    assert suppressed_by_question[
        "Can I rename one workspace?"
    ]["suppression_reason"] == "too_low_volume"


def test_deflection_priority_score_keeps_cost_ahead_of_resolvability() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=37,
        ticket_source_count=37,
        output_checks={"condensed": True},
        items=(
            {
                "question": "Why does the nightly sync fail for enterprise workspaces?",
                "customer_wording": "nightly sync keeps failing",
                "topic": "integrations",
                "weighted_frequency": 8,
                "ticket_count": 8,
                "opportunity_score": 4,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": tuple(f"ticket-sync-{index}" for index in range(8)),
            },
            {
                "question": "How do I reduce repeated invoice export tickets?",
                "customer_wording": "invoice export tickets keep repeating",
                "topic": "billing",
                "weighted_frequency": 7,
                "ticket_count": 7,
                "opportunity_score": 0,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": tuple(
                    f"ticket-invoice-repeat-{index}" for index in range(7)
                ),
            },
            {
                "question": "How do I reopen an attribution export case?",
                "customer_wording": "attribution export answer still fails",
                "topic": "exports",
                "weighted_frequency": 5,
                "ticket_count": 5,
                "opportunity_score": 50,
                "answer": "Open Attribution, select the export, and rerun the report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "outcome_diagnostics": {
                    "negative_csat_ticket_count": 2,
                    "reopened_ticket_count": 2,
                    "ticket_status_summary": {"reopened": 2, "resolved": 3},
                },
                "source_ids": tuple(
                    f"ticket-attribution-risk-{index}" for index in range(5)
                ),
            },
            {
                "question": "How do I reopen an attribution export with low CSAT?",
                "customer_wording": "attribution export answer has low csat",
                "topic": "exports",
                "weighted_frequency": 5,
                "ticket_count": 5,
                "opportunity_score": 50,
                "answer": "Open Attribution, select the export, and rerun the report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "outcome_diagnostics": {
                    "csat_present_count": 5,
                    "csat_score_average": 1.0,
                    "negative_csat_ticket_count": 2,
                    "reopened_ticket_count": 2,
                    "ticket_status_summary": {"reopened": 2, "resolved": 3},
                },
                "source_ids": tuple(
                    f"ticket-attribution-low-csat-{index}" for index in range(5)
                ),
            },
            {
                "question": "How do I reopen an attribution export with malformed CSAT?",
                "customer_wording": "attribution export answer has malformed csat",
                "topic": "exports",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 40,
                "answer": "Open Attribution, select the export, and rerun the report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "outcome_diagnostics": {
                    "csat_present_count": 5,
                    "csat_score_average": "not-a-number",
                    "ticket_status_summary": {"resolved": 5},
                },
                "source_ids": tuple(
                    f"ticket-attribution-malformed-csat-{index}"
                    for index in range(3)
                ),
            },
            {
                "question": "How do I reopen an attribution export with negative CSAT?",
                "customer_wording": "attribution export answer has negative csat",
                "topic": "exports",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 40,
                "answer": "Open Attribution, select the export, and rerun the report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "outcome_diagnostics": {
                    "csat_present_count": 5,
                    "csat_score_average": -1.0,
                    "ticket_status_summary": {"resolved": 5},
                },
                "source_ids": tuple(
                    f"ticket-attribution-negative-csat-{index}"
                    for index in range(3)
                ),
            },
            {
                "question": "How do I export a quarterly billing report?",
                "customer_wording": "quarterly billing export",
                "topic": "billing",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 150,
                "answer": "Open Billing, choose Reports, then export the quarter.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": tuple(f"ticket-billing-{index}" for index in range(3)),
            },
            {
                "question": "How do I find the workspace invite article?",
                "customer_wording": "invite article still confusing",
                "topic": "workspace",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 145,
                "answer": "Open Workspace settings and choose Invitations.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "outcome_diagnostics": {
                    "reopened_ticket_count": 1,
                    "ticket_status_summary": {"reopened": 1, "resolved": 2},
                },
                "source_ids": tuple(f"ticket-workspace-{index}" for index in range(3)),
            },
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    priority_items = sections["priority_fix_queue"]["data"]["items"]
    by_question = {item["question"]: item for item in priority_items}
    unresolved = by_question[
        "Why does the nightly sync fail for enterprise workspaces?"
    ]
    lower_cost_dissatisfied = by_question[
        "How do I reopen an attribution export case?"
    ]
    lower_cost_low_average = by_question[
        "How do I reopen an attribution export with low CSAT?"
    ]
    lower_cost_malformed_average = by_question[
        "How do I reopen an attribution export with malformed CSAT?"
    ]
    lower_cost_negative_average = by_question[
        "How do I reopen an attribution export with negative CSAT?"
    ]
    higher_cost_clean = by_question[
        "How do I reduce repeated invoice export tickets?"
    ]

    assert priority_items[0]["question"] == (
        "Why does the nightly sync fail for enterprise workspaces?"
    )
    assert unresolved["estimated_support_cost"] == 108.0
    assert unresolved["status"] == "Needs answer"
    assert unresolved["fix_type"] == "create_missing_answer"
    assert "missing_answer" in unresolved["priority_drivers"]
    assert "answer" not in unresolved
    assert higher_cost_clean["estimated_support_cost"] == 94.5
    assert lower_cost_dissatisfied["estimated_support_cost"] == 67.5
    assert higher_cost_clean["priority_score"] == 307
    assert lower_cost_dissatisfied["priority_score"] == 304
    assert lower_cost_low_average["priority_score"] == 304
    assert lower_cost_malformed_average["priority_score"] == 172
    assert lower_cost_malformed_average["status"] == "Draft ready"
    assert lower_cost_malformed_average["csat_signal"] == {
        "status": "sparse",
        "csat_present_count": 5,
        "negative_csat_ticket_count": 0,
        "numeric_average": None,
    }
    assert lower_cost_negative_average["priority_score"] == 172
    assert lower_cost_negative_average["status"] == "Draft ready"
    assert lower_cost_negative_average["csat_signal"] == {
        "status": "sparse",
        "csat_present_count": 5,
        "negative_csat_ticket_count": 0,
        "numeric_average": None,
    }
    assert (
        higher_cost_clean["priority_score"]
        > lower_cost_dissatisfied["priority_score"]
    )
    assert (
        higher_cost_clean["priority_score"]
        > lower_cost_low_average["priority_score"]
    )
    assert by_question[
        "How do I export a quarterly billing report?"
    ]["status"] == "Draft ready"
    assert by_question[
        "How do I find the workspace invite article?"
    ]["status"] == "Already covered but still recurring"
    assert sections["top_unresolved_repeats"]["data"]["items"][0]["question"] == (
        "Why does the nightly sync fail for enterprise workspaces?"
    )
    assert sections["drafted_resolutions"]["data"]["items"][0]["question"] == (
        "How do I export a quarterly billing report?"
    )
    assert sections["already_covered_still_recurring"]["data"]["items"][0][
        "question"
    ] == "How do I reopen an attribution export case?"


def test_deflection_suppressed_repeat_review_queue_explains_hidden_rows() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=10,
        ticket_source_count=10,
        output_checks={"condensed": True},
        items=(
            {
                "question": "",
                "customer_wording": "",
                "topic": "unknown",
                "weighted_frequency": 8,
                "ticket_count": 3,
                "opportunity_score": 30,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (
                    "ticket-missing-question-1",
                    "ticket-missing-question-2",
                ),
            },
            {
                "question": "Can I rename one workspace?",
                "customer_wording": "rename one workspace",
                "topic": "workspace",
                "weighted_frequency": 1,
                "ticket_count": 1,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-one-off",),
            },
            {
                "question": "Why did my imported contacts duplicate?",
                "customer_wording": "contacts imported twice",
                "topic": "imports",
                "weighted_frequency": 5,
                "ticket_count": 4,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_count": 1,
                "source_ids": ("ticket-sparse-only",),
            },
            {
                "question": "How do I configure SCIM groups?",
                "customer_wording": "scim group setup",
                "topic": "identity",
                "weighted_frequency": 7,
                "ticket_count": 3,
                "opportunity_score": 18,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (
                    "ticket-scim-1",
                    "ticket-scim-2",
                    "ticket-scim-3",
                ),
            },
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    unresolved_questions = {
        item["question"]
        for item in sections["top_unresolved_repeats"]["data"]["items"]
    }
    queue = sections["suppressed_repeat_review_queue"]["data"]
    by_reason = {
        item["suppression_reason"]: item
        for item in queue["items"]
    }
    review_keys = {item["review_key"] for item in queue["items"]}

    assert unresolved_questions == {"How do I configure SCIM groups?"}
    assert queue["total_item_count"] == 3
    assert queue["reason_counts"] == {
        "insufficient_source_support": 1,
        "missing_question": 1,
        "too_low_volume": 1,
    }
    assert set(by_reason) == set(queue["reason_counts"])
    assert len(review_keys) == 3
    assert all(key.startswith("review_") and len(key) == 31 for key in review_keys)
    assert by_reason["missing_question"]["question"] == ""
    assert by_reason["missing_question"]["status"] == "Low confidence"
    assert "No normalized customer question" in by_reason[
        "missing_question"
    ]["suppression_reason_label"]
    assert by_reason["too_low_volume"]["question"] == "Can I rename one workspace?"
    assert by_reason["insufficient_source_support"]["question"] == (
        "Why did my imported contacts duplicate?"
    )


def test_deflection_priority_fix_queue_keeps_pdf_limit_items() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=12,
        ticket_source_count=12,
        output_checks={"condensed": True},
        items=tuple(
            {
                "question": f"How do I resolve repeat issue {index}?",
                "customer_wording": f"repeat issue {index}",
                "topic": "support",
                "weighted_frequency": index,
                "ticket_count": index,
                "opportunity_score": index,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": tuple(
                    f"ticket-{index}-{source}" for source in range(1, index + 1)
                ),
            }
            for index in range(1, 13)
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    priority = sections["priority_fix_queue"]["data"]

    assert priority["result_page_limit"] == 3
    assert priority["pdf_limit"] == 10
    assert len(priority["items"]) == 10
    assert priority["items"][0]["question"] == "How do I resolve repeat issue 12?"


def test_deflection_top_unresolved_repeats_keeps_pdf_limit_items() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=13,
        ticket_source_count=13,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I handle a one-off invoice question?",
                "customer_wording": "one-off invoice question",
                "topic": "billing",
                "weighted_frequency": 99,
                "ticket_count": 1,
                "opportunity_score": 99,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-singleton-1",),
            },
            *(
                {
                    "question": f"How do I resolve unresolved repeat {index}?",
                    "customer_wording": f"unresolved repeat {index}",
                    "topic": "billing",
                    "weighted_frequency": index,
                    "ticket_count": index,
                    "opportunity_score": index,
                    "answer_evidence_status": "draft_needs_review",
                    "source_ids": tuple(
                        f"ticket-unresolved-{index}-{source}"
                        for source in range(1, index + 1)
                    ),
                }
                for index in range(2, 14)
            ),
        ),
    )

    artifact = build_deflection_report_artifact(result).as_dict()
    sections = {
        section["id"]: section
        for section in artifact["report_model"]["sections"]
    }
    unresolved = sections["top_unresolved_repeats"]["data"]
    snapshot = build_deflection_snapshot(artifact, top_n=5).as_dict()

    assert unresolved["result_page_limit"] == 3
    assert unresolved["pdf_limit"] == 10
    assert unresolved["top_item_count"] == 10
    assert len(unresolved["items"]) == 10
    assert unresolved["items"][0]["question"] == (
        "How do I resolve unresolved repeat 13?"
    )
    assert len(snapshot["top_blind_spots"]) == unresolved["result_page_limit"]
    assert [
        row["question"] for row in snapshot["top_blind_spots"]
    ] == [
        "How do I resolve unresolved repeat 13?",
        "How do I resolve unresolved repeat 12?",
        "How do I resolve unresolved repeat 11?",
    ]
    assert "How do I resolve unresolved repeat 10?" not in json.dumps(
        snapshot["top_blind_spots"],
        sort_keys=True,
    )
    assert "one-off" not in json.dumps(unresolved, sort_keys=True)


def test_deflection_drafted_resolutions_keep_pdf_limit_items() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=12,
        ticket_source_count=12,
        output_checks={"condensed": True},
        items=tuple(
            {
                "question": f"How do I publish drafted answer {index}?",
                "customer_wording": f"drafted answer {index}",
                "topic": "support",
                "weighted_frequency": index,
                "ticket_count": index,
                "opportunity_score": index,
                "answer": f"Use the drafted answer for issue {index}.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": tuple(
                    f"ticket-drafted-{index}-{source}"
                    for source in range(1, index + 1)
                ),
            }
            for index in range(1, 13)
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    drafted = sections["drafted_resolutions"]["data"]

    assert drafted["result_page_limit"] == 3
    assert drafted["pdf_limit"] == 10
    assert drafted["top_item_count"] == 10
    assert len(drafted["items"]) == 10
    assert drafted["items"][0]["question"] == "How do I publish drafted answer 12?"


def test_deflection_already_covered_still_recurring_keeps_pdf_limit_items() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=12,
        ticket_source_count=12,
        output_checks={"condensed": True},
        items=tuple(
            {
                "question": f"How do I find covered answer {index}?",
                "customer_wording": f"covered answer {index}",
                "topic": "support",
                "weighted_frequency": index,
                "ticket_count": index,
                "opportunity_score": index,
                "answer": f"Use the published answer for issue {index}.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "outcome_diagnostics": {
                    "diagnostic_ticket_count": index,
                    "outcome_risk_ticket_count": 1,
                    "reopened_ticket_count": 1,
                    "negative_csat_ticket_count": 0,
                },
                "source_ids": tuple(
                    f"ticket-covered-{index}-{source}"
                    for source in range(1, index + 1)
                ),
            }
            for index in range(1, 13)
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    recurring = sections["already_covered_still_recurring"]["data"]

    assert recurring["result_page_limit"] == 3
    assert recurring["pdf_limit"] == 10
    assert recurring["top_item_count"] == 10
    assert len(recurring["items"]) == 10
    assert recurring["items"][0]["question"] == "How do I find covered answer 12?"


def test_deflection_action_evidence_quotes_match_source_ids() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=3,
        ticket_source_count=3,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I export reports?",
                "customer_wording": "export reports",
                "topic": "exports",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 11,
                "answer": "Open Analytics and export the report.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "source_ids": ("ticket-a", "ticket-b", "ticket-c"),
                "evidence_quotes": (
                    "`ticket-c` - third ticket quote",
                    "`ticket-a` - first ticket quote",
                    "`other-ticket` - unrelated quote",
                ),
            },
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).as_dict()[
            "report_model"
        ]["sections"]
    }
    top_evidence = sections["priority_fix_queue"]["data"]["items"][0]["top_evidence"]

    assert top_evidence == [
        {"source_id": "ticket-a", "evidence_quote": "`ticket-a` - first ticket quote"},
        {"source_id": "ticket-b", "evidence_quote": ""},
        {"source_id": "ticket-c", "evidence_quote": "`ticket-c` - third ticket quote"},
    ]


def test_deflection_action_identity_is_stable_when_rank_changes() -> None:
    base = _structured_report_fixture_result()
    reordered = TicketFAQMarkdownResult(
        markdown=base.markdown,
        source_count=base.source_count,
        ticket_source_count=base.ticket_source_count,
        output_checks=base.output_checks,
        items=tuple(reversed(base.items)),
    )

    base_artifact = build_deflection_report_artifact(base)
    reordered_artifact = build_deflection_report_artifact(reordered)
    base_export = build_deflection_evidence_export(base_artifact)
    reordered_export = build_deflection_evidence_export(reordered_artifact)

    base_questions = {
        question["question"]: question
        for question in base_export["questions"]
    }
    reordered_questions = {
        question["question"]: question
        for question in reordered_export["questions"]
    }

    for question, base_row in base_questions.items():
        reordered_row = reordered_questions[question]
        assert reordered_row["repeat_key"] == base_row["repeat_key"]
        assert reordered_row["cluster_id"] == base_row["cluster_id"]
        assert reordered_row["identity_basis"] == "question_topic"
        assert reordered_row["identity_confidence"] == "high"

    assert (
        base_questions["How do I export attribution reports?"]["question_id"]
        != reordered_questions["How do I export attribution reports?"]["question_id"]
    )

    base_actions = {
        item["question"]: item
        for section in base_artifact.report_model.as_dict()["sections"]
        if section["id"] == "backlog_table"
        for item in section["data"]["items"]
    }
    reordered_actions = {
        item["question"]: item
        for section in reordered_artifact.report_model.as_dict()["sections"]
        if section["id"] == "backlog_table"
        for item in section["data"]["items"]
    }
    assert (
        base_actions["How do I export attribution reports?"]["repeat_key"]
        == reordered_actions["How do I export attribution reports?"]["repeat_key"]
    )


def test_deflection_suppressed_review_key_is_stable_when_rank_changes() -> None:
    base = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=6,
        ticket_source_count=6,
        output_checks={"condensed": True},
        items=(
            {
                "question": "Can I rename one workspace?",
                "customer_wording": "rename one workspace",
                "topic": "workspace",
                "weighted_frequency": 1,
                "ticket_count": 1,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-one-off",),
            },
            {
                "question": "Why did my imported contacts duplicate?",
                "customer_wording": "contacts imported twice",
                "topic": "imports",
                "weighted_frequency": 5,
                "ticket_count": 4,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_count": 1,
                "source_ids": ("ticket-sparse-only",),
            },
        ),
    )
    reordered = TicketFAQMarkdownResult(
        markdown=base.markdown,
        source_count=base.source_count,
        ticket_source_count=base.ticket_source_count,
        output_checks=base.output_checks,
        items=tuple(reversed(base.items)),
    )

    def review_keys_by_question(result: TicketFAQMarkdownResult) -> dict[str, str]:
        sections = {
            section["id"]: section
            for section in build_deflection_report_artifact(result).report_model.as_dict()[
                "sections"
            ]
        }
        return {
            item["question"]: item["review_key"]
            for item in sections["suppressed_repeat_review_queue"]["data"]["items"]
        }

    assert review_keys_by_question(base) == review_keys_by_question(reordered)


def test_deflection_suppressed_review_key_disambiguates_identityless_rows() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={"condensed": True},
        items=(
            {
                "question": "",
                "customer_wording": "",
                "topic": "",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 30,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (),
            },
            {
                "question": "",
                "customer_wording": "",
                "topic": "",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 25,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (),
            },
        ),
    )

    sections = {
        section["id"]: section
        for section in build_deflection_report_artifact(result).report_model.as_dict()[
            "sections"
        ]
    }
    items = sections["suppressed_repeat_review_queue"]["data"]["items"]
    review_keys = [item["review_key"] for item in items]

    assert len(items) == 2
    assert [item["identity_basis"] for item in items] == [
        "insufficient_identity",
        "insufficient_identity",
    ]
    assert [item["suppression_reason"] for item in items] == [
        "missing_question",
        "missing_question",
    ]
    assert len(set(review_keys)) == len(review_keys)


def test_deflection_action_identity_ignores_ticket_id_rollover() -> None:
    base = _structured_report_fixture_result()
    next_period_items: list[dict[str, object]] = []
    for item in base.items:
        updated = dict(item)
        source_ids = item.get("source_ids")
        source_count = len(source_ids) if isinstance(source_ids, tuple) else 0
        updated["source_ids"] = tuple(
            f"next-month-{index}"
            for index in range(1, source_count + 1)
        )
        next_period_items.append(updated)
    next_period = TicketFAQMarkdownResult(
        markdown=base.markdown,
        source_count=base.source_count,
        ticket_source_count=base.ticket_source_count,
        output_checks=base.output_checks,
        items=tuple(next_period_items),
    )

    base_export = build_deflection_evidence_export(
        build_deflection_report_artifact(base)
    )
    next_export = build_deflection_evidence_export(
        build_deflection_report_artifact(next_period)
    )

    base_questions = {
        question["question"]: question
        for question in base_export["questions"]
    }
    next_questions = {
        question["question"]: question
        for question in next_export["questions"]
    }

    for question, base_row in base_questions.items():
        next_row = next_questions[question]
        assert next_row["repeat_key"] == base_row["repeat_key"]
        assert next_row["cluster_id"] == base_row["cluster_id"]
        assert next_row["identity_basis"] == "question_topic"
        assert next_row["identity_confidence"] == "high"
        assert next_row["source_ids"] != base_row["source_ids"]


def test_deflection_snapshot_projection_is_allowlist_only() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    report_model = artifact.report_model.as_dict()
    sections = {
        section["id"]: section
        for section in report_model["sections"]
    }
    sections["support_tax"]["data"]["paid_only_metric"] = "must not project"
    sections["ranked_questions"]["data"]["rows"][0]["source_ids"] = [
        "ticket-export-1"
    ]
    sections["priority_fix_queue"]["data"]["items"][0]["paid_only_note"] = (
        "hidden action detail"
    )
    sections["top_unresolved_repeats"]["data"]["items"][0]["paid_only_note"] = (
        "hidden unresolved action detail"
    )
    sections["drafted_resolutions"]["data"]["items"][0]["paid_only_note"] = (
        "hidden drafted action detail"
    )
    sections["question_details"]["data"]["rows"][1]["answer"] = (
        "LOCKED PAID ANSWER SHOULD NOT PROJECT"
    )
    sections["question_details"]["data"]["rows"][1]["steps"] = [
        "LOCKED PAID STEP SHOULD NOT PROJECT"
    ]

    snapshot = build_deflection_snapshot({"report_model": report_model}, top_n=1).as_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert "paid_only_metric" not in encoded
    assert "source_ids" not in encoded
    assert "ticket-export-1" not in encoded
    assert "priority_fix_queue" not in encoded
    assert "repeat_key" not in encoded
    assert "cluster_id" not in encoded
    assert "identity_basis" not in encoded
    assert "identity_confidence" not in encoded
    assert "hidden action detail" not in encoded
    assert "hidden unresolved action detail" not in encoded
    assert "hidden drafted action detail" not in encoded
    assert "LOCKED PAID ANSWER SHOULD NOT PROJECT" not in encoded
    assert "LOCKED PAID STEP SHOULD NOT PROJECT" not in encoded
    assert snapshot["top_questions"][0] == {
        "rank": 1,
        "question": "How do I export attribution reports?",
        "ticket_count": 5,
        "weighted_frequency": 8,
        "customer_wording": "export attribution reports",
    }
    assert snapshot["top_blind_spots"] == [
        {
            "rank": 2,
            "question": "Can I turn on SSO for all users?",
            "ticket_count": 3,
        }
    ]


def test_deflection_snapshot_falls_back_when_legacy_model_lacks_row_fields() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    payload = artifact.as_dict()
    report_model = payload["report_model"]
    sections = {
        section["id"]: section
        for section in report_model["sections"]
    }
    for row in sections["ranked_questions"]["data"]["rows"]:
        row.pop("weighted_frequency", None)
        row.pop("customer_wording", None)
    for row in sections["question_details"]["data"]["rows"]:
        row.pop("source_count", None)

    snapshot = build_deflection_snapshot(payload, top_n=1).as_dict()

    assert snapshot["top_questions"][0] == {
        "rank": 1,
        "question": "How do I export attribution reports?",
        "ticket_count": 5,
        "weighted_frequency": 8,
        "customer_wording": "export attribution reports",
    }


def test_deflection_report_model_contract_shape_requires_version_bump() -> None:
    shape = deflection_report_model_contract_shape()

    assert shape["schema_version"] == DEFLECTION_REPORT_SCHEMA_VERSION
    assert shape["model_fields"] == [
        "schema_version",
        "title",
        "summary",
        "sections",
    ]
    assert shape["section_fields"] == [
        "id",
        "title",
        "priority",
        "surfaces",
        "default_limit",
        "required_data",
        "snapshot_safe_fields",
        "data",
    ]
    assert [
        (
            section["id"],
            section["priority"],
            section["surfaces"],
            section["default_limit"],
            section["required_data"],
            section["snapshot_safe_fields"],
        )
        for section in shape["sections"]
    ] == [
        (
            "support_tax",
            10,
            ["web", "pdf", "email_summary", "markdown"],
            None,
            [
                "repeat_ticket_count",
                "non_repeat_ticket_count",
                "generated_question_count",
                "assisted_contact_cost",
                "estimated_support_cost",
                "source_date_window",
                "drafted_answer_count",
                "no_proven_answer_count",
                "ticket_source_count",
            ],
            [
                "repeat_ticket_count",
                "non_repeat_ticket_count",
                "generated_question_count",
                "drafted_answer_count",
                "no_proven_answer_count",
                "ticket_source_count",
                "source_date_window",
            ],
        ),
        ("source_file", 15, ["web", "pdf", "markdown"], None, ["source_label"], []),
        (
            "seo_targets",
            20,
            ["web", "pdf", "markdown"],
            DEFAULT_DEFLECTION_SEO_TARGET_LIMIT,
            [
                "phrases",
                "total_phrase_count",
                "displayed_phrase_count",
                "omitted_phrase_count",
                "limit",
            ],
            [],
        ),
        (
            "ranked_questions",
            30,
            ["web", "pdf", "markdown"],
            None,
            ["rows"],
            [
                "rows.rank",
                "rows.question",
                "rows.ticket_count",
                "rows.weighted_frequency",
                "rows.customer_wording",
            ],
        ),
        (
            "priority_fix_queue",
            35,
            ["web", "pdf", "email_summary"],
            3,
            [
                "items",
                "status_counts",
                "result_page_limit",
                "pdf_limit",
                "backlog_limit",
                "support_cost_basis",
            ],
            [],
        ),
        (
            "top_unresolved_repeats",
            36,
            ["web", "pdf"],
            3,
            [
                "items",
                "top_item_count",
                "result_page_limit",
                "pdf_limit",
                "support_cost_basis",
            ],
            [
                "items.rank",
                "items.question",
                "items.ticket_count",
            ],
        ),
        (
            "drafted_resolutions",
            37,
            ["web", "pdf", "email_summary"],
            3,
            [
                "items",
                "top_item_count",
                "result_page_limit",
                "pdf_limit",
            ],
            [],
        ),
        (
            "already_covered_still_recurring",
            38,
            ["web", "pdf"],
            3,
            [
                "items",
                "top_item_count",
                "result_page_limit",
                "pdf_limit",
            ],
            [],
        ),
        (
            "backlog_table",
            39,
            ["web", "pdf", "export"],
            25,
            ["items", "total_item_count", "default_limit"],
            [],
        ),
        (
            "outcome_diagnostics",
            40,
            ["web", "pdf", "markdown"],
            None,
            [
                "outcome_diagnostic_ticket_count",
                "outcome_risk_ticket_count",
                "reopened_ticket_count",
                "negative_csat_ticket_count",
                "rows",
            ],
            [],
        ),
        (
            "suppressed_repeat_review_queue",
            41,
            ["web", "export"],
            25,
            ["items", "total_item_count", "default_limit", "reason_counts"],
            [],
        ),
        (
            "question_details",
            50,
            ["web", "pdf", "markdown"],
            None,
            ["rows"],
            [
                "rows.rank",
                "rows.question",
                "rows.answer_evidence_status",
                "rows.resolution_evidence_scope",
                "rows.weighted_frequency",
                "rows.source_count",
            ],
        ),
        (
            "complete_evidence",
            90,
            ["export"],
            None,
            [
                "question_count",
                "evidence_row_count",
                "source_id_count",
                "surfaces",
            ],
            [],
        ),
    ]


def test_deflection_snapshot_projection_contract_is_registry_derived() -> None:
    shape = deflection_report_model_contract_shape()
    projection = shape["snapshot_projection"]
    fields = {field["field"]: field for field in projection["fields"]}
    encoded = json.dumps(projection, sort_keys=True)

    assert projection["schema_version"] == DEFLECTION_REPORT_SCHEMA_VERSION
    assert projection["top_level_fields"] == [
        "summary",
        "top_questions",
        "locked_questions",
        "top_blind_spots",
        "teaser",
    ]
    assert list(fields) == projection["top_level_fields"]
    assert fields["summary"]["source_section"] == "support_tax"
    assert fields["summary"]["snapshot_safe_fields"] == list(
        DEFLECTION_REPORT_SECTION_REGISTRY["support_tax"].snapshot_safe_fields
    )
    assert fields["summary"]["optional_projected_fields"] == [
        "source_date_start",
        "source_date_end",
        "source_window_days",
    ]
    assert fields["top_questions"] == {
        "field": "top_questions",
        "source_section": "ranked_questions",
        "snapshot_safe_fields": list(
            DEFLECTION_REPORT_SECTION_REGISTRY[
                "ranked_questions"
            ].snapshot_safe_fields
        ),
        "projected_fields": [
            "rank",
            "question",
            "ticket_count",
            "weighted_frequency",
            "customer_wording",
        ],
        "source_collection": "rows",
        "limit": "top_n",
    }
    assert fields["top_blind_spots"] == {
        "field": "top_blind_spots",
        "source_section": "top_unresolved_repeats",
        "snapshot_safe_fields": list(
            DEFLECTION_REPORT_SECTION_REGISTRY[
                "top_unresolved_repeats"
            ].snapshot_safe_fields
        ),
        "projected_fields": ["rank", "question", "ticket_count"],
        "source_collection": "items",
        "limit": "top_unresolved_repeats.result_page_limit",
    }
    assert fields["teaser"]["source_section"] == "question_details"
    assert fields["teaser"]["policy"] == "scoped_resolution_evidence_only"
    assert fields["teaser"]["snapshot_safe_fields"] == list(
        DEFLECTION_REPORT_SECTION_REGISTRY["question_details"].snapshot_safe_fields
    )
    assert fields["teaser"]["full_answer_fields"] == [
        "rank",
        "question",
        "answer",
        "steps",
        "answer_evidence_status",
        "resolution_evidence_scope",
        "weighted_frequency",
        "source_count",
    ]
    for forbidden in (
        "source_ids",
        "top_evidence",
        "representative_phrasing",
        "priority_score",
        "repeat_key",
        "cluster_id",
        "identity_basis",
        "identity_confidence",
    ):
        assert forbidden not in encoded


def test_deflection_report_projection_contract_is_registry_derived() -> None:
    shape = deflection_report_model_contract_shape()
    projection = shape["report_projection"]
    sections = {section["id"]: section for section in projection["sections"]}

    assert projection["schema_version"] == DEFLECTION_REPORT_SCHEMA_VERSION
    assert projection["model_fields"] == [
        "schema_version",
        "title",
        "summary",
        "sections",
    ]
    assert projection["section_fields"] == [
        "id",
        "title",
        "priority",
        "surfaces",
        "default_limit",
        "required_data",
        "snapshot_safe_fields",
        "data",
    ]
    assert list(sections) == list(DEFLECTION_REPORT_SECTION_REGISTRY)

    for section_id, definition in DEFLECTION_REPORT_SECTION_REGISTRY.items():
        section = sections[section_id]
        assert section["title"] == definition.title
        assert section["priority"] == definition.priority
        assert section["surfaces"] == list(definition.surfaces)
        assert section["default_limit"] == definition.default_limit
        assert section["required_data"] == list(definition.required_data)
        assert section["snapshot_safe_fields"] == list(definition.snapshot_safe_fields)
        assert set(definition.required_data) <= set(section["projected_fields"])

    support_tax = sections["support_tax"]
    assert support_tax["optional_projected_fields"] == [
        "annualized_support_cost",
        "annualized_run_rate_support_cost",
    ]
    assert set(support_tax["required_data"]).isdisjoint(
        support_tax["optional_projected_fields"]
    )
    assert set(support_tax["optional_projected_fields"]) <= set(
        support_tax["projected_fields"]
    )
    support_tax_nested = {
        entry["field"]: entry for entry in support_tax["nested_object_fields"]
    }
    assert support_tax_nested["source_date_window"]["projected_fields"] == [
        "source_date_start",
        "source_date_end",
        "source_window_days",
    ]
    assert support_tax_nested["source_date_window"]["hosted_consumer_safe_fields"] == [
        "source_date_start",
        "source_date_end",
        "source_window_days",
    ]

    source_file = sections["source_file"]
    assert source_file["presence"] == {
        "mode": "conditional",
        "condition": "source_label_present",
    }
    assert source_file["hosted_consumer_safe_fields"] == []

    outcome_diagnostics = sections["outcome_diagnostics"]
    assert outcome_diagnostics["presence"] == {
        "mode": "conditional",
        "condition": "outcome_diagnostics_rendered",
    }
    outcome_rows = outcome_diagnostics["collection"]
    assert "status_mix" in outcome_rows["projected_fields"]
    assert "status_mix" in outcome_rows["hosted_consumer_safe_fields"]

    assert sections["support_tax"]["presence"] == {"mode": "required"}
    top_unresolved = sections["top_unresolved_repeats"]
    assert top_unresolved["hosted_consumer_safe_fields"] == [
        "items",
        "top_item_count",
        "support_cost_basis",
    ]
    support_cost_basis = {
        entry["field"]: entry
        for entry in top_unresolved["nested_object_fields"]
    }["support_cost_basis"]
    assert support_cost_basis["projected_fields"] == [
        "status",
        "assisted_contact_cost",
        "formula",
        "source",
    ]
    assert support_cost_basis["hosted_consumer_safe_fields"] == ["status"]


def test_deflection_report_projection_separates_paid_and_hosted_action_fields() -> None:
    projection = deflection_report_model_contract_shape()["report_projection"]
    sections = {section["id"]: section for section in projection["sections"]}
    action_section_ids = [
        "priority_fix_queue",
        "top_unresolved_repeats",
        "drafted_resolutions",
        "already_covered_still_recurring",
        "backlog_table",
        "suppressed_repeat_review_queue",
    ]
    paid_only_action_fields = {
        "repeat_key",
        "cluster_id",
        "identity_basis",
        "identity_confidence",
        "fix_type",
        "recommended_title",
        "representative_phrasing",
        "support_cost_formula",
        "support_cost_source",
        "opportunity_score",
        "top_evidence",
    }
    additive_action_context_fields = {
        "owner_category",
        "evidence_tier",
        "routing_signals",
        "product_gap_summary",
        "customer_vocabulary",
        "cost_period",
        "cost_confidence",
        "jira_template",
    }

    for section_id in action_section_ids:
        collection = sections[section_id]["collection"]
        projected = set(collection["projected_fields"])
        hosted_safe = set(collection["hosted_consumer_safe_fields"])
        optional = set(collection["optional_projected_fields"])
        nested = {entry["field"]: entry for entry in collection["nested_object_fields"]}
        nested_collections = {
            entry["field"]: entry
            for entry in collection["nested_collection_fields"]
        }

        assert collection["field"] == "items"
        assert paid_only_action_fields <= projected
        assert additive_action_context_fields <= optional
        assert optional <= projected
        assert hosted_safe < projected
        assert paid_only_action_fields.isdisjoint(hosted_safe)
        assert nested["csat_signal"]["hosted_consumer_safe_fields"] == [
            "status",
            "csat_present_count",
            "negative_csat_ticket_count",
            "numeric_average",
        ]
        assert "top_evidence" not in nested
        assert nested_collections["top_evidence"]["item_type"] == "object"
        assert nested_collections["top_evidence"]["projected_fields"] == [
            "source_id",
            "evidence_quote",
        ]
        assert nested_collections["top_evidence"]["hosted_consumer_safe_fields"] == []

    priority = sections["priority_fix_queue"]
    assert priority["record_fields"] == ["status_counts"]

    suppressed = sections["suppressed_repeat_review_queue"]
    suppressed_collection = suppressed["collection"]
    assert suppressed["record_fields"] == ["reason_counts"]
    assert {
        "review_key",
        "suppression_reason",
        "suppression_reason_label",
    } <= set(suppressed_collection["projected_fields"])
    assert {
        "review_key",
        "suppression_reason",
        "suppression_reason_label",
    } <= set(suppressed_collection["hosted_consumer_safe_fields"])


def test_deflection_report_projection_marks_raw_question_evidence_export_only() -> None:
    projection = deflection_report_model_contract_shape()["report_projection"]
    sections = {section["id"]: section for section in projection["sections"]}
    question_details = sections["question_details"]["collection"]
    complete_evidence = sections["complete_evidence"]

    assert question_details["field"] == "rows"
    assert {
        "source_ids",
        "evidence_quotes",
        "outcome_diagnostics",
    } <= set(question_details["projected_fields"])
    assert {
        "source_ids",
        "evidence_quotes",
        "outcome_diagnostics",
    }.isdisjoint(question_details["hosted_consumer_safe_fields"])
    assert "source_count" in question_details["hosted_consumer_safe_fields"]
    assert "evidence_tier" in question_details["hosted_consumer_safe_fields"]
    term_mapping_contract = {
        entry["field"]: entry
        for entry in question_details["nested_collection_fields"]
    }["term_mappings"]
    assert term_mapping_contract["item_type"] == "object"
    assert term_mapping_contract["projected_fields"] == [
        "customer_term",
        "documentation_term",
        "suggestion",
        "source_id_count",
    ]
    assert term_mapping_contract["hosted_consumer_safe_fields"] == [
        "customer_term",
        "documentation_term",
        "suggestion",
        "source_id_count",
    ]
    assert complete_evidence["projected_fields"] == [
        "question_count",
        "evidence_row_count",
        "source_id_count",
        "surfaces",
    ]
    assert complete_evidence["hosted_consumer_safe_fields"] == []


def _report_projection_drift_errors(
    report_model: Mapping[str, object],
    *,
    expected_present: set[str] | None = None,
    expected_absent: set[str] | None = None,
    require_non_empty_collections: bool = True,
) -> list[str]:
    projection = deflection_report_model_contract_shape()["report_projection"]
    errors = _field_set_errors(
        "report_model",
        set(report_model),
        set(_string_list(projection.get("model_fields"))),
    )
    contract_sections = {
        str(section["id"]): section
        for section in projection["sections"]
        if isinstance(section, Mapping)
    }
    section_fields = set(_string_list(projection.get("section_fields")))
    raw_sections = report_model.get("sections")
    if not isinstance(raw_sections, list):
        errors.append("report_model.sections is not a list")
        return errors
    actual_sections: dict[str, Mapping[str, object]] = {}
    for index, section in enumerate(raw_sections):
        if not isinstance(section, Mapping):
            errors.append(f"report_model.sections[{index}] is not an object")
            continue
        section_id = section.get("id")
        if not isinstance(section_id, str) or not section_id:
            errors.append(f"report_model.sections[{index}] missing string id")
            continue
        if section_id in actual_sections:
            errors.append(f"duplicate section emitted: {section_id}")
            continue
        actual_sections[section_id] = section
        errors.extend(
            _field_set_errors(
                f"section {section_id} envelope",
                set(section),
                section_fields,
            )
        )
    expected_present = expected_present or set()
    expected_absent = expected_absent or set()

    for section_id in sorted(set(actual_sections) - set(contract_sections)):
        errors.append(f"unknown section emitted: {section_id}")
    for section_id, contract in contract_sections.items():
        presence = contract.get("presence")
        mode = (
            str(presence.get("mode"))
            if isinstance(presence, Mapping)
            else "required"
        )
        is_present = section_id in actual_sections
        if section_id in expected_absent and is_present:
            errors.append(f"section {section_id} should be absent")
        if (mode == "required" or section_id in expected_present) and not is_present:
            errors.append(f"section {section_id} missing")
        if not is_present:
            continue

        section = actual_sections[section_id]
        data = section.get("data")
        if not isinstance(data, Mapping):
            errors.append(f"section {section_id} data is not an object")
            continue
        errors.extend(
            _report_projection_data_errors(
                section_id,
                data,
                contract,
                require_non_empty_collections=require_non_empty_collections,
            )
        )
    return errors


def _report_projection_data_errors(
    section_id: str,
    data: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    require_non_empty_collections: bool,
) -> list[str]:
    expected = set(_string_list(contract.get("projected_fields")))
    optional = set(_string_list(contract.get("optional_projected_fields")))
    actual = set(data)
    errors = _field_set_errors(
        f"section {section_id} data",
        actual,
        expected,
        optional=optional,
    )
    collection = contract.get("collection")
    if isinstance(collection, Mapping):
        errors.extend(
            _report_projection_collection_errors(
                f"section {section_id}",
                data,
                collection,
                require_non_empty=require_non_empty_collections,
            )
        )
    errors.extend(
        _report_projection_nested_object_errors(
            f"section {section_id} data",
            data,
            contract.get("nested_object_fields"),
        )
    )
    errors.extend(_report_projection_record_field_errors(section_id, data, contract))
    return errors


def _report_projection_collection_errors(
    label: str,
    container: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    require_non_empty: bool,
) -> list[str]:
    field = str(contract.get("field"))
    item_type = str(contract.get("item_type") or "object")
    value = container.get(field)
    if not isinstance(value, list):
        return [f"{label}.{field} is not a list"]
    errors: list[str] = []
    if require_non_empty and not value:
        errors.append(f"{label}.{field} is empty")
    if item_type == "string":
        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(f"{label}.{field}[{index}] is not a string")
        return errors

    expected = set(_string_list(contract.get("projected_fields")))
    for index, item in enumerate(value):
        item_label = f"{label}.{field}[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{item_label} is not an object")
            continue
        errors.extend(_field_set_errors(item_label, set(item), expected))
        errors.extend(
            _report_projection_nested_object_errors(
                item_label,
                item,
                contract.get("nested_object_fields"),
            )
        )
        errors.extend(
            _report_projection_nested_collection_errors(
                item_label,
                item,
                contract.get("nested_collection_fields"),
                require_non_empty=require_non_empty,
            )
        )
    return errors


def _report_projection_nested_object_errors(
    label: str,
    container: Mapping[str, object],
    entries: object,
) -> list[str]:
    errors: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping):
            continue
        field = str(entry.get("field"))
        value = container.get(field)
        if not isinstance(value, Mapping):
            errors.append(f"{label}.{field} is not an object")
            continue
        errors.extend(
            _field_set_errors(
                f"{label}.{field}",
                set(value),
                set(_string_list(entry.get("projected_fields"))),
            )
        )
    return errors


def _report_projection_nested_collection_errors(
    label: str,
    container: Mapping[str, object],
    entries: object,
    *,
    require_non_empty: bool,
) -> list[str]:
    errors: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping):
            continue
        field = str(entry.get("field"))
        value = container.get(field)
        if not isinstance(value, list):
            errors.append(f"{label}.{field} is not a list")
            continue
        if require_non_empty and not value:
            errors.append(f"{label}.{field} is empty")
        expected = set(_string_list(entry.get("projected_fields")))
        for index, item in enumerate(value):
            item_label = f"{label}.{field}[{index}]"
            if not isinstance(item, Mapping):
                errors.append(f"{item_label} is not an object")
                continue
            errors.extend(_field_set_errors(item_label, set(item), expected))
    return errors


def _report_projection_record_field_errors(
    section_id: str,
    data: Mapping[str, object],
    contract: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    for field in _string_list(contract.get("record_fields")):
        value = data.get(field)
        if not isinstance(value, Mapping):
            errors.append(f"section {section_id} data.{field} is not an object")
    return errors


def _field_set_errors(
    label: str,
    actual: set[str],
    expected: set[str],
    *,
    optional: set[str] | None = None,
) -> list[str]:
    optional = optional or set()
    missing = sorted((expected - optional) - actual)
    extra = sorted(actual - expected)
    errors: list[str] = []
    if missing:
        errors.append(f"{label} missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{label} has unexpected fields: {', '.join(extra)}")
    return errors


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def test_deflection_report_projection_fields_match_runtime_output() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_runtime_fixture_result(),
        source_label="zendesk-trial-export.json",
    ).as_dict()

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_present={"source_file", "outcome_diagnostics"},
    )

    assert errors == []


def test_deflection_report_projection_conditional_sections_match_runtime_output() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_no_conditionals_fixture_result()
    ).as_dict()

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_absent={"source_file", "outcome_diagnostics"},
        require_non_empty_collections=False,
    )

    assert errors == []


def test_deflection_report_projection_checker_fails_on_field_drift() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_runtime_fixture_result(),
        source_label="zendesk-trial-export.json",
    ).as_dict()
    sections = {
        section["id"]: section
        for section in artifact["report_model"]["sections"]
    }
    sections["priority_fix_queue"]["data"]["items"][0].pop("top_evidence")
    sections["question_details"]["data"]["rows"][0]["surprise_raw_payload"] = {
        "ticket": "raw"
    }

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_present={"source_file", "outcome_diagnostics"},
    )

    assert (
        "section priority_fix_queue.items[0] missing fields: top_evidence"
        in errors
    )
    assert (
        "section question_details.rows[0] has unexpected fields: "
        "surprise_raw_payload"
    ) in errors


def test_deflection_report_projection_checker_fails_on_nested_drift() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_runtime_fixture_result(),
        source_label="zendesk-trial-export.json",
    ).as_dict()
    sections = {
        section["id"]: section
        for section in artifact["report_model"]["sections"]
    }
    item = sections["priority_fix_queue"]["data"]["items"][0]
    item["csat_signal"].pop("numeric_average")
    item["top_evidence"][0]["raw_ticket_body"] = "do not project"

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_present={"source_file", "outcome_diagnostics"},
    )

    assert (
        "section priority_fix_queue.items[0].csat_signal missing fields: "
        "numeric_average"
    ) in errors
    assert (
        "section priority_fix_queue.items[0].top_evidence[0] has unexpected "
        "fields: raw_ticket_body"
    ) in errors


def test_deflection_report_projection_checker_fails_on_presence_drift() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_no_conditionals_fixture_result()
    ).as_dict()
    artifact["report_model"]["sections"].append({
        "id": "outcome_diagnostics",
        "title": "Outcome Diagnostics",
        "priority": 90,
        "surfaces": ["web", "pdf", "markdown"],
        "default_limit": None,
        "required_data": [],
        "snapshot_safe_fields": [],
        "data": {
            "outcome_diagnostic_ticket_count": 0,
            "outcome_risk_ticket_count": 0,
            "reopened_ticket_count": 0,
            "negative_csat_ticket_count": 0,
            "rows": [],
        },
    })

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_absent={"source_file", "outcome_diagnostics"},
    )

    assert "section outcome_diagnostics should be absent" in errors


def test_deflection_report_projection_checker_fails_on_envelope_drift() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_runtime_fixture_result(),
        source_label="zendesk-trial-export.json",
    ).as_dict()
    report_model = artifact["report_model"]
    report_model.pop("title")
    report_model["debug_payload"] = {"raw": "do not project"}
    sections = report_model["sections"]
    first_section = sections[0]
    first_section["debug_payload"] = {"raw": "do not project"}
    sections.append(dict(first_section))
    sections.append("not-a-section")

    errors = _report_projection_drift_errors(
        report_model,
        expected_present={"source_file", "outcome_diagnostics"},
    )

    assert "report_model missing fields: title" in errors
    assert "report_model has unexpected fields: debug_payload" in errors
    assert (
        f"section {first_section['id']} envelope has unexpected fields: "
        "debug_payload"
    ) in errors
    assert f"duplicate section emitted: {first_section['id']}" in errors
    assert "report_model.sections[14] is not an object" in errors


def test_deflection_report_projection_checker_fails_on_record_field_drift() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_runtime_fixture_result(),
        source_label="zendesk-trial-export.json",
    ).as_dict()
    sections = {
        section["id"]: section
        for section in artifact["report_model"]["sections"]
    }
    sections["priority_fix_queue"]["data"]["status_counts"] = [
        "Draft ready",
        "Needs answer",
    ]

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_present={"source_file", "outcome_diagnostics"},
    )

    assert "section priority_fix_queue data.status_counts is not an object" in errors


def test_deflection_report_projection_checker_fails_on_empty_collections() -> None:
    artifact = build_deflection_report_artifact(
        _report_projection_runtime_fixture_result(),
        source_label="zendesk-trial-export.json",
    ).as_dict()
    sections = {
        section["id"]: section
        for section in artifact["report_model"]["sections"]
    }
    sections["ranked_questions"]["data"]["rows"] = []
    sections["priority_fix_queue"]["data"]["items"][0]["top_evidence"] = []

    errors = _report_projection_drift_errors(
        artifact["report_model"],
        expected_present={"source_file", "outcome_diagnostics"},
    )

    assert "section ranked_questions.rows is empty" in errors
    assert "section priority_fix_queue.items[0].top_evidence is empty" in errors


def test_deflection_snapshot_projected_fields_match_runtime_output() -> None:
    fixture = _structured_report_fixture_result()
    fixture = replace(
        fixture,
        source_count=fixture.source_count + 2,
        ticket_source_count=fixture.ticket_source_count + 2,
        items=fixture.items
        + (
            {
                "question": "How do I resend an invoice receipt?",
                "customer_wording": "resend invoice receipt",
                "topic": "billing",
                "weighted_frequency": 2,
                "ticket_count": 2,
                "opportunity_score": 8,
                "answer": "Open Billing, choose Receipts, then resend the receipt.",
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
                "steps": ["Open Billing and select Resend receipt."],
                "source_ids": ("ticket-receipt-1", "ticket-receipt-2"),
                "source_date_span": {
                    "start": "2026-05-16",
                    "end": "2026-05-17",
                    "missing_source_count": 0,
                },
            },
        ),
    )
    artifact = build_deflection_report_artifact(fixture)
    snapshot = build_deflection_snapshot(artifact, top_n=1).as_dict()
    projection = deflection_report_model_contract_shape()["snapshot_projection"]
    fields = {field["field"]: field for field in projection["fields"]}
    summary_optional = set(fields["summary"]["optional_projected_fields"])

    assert snapshot["top_questions"]
    assert snapshot["top_blind_spots"]
    assert snapshot["teaser"]["full_answer"] is not None
    assert snapshot["teaser"]["previews"]

    assert set(snapshot["summary"]) == set(fields["summary"]["projected_fields"])
    assert set(snapshot["top_questions"][0]) == set(
        fields["top_questions"]["projected_fields"]
    )
    for locked in snapshot["locked_questions"]:
        assert set(locked) == set(fields["locked_questions"]["projected_fields"])
    assert set(snapshot["top_blind_spots"][0]) == set(
        fields["top_blind_spots"]["projected_fields"]
    )
    assert set(snapshot["teaser"]) == set(fields["teaser"]["projected_fields"])
    assert set(snapshot["teaser"]["full_answer"]) == set(
        fields["teaser"]["full_answer_fields"]
    )
    assert set(snapshot["teaser"]["previews"][0]) == set(
        fields["teaser"]["preview_fields"]
    )

    no_window_fixture = replace(
        fixture,
        items=tuple(
            {
                key: value
                for key, value in item.items()
                if key != "source_date_span"
            }
            for item in fixture.items
        ),
    )
    no_window_snapshot = build_deflection_snapshot(
        build_deflection_report_artifact(no_window_fixture),
        top_n=1,
    ).as_dict()
    no_window_summary_fields = set(no_window_snapshot["summary"])

    assert no_window_summary_fields == (
        set(fields["summary"]["projected_fields"]) - summary_optional
    )
    assert summary_optional.isdisjoint(no_window_summary_fields)


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
        "repeat_key": "repeat_b110111100110011101111100111001111001111111111110",
        "cluster_id": "repeat_b110111100110011101111100111001111001111111111110",
        "identity_basis": "question_topic",
        "identity_confidence": "high",
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


def test_deflection_full_report_qa_scorecard_checks_action_section_caps() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)
    model = json.loads(json.dumps(artifact.report_model.as_dict()))
    sections = {section["id"]: section for section in model["sections"]}
    priority_rows = sections["priority_fix_queue"]["data"]["items"]
    selected_email_rows = deflection_report_email_action_rows(model)
    expected_email_priority_rows = len(selected_email_rows["priority_fix_queue"])
    expected_email_drafted_rows = len(selected_email_rows["drafted_resolutions"])
    expected_pdf_rows = min(len(priority_rows), 10)

    assert expected_pdf_rows > 0
    assert expected_email_priority_rows > 0
    assert expected_email_drafted_rows > 0
    assert expected_email_priority_rows < min(len(priority_rows), 3)
    assert [
        row["question"] for row in selected_email_rows["priority_fix_queue"]
    ] == ["Can I turn on SSO for all users?"]
    assert [
        row["question"] for row in selected_email_rows["drafted_resolutions"]
    ] == ["How do I export attribution reports?"]

    scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=export,
        surface_observations={
            "email": {
                "displayed_rows": {
                    "priority_fix_queue": expected_email_priority_rows,
                    "drafted_resolutions": expected_email_drafted_rows,
                },
            },
            "pdf": {"displayed_rows": {"priority_fix_queue": expected_pdf_rows}},
        },
    )
    bad_scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=export,
        surface_observations={
            "email": {
                "displayed_rows": {
                    "priority_fix_queue": expected_email_priority_rows - 1,
                    "drafted_resolutions": expected_email_drafted_rows,
                },
            },
            "pdf": {"displayed_rows": {"priority_fix_queue": expected_pdf_rows}},
        },
    )
    raw_min_scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=export,
        surface_observations={
            "email": {
                "displayed_rows": {
                    "priority_fix_queue": min(len(priority_rows), 3),
                    "drafted_resolutions": expected_email_drafted_rows,
                },
            },
            "pdf": {"displayed_rows": {"priority_fix_queue": expected_pdf_rows}},
        },
    )
    harness = build_deflection_full_report_qa_deterministic_harness(
        model,
        evidence_export=export,
    )

    assert scorecard["ok"] is True
    assert bad_scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in bad_scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.email.displayed_rows.priority_fix_queue" in failed
    raw_min_failed = {
        assertion["id"]
        for assertion in raw_min_scorecard["assertions"]
        if not assertion["ok"]
    }
    assert raw_min_scorecard["ok"] is False
    assert "surface.email.displayed_rows.priority_fix_queue" in raw_min_failed
    assert harness["counts"]["priority_fix_queue_count"] == len(priority_rows)
    email_priority_assertion = next(
        assertion
        for assertion in harness["assertions"]
        if assertion["id"] == "surface.email.displayed_rows.priority_fix_queue"
    )
    assert email_priority_assertion["expected"] == expected_email_priority_rows
    assert email_priority_assertion["actual"] == expected_email_priority_rows
    assert any(
        assertion["id"] == "harness.surface.pdf.displayed_rows.priority_fix_queue.present"
        and assertion["ok"] is True
        for assertion in harness["assertions"]
    )
    assert any(
        assertion["id"] == "harness.surface.email.displayed_rows.priority_fix_queue.present"
        and assertion["ok"] is True
        for assertion in harness["assertions"]
    )
    assert any(
        assertion["id"] == "harness.surface.email.displayed_rows.drafted_resolutions.present"
        and assertion["ok"] is True
        for assertion in harness["assertions"]
    )


def test_deflection_full_report_qa_scorecard_mirrors_email_action_limits() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)
    model = json.loads(json.dumps(artifact.report_model.as_dict()))
    sections = {section["id"]: section for section in model["sections"]}
    sections["priority_fix_queue"]["data"]["result_page_limit"] = 0
    sections["drafted_resolutions"]["data"]["result_page_limit"] = 0

    scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=export,
        surface_observations={
            "email": {
                "displayed_rows": {
                    "priority_fix_queue": 0,
                    "drafted_resolutions": 0,
                },
            },
        },
    )
    bad_scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=export,
        surface_observations={
            "email": {
                "displayed_rows": {
                    "priority_fix_queue": 1,
                    "drafted_resolutions": 1,
                },
            },
        },
    )

    assert scorecard["ok"] is True
    assert bad_scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in bad_scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.email.displayed_rows.priority_fix_queue" in failed
    assert "surface.email.displayed_rows.drafted_resolutions" in failed


def test_deflection_full_report_qa_scorecard_requires_action_sections() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    model = json.loads(json.dumps(artifact.report_model.as_dict()))
    model["sections"] = [
        section
        for section in model["sections"]
        if section["id"]
        not in {
            "priority_fix_queue",
            "top_unresolved_repeats",
            "drafted_resolutions",
            "already_covered_still_recurring",
        }
    ]

    scorecard = build_deflection_full_report_qa_deterministic_harness(model)

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "model.section.priority_fix_queue.present" in failed
    assert "model.section.top_unresolved_repeats.present" in failed
    assert "model.section.drafted_resolutions.present" in failed
    assert "model.section.already_covered_still_recurring.present" in failed


def test_deflection_full_report_qa_harness_defers_result_page_action_row_observer() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)
    model = artifact.report_model.as_dict()
    sections = {section["id"]: section for section in model["sections"]}
    priority_rows = sections["priority_fix_queue"]["data"]["items"]
    unresolved_rows = sections["top_unresolved_repeats"]["data"]["items"]
    drafted_rows = sections["drafted_resolutions"]["data"]["items"]
    counts = build_deflection_full_report_qa_scorecard(model)["counts"]

    scorecard = build_deflection_full_report_qa_deterministic_harness(
        model,
        evidence_export=export,
        surface_observations={
            "result_page": {
                "counts": {
                    "repeat_ticket_count": counts["repeat_ticket_count"],
                    "generated_question_count": counts["generated_question_count"],
                    "ranked_question_count": counts["ranked_question_count"],
                    "drafted_answer_count": counts["drafted_answer_count"],
                    "no_proven_answer_count": counts["no_proven_answer_count"],
                    "ticket_source_count": counts["ticket_source_count"],
                    "estimated_support_cost": counts["estimated_support_cost"],
                    "evidence_row_count": len(export["evidence_rows"]),
                    "source_id_count": export["summary"]["source_id_count"],
                    "product_gap_card_count": counts["product_gap_card_count"],
                    "jira_handoff_count": counts["jira_handoff_count"],
                },
                "displayed_rows": {
                    "ranked_questions": counts["ranked_question_count"],
                    "question_details": counts["question_detail_count"],
                    "product_gap_cards": counts["product_gap_card_count"],
                    "jira_handoffs": counts["jira_handoff_count"],
                    "seo_targets": counts["seo_total_phrase_count"],
                    "outcome_diagnostics": counts["outcome_diagnostic_row_count"],
                },
            },
            "email": {
                "counts": {
                    "repeat_ticket_count": counts["repeat_ticket_count"],
                    "generated_question_count": counts["generated_question_count"],
                    "drafted_answer_count": counts["drafted_answer_count"],
                    "no_proven_answer_count": counts["no_proven_answer_count"],
                    "ticket_source_count": counts["ticket_source_count"],
                    "estimated_support_cost": counts["estimated_support_cost"],
                },
                "displayed_rows": {
                    "priority_fix_queue": 1,
                    "drafted_resolutions": min(len(drafted_rows), 3),
                },
            },
            "pdf": {
                "counts": {
                    "repeat_ticket_count": counts["repeat_ticket_count"],
                    "generated_question_count": counts["generated_question_count"],
                    "ranked_question_count": counts["ranked_question_count"],
                    "drafted_answer_count": counts["drafted_answer_count"],
                    "no_proven_answer_count": counts["no_proven_answer_count"],
                    "ticket_source_count": counts["ticket_source_count"],
                    "estimated_support_cost": counts["estimated_support_cost"],
                },
                "displayed_rows": {
                    "ranked_questions": counts["ranked_question_count"],
                    "question_details": counts["question_detail_count"],
                    "priority_fix_queue": len(priority_rows),
                    "top_unresolved_repeats": len(unresolved_rows),
                    "drafted_resolutions": len(drafted_rows),
                },
            },
            "evidence_export": {
                "counts": {
                    "evidence_question_count": len(export["questions"]),
                    "evidence_row_count": len(export["evidence_rows"]),
                    "source_id_count": export["summary"]["source_id_count"],
                    "drafted_answer_count": counts["drafted_answer_count"],
                    "no_proven_answer_count": counts["no_proven_answer_count"],
                },
            },
        },
    )

    assert scorecard["ok"] is True
    assert not any(
        assertion["id"]
        == "harness.surface.result_page.displayed_rows.priority_fix_queue.present"
        for assertion in scorecard["assertions"]
    )


def test_deflection_full_report_qa_scorecard_honors_zero_row_surface_caps() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "email_summary": {"displayed_rows": {"question_details": 0}},
        },
        surface_caps={"email_summary": {"question_details": 0}},
    )

    assert scorecard["ok"] is True


def test_deflection_full_report_qa_scorecard_fails_empty_surface_observation() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=export,
        surface_observations={"result_page": {}},
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.result_page.observation_has_data" in failed


def test_deflection_full_report_qa_scorecard_rejects_boolean_count_observation() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "result_page": {"counts": {"drafted_answer_count": True}},
        },
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.result_page.count.drafted_answer_count" in failed


def test_deflection_full_report_qa_scorecard_fails_missing_zero_export_fields() -> None:
    artifact = build_deflection_report_artifact(
        TicketFAQMarkdownResult(
            markdown="# FAQ",
            source_count=0,
            ticket_source_count=0,
            output_checks={"condensed": True},
            items=(),
        )
    )
    malformed_export = {
        "schema_version": DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
        "summary": {},
    }

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=malformed_export,
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "evidence_export.summary.question_count" in failed
    assert "evidence_export.questions.present" in failed
    assert "evidence_export.evidence_rows.present" in failed


def test_deflection_full_report_qa_scorecard_redacts_bad_observed_strings() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)
    raw_request_id = "content-ops-" + "45c06a6950ec4677a214368d6e4dc44f"
    raw_result_url = f"https://www.juancanfield.com/results/{raw_request_id}"

    scorecard = build_deflection_full_report_qa_scorecard(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            raw_result_url: {
                "counts": {"ticket-export-1": "ticket-export-1"},
            }
        },
    )

    assert scorecard["ok"] is False
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-export-1" not in encoded
    assert raw_request_id not in encoded
    assert "juancanfield.com" not in encoded
    assert "surface.surface_1.count.count_1" in encoded
    assert "<redacted-string>" in encoded


def test_deflection_full_report_qa_deterministic_harness_composes_all_surfaces() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_deterministic_harness(
        artifact.report_model,
        evidence_export=export,
    )

    assert scorecard["schema_version"] == (
        DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION
    )
    assert scorecard["ok"] is True
    assert scorecard["surfaces"] == {
        "required": ["email", "result_page", "pdf", "evidence_export"],
        "observed": ["email", "evidence_export", "pdf", "result_page"],
    }
    assertion_ids = {assertion["id"] for assertion in scorecard["assertions"]}
    assert "harness.surface.email.present" in assertion_ids
    assert "harness.surface.result_page.present" in assertion_ids
    assert "harness.surface.pdf.present" in assertion_ids
    assert "harness.surface.evidence_export.present" in assertion_ids
    assert "surface.email.count.repeat_ticket_count" in assertion_ids
    assert "surface.email.displayed_rows.priority_fix_queue" in assertion_ids
    assert "surface.email.displayed_rows.drafted_resolutions" in assertion_ids
    assert "harness.surface.email.displayed_rows.priority_fix_queue.present" in assertion_ids
    assert "harness.surface.email.displayed_rows.drafted_resolutions.present" in assertion_ids
    assert "surface.result_page.displayed_rows.seo_targets" in assertion_ids
    assert "surface.pdf.displayed_rows.ranked_questions" in assertion_ids
    assert "surface.evidence_export.count.evidence_row_count" in assertion_ids

    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-export-1" not in encoded
    assert "Export attribution" not in encoded


def test_deflection_full_report_qa_deterministic_harness_requires_each_surface() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_deterministic_harness(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "email": {"counts": {"repeat_ticket_count": 8}},
            "result_page": {"counts": {"repeat_ticket_count": 8}},
            "evidence_export": {"counts": {"evidence_row_count": 8}},
        },
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "harness.surface.pdf.present" in failed


def test_deflection_full_report_qa_deterministic_harness_requires_surface_metrics() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_deterministic_harness(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "email": {"counts": {"repeat_ticket_count": 8}},
            "result_page": {
                "counts": {"repeat_ticket_count": 8},
                "displayed_rows": {"ranked_questions": 2},
            },
            "pdf": {"counts": {"repeat_ticket_count": 8}},
            "evidence_export": {"counts": {"evidence_row_count": 8}},
        },
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "harness.surface.email.count.generated_question_count.present" in failed
    assert (
        "harness.surface.email.displayed_rows.priority_fix_queue.present"
        in failed
    )
    assert (
        "harness.surface.email.displayed_rows.drafted_resolutions.present"
        in failed
    )
    assert "harness.surface.result_page.count.evidence_row_count.present" in failed
    assert (
        "harness.surface.result_page.displayed_rows.question_details.present"
        in failed
    )
    assert "harness.surface.pdf.displayed_rows.ranked_questions.present" in failed
    assert (
        "harness.surface.evidence_export.count.evidence_question_count.present"
        in failed
    )


def test_deflection_full_report_qa_deterministic_harness_fails_surface_mismatches() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)

    scorecard = build_deflection_full_report_qa_deterministic_harness(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "email": {"counts": {"repeat_ticket_count": 9}},
            "result_page": {
                "counts": {"repeat_ticket_count": 8},
                "displayed_rows": {"ranked_questions": 3},
            },
            "pdf": {
                "counts": {"repeat_ticket_count": 8},
                "displayed_rows": {"ranked_questions": 2},
            },
            "evidence_export": {"counts": {"evidence_row_count": 8}},
        },
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.email.count.repeat_ticket_count" in failed
    assert "surface.result_page.displayed_rows.ranked_questions" in failed


def test_deflection_full_report_qa_deterministic_harness_redacts_unknown_surface_names() -> None:
    artifact = build_deflection_report_artifact(_structured_report_fixture_result())
    export = build_deflection_evidence_export(artifact)
    raw_request_id = "content-ops-" + "45c06a6950ec4677a214368d6e4dc44f"
    raw_result_url = f"https://www.juancanfield.com/results/{raw_request_id}"

    scorecard = build_deflection_full_report_qa_deterministic_harness(
        artifact.report_model,
        evidence_export=export,
        surface_observations={
            "email": {"counts": {"repeat_ticket_count": 8}},
            "result_page": {"counts": {"repeat_ticket_count": 8}},
            "pdf": {"counts": {"repeat_ticket_count": 8}},
            "evidence_export": {"counts": {"evidence_row_count": 8}},
            raw_result_url: {"counts": {"repeat_ticket_count": 8}},
        },
    )

    encoded = json.dumps(scorecard, sort_keys=True)
    assert raw_request_id not in encoded
    assert "juancanfield.com" not in encoded
    assert "surface_" in encoded


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
        "top_blind_spots": [],
        "teaser": {"full_answer": None, "previews": []},
    }
    assert "Open Analytics" not in encoded
    assert "ticket-export-1" not in encoded
    assert "evidence_quotes" not in encoded
    assert "source_ids" not in encoded


def test_deflection_report_payload_scrubs_supported_pii_before_snapshot_projection() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=2,
        ticket_source_count=2,
        output_checks={"condensed": True},
        items=(
            {
                "question": "How do I reset access for jane.doe@acme.com?",
                "question_source": "customer_wording",
                "customer_wording": (
                    "Jane can be reached at 555-123-4567 for account 4829103."
                ),
                "topic": "Password reset for jane.doe@acme.com",
                "weighted_frequency": 4,
                "ticket_count": 2,
                "answer": "Send jane.doe@acme.com a reset link.",
                "steps": [
                    "Confirm account 4829103 before resetting access.",
                    "Call (555) 123-4567 if the customer is locked out.",
                    "Escalate claim 999999 if the reset still fails.",
                ],
                "term_mappings": [
                    {
                        "customer_term": "Jane's account 4829103",
                        "documentation_term": "password reset",
                        "suggestion": "Remove XXXXX before publishing.",
                        "source_id_count": 2,
                    }
                ],
                "source_ids": ("ticket-source-a", "ticket-source-b"),
                "evidence_quotes": (
                    "`ticket-source-a`: jane.doe@acme.com mentioned XXXXX.",
                    "`ticket-source-b`: Call 555-123-4567 for ref 777777.",
                ),
                "answer_evidence_status": "resolution_evidence",
                "resolution_evidence_scope": "scoped",
            },
        ),
    )
    artifact = build_deflection_report_artifact(result).as_dict()

    scrubbed_artifact = scrub_deflection_report_payload(artifact)
    snapshot = build_deflection_snapshot(scrubbed_artifact).as_dict()
    encoded = json.dumps(
        {"artifact": scrubbed_artifact, "snapshot": snapshot},
        sort_keys=True,
    ).lower()

    for raw_fragment in (
        "jane.doe",
        "acme.com",
        "555-123-4567",
        "(555) 123-4567",
        "4829103",
        "777777",
        "999999",
        "xxxxx",
    ):
        assert raw_fragment not in encoded
    assert scrubbed_artifact["faq_result"]["items"][0]["source_ids"] == [
        "ticket-source-a",
        "ticket-source-b",
    ]
    assert "[redacted-email]" in encoded
    assert "[redacted-phone]" in encoded
    assert "[redacted-identifier]" in encoded
    assert "[redacted-text]" in encoded


def test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys() -> None:
    scrubbed = scrub_deflection_report_payload(
        {
            "source_id": "4829103",
            "source_ids": [
                "777777",
                "ticket-4829103",
                "owner@example.com",
                "123-45-6789",
                "4111 1111 1111 1111",
            ],
            "first_source_id": "888888",
            "account_id": 5550000,
            "id": "question_details",
            "note": (
                "Sources 4829103 and ticket-4829103 reached "
                "_jane.doe@acme.com_ at _555-123-4567_ about case 999999."
            ),
            "jane.doe@acme.com": {"_555-123-4567_": "visible"},
        }
    )

    encoded = json.dumps(scrubbed, sort_keys=True).lower()

    for raw_fragment in (
        "5550000",
        "999999",
        "jane.doe",
        "acme.com",
        "owner@example.com",
        "123-45-6789",
        "4111 1111 1111 1111",
        "555-123-4567",
    ):
        assert raw_fragment not in encoded
    assert scrubbed["source_id"] == "4829103"
    assert scrubbed["source_ids"] == [
        "777777",
        "ticket-4829103",
        "[redacted-email]",
        "[redacted-ssn]",
        "[redacted-payment-card]",
    ]
    assert scrubbed["first_source_id"] == "888888"
    assert scrubbed["account_id"].startswith("deflection-ref-")
    assert scrubbed["id"] == "question_details"
    assert "[redacted-email]" in scrubbed
    assert any("[redacted-phone]" in key for key in scrubbed["[redacted-email]"])
    assert "[redacted-email]" in encoded
    assert "[redacted-phone]" in encoded
    assert "[redacted-identifier]" in encoded


def test_deflection_report_payload_scrubs_local_entity_shapes() -> None:
    scrubbed = scrub_deflection_report_payload(
        {
            "source_id": "ticket-source-a",
            "source_ids": [
                "ticket-source-a",
                "owner@example.com",
                "123 Maple Street",
                "customer: jane doe",
            ],
            "answer": (
                "Customer: Jane Doe asked us to ship the replacement to "
                "123 Maple Street Apt 4B. Their migration token is "
                "CUST-9XQ7-ABCD."
            ),
            "steps": [
                "Requester name is Maria Garcia.",
                "requester name is maria garcia.",
                "Mail the label to 44 W 9th St Unit 12.",
                "Search for profile code A9X4Q7Z2 before replying.",
            ],
        }
    )

    encoded = json.dumps(scrubbed, sort_keys=True)

    for raw_fragment in (
        "Jane Doe",
        "123 Maple",
        "CUST-9XQ7-ABCD",
        "Maria Garcia",
        "maria garcia",
        "44 W 9th",
        "A9X4Q7Z2",
    ):
        assert raw_fragment not in encoded
    assert scrubbed["source_id"] == "ticket-source-a"
    assert scrubbed["source_ids"] == [
        "ticket-source-a",
        "[redacted-email]",
        "[redacted-address]",
        "customer: [redacted-name]",
    ]
    assert "[redacted-name]" in encoded
    assert "[redacted-address]" in encoded
    assert "[redacted-identifier]" in encoded


@pytest.mark.parametrize(
    ("raw_text", "raw_fragment", "expected_token"),
    (
        ("SSN 123-45-6789 was in the note.", "123-45-6789", "[redacted-ssn]"),
        ("Social security 987-65-4321 appeared.", "987-65-4321", "[redacted-ssn]"),
        ("Customer record had 212-55-0199.", "212-55-0199", "[redacted-ssn]"),
        (
            "Card 4111 1111 1111 1111 was pasted.",
            "4111 1111 1111 1111",
            "[redacted-payment-card]",
        ),
        (
            "Card 5555-5555-5555-4444 was pasted.",
            "5555-5555-5555-4444",
            "[redacted-payment-card]",
        ),
        (
            "Amex 378282246310005 was pasted.",
            "378282246310005",
            "[redacted-payment-card]",
        ),
        (
            "Discover 6011 1111 1111 1117 was pasted.",
            "6011 1111 1111 1117",
            "[redacted-payment-card]",
        ),
        (
            "Diners 30569309025904 was pasted.",
            "30569309025904",
            "[redacted-payment-card]",
        ),
    ),
)
def test_deflection_report_payload_scrubs_ssn_and_payment_card_shapes(
    raw_text: str,
    raw_fragment: str,
    expected_token: str,
) -> None:
    scrubbed = scrub_deflection_report_payload({raw_text: raw_text})

    encoded = json.dumps(scrubbed, sort_keys=True)
    assert raw_fragment not in encoded
    assert expected_token in encoded


@pytest.mark.parametrize(
    ("raw_text", "raw_fragment"),
    (
        ("DOB 1990-04-17 was pasted.", "1990-04-17"),
        ("dob: 1984/11/02 was pasted.", "1984/11/02"),
        ("Date of birth is 04/17/1990.", "04/17/1990"),
        ("Birthdate=11-02-1984 appeared.", "11-02-1984"),
        ("Customer was born on April 17, 1990.", "April 17, 1990"),
        ("Customer was born on 17 April 1990.", "17 April 1990"),
        ("birthday is 1990-05-03.", "1990-05-03"),
        ("Customer was born 05/03/1990.", "05/03/1990"),
    ),
)
def test_deflection_report_payload_scrubs_context_cued_dob_shapes(
    raw_text: str,
    raw_fragment: str,
) -> None:
    scrubbed = scrub_deflection_report_payload({raw_text: raw_text})

    encoded = json.dumps(scrubbed, sort_keys=True)
    assert raw_fragment not in encoded
    assert "[redacted-dob]" in encoded


@pytest.mark.parametrize(
    "raw_text",
    (
        "Report date 1990-04-17 stayed readable.",
        "The launch happened on April 17, 1990.",
        "CVE-2021-44228 and ISO 27001 stayed readable.",
        "The 2026 count is 1234567 and the price is $49.",
        "DOB policy changed in 2026 without listing a birth date.",
        "Born to lead in 2024 stayed readable.",
        "The birthday reminder for 2026-01-15 stayed readable.",
    ),
)
def test_deflection_report_payload_preserves_dob_near_misses(raw_text: str) -> None:
    scrubbed = scrub_deflection_report_payload({"answer": raw_text})

    assert scrubbed["answer"] == raw_text
    assert "[redacted-dob]" not in scrubbed["answer"]


def test_deflection_report_payload_preserves_card_number_near_misses() -> None:
    raw_text = (
        "CVE-2021-44228, ISO 27001, SKU-12345678, price $49, "
        "batch number 1234 5678 9012 3456, and identifier "
        "11111111-1111-4111-8111-111111111111 stayed readable."
    )

    scrubbed = scrub_deflection_report_payload({"answer": raw_text})

    assert scrubbed["answer"] == raw_text
    assert "[redacted-payment-card]" not in scrubbed["answer"]
    assert "[redacted-ssn]" not in scrubbed["answer"]


def test_deflection_report_payload_scrubs_numeric_email_as_email() -> None:
    scrubbed = scrub_deflection_report_payload(
        {"answer": "Reply to 4111111111111111@example.com about the ticket."}
    )

    assert scrubbed["answer"] == "Reply to [redacted-email] about the ticket."
    assert "@example.com" not in scrubbed["answer"]
    assert "[redacted-payment-card]" not in scrubbed["answer"]


@pytest.mark.parametrize(
    "raw_text",
    (
        "Card 4111 1111 1111 1111 05/26 was pasted.",
        "Card 4111 1111 1111 1111 123 was pasted.",
    ),
)
def test_deflection_report_payload_scrubs_card_before_adjacent_digits(
    raw_text: str,
) -> None:
    scrubbed = scrub_deflection_report_payload({"answer": raw_text})

    assert "4111 1111 1111 1111" not in scrubbed["answer"]
    assert "[redacted-payment-card]" in scrubbed["answer"]


@pytest.mark.parametrize(
    ("raw_text", "expected_text"),
    (
        ("Customer Jordan Lee asked for help.", "Customer [redacted-name] asked for help."),
        ("Requester Maria Garcia reopened it.", "Requester [redacted-name] reopened it."),
        ("Client Priya Shah needs a refund.", "Client [redacted-name] needs a refund."),
        ("User Alex Chen reported a bug.", "User [redacted-name] reported a bug."),
        ("Agent Sam Rivera replied.", "Agent [redacted-name] replied."),
        ("Member Pat Morgan updated billing.", "Member [redacted-name] updated billing."),
        (
            "Customer Jane Smith Account was closed.",
            "Customer [redacted-name] Account was closed.",
        ),
        ("Customer Maya Chen Report", "Customer [redacted-name] Report"),
        (
            "Customer Mary Jane Watson Report plan was upgraded.",
            "Customer [redacted-name] Report plan was upgraded.",
        ),
        (
            "Customer Jane Smith Premium plan was upgraded.",
            "Customer [redacted-name] Premium plan was upgraded.",
        ),
        (
            "Customer Taylor Morgan Platinum account was upgraded.",
            "Customer [redacted-name] Platinum account was upgraded.",
        ),
        (
            "Customer Jane Smith Gold plan was upgraded.",
            "Customer [redacted-name] Gold plan was upgraded.",
        ),
    ),
)
def test_deflection_report_payload_scrubs_plain_role_cued_names(
    raw_text: str,
    expected_text: str,
) -> None:
    scrubbed = scrub_deflection_report_payload({"answer": raw_text})

    assert scrubbed["answer"] == expected_text


@pytest.mark.parametrize(
    "raw_text",
    (
        "Customer Reset Password is a heading.",
        "Customer Support Portal stayed online.",
        "Agent Support Desk replied.",
        "Customer Premium Plan upgrade stayed readable.",
        "customer Annual Subscription stayed readable.",
        "Customer Success Manager replied.",
        "Member Benefits Team answered.",
        "User Experience Research finished.",
        "Customer Annual Subscription Premium stayed readable.",
    ),
)
def test_deflection_report_payload_preserves_plain_role_cue_near_misses(
    raw_text: str,
) -> None:
    scrubbed = scrub_deflection_report_payload({"answer": raw_text})

    assert scrubbed["answer"] == raw_text
    assert "[redacted-name]" not in scrubbed["answer"]


def test_deflection_report_payload_scrubs_standard_prefixed_identifier_leaks() -> None:
    scrubbed = scrub_deflection_report_payload(
        {
            "source_id": "ISO customer 123456 should be scrubbed.",
            "answer": (
                "customer id is ISO 9X4Q7ABCD12. "
                "reference ISO ABC123XYZ7. "
                "HIPAA 4829103 case. "
                "ISO customer 123456 should be scrubbed. "
                "ISO 27001 references and reference ISO-27001 stay readable."
            ),
        }
    )

    encoded = json.dumps(scrubbed, sort_keys=True)
    for raw_fragment in (
        "ISO 9X4Q7ABCD12",
        "ISO ABC123XYZ7",
        "HIPAA 4829103",
        "customer 123456",
    ):
        assert raw_fragment not in encoded
    assert scrubbed["source_id"] == "ISO [redacted-identifier] should be scrubbed."
    assert "customer id is [redacted-identifier]" in scrubbed["answer"]
    assert "reference [redacted-identifier]" in scrubbed["answer"]
    assert "HIPAA [redacted-identifier]" in scrubbed["answer"]
    assert "ISO [redacted-identifier] should be scrubbed" in scrubbed["answer"]
    assert "ISO 27001 references" in scrubbed["answer"]
    assert "reference ISO-27001" in scrubbed["answer"]


def test_deflection_report_payload_scrubs_pii_regression_shapes() -> None:
    uuid_value = "123e4567-e89b-12d3-a456-426614174000"
    scrubbed = scrub_deflection_report_payload(
        {
            "source_id": "customer is Jane Smith ticket",
            "source_ids": [
                "account is 1234567",
                "customer is John Member",
                "ticket-source-a",
            ],
            "answer": (
                "The customer is Jane Smith ticket was closed. "
                "Requester name is Maria Elena Garcia account is active. "
                "Requester name is Jane Client. "
                f"The customer id is {uuid_value}. "
                "customer id is 12345678 and the ordinary count is 12345678. "
                "customer id is ISO9X4Q7ABCD, session token is SKU90X4Q7AB, "
                "and account id is HIPAA9X4Q7AB. "
                "case CVE-2021-44228, order SKU-12345678, and "
                "reference ISO-27001 should stay readable. ISO 27001 references "
                "should stay readable too. HIPAA2026 too."
            ),
            "steps": [
                "agent is Alex Chen member was reassigned.",
                "session token is CUST-9XQ7-ABCD.",
                "reference ISO27001 remains a standard.",
            ],
        }
    )

    answer = scrubbed["answer"]
    encoded = json.dumps(scrubbed, sort_keys=True)
    for raw_fragment in (
        "Jane Smith",
        "Maria Elena Garcia",
        "Jane Client",
        "John Member",
        uuid_value,
        "customer id is 12345678",
        "account is 1234567",
        "ISO9X4Q7ABCD",
        "SKU90X4Q7AB",
        "HIPAA9X4Q7AB",
        "Alex Chen",
        "CUST-9XQ7-ABCD",
    ):
        assert raw_fragment not in encoded
    assert "customer is [redacted-name] ticket was closed" in answer
    assert "Requester name is [redacted-name] account is active" in answer
    assert "Requester name is [redacted-name]" in answer
    assert "customer id is [redacted-identifier]" in answer
    assert "ordinary count is 12345678" in answer
    assert "case CVE-2021-44228" in answer
    assert "order SKU-12345678" in answer
    assert "reference ISO-27001" in answer
    assert "ISO 27001 references" in answer
    assert "HIPAA2026" in answer
    assert scrubbed["source_id"] == "customer is [redacted-name] ticket"
    assert scrubbed["source_ids"] == [
        "[redacted-identifier]",
        "customer is [redacted-name]",
        "ticket-source-a",
    ]
    assert scrubbed["steps"][0] == "agent is [redacted-name] member was reassigned."
    assert scrubbed["steps"][1] == "session token is [redacted-identifier]."
    assert scrubbed["steps"][2] == "reference ISO27001 remains a standard."


def test_deflection_report_payload_preserves_local_entity_near_misses() -> None:
    scrubbed = scrub_deflection_report_payload(
        {
            "source_id": "ticket-source-a",
            "answer": (
                "Northstar Analytics follows ISO-9001. Customer: Reset Password "
                "is a heading, not a person. The 2026 count is 1234567, the "
                "price is $49, and ticket-source-a stays usable. "
                "CVE-2021-44228 affected a dependency, SKU ABC12345 is active, "
                "iPhone15Pro and Windows11 are supported, ISO27001 and HIPAA2026 "
                "remain valid, ISO 27001 references remain valid, Q4FY2026 is a "
                "reporting label, campaign2026 is a campaign tag, 4 tickets are "
                "on the way, 4 invoices in one place, and 4 cases in court."
            ),
        }
    )

    answer = scrubbed["answer"]
    assert "Northstar Analytics" in answer
    assert "ISO-9001" in answer
    assert "Customer: Reset Password" in answer
    assert "2026" in answer
    assert "1234567" in answer
    assert "$49" in answer
    assert "ticket-source-a" in answer
    assert "CVE-2021-44228" in answer
    assert "SKU ABC12345" in answer
    assert "iPhone15Pro" in answer
    assert "Windows11" in answer
    assert "ISO27001" in answer
    assert "ISO 27001 references" in answer
    assert "HIPAA2026" in answer
    assert "Q4FY2026" in answer
    assert "campaign2026" in answer
    assert "4 tickets are on the way" in answer
    assert "4 invoices in one place" in answer
    assert "4 cases in court" in answer
    assert "[redacted-name]" not in answer
    assert "[redacted-address]" not in answer
    assert "[redacted-identifier]" not in answer


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
    assert [row["rank"] for row in snapshot["top_blind_spots"]] == [2]
    assert snapshot["locked_questions"] == [
        {"rank": 3, "ticket_count": 0},
    ]
    assert snapshot["top_blind_spots"] == [
        {"rank": 2, "question": "Can I enable SSO?", "ticket_count": 2}
    ]
    assert "question" not in snapshot["locked_questions"][0]
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
    assert [preview["rank"] for preview in snapshot["teaser"]["previews"]] == [4]
    assert snapshot["locked_questions"] == [
        {"rank": 2, "ticket_count": 1},
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
    assert locked.report_model() is not None
    assert locked.report_model()["schema_version"] == artifact["report_model"]["schema_version"]
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
    assert unlocked.report_model() is not None
    assert unlocked.report_model()["schema_version"] == artifact["report_model"]["schema_version"]
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
    assert record.report_model() is not None
    assert record.report_model()["schema_version"] == artifact["report_model"]["schema_version"]


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
                "snapshot_safe_fields": [],
                "data": {"legacy_count": 3},
            },
            {
                "id": "future_section",
                "title": "Future section",
                "priority": 70,
                "surfaces": ["web", "pdf"],
                "default_limit": None,
                "required_data": ["future_rows"],
                "snapshot_safe_fields": [],
                "data": {"future_rows": [{"id": "future-1"}]},
            },
        ],
    }


def test_stored_deflection_report_model_backfills_legacy_action_limits() -> None:
    artifact = build_deflection_report_artifact(
        _structured_report_fixture_result()
    ).as_dict()
    legacy_artifact = json.loads(json.dumps(artifact))
    section_by_id = {
        section["id"]: section
        for section in legacy_artifact["report_model"]["sections"]
    }
    expected_limits = {
        "priority_fix_queue": {
            "result_page_limit": 3,
            "pdf_limit": 10,
            "backlog_limit": 25,
        },
        "top_unresolved_repeats": {
            "result_page_limit": 3,
            "pdf_limit": 10,
        },
        "drafted_resolutions": {
            "result_page_limit": 3,
            "pdf_limit": 10,
        },
        "already_covered_still_recurring": {
            "result_page_limit": 3,
            "pdf_limit": 10,
        },
    }
    for section_id, limits in expected_limits.items():
        section = section_by_id[section_id]
        for key in limits:
            section["data"].pop(key, None)
            section["required_data"] = [
                required for required in section["required_data"]
                if required != key
            ]

    payload = stored_deflection_report_model(legacy_artifact)
    assert payload is not None
    normalized_sections = {
        section["id"]: section
        for section in payload["sections"]
    }
    for section_id, limits in expected_limits.items():
        section = normalized_sections[section_id]
        for key, expected in limits.items():
            assert section["data"][key] == expected
            assert key in section["required_data"]
            assert key not in section_by_id[section_id]["data"]
            assert key not in section_by_id[section_id]["required_data"]


def test_stored_deflection_report_model_backfills_legacy_action_owner_metadata() -> None:
    artifact = build_deflection_report_artifact(
        _structured_report_fixture_result()
    ).as_dict()
    legacy_artifact = json.loads(json.dumps(artifact))
    priority_section = next(
        section
        for section in legacy_artifact["report_model"]["sections"]
        if section["id"] == "priority_fix_queue"
    )
    item = priority_section["data"]["items"][0]
    item.pop("owner_category", None)
    item.pop("evidence_tier", None)
    item["jira_template"].pop("owner_category", None)
    item["routing_signals"] = {
        "group": ["Support Queue"],
        "assignee": ["Agent Export"],
        "tags": ["login"],
        "brand": ["Admin Co"],
        "organization": ["Billing Team LLC"],
        "product_area": ["Authentication"],
        "custom_product_area": [],
    }

    payload = stored_deflection_report_model(legacy_artifact)
    assert payload is not None
    section = next(
        section
        for section in payload["sections"]
        if section["id"] == "priority_fix_queue"
    )
    normalized_item = section["data"]["items"][0]

    assert normalized_item["owner_category"] == "Content / Support Enablement"
    assert normalized_item["jira_template"]["owner_category"] == (
        "Content / Support Enablement"
    )
    assert normalized_item["evidence_tier"] == "csv_index_metadata_only"
    assert normalized_item["routing_signals"] == {
        "tags": ["login"],
        "product_area": ["Authentication"],
        "custom_product_area": [],
    }


def test_stored_deflection_report_model_backfills_legacy_suppressed_review_keys() -> None:
    result = TicketFAQMarkdownResult(
        markdown="# FAQ",
        source_count=4,
        ticket_source_count=4,
        output_checks={"condensed": True},
        items=(
            {
                "question": "",
                "customer_wording": "",
                "topic": "",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 30,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (),
            },
            {
                "question": "",
                "customer_wording": "",
                "topic": "",
                "weighted_frequency": 3,
                "ticket_count": 3,
                "opportunity_score": 25,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": (),
            },
            {
                "question": "Can I rename one workspace?",
                "customer_wording": "rename one workspace",
                "topic": "workspace",
                "weighted_frequency": 1,
                "ticket_count": 1,
                "opportunity_score": 20,
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ("ticket-one-off",),
            },
        ),
    )
    artifact = build_deflection_report_artifact(result).as_dict()
    legacy_artifact = json.loads(json.dumps(artifact))
    expected_keys_by_rank: dict[int, str] = {}
    legacy_section = next(
        section
        for section in legacy_artifact["report_model"]["sections"]
        if section["id"] == "suppressed_repeat_review_queue"
    )
    original_section = next(
        section
        for section in artifact["report_model"]["sections"]
        if section["id"] == "suppressed_repeat_review_queue"
    )
    for item in original_section["data"]["items"]:
        expected_keys_by_rank[item["rank"]] = item["review_key"]
    for item in legacy_section["data"]["items"]:
        item.pop("review_key", None)

    payload = stored_deflection_report_model(legacy_artifact)
    assert payload is not None
    section = next(
        section
        for section in payload["sections"]
        if section["id"] == "suppressed_repeat_review_queue"
    )
    items = section["data"]["items"]
    review_keys = [item["review_key"] for item in items]

    assert review_keys == [expected_keys_by_rank[item["rank"]] for item in items]
    assert len(set(review_keys)) == len(review_keys)
    assert "review_key" not in legacy_section["data"]["items"][0]


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
async def test_in_memory_deflection_report_store_deletes_report_and_referencing_deltas() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    for account_id, request_id, question in (
        ("acct-1", "current", "Delete me"),
        ("acct-1", "baseline", "Baseline"),
        ("acct-2", "current", "Keep me"),
    ):
        await store.save_report(
            account_id=account_id,
            request_id=request_id,
            snapshot=_report_access_snapshot(question),
            artifact={},
        )
    await store.save_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
        delta={"summary": {"matched_item_count": 1}},
    )

    assert await store.delete_report(account_id="acct-1", request_id="current") is True

    assert await store.get_artifact_record(
        account_id="acct-1",
        request_id="current",
    ) is None
    assert await store.get_artifact_record(
        account_id="acct-2",
        request_id="current",
    ) is not None
    assert await store.get_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
    ) is None
    assert await store.delete_report(account_id="acct-1", request_id="current") is False


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


@pytest.mark.asyncio
async def test_postgres_delete_report_scopes_to_account_and_request_id() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
            self.fetchval_result = 1

        async def fetchval(self, query: str, *args: object) -> int:
            self.fetchval_calls.append((query, args))
            return self.fetchval_result

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    assert await store.delete_report(
        account_id=" acct-1 ",
        request_id=" request-delete ",
    ) is True
    query, args = pool.fetchval_calls[0]
    assert args == ("acct-1", "request-delete")
    assert "WITH target AS" in query
    assert "DELETE FROM content_ops_deflection_report_deliveries" in query
    assert "DELETE FROM content_ops_deflection_reports" in query
    assert "WHERE account_id = $1 AND request_id = $2" in query

    pool.fetchval_result = 0
    assert await store.delete_report(account_id="acct-1", request_id="missing") is False


@pytest.mark.asyncio
async def test_in_memory_report_retention_deletes_only_old_rows_with_limit() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    now = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
    await _save_retention_report(store, "acct-1", "oldest")
    await _save_retention_report(store, "acct-1", "old")
    await _save_retention_report(store, "acct-1", "fresh")
    await _save_retention_report(store, "acct-2", "other-old")
    _set_report_created_at(store, "acct-1", "oldest", now - timedelta(days=70))
    _set_report_created_at(store, "acct-1", "old", now - timedelta(days=45))
    _set_report_created_at(store, "acct-1", "fresh", now - timedelta(days=2))
    _set_report_created_at(store, "acct-2", "other-old", now - timedelta(days=50))
    cutoff = now - timedelta(days=30)

    assert await store.count_reports_older_than(cutoff=cutoff) == 3
    assert await store.delete_reports_older_than(cutoff=cutoff, limit=2) == 2

    assert await store.get_artifact_record(account_id="acct-1", request_id="oldest") is None
    assert await store.get_artifact_record(account_id="acct-2", request_id="other-old") is None
    assert await store.get_artifact_record(account_id="acct-1", request_id="old") is not None
    assert await store.get_artifact_record(account_id="acct-1", request_id="fresh") is not None
    assert await store.count_reports_older_than(cutoff=cutoff) == 1
    assert await store.delete_reports_older_than(cutoff=cutoff) == 1
    assert await store.get_artifact_record(account_id="acct-1", request_id="old") is None
    assert await store.get_artifact_record(account_id="acct-1", request_id="fresh") is not None


@pytest.mark.asyncio
async def test_report_retention_store_rejects_unsafe_boundaries() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    aware_cutoff = datetime(2026, 6, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="timezone-aware"):
        await store.count_reports_older_than(cutoff=datetime(2026, 6, 1))
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await store.delete_reports_older_than(cutoff=aware_cutoff, limit=0)
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await store.delete_reports_older_than(cutoff=aware_cutoff, limit=False)


@pytest.mark.asyncio
async def test_postgres_report_retention_uses_cutoff_and_limited_delete() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
            self.delete_result = 2

        async def fetchval(self, query: str, *args: object) -> int:
            self.fetchval_calls.append((query, args))
            if "SELECT COUNT(*)\n            FROM content_ops_deflection_reports" in query:
                return 3
            return self.delete_result

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)
    cutoff = datetime(2026, 5, 1, tzinfo=timezone.utc)

    assert await store.count_reports_older_than(cutoff=cutoff) == 3
    assert pool.fetchval_calls[0][1] == (cutoff,)
    assert "created_at < $1" in pool.fetchval_calls[0][0]
    assert await store.delete_reports_older_than(cutoff=cutoff, limit=25) == 2
    limited_query, limited_args = pool.fetchval_calls[1]
    assert limited_args == (cutoff, 25)
    assert "WITH doomed AS" in limited_query
    assert "DELETE FROM content_ops_deflection_report_deliveries" in limited_query
    assert "DELETE FROM content_ops_deflection_reports" in limited_query
    assert "LIMIT $2" in limited_query
    pool.delete_result = 4
    assert await store.delete_reports_older_than(cutoff=cutoff) == 4
    unbounded_query, unbounded_args = pool.fetchval_calls[2]
    assert unbounded_args == (cutoff,)
    assert "DELETE FROM content_ops_deflection_reports" in unbounded_query
    assert "LIMIT" not in unbounded_query


@pytest.mark.asyncio
async def test_postgres_report_retention_fails_closed_on_unparseable_delete_count() -> None:
    class _Pool:
        async def fetchval(self, _query: str, *_args: object) -> str:
            return "unknown"

    store = PostgresDeflectionReportArtifactStore(pool=_Pool())
    cutoff = datetime(2026, 5, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="invalid literal"):
        await store.delete_reports_older_than(cutoff=cutoff)


@pytest.mark.asyncio
async def test_retention_runner_dry_run_counts_without_deleting_or_exposing_payload() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    now = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
    await _save_retention_report(store, "acct-1", "old", delivery_email="buyer@example.com")
    await _save_retention_report(store, "acct-1", "fresh")
    _set_report_created_at(store, "acct-1", "old", now - timedelta(days=40))
    _set_report_created_at(store, "acct-1", "fresh", now - timedelta(days=2))

    code, payload = await RETENTION_MODULE.run_deflection_report_retention_purge(
        _retention_args(retention_days=30),
        object(),
        store=store,
        now=now,
    )

    assert code == 0
    assert payload == {
        "ok": True,
        "dry_run": True,
        "retention_days": 30,
        "cutoff": "2026-05-21T12:00:00+00:00",
        "candidate_count": 1,
        "deleted_count": 0,
        "limit": None,
    }
    assert "buyer@example.com" not in json.dumps(payload)
    assert await store.get_artifact_record(account_id="acct-1", request_id="old") is not None


@pytest.mark.asyncio
async def test_retention_runner_requires_confirm_delete_and_valid_bounds() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    now = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
    await _save_retention_report(store, "acct-1", "oldest")
    await _save_retention_report(store, "acct-1", "old")
    _set_report_created_at(store, "acct-1", "oldest", now - timedelta(days=80))
    _set_report_created_at(store, "acct-1", "old", now - timedelta(days=40))

    code, payload = await RETENTION_MODULE.run_deflection_report_retention_purge(
        _retention_args(retention_days=30, limit=1, confirm_delete=True),
        object(),
        store=store,
        now=now,
    )

    assert code == 0
    assert payload["dry_run"] is False
    assert payload["candidate_count"] == 2
    assert payload["deleted_count"] == 1
    assert payload["limit"] == 1
    assert await store.get_artifact_record(account_id="acct-1", request_id="oldest") is None
    assert await store.get_artifact_record(account_id="acct-1", request_id="old") is not None
    with pytest.raises(SystemExit, match="--retention-days must be greater than 0"):
        RETENTION_MODULE._validate_args(_retention_args(retention_days=0))
    with pytest.raises(SystemExit, match="--limit must be greater than 0"):
        RETENTION_MODULE._validate_args(_retention_args(retention_days=30, limit=0))
    with pytest.raises(ValueError, match="timezone-aware"):
        await RETENTION_MODULE.run_deflection_report_retention_purge(
            _retention_args(retention_days=30),
            object(),
            store=store,
            now=datetime(2026, 6, 20, 12, 0),
        )


@pytest.mark.asyncio
async def test_retention_runner_rejects_invalid_args_before_opening_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_create_pool(_database_url: str) -> object:
        raise AssertionError("pool must not open before argument validation")

    monkeypatch.setattr(RETENTION_MODULE, "_create_pool", fail_create_pool)

    with pytest.raises(SystemExit, match="--retention-days must be greater than 0"):
        await RETENTION_MODULE._main([
            "--database-url",
            "postgres://example",
            "--retention-days",
            "0",
            "--confirm-delete",
        ])


def test_retention_runner_resolves_database_url_from_non_argv_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATLAS_TEST_RETENTION_DSN", "postgres://env-secret")
    dsn_file = tmp_path / "dsn.txt"
    dsn_file.write_text("postgres://file-secret\n")

    assert (
        RETENTION_MODULE._resolve_database_url(
            _retention_args(database_url=None, database_url_env="ATLAS_TEST_RETENTION_DSN")
        )
        == "postgres://env-secret"
    )
    assert (
        RETENTION_MODULE._resolve_database_url(
            _retention_args(database_url=None, database_url_file=dsn_file)
        )
        == "postgres://file-secret"
    )
    with pytest.raises(SystemExit, match="provide exactly one database URL source"):
        RETENTION_MODULE._validate_args(
            _retention_args(
                database_url="postgres://argv-secret",
                database_url_env="ATLAS_TEST_RETENTION_DSN",
            )
        )


@pytest.mark.asyncio
async def test_retention_runner_preflights_output_before_opening_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("already a file")

    async def fail_create_pool(_database_url: str) -> object:
        raise AssertionError("pool must not open before output preflight")

    monkeypatch.setattr(RETENTION_MODULE, "_create_pool", fail_create_pool)
    monkeypatch.setenv("ATLAS_TEST_RETENTION_DSN", "postgres://env-secret")

    with pytest.raises(SystemExit, match="could not prepare --output path"):
        await RETENTION_MODULE._main(
            [
                "--database-url-env",
                "ATLAS_TEST_RETENTION_DSN",
                "--retention-days",
                "30",
                "--confirm-delete",
                "--output",
                str(blocked_parent / "summary.json"),
            ]
        )


def test_retention_runner_output_failure_after_purge_falls_back_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_emit(_payload: Mapping[str, object], *, output: Path | None = None) -> None:
        raise OSError("late write failure")

    monkeypatch.setattr(RETENTION_MODULE, "_emit_payload", fail_emit)

    RETENTION_MODULE._emit_payload_after_purge(
        {"ok": True, "deleted_count": 1},
        output=Path("summary.json"),
    )

    fallback = json.loads(capsys.readouterr().out)
    assert fallback["ok"] is True
    assert fallback["deleted_count"] == 1
    assert fallback["output_error"] == "failed to write --output: OSError"


async def _save_retention_report(
    store: InMemoryDeflectionReportArtifactStore,
    account_id: str,
    request_id: str,
    *,
    delivery_email: str | None = None,
) -> None:
    await store.save_report(
        account_id=account_id,
        request_id=request_id,
        snapshot=_report_access_snapshot(f"Question {request_id}"),
        artifact={"markdown": f"# Report {request_id}"},
        delivery_email=delivery_email,
    )


def _set_report_created_at(
    store: InMemoryDeflectionReportArtifactStore,
    account_id: str,
    request_id: str,
    created_at: datetime,
) -> None:
    store._created_at_by_key[(account_id, request_id)] = created_at


def _retention_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "database_url": "postgres://example",
        "database_url_env": None,
        "database_url_file": None,
        "retention_days": 30,
        "limit": None,
        "confirm_delete": False,
        "output": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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
