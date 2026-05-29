from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from extracted_content_pipeline.faq_deflection_report import (
    build_deflection_report_artifact,
)
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownResult


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
                "answer_evidence_status": "resolution_evidence",
                "steps": [
                    "Use the uploaded resolution evidence: Open Analytics, choose Attribution, then select Download report.",
                    "Confirm the answer matches the customer's support record before publishing it.",
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
    assert "## Ranked Question Opportunities" in artifact.markdown
    assert "## Drafted Answers With Proven Solutions" in artifact.markdown
    assert "How do I export attribution reports?" in artifact.markdown
    assert "Use the uploaded resolution evidence: Open Analytics" in artifact.markdown
    assert "## No Proven Answer Yet" in artifact.markdown
    assert "Why is the dashboard stale?" in artifact.markdown
    assert "Customers repeatedly asked this question" in artifact.markdown
    assert "No verified support resolution was present" in artifact.markdown
    assert "export" in artifact.markdown
    assert "Download report" in artifact.markdown


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
    drafted_section = markdown.split("## No Proven Answer Yet", 1)[0]

    assert "No FAQ gap in this run included uploaded resolution evidence." in drafted_section
    assert "Review the cited ticket evidence" not in drafted_section
    assert "Can I update permissions?" in markdown.split("## No Proven Answer Yet", 1)[1]


def test_deflection_report_cli_builds_saas_demo_artifact(tmp_path: Path) -> None:
    output = tmp_path / "deflection-report.md"
    summary_output = tmp_path / "deflection-report-summary.json"

    exit_code = MODULE.main([
        str(SAAS_DEMO),
        "--source-format",
        "csv",
        "--output",
        str(output),
        "--summary-output",
        str(summary_output),
    ])

    assert exit_code == 0
    markdown = output.read_text()
    summary = json.loads(summary_output.read_text())
    assert summary["generated"] >= 7
    assert summary["source_count"] == 36
    assert summary["no_proven_answer_count"] >= 1
    assert "## Ranked Question Opportunities" in markdown
    assert "## No Proven Answer Yet" in markdown
    assert "support_ticket_saas_demo_sources.csv" in markdown
    assert "tell users exactly what to try next" not in markdown


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
    drafted = markdown.split("## Drafted Answers With Proven Solutions", 1)[1].split(
        "## No Proven Answer Yet",
        1,
    )[0]
    no_proven = markdown.split("## No Proven Answer Yet", 1)[1].split(
        "## Vocabulary Gaps",
        1,
    )[0]

    assert summary["generated"] == 2
    assert summary["source_count"] == 7
    assert summary["drafted_answer_count"] == 1
    assert summary["no_proven_answer_count"] == 1
    assert "Open Analytics then Attribution then click Download report" in drafted
    assert "How do I enable SSO for my team?" in no_proven
    assert "Open Analytics then Attribution then click Download report" not in no_proven
    assert "tell users exactly what to try next" not in drafted
