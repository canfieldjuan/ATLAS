from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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

    assert str(exc_info.value) == "Deflection report output checks failed: condensed"
    assert not output.exists()
    assert not summary_output.exists()
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["status"] == "failed_output_checks"
    assert result["failed_output_checks"] == ["condensed"]
    assert result["summary"]["source_count"] == 2
    assert result["summary"]["ticket_source_count"] == 2
    assert result["output"]["markdown_path"] == str(output)
    assert result["output"]["summary_path"] == str(summary_output)
    assert result["output"]["result_path"] == str(result_output)
    assert result["diagnostics"]["item_count"] == 2
    assert result["diagnostics"]["items"][0]["source_id_count"] == 1
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
                "message": "The invite link expired before a contractor could join.",
            },
            {
                "ticket_id": "ticket-access-2",
                "source_type": "support_ticket",
                "subject": "Login email",
                "message": "How do I change the login email for a new teammate?",
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
                "message": "Can connector lag delay SSO provisioning?",
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
