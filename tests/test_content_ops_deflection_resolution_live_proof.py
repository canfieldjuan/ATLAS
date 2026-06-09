from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_content_ops_deflection_report.py"
FIXTURE_DIR = (
    ROOT
    / "docs"
    / "extraction"
    / "validation"
    / "fixtures"
    / "deflection_resolution_evidence_live_proof_20260609"
)
SOURCE = FIXTURE_DIR / "source.csv"
REPORT = FIXTURE_DIR / "report.md"
SUMMARY = FIXTURE_DIR / "summary.json"
RESULT = FIXTURE_DIR / "result.json"
EXPECTED_HEADER = [
    "ticket_id",
    "created_at",
    "subject",
    "message",
    "pain_category",
    "resolution_text",
    "status",
    "tags",
]

SPEC = importlib.util.spec_from_file_location(
    "build_content_ops_deflection_report",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_HEADER
        return list(reader)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_resolution_live_proof_fixture_has_resolved_and_unresolved_lanes() -> None:
    rows = _rows()

    assert len(rows) == 12
    assert {row["status"] for row in rows} == {"closed"}
    assert len({row["ticket_id"] for row in rows}) == len(rows)
    assert all(row["message"].strip().endswith("?") for row in rows)
    assert sum(1 for row in rows if row["resolution_text"].strip()) == 7
    assert sum(1 for row in rows if not row["resolution_text"].strip()) == 5

    package = build_support_ticket_input_package(rows)
    assert package.metadata["source_row_count"] == 12
    assert package.metadata["included_row_count"] == 12
    assert package.metadata["support_ticket_resolution_evidence_present"] is True
    assert package.metadata["support_ticket_resolution_evidence_count"] == 7
    assert package.inputs["support_ticket_resolution_evidence_present"] is True
    assert package.inputs["support_ticket_resolution_evidence_count"] == 7
    assert package.warnings == ()


def test_resolution_live_proof_artifacts_show_publishable_and_gap_lanes() -> None:
    summary = _json(SUMMARY)
    result = _json(RESULT)
    report = REPORT.read_text(encoding="utf-8")

    assert summary["generated"] == 4
    assert summary["source_count"] == 12
    assert summary["drafted_answer_count"] == 2
    assert summary["no_proven_answer_count"] == 2
    assert summary["support_ticket_resolution_evidence_present"] is True
    assert summary["support_ticket_resolution_evidence_count"] == 2
    assert result["status"] == "ok"
    assert result["failed_output_checks"] == []
    assert result["summary"] == summary

    statuses = [
        item["answer_evidence_status"]
        for item in result["diagnostics"]["items"]
    ]
    assert statuses == [
        "resolution_evidence",
        "resolution_evidence",
        "draft_needs_review",
        "draft_needs_review",
    ]
    assert {
        item["first_source_id"]
        for item in result["diagnostics"]["items"]
        if item["answer_evidence_status"] == "resolution_evidence"
    } == {"zd-4101", "zd-4201"}
    assert {
        item["first_source_id"]
        for item in result["diagnostics"]["items"]
        if item["answer_evidence_status"] == "draft_needs_review"
    } == {"zd-4301", "zd-4401"}

    drafted = report.split(
        "## Publishable Help-Center Copy From Proven Resolutions",
        1,
    )[1].split("## No Proven Answer Yet", 1)[0]
    no_proven = report.split("## No Proven Answer Yet", 1)[1].split(
        "## Vocabulary Gaps",
        1,
    )[0]
    assert "Open Analytics then Attribution then click Download report" in drafted
    assert "Open Billing then Invoices" in drafted
    assert "Where do I upload the new SSO certificate before it expires?" in no_proven
    assert "Why did the CRM integration pause after the field mapping changed?" in no_proven
    assert "Open Analytics then Attribution then click Download report" not in no_proven
    assert "Open Billing then Invoices" not in no_proven


def test_resolution_live_proof_regenerates_from_committed_csv(tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    summary_output = tmp_path / "summary.json"
    result_output = tmp_path / "result.json"

    exit_code = MODULE.main([
        str(SOURCE),
        "--source-format",
        "csv",
        "--output",
        str(output),
        "--summary-output",
        str(summary_output),
        "--result-output",
        str(result_output),
        "--require-output-checks",
        "--json",
    ])

    assert exit_code == 0
    assert _json(summary_output) == _json(SUMMARY)
    regenerated = _json(result_output)
    assert regenerated["status"] == "ok"
    assert regenerated["failed_output_checks"] == []
    assert regenerated["summary"] == _json(SUMMARY)
    assert "Publishable Help-Center Copy From Proven Resolutions" in output.read_text(
        encoding="utf-8"
    )
