from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_deflection_full_report_pdf_export_artifacts.py"
SPEC = importlib.util.spec_from_file_location(
    "check_deflection_full_report_pdf_export_artifacts",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _report_model() -> dict[str, object]:
    return {
        "schema_version": "deflection.v1",
        "title": "Support Ticket Deflection Report",
        "summary": {"generated": 2},
        "sections": [
            {
                "id": "support_tax",
                "data": {
                    "repeat_ticket_count": 8,
                    "non_repeat_ticket_count": 0,
                    "generated_question_count": 2,
                    "assisted_contact_cost": 13.5,
                    "estimated_support_cost": 108.0,
                    "source_date_window": {},
                    "drafted_answer_count": 1,
                    "no_proven_answer_count": 1,
                    "ticket_source_count": 8,
                },
            },
            {
                "id": "seo_targets",
                "data": {
                    "phrases": ["export attribution reports", "report download"],
                    "total_phrase_count": 2,
                    "displayed_phrase_count": 2,
                    "omitted_phrase_count": 0,
                    "limit": 50,
                },
            },
            {
                "id": "ranked_questions",
                "data": {"rows": [{"rank": 1}, {"rank": 2}]},
            },
            {
                "id": "question_details",
                "data": {"rows": [{"rank": 1}, {"rank": 2}]},
            },
            {
                "id": "complete_evidence",
                "data": {
                    "question_count": 2,
                    "evidence_row_count": 8,
                    "source_id_count": 8,
                    "surfaces": ["export"],
                },
            },
        ],
    }


def _evidence_export() -> dict[str, object]:
    return {
        "schema_version": "deflection_evidence.v1",
        "summary": {
            "question_count": 2,
            "evidence_row_count": 8,
            "source_id_count": 8,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 1,
        },
        "questions": [{}, {}],
        "evidence_rows": [{} for _ in range(8)],
    }


def _pdf_bytes() -> bytes:
    return b"%PDF-1.7\n% synthetic test bytes for artifact validator\n%%EOF\n"


def _pdf_text() -> str:
    return """
    Support Ticket Deflection Report

    Support Tax Confirmation
    This report found 8 question-level repeat tickets across 2 ranked questions.
    At the benchmark, that repeated-question work sizes to about $108.
    1 questions have publishable answers and 1 questions still have no proven answer.
    The report is grounded in 8 source tickets.

    Ranked Question Opportunities
    2 ranked questions appear in the curated PDF.

    Question Details and Evidence
    The PDF is curated for sharing. Use the complete evidence export for the
    uncapped audit trail.
    """


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _failed_ids(scorecard: dict[str, object]) -> set[str]:
    return {
        assertion["id"]
        for assertion in scorecard["assertions"]  # type: ignore[index]
        if assertion["ok"] is False
    }


def test_pdf_export_validator_passes_sanitized_artifacts() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=_pdf_text(),
    )

    assert scorecard["ok"] is True
    assert scorecard["surfaces"]["required"] == ["pdf", "evidence_export"]
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-export-1" not in encoded
    assert "raw quoted evidence" not in encoded
    assert "content-ops-" not in encoded


def test_pdf_export_validator_fails_bad_pdf_bytes() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=b"not a pdf",
        pdf_text=_pdf_text(),
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.bytes.header" in failed
    assert "harness.surface.pdf.count.repeat_ticket_count.present" in failed


def test_pdf_export_validator_fails_missing_visible_model_count() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=_pdf_text().replace("$108", "the expected support cost"),
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.text.count.estimated_support_cost" in failed
    assert "harness.surface.pdf.count.estimated_support_cost.present" in failed


def test_pdf_export_validator_fails_leak_strings_without_echoing_them() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=f"{_pdf_text()}\nSource IDs: ticket-export-1\nraw quoted evidence",
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.leak.source_id_list" in failed
    assert "artifact.pdf.leak.raw_evidence_quote" in failed
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-export-1" not in encoded
    assert "raw quoted evidence" not in encoded


def test_pdf_export_validator_cli_fails_mismatched_export_totals(tmp_path: Path) -> None:
    report_model = tmp_path / "report_model.json"
    export = tmp_path / "evidence_export.json"
    pdf_bytes = tmp_path / "report.pdf"
    pdf_text = tmp_path / "report_pdf_text.txt"
    output = tmp_path / "scorecard.json"
    bad_export = _evidence_export()
    bad_export["summary"]["evidence_row_count"] = 7  # type: ignore[index]

    _write_json(report_model, _report_model())
    _write_json(export, bad_export)
    pdf_bytes.write_bytes(_pdf_bytes())
    pdf_text.write_text(_pdf_text(), encoding="utf-8")

    code = checker.main([
        "--report-model",
        str(report_model),
        "--evidence-export",
        str(export),
        "--pdf-bytes",
        str(pdf_bytes),
        "--pdf-text",
        str(pdf_text),
        "--output",
        str(output),
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    failed = _failed_ids(payload)
    assert "evidence_export.summary.evidence_row_count" in failed
    assert "surface.evidence_export.count.evidence_row_count" in failed
