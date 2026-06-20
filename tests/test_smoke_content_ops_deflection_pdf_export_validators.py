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
                "data": {
                    "rows": [
                        {"rank": 1, "question": "How do I export attribution reports?"},
                        {"rank": 2, "question": "How do I invite teammates?"},
                    ],
                },
            },
            {
                "id": "question_details",
                "data": {
                    "rows": [
                        {
                            "rank": 1,
                            "question": "How do I export attribution reports?",
                            "source_ids": ["zd-100"],
                        },
                        {
                            "rank": 2,
                            "question": "How do I invite teammates?",
                            "source_ids": ["fd-200"],
                        },
                    ],
                },
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
        "evidence_rows": [
            {"source_id": "zd-100"},
            {"source_id": "fd-200"},
            {},
            {},
            {},
            {},
            {},
            {},
        ],
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
    How do I export attribution reports?
    How do I invite teammates?

    Question Details and Evidence
    How do I export attribution reports?
    How do I invite teammates?
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


def test_pdf_export_validator_detects_live_urls_and_request_ids() -> None:
    leaked_text = (
        f"{_pdf_text()}\n"
        "https://juancanfield.com/systems/support-ticket-deflection/results/"
        "content-ops-1234567890abcdef\n"
        "content-ops-1234567890abcdef"
    )

    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=leaked_text,
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.leak.result_url" in failed
    assert "artifact.pdf.leak.request_id" in failed
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "content-ops-1234567890abcdef" not in encoded


def test_pdf_export_validator_detects_content_ops_report_urls() -> None:
    leaked_id = "content-ops-fedcba0987654321"
    leaked_text = (
        f"{_pdf_text()}\n"
        f"https://juancanfield.com/content-ops/deflection-reports/{leaked_id}"
    )

    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=leaked_text,
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.leak.result_url" in failed
    encoded = json.dumps(scorecard, sort_keys=True)
    assert leaked_id not in encoded


def test_pdf_export_validator_detects_dynamic_source_id_leaks_without_echoing_them() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=f"{_pdf_text()}\nSource IDs: zd-100, fd-200\nraw quoted evidence",
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.leak.source_id_list" in failed
    assert "artifact.pdf.leak.raw_evidence_quote" in failed
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "zd-100" not in encoded
    assert "fd-200" not in encoded
    assert "raw quoted evidence" not in encoded


def test_pdf_export_validator_detects_numeric_source_id_leaks_without_echoing_them() -> None:
    model = _report_model()
    question_details = model["sections"][3]["data"]["rows"]  # type: ignore[index]
    question_details[0]["source_ids"] = ["360098765432"]
    question_details[1]["source_ids"] = ["150042"]
    export = _evidence_export()
    export["questions"] = [  # type: ignore[index]
        {"source_ids": ["360098765432"]},
        {"source_ids": ["150042"]},
    ]
    export["evidence_rows"] = [  # type: ignore[index]
        {"source_id": "360098765432"},
        {"source_id": "150042"},
    ]

    scorecard = checker.build_pdf_export_scorecard(
        report_model=model,
        evidence_export=export,
        pdf_bytes=_pdf_bytes(),
        pdf_text=f"{_pdf_text()}\nSource IDs: 360098765432 and 150042",
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.leak.source_id_list" in failed
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "360098765432" not in encoded
    assert "150042" not in encoded


def test_pdf_export_validator_detects_actual_evidence_quote_leak_without_echoing_it() -> None:
    leaked_quote = "Customer says the export failed after upgrading the workspace"
    export = _evidence_export()
    export["evidence_rows"][0]["evidence_quote"] = leaked_quote  # type: ignore[index]

    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=export,
        pdf_bytes=_pdf_bytes(),
        pdf_text=f"{_pdf_text()}\n{leaked_quote}",
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.leak.raw_evidence_quote" in failed
    encoded = json.dumps(scorecard, sort_keys=True)
    assert leaked_quote not in encoded


def test_pdf_export_validator_anchors_count_checks_to_labels() -> None:
    text = """
    Support Tax Confirmation
    This report found 8 question-level repeat tickets across 2 ranked questions.
    The expected support cost is $108.
    Page 1

    Ranked Question Opportunities
    How do I export attribution reports?
    How do I invite teammates?

    Question Details and Evidence
    How do I export attribution reports?
    How do I invite teammates?
    Use the complete evidence export for the uncapped audit trail.
    """

    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=text,
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.text.count.drafted_answer_count" in failed
    assert "artifact.pdf.text.count.no_proven_answer_count" in failed


def test_pdf_export_validator_rejects_swapped_values_on_shared_count_line() -> None:
    model = _report_model()
    support_tax = model["sections"][0]["data"]  # type: ignore[index]
    support_tax["repeat_ticket_count"] = 2  # type: ignore[index]
    support_tax["generated_question_count"] = 8  # type: ignore[index]
    ranked_questions = model["sections"][2]["data"]["rows"]  # type: ignore[index]
    ranked_questions.extend(
        {"rank": rank, "question": f"Held-out question {rank}?"}
        for rank in range(3, 9)
    )

    scorecard = checker.build_pdf_export_scorecard(
        report_model=model,
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=_pdf_text(),
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "artifact.pdf.text.count.repeat_ticket_count" in failed
    assert "artifact.pdf.text.count.generated_question_count" in failed
    assert "artifact.pdf.text.count.ranked_question_count" in failed


def test_pdf_export_validator_accepts_renderer_count_and_money_display_forms() -> None:
    model = _report_model()
    support_tax = model["sections"][0]["data"]  # type: ignore[index]
    support_tax["repeat_ticket_count"] = 1234  # type: ignore[index]
    support_tax["generated_question_count"] = 1234  # type: ignore[index]
    support_tax["ticket_source_count"] = 1234  # type: ignore[index]
    support_tax["estimated_support_cost"] = 67.5  # type: ignore[index]
    text = _pdf_text().replace(
        "8 question-level repeat tickets across 2 ranked questions",
        "1,234 question-level repeat tickets across 1,234 ranked questions",
    ).replace(
        "$108",
        "$68",
    ).replace(
        "grounded in 8 source tickets",
        "grounded in 1,234 source tickets",
    )

    scorecard = checker.build_pdf_export_scorecard(
        report_model=model,
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=text,
    )

    failed = _failed_ids(scorecard)
    assert "artifact.pdf.text.count.repeat_ticket_count" not in failed
    assert "artifact.pdf.text.count.generated_question_count" not in failed
    assert "artifact.pdf.text.count.ticket_source_count" not in failed
    assert "artifact.pdf.text.count.estimated_support_cost" not in failed


def test_pdf_export_validator_keeps_pdf_row_cap_observations_enabled() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=_pdf_text().replace("How do I invite teammates?", ""),
    )

    assert scorecard["ok"] is False
    failed = _failed_ids(scorecard)
    assert "surface.pdf.displayed_rows.ranked_questions" in failed
    assert "surface.pdf.displayed_rows.question_details" in failed


def test_pdf_export_validator_does_not_treat_body_copy_as_section_end() -> None:
    text = """
    Support Ticket Deflection Report

    Support Tax Confirmation
    This report found 8 question-level repeat tickets across 2 ranked questions.
    At the benchmark, that repeated-question work sizes to about $108.
    Publishable answers drafted from proven resolutions: 1
    Questions still needing an approved resolution: 1
    Ticket sources represented: 8

    Ranked Question Opportunities
    1 | How do I export attribution reports? | 0 | $0 | 0
    2 | How do I invite teammates? | 0 | $0 | 0

    Question Details and Evidence
    How do I export attribution reports?
    Ticket backing: 1 source tickets; complete source details are in the
    complete evidence export.
    How do I invite teammates?
    Complete Evidence export details stay available in the downloadable JSON,
    but this sentence is body copy, not a section heading.
    Ticket backing: 1 source tickets; complete source details are in the
    complete evidence export.
    Complete Evidence
    """

    scorecard = checker.build_pdf_export_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        pdf_bytes=_pdf_bytes(),
        pdf_text=text,
    )

    assert scorecard["ok"] is True
    failed = _failed_ids(scorecard)
    assert "surface.pdf.displayed_rows.question_details" not in failed


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


def test_pdf_export_validator_cli_fails_malformed_export_arrays_without_crashing(
    tmp_path: Path,
) -> None:
    report_model = tmp_path / "report_model.json"
    export = tmp_path / "evidence_export.json"
    pdf_bytes = tmp_path / "report.pdf"
    pdf_text = tmp_path / "report_pdf_text.txt"
    output = tmp_path / "scorecard.json"
    bad_export = _evidence_export()
    bad_export["questions"] = {"not": "a list"}
    bad_export["evidence_rows"] = 7

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
    assert payload["artifacts"]["evidence_export"] == {
        "questions": 0,
        "evidence_rows": 0,
    }
    failed = _failed_ids(payload)
    assert "evidence_export.questions.present" in failed
    assert "evidence_export.evidence_rows.present" in failed


def test_pdf_export_validator_handles_non_mapping_payloads_without_crashing() -> None:
    scorecard = checker.build_pdf_export_scorecard(
        report_model=["not", "a", "mapping"],
        evidence_export=["not", "a", "mapping"],
        pdf_bytes=_pdf_bytes(),
        pdf_text=_pdf_text(),
    )

    assert scorecard["ok"] is False
    assert isinstance(scorecard["assertions"], list)
