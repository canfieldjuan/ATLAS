from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_deflection_full_report_hosted_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "check_deflection_full_report_hosted_smoke",
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


def _observations() -> dict[str, object]:
    return {
        "result_page": {
            "counts": {
                "repeat_ticket_count": 8,
                "generated_question_count": 2,
                "ranked_question_count": 2,
                "drafted_answer_count": 1,
                "no_proven_answer_count": 1,
                "ticket_source_count": 8,
                "estimated_support_cost": 108.0,
                "evidence_row_count": 8,
                "source_id_count": 8,
            },
            "displayed_rows": {
                "ranked_questions": 2,
                "question_details": 2,
                "seo_targets": 2,
            },
        },
        "evidence_export": {
            "counts": {
                "evidence_question_count": 2,
                "evidence_row_count": 8,
                "source_id_count": 8,
                "drafted_answer_count": 1,
                "no_proven_answer_count": 1,
            },
        },
    }


def _surface_caps() -> dict[str, object]:
    return {
        "result_page": {
            "ranked_questions": 8,
            "question_details": 10,
            "seo_targets": 10,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_hosted_smoke_scorecard_passes_sanitized_surface_observations() -> None:
    scorecard = checker.build_hosted_smoke_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        surface_observations=_observations(),
        surface_caps=_surface_caps(),
    )

    assert scorecard["ok"] is True
    assert scorecard["surfaces"]["required"] == ["result_page", "evidence_export"]
    encoded = json.dumps(scorecard, sort_keys=True)
    assert "ticket-" not in encoded
    assert "evidence_quote" not in encoded


def test_hosted_smoke_scorecard_fails_present_but_incomplete_surface() -> None:
    observations = _observations()
    del observations["result_page"]["counts"]["source_id_count"]  # type: ignore[index]

    scorecard = checker.build_hosted_smoke_scorecard(
        report_model=_report_model(),
        evidence_export=_evidence_export(),
        surface_observations=observations,
        surface_caps=_surface_caps(),
    )

    assert scorecard["ok"] is False
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "harness.surface.result_page.count.source_id_count.present" in failed


def test_hosted_smoke_cli_exits_nonzero_on_count_mismatch(tmp_path: Path) -> None:
    report_model = tmp_path / "report_model.json"
    export = tmp_path / "evidence_export.json"
    observations = _observations()
    observations["result_page"]["counts"]["repeat_ticket_count"] = 7  # type: ignore[index]
    observation_path = tmp_path / "observations.json"
    caps = tmp_path / "caps.json"
    output = tmp_path / "scorecard.json"
    _write_json(report_model, _report_model())
    _write_json(export, _evidence_export())
    _write_json(observation_path, observations)
    _write_json(caps, _surface_caps())

    code = checker.main([
        "--report-model",
        str(report_model),
        "--evidence-export",
        str(export),
        "--surface-observations",
        str(observation_path),
        "--surface-caps",
        str(caps),
        "--output",
        str(output),
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    failed = {
        assertion["id"]
        for assertion in payload["assertions"]
        if not assertion["ok"]
    }
    assert "surface.result_page.count.repeat_ticket_count" in failed
