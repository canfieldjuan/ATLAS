from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_deflection_pii_recall.py"
TINY_FIXTURE = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json"
)
SPEC = importlib.util.spec_from_file_location("score_deflection_pii_recall", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)


def _tiny_corpus() -> dict:
    return json.loads(TINY_FIXTURE.read_text(encoding="utf-8"))


def test_tiny_fixture_scores_all_surfaces_without_echoing_spans() -> None:
    summary = CLI.score_corpus(_tiny_corpus())

    assert summary["status"] == "ok"
    assert summary["input"] == {
        "schema_version": "deflection_pii_eval_corpus.v1",
        "ticket_count": 2,
        "label_count": 9,
        "must_survive_count": 3,
    }
    paid_pdf = summary["surface_generation"]["paid_pdf"]
    if paid_pdf["rendered"]:
        assert paid_pdf["skipped"] is False
        assert paid_pdf["byte_count"] > 1000
    else:
        assert paid_pdf == {
            "rendered": False,
            "skipped": True,
            "skip_reason": "missing_optional_dependency:fpdf",
            "byte_count": 0,
            "scored_text_source": None,
        }
    assert set(summary["surfaces"]) == {
        "free_snapshot",
        "free_teaser",
        "paid_artifact",
        "paid_pdf",
    }
    assert summary["surfaces"]["free_snapshot"]["email"]["expected"] == 1
    assert summary["surfaces"]["paid_artifact"]["ssn"] == {
        "expected": 1,
        "redacted": 1,
        "leaks": 0,
        "recall": 1.0,
    }
    assert summary["surfaces"]["paid_artifact"]["payment_card"] == {
        "expected": 1,
        "redacted": 1,
        "leaks": 0,
        "recall": 1.0,
    }
    assert set(summary["person_name"]) == {"cue_less", "cue_prefixed"}
    assert summary["headline"] == {
        "free_high_severity_leak_count": 1,
        "free_high_severity_pass": False,
    }
    if paid_pdf["rendered"]:
        assert summary["person_name"]["cue_prefixed"] == {
            "expected": 5,
            "redacted": 3,
            "leaks": 2,
            "recall": 0.6,
        }
    else:
        assert summary["person_name"]["cue_prefixed"] == {
            "expected": 3,
            "redacted": 2,
            "leaks": 1,
            "recall": 0.6667,
        }
    assert summary["person_name"]["cue_less"]["leaks"] > 0
    assert summary["must_survive"]["violation_count"] == 0
    assert summary["leak_samples"]
    assert all(
        sample["surrogate_id"] != "person_name-002"
        for sample in summary["leak_samples"]
    )
    assert all(
        sample["surrogate_id"] not in {"ssn-001", "payment_card-001"}
        for sample in summary["leak_samples"]
    )
    partial_name_leaks = [
        sample
        for sample in summary["leak_samples"]
        if sample["surrogate_id"] == "person_name-003"
    ]
    expected_partial_surfaces = {"paid_artifact"}
    if summary["surface_generation"]["paid_pdf"]["rendered"]:
        expected_partial_surfaces.add("paid_pdf")
    assert {
        sample["surface"]
        for sample in partial_name_leaks
        if sample["leak_kind"] == "partial_name_token"
    } == expected_partial_surfaces
    assert all("span" not in sample for sample in summary["leak_samples"])
    assert all("token" not in sample for sample in summary["leak_samples"])
    rendered_samples = json.dumps(summary["leak_samples"], sort_keys=True)
    assert "Mary" not in rendered_samples
    assert "Jane" not in rendered_samples
    assert "Watson" not in rendered_samples


def test_partial_name_token_detection_is_scoped_to_own_ticket() -> None:
    corpus = _tiny_corpus()
    corpus["tickets"].append(
        {
            "fields": {
                "agent_reply": "Customer Alice Watson Premium was upgraded.",
                "customer_message": "Premium update completed.",
                "source_id": "ticket-eval-safe-003",
                "subject": "Premium update",
            },
            "labels": [
                {
                    "class": "person_name",
                    "name_subtype": "cue_prefixed",
                    "origin_field": "agent_reply",
                    "severity": "high",
                    "span": "Alice Watson",
                    "surrogate_id": "person_name-004",
                }
            ],
            "must_survive": [],
            "ticket_id": "pii-eval-003",
        }
    )

    summary = CLI.score_corpus(corpus)

    assert summary["status"] == "ok"
    assert [
        sample
        for sample in summary["leak_samples"]
        if sample["surrogate_id"] == "person_name-004"
    ] == []
    assert any(
        sample["surrogate_id"] == "person_name-003"
        and sample["leak_kind"] == "partial_name_token"
        for sample in summary["leak_samples"]
    )


def test_forced_leak_reports_surface_and_surrogate_without_span() -> None:
    corpus = _tiny_corpus()
    ticket = corpus["tickets"][0]
    ticket["fields"]["subject"] = "UNSCRUBBED_SENTINEL refund status"
    ticket["labels"].append(
        {
            "class": "custom_token",
            "origin_field": "subject",
            "severity": "low",
            "span": "UNSCRUBBED_SENTINEL",
            "surrogate_id": "custom-token-001",
        }
    )

    summary = CLI.score_corpus(corpus)

    matching = [
        sample
        for sample in summary["leak_samples"]
        if sample["surrogate_id"] == "custom-token-001"
    ]
    expected_surfaces = {
        "free_snapshot",
        "paid_artifact",
    }
    if summary["surface_generation"]["paid_pdf"]["rendered"]:
        expected_surfaces.add("paid_pdf")
    assert {sample["surface"] for sample in matching} >= expected_surfaces
    assert all("span" not in sample for sample in matching)
    assert "UNSCRUBBED_SENTINEL" not in json.dumps(
        summary["leak_samples"],
        sort_keys=True,
    )


def test_must_survive_violation_reports_precision_loss() -> None:
    corpus = _tiny_corpus()
    ticket = corpus["tickets"][0]
    ticket["must_survive"].append(
        {
            "origin_field": "customer_message",
            "reason": "forced_email_precision_probe",
            "span": "alex.rivera@example.test",
        }
    )

    summary = CLI.score_corpus(corpus)

    assert summary["must_survive"]["violation_count"] > 0
    assert {
        violation["reason"]
        for violation in summary["must_survive"]["violations"]
    } >= {"forced_email_precision_probe"}


def test_invalid_corpus_fails_with_safe_error_codes() -> None:
    summary = CLI.score_corpus({"schema_version": "wrong", "tickets": []})

    assert summary["status"] == "failed"
    assert set(summary["blocking_error_codes"]) == {
        "corpus_empty_tickets",
        "corpus_schema_version_mismatch",
    }
    assert "alex.rivera@example.test" not in json.dumps(summary, sort_keys=True)


@pytest.mark.parametrize(
    ("bad_label", "error_code"),
    (
        (
            {
                "class": "email",
                "origin_field": "customer_message",
                "severity": "high",
                "surrogate_id": "email-bad",
            },
            "label_missing_span",
        ),
        (
            {
                "class": "email",
                "origin_field": "customer_message",
                "severity": "high",
                "span": "",
                "surrogate_id": "email-bad",
            },
            "label_missing_span",
        ),
        (
            {
                "class": "email",
                "origin_field": "customer_message",
                "severity": "high",
                "span": 12345,
                "surrogate_id": "email-bad",
            },
            "label_missing_span",
        ),
        ("not-a-label-object", "label_not_object"),
    ),
)
def test_malformed_label_span_fails_closed_without_scoring(
    bad_label: object,
    error_code: str,
) -> None:
    corpus = _tiny_corpus()
    corpus["tickets"][0]["labels"].append(bad_label)

    summary = CLI.score_corpus(corpus)

    assert summary["status"] == "failed"
    assert summary["blocking_error_codes"] == [error_code]
    assert summary["errors"] == [
        {
            "code": error_code,
            "ticket_index": 1,
            "label_index": 9,
        }
    ]
    assert "email-bad" not in json.dumps(summary, sort_keys=True)


@pytest.mark.parametrize(
    ("bad_record", "error_code"),
    (
        (
            {
                "origin_field": "customer_message",
                "reason": "bad_precision_record",
            },
            "must_survive_missing_span",
        ),
        (
            {
                "origin_field": "customer_message",
                "reason": "bad_precision_record",
                "span": "   ",
            },
            "must_survive_missing_span",
        ),
        (
            {
                "origin_field": "customer_message",
                "reason": "bad_precision_record",
                "span": 12345,
            },
            "must_survive_missing_span",
        ),
        ("not-a-must-survive-object", "must_survive_not_object"),
    ),
)
def test_malformed_must_survive_span_fails_closed_without_scoring(
    bad_record: object,
    error_code: str,
) -> None:
    corpus = _tiny_corpus()
    corpus["tickets"][0]["must_survive"].append(bad_record)

    summary = CLI.score_corpus(corpus)

    assert summary["status"] == "failed"
    assert summary["blocking_error_codes"] == [error_code]
    assert summary["errors"] == [
        {
            "code": error_code,
            "ticket_index": 1,
            "record_index": 4,
        }
    ]
    assert "bad_precision_record" not in json.dumps(summary, sort_keys=True)


def test_missing_pdf_dependency_skips_paid_pdf_scoring(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'fpdf'")
    error.name = "fpdf"
    monkeypatch.setattr(CLI, "_PDF_RENDERER_IMPORT_ERROR", error)
    monkeypatch.setattr(CLI, "_artifact_report_model_pdf_markdown", None)
    monkeypatch.setattr(CLI, "render_deflection_full_report_pdf", None)

    summary = CLI.score_corpus(_tiny_corpus())

    assert summary["status"] == "ok"
    assert summary["surface_generation"]["paid_pdf"] == {
        "rendered": False,
        "skipped": True,
        "skip_reason": "missing_optional_dependency:fpdf",
        "byte_count": 0,
        "scored_text_source": None,
    }
    assert summary["surfaces"]["paid_pdf"] == {}


def test_cli_writes_failure_summary_and_returns_nonzero(tmp_path: Path) -> None:
    corpus_path = tmp_path / "bad.json"
    output_path = tmp_path / "summary.json"
    corpus_path.write_text('{"schema_version": "wrong", "tickets": []}', encoding="utf-8")

    exit_code = CLI.main([
        "--corpus",
        str(corpus_path),
        "--output",
        str(output_path),
    ])

    assert exit_code == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "failed"
    assert written["blocking_error_codes"]
