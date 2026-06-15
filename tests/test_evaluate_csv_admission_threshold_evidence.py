from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_csv_admission_threshold_evidence",
    ROOT / "scripts" / "evaluate_csv_admission_threshold_evidence.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_zendesk_product_proof_corpus_records_full_csv_admission() -> None:
    corpus = json.loads(
        (
            ROOT
            / "docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json"
        ).read_text(encoding="utf-8")
    )

    summary = MOD.evaluate_zendesk_corpus(corpus)

    assert summary["status"] == "ok"
    assert summary["blocking_violation_codes"] == []
    assert summary["corpus"]["ticket_count"] == 50
    assert summary["case_count"] == 2
    assert summary["observed_min_usable_source_ratio"] == 1.0
    assert {
        case["name"]: case["usable_source_row_count"] for case in summary["cases"]
    } == {
        "zendesk_public_comments_csv": 50,
        "zendesk_description_csv": 50,
    }
    for case in summary["cases"]:
        assert case["admission_status"] == "ACCEPT"
        assert case["raw_source_row_count"] == 50
        assert case["usable_source_ratio"] == 1.0
        assert case["coverage_warnings"] == []


def test_public_comments_projection_ignores_private_note_column() -> None:
    corpus = {
        "source": "unit",
        "run_tag": "unit",
        "tickets": [{
            "id": "zd-proof-001",
            "subject": "Duplicate charge",
            "description": "Why was I charged twice?",
            "comments": [
                {
                    "public": True,
                    "body": "Why was I charged twice?",
                },
                {
                    "public": False,
                    "body": "Known proration bug. Do not publish this detail.",
                },
            ],
        }],
    }

    summary = MOD.evaluate_zendesk_corpus(corpus)
    public_case = {
        case["name"]: case for case in summary["cases"]
    }["zendesk_public_comments_csv"]

    assert public_case["admission_status"] == "ACCEPT"
    assert public_case["mapped_fields"] == {
        "source_id": ["Ticket ID"],
        "source_title": ["Subject"],
        "thread_text": ["Public Comments"],
    }
    assert public_case["ignored_private_fields"] == ["Internal Notes"]
    assert public_case["populated_unmapped_fields"] == []


def test_partial_coverage_case_records_warning_without_rejecting() -> None:
    case = MOD.CsvAdmissionCase(
        name="partial_product_shape_probe",
        description="Synthetic partial-coverage probe for the evidence runner.",
        fieldnames=("Ticket ID", "Subject", "Public Comments"),
        rows=(
            {
                "Ticket ID": "T-1",
                "Subject": "Export failed",
                "Public Comments": "Customer cannot export reports.",
            },
            {
                "Ticket ID": "T-2",
                "Subject": "Invite failed",
                "Public Comments": "",
            },
        ),
        expected_full_coverage=False,
    )

    result = MOD.evaluate_csv_case(case)

    assert result["case_status"] == "observed"
    assert result["admission_status"] == "ACCEPT"
    assert result["raw_source_row_count"] == 2
    assert result["usable_source_row_count"] == 1
    assert result["usable_source_ratio"] == 0.5
    assert result["coverage_warnings"] == [{
        "code": "partial_source_row_coverage",
        "location": "source_row_csv",
        "raw_source_row_count": 2,
        "usable_source_row_count": 1,
        "skipped_source_row_count": 1,
        "usable_source_ratio": 0.5,
    }]


def test_expected_full_coverage_partial_csv_fails_closed(tmp_path: Path) -> None:
    corpus = {
        "source": "unit",
        "run_tag": "unit",
        "tickets": [
            {
                "id": "zd-proof-001",
                "subject": "Export failed",
                "description": "Customer cannot export reports.",
                "comments": [{
                    "public": True,
                    "body": "Customer cannot export reports.",
                }],
            },
            {
                "id": "zd-proof-002",
                "subject": "Invite failed",
                "description": "",
                "comments": [],
            },
        ],
    }
    corpus_path = tmp_path / "bad-corpus.json"
    out_dir = tmp_path / "artifact"
    doc = tmp_path / "proof.md"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

    summary = MOD.evaluate_zendesk_corpus(corpus)
    code = MOD.main([
        "--corpus",
        str(corpus_path),
        "--out-dir",
        str(out_dir),
        "--doc",
        str(doc),
    ])

    assert summary["status"] == "failed"
    assert summary["blocking_violation_codes"] == [
        "zendesk_public_comments_csv:unexpected_admission_result",
        "zendesk_description_csv:unexpected_admission_result",
    ]
    assert [case["case_status"] for case in summary["cases"]] == ["failed", "failed"]
    assert code == 1
    written = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert written["status"] == "failed"


def test_main_writes_summary_and_doc(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifact"
    doc = tmp_path / "proof.md"

    code = MOD.main(["--out-dir", str(out_dir), "--doc", str(doc)])

    assert code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["observed_min_usable_source_ratio"] == 1.0
    proof = doc.read_text(encoding="utf-8")
    assert "does not choose a low non-zero reject threshold" in proof
