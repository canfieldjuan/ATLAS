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
    assert summary["breakage_matrix"]["case_count"] == 6
    assert summary["breakage_matrix"]["blocking_case_count"] == 0
    assert summary["breakage_matrix"]["known_gap_count"] == 1


def test_breakage_matrix_scores_fail_closed_warning_and_known_gap() -> None:
    cases = {case["name"]: case for case in MOD.evaluate_breakage_matrix()}

    assert cases["unknown_body_like_column_rejects_zero_usable"][
        "observed_outcome"
    ] == "REJECT"
    assert cases["unknown_body_like_column_rejects_zero_usable"][
        "admission_status"
    ] == "REJECT"
    assert cases["unknown_body_like_column_rejects_zero_usable"][
        "admission_decision_reason"
    ] == "no_usable_source_rows"
    assert cases["unknown_body_like_column_rejects_zero_usable"][
        "admission_decision_location"
    ] == "source_row_csv"
    assert cases["unknown_body_like_column_rejects_zero_usable"][
        "decision_matched"
    ] is True
    assert cases["unknown_body_like_column_rejects_zero_usable"][
        "populated_unmapped_fields"
    ] == ["Conversation Text"]

    assert cases["private_note_only_rejects_zero_usable"]["observed_outcome"] == (
        "REJECT"
    )
    assert cases["private_note_only_rejects_zero_usable"][
        "ignored_private_fields"
    ] == ["Internal Notes"]
    assert cases["private_note_only_rejects_zero_usable"][
        "admission_decision_reason"
    ] == "no_usable_source_rows"
    assert cases["private_note_only_rejects_zero_usable"][
        "admission_decision_location"
    ] == "source_row_csv"

    assert cases["status_timestamp_only_rejects_zero_usable"][
        "observed_outcome"
    ] == "REJECT"
    assert cases["status_timestamp_only_rejects_zero_usable"][
        "populated_unmapped_fields"
    ] == ["Status", "First Response", "Last Response"]
    assert cases["status_timestamp_only_rejects_zero_usable"][
        "admission_decision_reason"
    ] == "no_usable_source_rows"
    assert cases["status_timestamp_only_rejects_zero_usable"][
        "admission_decision_location"
    ] == "source_row_csv"

    partial = cases["partial_blank_rows_warns_without_rejecting"]
    assert partial["observed_outcome"] == "ACCEPT_WITH_WARNING"
    assert partial["coverage_warnings"] == [{
        "code": "partial_source_row_coverage",
        "location": "source_row_csv",
        "raw_source_row_count": 2,
        "usable_source_row_count": 1,
        "skipped_source_row_count": 1,
        "usable_source_ratio": 0.5,
    }]

    assert cases["header_only_csv_has_no_policy_decision"]["observed_outcome"] == (
        "NO_POLICY_DECISION"
    )
    known_gap = cases["json_blob_message_known_fail_open"]
    assert known_gap["case_status"] == "known_gap"
    assert known_gap["known_gap"] is True
    assert known_gap["observed_outcome"] == "ACCEPT_CLEAN"
    assert known_gap["admission_status"] == "ACCEPT"


def test_breakage_matrix_expected_guard_mismatch_blocks() -> None:
    case = MOD.CsvBreakageCase(
        name="broken_expectation",
        description="A mismatch should fail the evidence runner.",
        fieldnames=("Ticket ID", "Message"),
        rows=({"Ticket ID": "T-1", "Message": "Customer cannot export reports."},),
        expected_outcome="REJECT",
    )

    result = MOD.evaluate_breakage_case(case)

    assert result["observed_outcome"] == "ACCEPT_CLEAN"
    assert result["case_status"] == "failed"
    assert MOD._breakage_blocking_codes([result]) == [
        "broken_expectation:unexpected_breakage_outcome"
    ]


def test_breakage_matrix_reject_reason_mismatch_blocks() -> None:
    case = MOD.CsvBreakageCase(
        name="wrong_reject_reason",
        description="Reject cases must assert the exact reason and location.",
        fieldnames=("Ticket ID", "Conversation Text"),
        rows=({"Ticket ID": "T-1", "Conversation Text": "Cannot export reports."},),
        expected_outcome="REJECT",
        expected_decision_reason="wrong_reason",
        expected_decision_location="source_row_csv",
    )

    result = MOD.evaluate_breakage_case(case)

    assert result["observed_outcome"] == "REJECT"
    assert result["admission_decision_reason"] == "no_usable_source_rows"
    assert result["admission_decision_location"] == "source_row_csv"
    assert result["decision_matched"] is False
    assert result["case_status"] == "failed"
    assert MOD._breakage_blocking_codes([result]) == [
        "wrong_reject_reason:unexpected_breakage_outcome"
    ]


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


def test_observed_partial_csv_is_recorded_without_blocking(tmp_path: Path) -> None:
    corpus = {
        "source": "unit",
        "run_tag": "unit",
        "tickets": [{
            "id": "zd-proof-001",
            "subject": "Export failed",
            "description": "Customer cannot export reports.",
            "comments": [{
                "public": True,
                "body": "Customer cannot export reports.",
            }],
        }],
    }
    corpus_path = tmp_path / "corpus.json"
    observed_path = tmp_path / "partial.csv"
    out_dir = tmp_path / "artifact"
    doc = tmp_path / "proof.md"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    observed_path.write_text(
        "Ticket ID,Subject,Public Comments\n"
        "T-1,Export failed,Customer cannot export reports.\n"
        "T-2,Invite failed,\n",
        encoding="utf-8",
    )

    code = MOD.main([
        "--corpus",
        str(corpus_path),
        "--observed-csv",
        f"observed_partial={observed_path}",
        "--out-dir",
        str(out_dir),
        "--doc",
        str(doc),
    ])

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    observed = {
        case["name"]: case for case in summary["cases"]
    }["observed_partial"]
    assert code == 0
    assert summary["status"] == "ok"
    assert summary["blocking_violation_codes"] == []
    assert summary["observed_case_count"] == 1
    assert summary["blocking_case_count"] == 0
    assert observed["case_status"] == "observed"
    assert observed["admission_status"] == "ACCEPT"
    assert observed["raw_source_row_count"] == 2
    assert observed["usable_source_row_count"] == 1
    assert observed["coverage_warnings"] == [{
        "code": "partial_source_row_coverage",
        "location": "source_row_csv",
        "raw_source_row_count": 2,
        "usable_source_row_count": 1,
        "skipped_source_row_count": 1,
        "usable_source_ratio": 0.5,
    }]
    assert "Observed evidence cases: `1`" in doc.read_text(encoding="utf-8")


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


def test_observed_csv_argument_requires_name_and_existing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    try:
        MOD.main(["--observed-csv", str(missing)])
    except SystemExit as exc:
        assert str(exc) == "--observed-csv values must use NAME=PATH"
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("missing NAME=PATH did not fail")

    try:
        MOD.main(["--observed-csv", f"observed={missing}"])
    except SystemExit as exc:
        assert str(exc) == f"--observed-csv path does not exist: {missing}"
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("missing observed path did not fail")


def test_main_writes_summary_and_doc(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifact"
    doc = tmp_path / "proof.md"

    code = MOD.main(["--out-dir", str(out_dir), "--doc", str(doc)])

    assert code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["observed_case_count"] == 0
    assert summary["observed_min_usable_source_ratio"] == 1.0
    proof = doc.read_text(encoding="utf-8")
    assert "does not choose a low non-zero reject threshold" in proof
    assert "## Parser Breakage Matrix" in proof
    assert "json_blob_message_known_fail_open" in proof
    assert "Known fail-open gaps: `1`" in proof
