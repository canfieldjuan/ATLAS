from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_deflection_pii_source_decision.py"
SPEC = importlib.util.spec_from_file_location("check_deflection_pii_source_decision", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)


def _valid_decision() -> dict:
    return {
        "schema_version": CLI.SCHEMA_VERSION,
        "source": {
            "kind": "zendesk_export",
            "supply": "transient_local_file",
            "reference": "support-q2-sample",
        },
        "corpus": {
            "target_ticket_count": 50,
            "minimum_ticket_count": 25,
            "pii_class_targets": [
                "dob",
                "email",
                "payment_card",
                "person_name",
                "phone",
                "ssn",
                "street_address",
            ],
            "person_name_subtypes": ["cue_prefixed", "cue_less"],
        },
        "labeling": {
            "owner": "support-ops",
            "reviewer": "privacy-review",
            "quality_review": "completed",
        },
    }


def _run_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: dict) -> tuple[int, dict]:
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = CLI.main([str(decision_path)])
    captured = capsys.readouterr()
    rendered = captured.out if exit_code == 0 else captured.err
    return exit_code, json.loads(rendered)


def _codes(payload: dict) -> set[str]:
    return {error["code"] for error in payload.get("errors", [])}


def test_valid_source_decision_returns_sanitized_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, payload = _run_cli(tmp_path, capsys, _valid_decision())

    assert exit_code == 0
    assert payload == {
        "ok": True,
        "schema_version": CLI.SCHEMA_VERSION,
        "source": {
            "kind": "zendesk_export",
            "supply": "transient_local_file",
            "reference": "support-q2-sample",
        },
        "corpus": {
            "target_ticket_count": 50,
            "minimum_ticket_count": 25,
            "pii_class_targets": [
                "dob",
                "email",
                "payment_card",
                "person_name",
                "phone",
                "ssn",
                "street_address",
            ],
            "person_name_subtypes": ["cue_less", "cue_prefixed"],
        },
        "labeling": {
            "owner": "support-ops",
            "reviewer": "privacy-review",
            "quality_review": "completed",
        },
        "raw_source_persisted": False,
        "raw_label_spans_persisted": False,
    }


def test_missing_handoff_sections_fail_with_stable_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = _valid_decision()
    decision.pop("source")
    decision["corpus"].pop("person_name_subtypes")
    decision["labeling"].pop("reviewer")

    exit_code, payload = _run_cli(tmp_path, capsys, decision)

    assert exit_code == 1
    assert {
        "source_missing",
        "person_name_subtypes_invalid",
        "labeling_reviewer_missing",
    } <= _codes(payload)


@pytest.mark.parametrize(
    "bad_reference",
    [
        "/tmp/raw-deflection.json",
        "../raw-deflection",
        "https://example.com/raw-export",
        "alice@example.com",
        "202-555-0188",
        "111-22-3333",
        "4242 4242 4242 4242",
        "1977-06-05",
        "DOB-1977-06-05",
        "source_1977_06_05",
        "customer Alice Baker asked for a refund and gave DOB 1977-06-05",
    ],
)
def test_unsafe_source_reference_fails_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad_reference: str,
) -> None:
    decision = _valid_decision()
    decision["source"]["reference"] = bad_reference

    exit_code, payload = _run_cli(tmp_path, capsys, decision)
    rendered = json.dumps(payload, sort_keys=True)

    assert exit_code == 1
    assert "source_reference_unsafe" in _codes(payload)
    assert bad_reference not in rendered
    assert "Alice Baker" not in rendered
    assert "1977-06-05" not in rendered


def test_unexpected_manifest_fields_fail_closed_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = _valid_decision()
    decision["raw_tickets"] = ["Alice Baker alice@example.com"]
    decision["source"]["raw_path"] = "/tmp/raw-deflection.json"
    decision["corpus"]["label_spans"] = [{"text": "Jane Client"}]
    decision["labeling"]["raw_notes"] = "DOB 1977-06-05 reviewed by Alice"

    exit_code, payload = _run_cli(tmp_path, capsys, decision)
    rendered = json.dumps(payload, sort_keys=True)

    assert exit_code == 1
    assert _codes(payload) == {"unexpected_field"}
    assert rendered.count("unexpected_field") == 4
    assert "raw_tickets" not in rendered
    assert "raw_path" not in rendered
    assert "label_spans" not in rendered
    assert "raw_notes" not in rendered
    assert "Alice Baker" not in rendered
    assert "alice@example.com" not in rendered
    assert "/tmp/raw-deflection.json" not in rendered
    assert "Jane Client" not in rendered
    assert "1977-06-05" not in rendered


def test_corpus_targets_fail_closed_on_incomplete_or_inconsistent_mix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = _valid_decision()
    decision["corpus"] = {
        "target_ticket_count": 0,
        "minimum_ticket_count": 20,
        "pii_class_targets": ["email", "phone", "person_name"],
        "person_name_subtypes": ["cue_prefixed"],
    }

    exit_code, payload = _run_cli(tmp_path, capsys, decision)

    assert exit_code == 1
    assert {
        "target_ticket_count_invalid",
        "pii_class_targets_missing_required",
        "person_name_subtypes_missing_required",
    } <= _codes(payload)
    missing_class_error = next(
        error
        for error in payload["errors"]
        if error["code"] == "pii_class_targets_missing_required"
    )
    assert missing_class_error["missing"] == ["dob", "payment_card", "ssn"]


def test_corpus_metadata_lists_reject_unsupported_values_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = _valid_decision()
    decision["corpus"]["pii_class_targets"].append("alice@example.com")
    decision["corpus"]["person_name_subtypes"].append("/tmp/raw-labels.json")

    exit_code, payload = _run_cli(tmp_path, capsys, decision)
    rendered = json.dumps(payload, sort_keys=True)

    assert exit_code == 1
    assert {
        "pii_class_targets_unsupported",
        "person_name_subtypes_unsupported",
    } <= _codes(payload)
    assert "alice@example.com" not in rendered
    assert "/tmp/raw-labels.json" not in rendered


def test_corpus_minimum_count_cannot_exceed_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = _valid_decision()
    decision["corpus"]["target_ticket_count"] = 10
    decision["corpus"]["minimum_ticket_count"] = 11

    exit_code, payload = _run_cli(tmp_path, capsys, decision)

    assert exit_code == 1
    assert "minimum_ticket_count_exceeds_target" in _codes(payload)


def test_labeling_quality_requires_completed_independent_review(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = _valid_decision()
    decision["labeling"] = {
        "owner": "support-ops",
        "reviewer": "support-ops",
        "quality_review": "pending",
    }

    exit_code, payload = _run_cli(tmp_path, capsys, decision)

    assert exit_code == 1
    assert {
        "labeling_reviewer_matches_owner",
        "quality_review_not_completed",
    } <= _codes(payload)


@pytest.mark.parametrize(
    ("owner", "reviewer"),
    [
        ("Support-Ops", "support-ops"),
        ("support ops", "support_ops"),
    ],
)
def test_labeling_owner_reviewer_identity_comparison_is_normalized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    owner: str,
    reviewer: str,
) -> None:
    decision = _valid_decision()
    decision["labeling"]["owner"] = owner
    decision["labeling"]["reviewer"] = reviewer

    exit_code, payload = _run_cli(tmp_path, capsys, decision)

    assert exit_code == 1
    assert "labeling_reviewer_matches_owner" in _codes(payload)


def test_unreadable_json_failure_is_sanitized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision_path = tmp_path / "decision.json"
    raw_text = '{"source": "alice@example.com",'
    decision_path.write_text(raw_text, encoding="utf-8")

    exit_code = CLI.main([str(decision_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    rendered = json.dumps(payload, sort_keys=True)

    assert exit_code == 1
    assert _codes(payload) == {"decision_json_unreadable"}
    assert "alice@example.com" not in rendered
    assert str(decision_path) not in rendered
