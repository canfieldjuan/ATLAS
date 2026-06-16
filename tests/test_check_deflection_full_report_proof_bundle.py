from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_deflection_full_report_proof_bundle.py"
SPEC = importlib.util.spec_from_file_location(
    "check_deflection_full_report_proof_bundle",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


FORBIDDEN_FIXTURES = {
    "request_id": '"request_id": "content-ops-45c06a6950ec4677a214368d6e4dc44f"',
    "result_url": "https://www.juancanfield.com/systems/support-ticket-deflection/results/content-ops-45c06a6950ec4677a214368d6e4dc44f",
    "customer_email": "Delivered-To: buyer@realcustomer.com",
    "absolute_local_path": "artifact written to /home/juan-canfield/Desktop/live/report.pdf",
    "stripe_checkout_session_id": "checkout session cs_live_a1b2c3d4e5f6g7h8",
    "stripe_payment_intent_id": "payment intent pi_3Pmtlivea1b2c3d4",
    "raw_evidence_quote": '{"evidence_quote": "Customer says their invoice exposes PII"}',
    "source_id_list": '{"source_ids": ["zd-proof-001", "zd-proof-002"]}',
    "private_note": '{"public": false, "body": "private note from agent"}',
}

RAW_CUSTOMER_TEXT = "Customer says account 555-12-9999 cannot be published"


def _write_bundle(tmp_path: Path, text: str) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "artifact.txt").write_text(text, encoding="utf-8")
    return bundle


@pytest.mark.parametrize("expected_key,text", sorted(FORBIDDEN_FIXTURES.items()))
def test_checker_names_each_forbidden_class(expected_key: str, text: str, tmp_path: Path) -> None:
    result = checker.scan_bundle(_write_bundle(tmp_path, text))

    assert result["ok"] is False
    assert set(result["checks"]) == set(checker.CHECK_KEYS)
    assert result["checks"][expected_key]["ok"] is False
    assert result["checks"][expected_key]["findings"]


def test_checker_allows_synthetic_and_example_values(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        "\n".join([
            '"request_id": "synthetic-request-id"',
            "https://qa.example.com/systems/support-ticket-deflection/results/synthetic-request-id",
            "To: buyer@example.com",
            '{"scorecard": "only"}',
        ]),
    )

    result = checker.scan_bundle(bundle)

    assert result["ok"] is True
    assert all(check["ok"] for check in result["checks"].values())
    assert set(result["checks"]) == set(checker.CHECK_KEYS)


def test_cli_writes_assertions_without_echoing_raw_sensitive_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_request_id = "content-ops-45c06a6950ec4677a214368d6e4dc44f"
    bundle = _write_bundle(tmp_path, f'"request_id": "{raw_request_id}"')
    output = tmp_path / "assertions.json"

    code = checker.main([str(bundle), "--output", str(output), "--pretty"])
    printed = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["checks"]["request_id"]["ok"] is False
    assert raw_request_id not in output.read_text(encoding="utf-8")
    assert raw_request_id not in printed
    assert "[request_id]" in output.read_text(encoding="utf-8")


def test_raw_evidence_and_private_note_do_not_echo_customer_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _write_bundle(
        tmp_path,
        "\n".join([
            f'{{"evidence_quote": "{RAW_CUSTOMER_TEXT}"}}',
            f'{{"public": false, "body": "{RAW_CUSTOMER_TEXT}"}}',
        ]),
    )
    output = tmp_path / "assertions.json"

    code = checker.main([str(bundle), "--output", str(output), "--pretty"])
    printed = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["checks"]["raw_evidence_quote"]["findings"][0]["snippet"] == "[raw_evidence_quote]"
    assert payload["checks"]["private_note"]["findings"][0]["snippet"] == "[private_note]"
    assert RAW_CUSTOMER_TEXT not in output.read_text(encoding="utf-8")
    assert RAW_CUSTOMER_TEXT not in printed


def test_request_id_in_bundle_path_is_detected_and_redacted(tmp_path: Path) -> None:
    raw_request_id = "content-ops-45c06a6950ec4677a214368d6e4dc44f"
    bundle = tmp_path / raw_request_id
    bundle.mkdir()
    (bundle / "artifact.txt").write_text('{"scorecard": "only"}', encoding="utf-8")

    result = checker.scan_bundle(bundle)
    encoded = json.dumps(result)

    assert result["ok"] is False
    assert result["checks"]["request_id"]["ok"] is False
    assert raw_request_id not in encoded
    assert "[request_id]" in encoded


def test_finding_paths_are_redacted(tmp_path: Path) -> None:
    raw_request_id = "content-ops-45c06a6950ec4677a214368d6e4dc44f"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / f"{raw_request_id}.txt"
    artifact.write_text("Delivered-To: buyer@realcustomer.com", encoding="utf-8")

    result = checker.scan_bundle(bundle)
    finding = result["checks"]["customer_email"]["findings"][0]

    assert finding["path"] == "[request_id].txt"
    assert raw_request_id not in json.dumps(result)


def test_missing_bundle_path_fails_closed(tmp_path: Path) -> None:
    result = checker.scan_bundle(tmp_path / "missing")

    assert result["ok"] is False
    assert result["checks"]["artifact_readability"]["ok"] is False
    assert result["checks"]["artifact_readability"]["findings"][0]["snippet"] == "[missing_bundle_path]"


def test_empty_bundle_directory_fails_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    result = checker.scan_bundle(bundle)

    assert result["ok"] is False
    assert result["scanned_files"] == 0
    assert result["checks"]["artifact_readability"]["findings"][0]["snippet"] == "[empty_bundle]"


def test_pdf_artifact_is_rejected_fail_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "report.pdf").write_bytes(b"%PDF-1.7\ncompressed content")

    result = checker.scan_bundle(bundle)

    assert result["ok"] is False
    assert result["checks"]["artifact_readability"]["ok"] is False
    assert result["checks"]["artifact_readability"]["findings"][0]["snippet"] == "[unsupported_pdf_artifact]"


@pytest.mark.parametrize(
    "path_text",
    [
        "/Users/alice/Desktop/report.pdf",
        "/workspace/ATLAS/report.pdf",
        "/var/tmp/report.pdf",
        "/root/report.pdf",
        "C:\\Users\\Alice\\Desktop\\report.pdf",
    ],
)
def test_absolute_path_detector_covers_common_local_roots(path_text: str, tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, f"artifact written to {path_text}")

    result = checker.scan_bundle(bundle)

    assert result["ok"] is False
    assert result["checks"]["absolute_local_path"]["ok"] is False


@pytest.mark.parametrize(
    ("expected_key", "text"),
    [
        ("source_id_list", "source_id,body\nzd-proof-001,customer text\n"),
        ("source_id_list", '{"source_id": "zd-proof-001", "body": "customer text"}'),
        ("private_note", "public,false,do not publish this body\n"),
        ("private_note", "is_public,false,do not publish this body\n"),
        ("private_note", "visibility,private,do not publish this body\n"),
    ],
)
def test_csv_and_jsonl_export_markers_fail_closed(expected_key: str, text: str, tmp_path: Path) -> None:
    result = checker.scan_bundle(_write_bundle(tmp_path, text))

    assert result["ok"] is False
    assert result["checks"][expected_key]["ok"] is False


def test_real_result_url_with_example_query_is_not_allowlisted(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        "https://www.juancanfield.com/systems/support-ticket-deflection/results/req_123456789?account=exampleco",
    )

    result = checker.scan_bundle(bundle)

    assert result["ok"] is False
    assert result["checks"]["result_url"]["ok"] is False


def test_real_result_url_with_synthetic_result_id_is_allowed(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        "https://www.juancanfield.com/systems/support-ticket-deflection/results/synthetic-request-id",
    )

    result = checker.scan_bundle(bundle)

    assert result["ok"] is True


def test_absolute_path_redaction_preserves_neighboring_context(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, "before /tmp/live/report.pdf after")

    result = checker.scan_bundle(bundle)
    snippet = result["checks"]["absolute_local_path"]["findings"][0]["snippet"]

    assert snippet == "before [absolute_local_path] after"
