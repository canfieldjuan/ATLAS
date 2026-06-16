from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_content_ops_faq_macro_lifecycle_artifact.py"
SPEC = importlib.util.spec_from_file_location(
    "check_content_ops_faq_macro_lifecycle_artifact",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_validate_complete_lifecycle_artifact_is_green() -> None:
    result = checker.validate_lifecycle_artifact(_artifact())

    assert result == {
        "ok": True,
        "errors": [],
        "account_id": "acct-1",
        "faq_id": "faq-123",
        "zendesk_base_url": "https://sandbox.zendesk.com",
        "lifecycle_stage": "complete",
        "write_stage": "complete",
        "cleanup_stage": "complete",
        "publishable_count": 1,
        "deleted_faq_count": 1,
        "external_deletes": [
            {
                "external_id": "macro-123",
                "ok": True,
                "already_deleted": False,
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda artifact: artifact.update({"skipped": True}), "lifecycle_skipped"),
        (lambda artifact: artifact.update({"cleanup_skipped": True}), "cleanup_skipped"),
        (lambda artifact: artifact.update({"stage": "cleanup"}), "lifecycle_incomplete"),
        (lambda artifact: artifact["write"].update({"ok": False}), "write_not_ok"),
        (lambda artifact: artifact["write"].update({"stage": "live_smoke"}), "write_incomplete"),
        (lambda artifact: artifact["cleanup"].update({"ok": False}), "cleanup_not_ok"),
        (lambda artifact: artifact["cleanup"].update({"stage": "external_delete"}), "cleanup_incomplete"),
        (lambda artifact: artifact["cleanup"].update({"deleted_faq_count": 0}), "cleanup_deleted_faq_count_not_one"),
        (lambda artifact: artifact["cleanup"].update({"external_deletes": []}), "missing_external_delete_proof"),
        (lambda artifact: artifact["write"].pop("live_smoke"), "missing_live_smoke_payload"),
    ],
)
def test_validate_rejects_partial_or_misleading_lifecycle_states(
    mutator: Any,
    error: str,
) -> None:
    artifact = _artifact()
    mutator(artifact)

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert error in result["errors"]


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda artifact: artifact.update({"ok": "false"}), "lifecycle_not_ok"),
        (lambda artifact: artifact.update({"skipped": "false"}), "lifecycle_skipped"),
        (lambda artifact: artifact["write"].update({"ok": "false"}), "write_not_ok"),
        (lambda artifact: artifact["write"].update({"skipped": "false"}), "write_skipped"),
        (lambda artifact: artifact["cleanup"].update({"ok": "false"}), "cleanup_not_ok"),
        (lambda artifact: artifact["cleanup"].update({"skipped": "false"}), "cleanup_skipped"),
        (lambda artifact: artifact["write"]["live_smoke"].update({"ok": "false"}), "live_smoke_not_ok"),
        (lambda artifact: artifact["write"]["live_smoke"].update({"skipped": "false"}), "live_smoke_skipped"),
        (lambda artifact: artifact["cleanup"]["external_deletes"][0].update({"ok": "true"}), "external_delete_not_ok"),
    ],
)
def test_validate_rejects_string_typed_success_flags(
    mutator: Any,
    error: str,
) -> None:
    artifact = _artifact()
    mutator(artifact)

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert error in result["errors"]


def test_validate_rejects_mismatched_faq_ids() -> None:
    artifact = _artifact()
    artifact["write"]["faq_id"] = "faq-other"
    artifact["cleanup"]["faq_id"] = "faq-third"
    artifact["write"]["live_smoke"]["faq_id"] = "faq-fourth"

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert "write_faq_id_mismatch" in result["errors"]
    assert "cleanup_faq_id_mismatch" in result["errors"]
    assert "live_smoke_faq_id_mismatch" in result["errors"]


def test_validate_rejects_missing_live_smoke_faq_id() -> None:
    artifact = _artifact()
    artifact["write"]["live_smoke"].pop("faq_id")

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert "missing_live_smoke_faq_id" in result["errors"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda artifact: artifact.update({"zendesk_base_url": "https://other.zendesk.com"}),
        lambda artifact: artifact["write"].update({"zendesk_base_url": "https://other.zendesk.com"}),
        lambda artifact: artifact["cleanup"].update({"zendesk_base_url": "https://other.zendesk.com"}),
        lambda artifact: artifact["write"]["live_smoke"].update({"zendesk_base_url": "https://other.zendesk.com"}),
    ],
)
def test_validate_rejects_cross_instance_base_url_mismatch(mutator: Any) -> None:
    artifact = _artifact()
    artifact["write"]["live_smoke"]["zendesk_base_url"] = "https://sandbox.zendesk.com"
    mutator(artifact)

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert "zendesk_base_url_mismatch" in result["errors"]


def test_validate_rejects_blank_external_delete_id() -> None:
    artifact = _artifact()
    artifact["cleanup"]["external_deletes"] = [
        {"external_id": " ", "ok": True, "error": ""}
    ]

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert "missing_external_delete_id" in result["errors"]


def test_validate_preserves_stage_errors_without_raw_payload_dump() -> None:
    artifact = _artifact()
    artifact["cleanup"].update({
        "ok": False,
        "errors": ["zendesk_macro_delete_failed"],
        "external_deletes": [
            {
                "external_id": "macro-123",
                "ok": False,
                "error": "secret transport detail",
            }
        ],
    })

    result = checker.validate_lifecycle_artifact(artifact)

    assert result["ok"] is False
    assert "zendesk_macro_delete_failed" in result["errors"]
    assert "external_delete_not_ok" in result["errors"]
    assert result["external_deletes"] == [
        {
            "external_id": "macro-123",
            "ok": False,
            "already_deleted": False,
        }
    ]


def test_render_summary_excludes_seed_copy_and_next_command() -> None:
    artifact = _artifact()
    result = checker.validate_lifecycle_artifact(artifact)

    summary = checker.render_summary(result)

    assert "How do I refund a duplicate charge" not in summary
    assert "Open Billing" not in summary
    assert "next_command" not in summary
    assert "DATABASE_URL" not in summary
    assert "faq-123" in summary
    assert "macro-123: ok" in summary


def test_main_rejects_invalid_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{", encoding="utf-8")

    code = checker.main(["--artifact", str(artifact)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["errors"] == ["artifact_json_invalid"]


def test_main_rejects_non_utf8_artifact_as_invalid_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"\xff\xfe\x00")

    code = checker.main(["--artifact", str(artifact)])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["errors"] == ["artifact_json_invalid"]


def test_main_writes_sanitized_markdown_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact.json"
    summary = tmp_path / "summary.md"
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")

    code = checker.main([
        "--artifact",
        str(artifact),
        "--summary-output",
        str(summary),
    ])

    payload = json.loads(capsys.readouterr().out)
    summary_text = summary.read_text(encoding="utf-8")
    assert code == 0
    assert payload["ok"] is True
    assert "Status: PASS" in summary_text
    assert "How do I refund a duplicate charge" not in summary_text


def test_checker_does_not_import_live_zendesk_or_postgres_clients() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "ZendeskMacroPublishProvider" not in source
    assert "ZendeskHTTPMacroTransport" not in source
    assert "asyncpg" not in source
    assert "atlas_brain" not in source
    assert "extracted_content_pipeline" not in source


def _artifact() -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": False,
        "cleanup_skipped": False,
        "stage": "complete",
        "account_id": "acct-1",
        "faq_id": "faq-123",
        "zendesk_base_url": "https://sandbox.zendesk.com",
        "write": {
            "ok": True,
            "skipped": False,
            "stage": "complete",
            "account_id": "acct-1",
            "faq_id": "faq-123",
            "zendesk_base_url": "https://sandbox.zendesk.com",
            "publishable_count": 1,
            "seed": {
                "macro_titles": ["How do I refund a duplicate charge?"],
                "next_command": "DATABASE_URL=secret python smoke.py",
            },
            "live_smoke": {
                "ok": True,
                "skipped": False,
                "faq_id": "faq-123",
                "publishable_count": 1,
                "summary": {"published_count": 1},
            },
        },
        "cleanup": {
            "ok": True,
            "skipped": False,
            "stage": "complete",
            "account_id": "acct-1",
            "faq_id": "faq-123",
            "zendesk_base_url": "https://sandbox.zendesk.com",
            "deleted_faq_count": 1,
            "external_deletes": [
                {"external_id": "macro-123", "ok": True, "error": ""}
            ],
            "errors": [],
        },
    }
