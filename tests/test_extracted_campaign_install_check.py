from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import extracted_content_pipeline.campaign_install_check as install_check
from extracted_content_pipeline.campaign_install_check import check_campaign_install


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/check_extracted_content_install.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "check_extracted_content_install",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_present(name: str):
    return object()


def test_offline_profile_passes_without_database_env(monkeypatch) -> None:
    monkeypatch.setattr(install_check, "find_spec", _module_present)

    report = check_campaign_install(environ={}, profiles=("offline",))

    data = report.as_dict()
    assert report.passed is True
    assert data["counts"] == {"ok": 1, "warning": 0, "error": 0}
    assert [check.name for check in report.checks] == ["product_package"]


def test_generation_profile_requires_database_and_asyncpg(monkeypatch) -> None:
    monkeypatch.setattr(
        install_check,
        "find_spec",
        lambda name: object() if name == "extracted_content_pipeline" else None,
    )

    report = check_campaign_install(environ={}, profiles=("generation",))

    errors = [check.name for check in report.checks if check.status == "error"]
    warnings = [check.name for check in report.checks if check.status == "warning"]
    assert report.passed is False
    assert errors == ["database_url", "asyncpg"]
    assert warnings == ["campaign_llm_route"]


def test_generation_offline_llm_skips_llm_route_warning(monkeypatch) -> None:
    monkeypatch.setattr(install_check, "find_spec", _module_present)

    report = check_campaign_install(
        environ={"EXTRACTED_DATABASE_URL": "postgres://example"},
        profiles=("generation",),
        llm="offline",
    )

    assert report.passed is True
    assert "campaign_llm_route" not in [check.name for check in report.checks]


def test_send_resend_profile_requires_api_key_and_from_email(monkeypatch) -> None:
    monkeypatch.setattr(install_check, "find_spec", _module_present)

    report = check_campaign_install(
        environ={"EXTRACTED_DATABASE_URL": "postgres://example"},
        profiles=("send",),
        sender="resend",
    )

    errors = [check.name for check in report.checks if check.status == "error"]
    assert report.passed is False
    assert errors == ["resend_api_key", "from_email"]


def test_send_resend_profile_passes_with_campaign_env(monkeypatch) -> None:
    monkeypatch.setattr(install_check, "find_spec", _module_present)

    report = check_campaign_install(
        environ={
            "DATABASE_URL": "postgres://example",
            "EXTRACTED_CAMPAIGN_RESEND_API_KEY": "re_test",
            "EXTRACTED_CAMPAIGN_FROM_EMAIL": "sales@example.com",
        },
        profiles=("send",),
        sender="resend",
    )

    assert report.passed is True
    assert {check.name for check in report.checks} == {
        "product_package",
        "database_url",
        "asyncpg",
        "resend_api_key",
        "from_email",
    }


def test_webhook_profile_can_skip_secret_requirement(monkeypatch) -> None:
    monkeypatch.setattr(install_check, "find_spec", _module_present)

    report = check_campaign_install(
        environ={"DATABASE_URL": "postgres://example"},
        profiles=("webhooks",),
        require_webhook_secret=False,
    )

    assert report.passed is True
    webhook_check = [check for check in report.checks if check.name == "resend_webhook_secret"][0]
    assert webhook_check.status == "warning"


def test_cli_emits_json_and_returns_nonzero_for_missing_database(
    monkeypatch,
    capsys,
) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(
        install_check,
        "find_spec",
        lambda name: object() if name == "extracted_content_pipeline" else None,
    )
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    exit_code = cli.main(["--profile", "generation", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["passed"] is False
    assert output["counts"]["error"] == 2


def test_cli_accepts_all_profile_with_resend_env(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(install_check, "find_spec", _module_present)
    monkeypatch.setenv("EXTRACTED_DATABASE_URL", "postgres://example")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_RESEND_API_KEY", "re_test")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_FROM_EMAIL", "sales@example.com")
    monkeypatch.setenv("EXTRACTED_RESEND_WEBHOOK_SECRET", "whsec_test")

    exit_code = cli.main([
        "--profile",
        "all",
        "--sender",
        "resend",
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "passed=true" in output
    assert "profiles=generation,send,webhooks,analytics,export" in output
