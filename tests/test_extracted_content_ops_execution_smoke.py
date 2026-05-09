from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/smoke_extracted_content_ops_execution.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_extracted_content_ops_execution",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_content_ops_execution_smoke_cli_runs_full_campaign_preset() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AI Content Ops execution smoke passed" in completed.stdout
    assert "email_campaign" in completed.stdout
    assert "blog_post" in completed.stdout
    assert "report" in completed.stdout
    assert "landing_page" in completed.stdout
    assert "sales_brief" in completed.stdout


def test_content_ops_execution_smoke_cli_accepts_output_subset_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--outputs",
            "email_campaign,report",
            "--target-mode",
            "challenger_intel",
            "--no-quality-gates",
            "--limit",
            "2",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert [step["output"] for step in payload["steps"]] == [
        "email_campaign",
        "report",
    ]
    assert payload["steps"][0]["result"]["generated"] == 2
    assert payload["steps"][0]["result"]["target_mode"] == "challenger_intel"
    assert payload["steps"][1]["result"]["target_mode"] == "challenger_intel"
    assert payload["steps"][0]["result"]["quality_revalidation_enabled"] is False
    assert payload["steps"][1]["result"]["quality_gates_enabled"] is False
    assert (
        payload["plan"]["preview"]["normalized_request"]["require_quality_gates"]
        is False
    )


def test_content_ops_execution_smoke_cli_runs_signal_extraction_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--outputs",
            "signal_extraction",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["steps"][0]["output"] == "signal_extraction"
    assert payload["steps"][0]["result"]["generated"] == 1
    assert (
        payload["steps"][0]["result"]["opportunities"][0]["target_id"]
        == "source-smoke-1"
    )


def test_content_ops_execution_smoke_cli_parameterizes_signal_source_row() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--outputs",
            "signal_extraction",
            "--source-id",
            "source-42",
            "--source-vendor",
            "Salesforce",
            "--source-contact-email",
            "ops@example.com",
            "--source-max-text-chars",
            "11",
            "--source-material",
            "The renewal created finance pressure.",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    opportunity = payload["steps"][0]["result"]["opportunities"][0]
    assert opportunity["target_id"] == "source-42"
    assert opportunity["vendor"] == "Salesforce"
    assert opportunity["contact_email"] == "ops@example.com"
    assert opportunity["evidence"][0]["text"] == "The renewal"


def test_content_ops_execution_smoke_fails_when_required_inputs_missing() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--outputs",
            "email_campaign",
            "--target-account",
            "",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "expected completed status, got 'blocked'" in completed.stderr


def test_content_ops_execution_smoke_json_failure_keeps_stdout_parseable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--outputs",
            "email_campaign",
            "--target-account",
            "",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["status"] == "blocked"
    assert payload["smoke_errors"] == ["expected completed status, got 'blocked'"]
    assert "AI Content Ops execution smoke failed" in completed.stderr


def test_execution_errors_reports_empty_steps() -> None:
    smoke = _load_smoke_module()

    assert smoke._execution_errors({"status": "completed", "steps": []}) == [
        "result.steps is missing or empty"
    ]


def test_execution_errors_accepts_signal_extraction_opportunities() -> None:
    smoke = _load_smoke_module()

    assert smoke._execution_errors({
        "status": "completed",
        "steps": [
            {
                "output": "signal_extraction",
                "status": "completed",
                "result": {"opportunities": [{"target_id": "Acme"}]},
            }
        ],
    }) == []
