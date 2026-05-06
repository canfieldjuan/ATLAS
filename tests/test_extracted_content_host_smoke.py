from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/smoke_extracted_content_pipeline_host.py"
EXAMPLE_PAYLOAD = (
    ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.json"
)
EXAMPLE_CSV = ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.csv"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_extracted_content_pipeline_host",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_smoke_cli_generates_offline_draft() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_PAYLOAD),
            "--limit",
            "1",
            "--min-drafts",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AI Content Ops host smoke passed" in completed.stdout
    assert "generated=2" in completed.stdout
    assert "model=offline-deterministic" in completed.stdout


def test_host_smoke_cli_default_requires_both_default_channels() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_PAYLOAD),
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AI Content Ops host smoke passed" in completed.stdout
    assert "generated=2" in completed.stdout


def test_host_smoke_cli_accepts_customer_csv_export() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_CSV),
            "--format",
            "csv",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AI Content Ops host smoke passed" in completed.stdout
    assert "generated=2" in completed.stdout


def test_host_smoke_cli_fails_when_min_drafts_not_met() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_PAYLOAD),
            "--limit",
            "1",
            "--channels",
            "email_cold",
            "--min-drafts",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "expected at least 2 draft(s), got 1" in completed.stdout


def test_draft_errors_reports_missing_required_fields() -> None:
    smoke = _load_smoke_module()

    errors = smoke._draft_errors(
        {"drafts": [{"subject": "Subject", "body": "", "target_id": "t1"}]},
        min_drafts=1,
    )

    assert errors == ["draft 1 missing body", "draft 1 missing channel"]
