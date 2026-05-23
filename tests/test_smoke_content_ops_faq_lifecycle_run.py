from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_lifecycle_run.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_lifecycle_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"


def _args(tmp_path: Path, lifecycle_args: list[str] | None = None):
    return argparse.Namespace(
        path=SUPPORT_TICKET_CSV,
        artifact_dir=tmp_path / "artifacts",
        lifecycle_args=lifecycle_args or ["--account-id", "acct-smoke"],
    )


def _result_path(command: list[str]) -> Path:
    return Path(command[command.index("--output-result") + 1])


def test_lifecycle_artifact_smoke_writes_standard_artifacts(monkeypatch, tmp_path: Path) -> None:
    calls = []
    lifecycle_summary = {"status": "ok", "source_rows": 1000, "saved_faq_count": 1}

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        _result_path(command).write_text(
            json.dumps({"ok": True, "saved_ids": ["faq-uuid-1"], "lifecycle_summary": lifecycle_summary}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(lifecycle_summary) + "\n", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    code, summary = smoke.run_lifecycle_artifact_smoke(
        _args(tmp_path, ["--account-id", "acct-smoke", "--min-source-rows", "1000"])
    )

    artifact_dir = tmp_path / "artifacts"
    saved_summary = json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))
    command, kwargs = calls[0]
    assert code == 0
    assert summary["failure"] is None
    assert saved_summary["lifecycle_summary"] == lifecycle_summary
    assert json.loads((artifact_dir / "summary_stdout.json").read_text(encoding="utf-8")) == lifecycle_summary
    assert json.loads((artifact_dir / "lifecycle_result.json").read_text(encoding="utf-8"))["saved_ids"] == ["faq-uuid-1"]
    assert saved_summary["artifact_details"]["summary_stdout"]["exists"] is True
    assert saved_summary["artifact_details"]["result"]["bytes"] > 0
    assert kwargs == {"check": False, "capture_output": True, "text": True}
    assert command[command.index("--min-source-rows") + 1] == "1000"
    assert "--summary-json" in command


def test_lifecycle_artifact_smoke_classifies_lifecycle_failure(monkeypatch, tmp_path: Path) -> None:
    errors = ["expected at least 500 source row(s), got 46"]
    lifecycle_summary = {"status": "failed", "source_rows": 46, "errors": errors}

    def fake_run(command, **kwargs):
        _result_path(command).write_text(
            json.dumps({"ok": False, "errors": errors, "lifecycle_summary": lifecycle_summary}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(lifecycle_summary) + "\n",
            stderr="Content Ops FAQ lifecycle smoke failed\n",
        )

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    code, summary = smoke.run_lifecycle_artifact_smoke(_args(tmp_path))

    assert code == 1
    assert summary["failure"]["type"] == "lifecycle_error"
    assert summary["failure"]["errors"] == errors
    assert summary["failure"]["summary_status"] == "failed"


def test_lifecycle_artifact_smoke_classifies_hard_cli_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="Missing --database-url\n")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    code, summary = smoke.run_lifecycle_artifact_smoke(_args(tmp_path))

    artifact_dir = tmp_path / "artifacts"
    assert code == 2
    assert summary["result"] is None
    assert summary["lifecycle_summary"] is None
    assert summary["failure"]["type"] == "missing_result"
    assert "Missing --database-url" in summary["failure"]["stderr_tail"]
    assert not (artifact_dir / "summary_stdout.json").exists()
    assert json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))["artifact_details"]["summary_stdout"]["exists"] is False


def test_lifecycle_artifact_smoke_main_and_reserved_flag_guard(monkeypatch, tmp_path: Path, capsys) -> None:
    lifecycle_summary = {"status": "ok", "source_rows": 4, "saved_faq_count": 1}

    def fake_run(command, **kwargs):
        _result_path(command).write_text(
            json.dumps({"ok": True, "lifecycle_summary": lifecycle_summary}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(lifecycle_summary) + "\n", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    code = smoke.main([
        str(SUPPORT_TICKET_CSV),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--account-id",
        "acct-smoke",
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert "Content Ops FAQ lifecycle artifact smoke passed:" in captured.out
    assert f"summary={tmp_path / 'artifacts' / 'run_summary.json'}" in captured.out
    with pytest.raises(SystemExit, match="artifact runner owns"):
        smoke._validate_args(_args(tmp_path, ["--json"]))
