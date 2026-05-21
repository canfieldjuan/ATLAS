from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_scale_run.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_scale_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"
TICKET_ROWS = (
    ("ticket-acme-1", "Change login email", "How do I change my login email?", "email and profile updates"),
    ("ticket-acme-2", "Update account email", "I need to update the email on my account.", "email and profile updates"),
    ("ticket-northstar-1", "Export campaign reports", "How do we export campaign attribution data before renewal?", "reporting friction"),
    ("ticket-northstar-2", "Reporting dashboard export", "We cannot export the reporting dashboard for analysts.", "reporting friction"),
)


def _args(tmp_path: Path, **overrides):
    values = {
        "path": SUPPORT_TICKET_CSV,
        "source_format": "auto",
        "artifact_dir": tmp_path / "artifacts",
        "title": "Customer Ticket FAQ Scale Smoke",
        "max_items": 12,
        "max_evidence_per_item": 5,
        "max_text_chars": 1200,
        "window_days": None,
        "as_of_date": None,
        "support_contact": None,
        "default_field": [],
        "allow_output_check_failures": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_ticket_csv(tmp_path: Path, *rows: str) -> Path:
    source = tmp_path / "tickets.csv"
    source.write_text(
        "Ticket ID,Created At,Subject,Description,Pain Category\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return source


def _write_source(tmp_path: Path, fmt: str) -> Path:
    dict_rows = [
        dict(zip(("Ticket ID", "Subject", "Description", "Pain Category"), row))
        for row in TICKET_ROWS
    ]
    if fmt == "jsonl":
        source = tmp_path / "tickets.jsonl"
        source.write_text(
            "\n".join(json.dumps(row) for row in dict_rows) + "\n",
            encoding="utf-8",
        )
        return source
    if fmt in {"auto", "json"}:
        source = tmp_path / "tickets.json"
        source.write_text(json.dumps(dict_rows) + "\n", encoding="utf-8")
        return source
    return _write_ticket_csv(
        tmp_path,
        *(",".join((ticket_id, "2026-05-01", subject, description, pain_category))
          for ticket_id, subject, description, pain_category in TICKET_ROWS),
    )


@pytest.mark.parametrize("fmt", ["auto", "csv", "json", "jsonl"])
def test_faq_scale_smoke_writes_standard_artifacts(tmp_path: Path, fmt: str) -> None:
    source = _write_source(tmp_path, fmt)
    code, summary = smoke.run_scale_smoke(_args(tmp_path, path=source, source_format=fmt))

    artifact_dir = tmp_path / "artifacts"
    assert code == 0
    assert summary["ok"] is True
    for name in ("faq.md", "faq_result.json", "stdout.txt", "stderr.txt", "run_summary.json"):
        assert (artifact_dir / name).exists()
    result = json.loads((artifact_dir / "faq_result.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["ticket_source_count"] == 4
    assert result["failed_output_checks"] == []
    assert saved_summary["exit_code"] == 0
    assert saved_summary["result"]["generated"] == result["generated"]
    assert saved_summary["source_format"] == fmt
    assert saved_summary["timing"]["elapsed_seconds"] >= 0
    assert saved_summary["failure"] is None
    assert saved_summary["artifact_details"]["markdown"]["exists"] is True
    assert saved_summary["artifact_details"]["markdown"]["bytes"] > 0
    assert saved_summary["artifact_details"]["summary"]["exists"] is True
    assert saved_summary["artifact_details"]["summary"]["bytes"] is None
    assert (artifact_dir / "faq.md").read_text(encoding="utf-8").startswith(
        "# Customer Ticket FAQ Scale Smoke"
    )


def test_faq_scale_smoke_preserves_fail_closed_exit_and_artifacts(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Unique one,The export button moved.,exports",
        "ticket-2,2026-05-01,Unique two,Billing receipt is missing.,billing",
    )

    code, summary = smoke.run_scale_smoke(_args(tmp_path, path=source))

    artifact_dir = tmp_path / "artifacts"
    assert code == 1
    assert summary["ok"] is False
    result = json.loads((artifact_dir / "faq_result.json").read_text(encoding="utf-8"))
    stderr = (artifact_dir / "stderr.txt").read_text(encoding="utf-8")
    assert result["status"] == "failed_output_checks"
    assert result["failed_output_checks"] == ["condensed"]
    assert "FAQ output checks failed: condensed" in stderr
    assert not (artifact_dir / "faq.md").exists()
    assert summary["artifacts"]["markdown"] is None
    assert summary["artifact_details"]["markdown"]["exists"] is False
    assert summary["artifact_details"]["result"]["bytes"] > 0
    assert summary["failure"]["type"] == "output_checks"
    assert summary["failure"]["failed_output_checks"] == ["condensed"]
    assert "FAQ output checks failed: condensed" in summary["failure"]["stderr_tail"]
    assert summary["result"]["diagnostics"]["rendered_ticket_source_count"] == 2


def test_faq_scale_smoke_can_allow_output_check_failures(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Unique one,The export button moved.,exports",
        "ticket-2,2026-05-01,Unique two,Billing receipt is missing.,billing",
    )

    code, summary = smoke.run_scale_smoke(
        _args(tmp_path, path=source, allow_output_check_failures=True)
    )

    assert code == 0
    assert summary["ok"] is False
    assert summary["exit_code"] == 1


def test_faq_scale_smoke_does_not_allow_hard_cli_failures(tmp_path: Path) -> None:
    code, summary = smoke.run_scale_smoke(
        _args(tmp_path, path=tmp_path / "missing.csv", allow_output_check_failures=True)
    )

    assert code != 0
    assert summary["ok"] is False
    assert summary["result"] is None
    assert summary["failure"]["type"] == "cli_error"
    assert summary["failure"]["result_status"] is None
    assert summary["artifact_details"]["result"]["exists"] is False
    assert "No such file or directory" in summary["failure"]["stderr_tail"]
    assert "No such file or directory" in (
        tmp_path / "artifacts" / "stderr.txt"
    ).read_text(encoding="utf-8")


def test_faq_scale_smoke_main_uses_cli_defaults(tmp_path: Path) -> None:
    code = smoke.main([str(SUPPORT_TICKET_CSV), "--artifact-dir", str(tmp_path / "artifacts")])

    result = json.loads((tmp_path / "artifacts" / "faq_result.json").read_text(encoding="utf-8"))
    assert code == 0
    assert result["input"]["source_format"] == "auto"
    assert result["config"]["max_items"] == 12


def test_text_tail_bounds_long_stderr() -> None:
    assert smoke._text_tail("\n".join(f"line{i}" for i in range(50))).splitlines() == [
        f"line{i}" for i in range(30, 50)
    ]
    assert len(smoke._text_tail("x" * 5000)) == 4000


@pytest.mark.parametrize("overrides,message", [
    ({"max_items": 0}, "--max-items must be positive"),
    ({"max_evidence_per_item": 0}, "--max-evidence-per-item must be positive"),
    ({"max_text_chars": 0}, "--max-text-chars must be positive"),
    ({"window_days": 0}, "--window-days must be positive"),
    ({"as_of_date": "2026-05-01"}, "--as-of-date requires --window-days"),
])
def test_faq_scale_smoke_rejects_invalid_args(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        smoke._validate_args(_args(tmp_path, **overrides))
