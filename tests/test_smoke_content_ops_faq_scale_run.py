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
FAILURE_EXAMPLES = {
    "density": ROOT / "extracted_content_pipeline/examples/faq_scale_density_limited_summary.json",
    "output_checks": ROOT / "extracted_content_pipeline/examples/faq_scale_output_check_failure_summary.json",
}
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


@pytest.mark.parametrize(("fmt", "raw_row_count_source"), [
    ("auto", "json_array"),
    ("csv", "csv_rows"),
    ("json", "json_array"),
    ("jsonl", "jsonl_lines"),
])
def test_faq_scale_smoke_writes_standard_artifacts(
    tmp_path: Path,
    fmt: str,
    raw_row_count_source: str,
) -> None:
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
    assert saved_summary["faq_run_summary"] == result["diagnostics"]["run_summary"]
    assert saved_summary["faq_run_summary"]["generated"] == result["generated"]
    assert saved_summary["faq_run_summary"]["weighted_source_volume"] == 4
    assert saved_summary["faq_run_summary"]["output_checks"]["failed"] == 0
    assert saved_summary["source_format"] == fmt
    assert saved_summary["input_profile"]["status"] == "ok"
    assert saved_summary["input_profile"]["raw_row_count"] == 4
    assert saved_summary["input_profile"]["raw_row_count_source"] == raw_row_count_source
    assert saved_summary["input_profile"]["usable_source_count"] == 4
    assert saved_summary["input_profile"]["missing_source_text_count"] == 0
    assert saved_summary["input_profile"]["usable_source_ratio"] == 1.0
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
        "ticket-3,2026-05-01,Empty text,,billing",
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
    assert summary["faq_run_summary"] == result["diagnostics"]["run_summary"]
    assert summary["faq_run_summary"]["status"] == "failed_output_checks"
    assert summary["faq_run_summary"]["output_checks"]["failed_checks"] == ["condensed"]
    assert summary["input_profile"]["raw_row_count"] == 3
    assert summary["input_profile"]["usable_source_count"] == 2
    assert summary["input_profile"]["warnings_by_code"]["missing_source_text"] == 1
    assert summary["input_profile"]["missing_source_text_count"] == 1
    assert summary["input_profile"]["skipped_row_count"] == 1
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
    assert summary["faq_run_summary"] is None
    assert summary["input_profile"]["status"] == "error"
    assert "No such file or directory" in summary["input_profile"]["error"]
    assert summary["failure"]["type"] == "cli_error"
    assert summary["failure"]["result_status"] is None
    assert summary["artifact_details"]["result"]["exists"] is False
    assert "No such file or directory" in summary["failure"]["stderr_tail"]
    assert "No such file or directory" in (
        tmp_path / "artifacts" / "stderr.txt"
    ).read_text(encoding="utf-8")


def test_faq_scale_smoke_main_uses_cli_defaults(tmp_path: Path, capsys) -> None:
    code = smoke.main([str(SUPPORT_TICKET_CSV), "--artifact-dir", str(tmp_path / "artifacts")])

    result = json.loads((tmp_path / "artifacts" / "faq_result.json").read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert code == 0
    assert "Content Ops FAQ scale smoke passed:" in captured.out
    assert "source_rows=4/4" in captured.out
    assert "faq=available" in captured.out
    assert "generated=2" in captured.out
    assert "weighted_volume=4" in captured.out
    assert "checks_failed=0/3" in captured.out
    assert "score_max=4" in captured.out
    assert "summary=" in captured.out
    assert captured.err == ""
    assert result["input"]["source_format"] == "auto"
    assert result["config"]["max_items"] == 12


def test_faq_scale_smoke_main_prints_profile_on_failure(tmp_path: Path, capsys) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Unique one,The export button moved.,exports",
        "ticket-2,2026-05-01,Unique two,Billing receipt is missing.,billing",
        "ticket-3,2026-05-01,Empty text,,billing",
    )

    code = smoke.main([str(source), "--artifact-dir", str(tmp_path / "artifacts")])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Content Ops FAQ scale smoke failed:" in captured.err
    assert "source_rows=2/3" in captured.err
    assert "faq=available" in captured.err
    assert "generated=2" in captured.err
    assert "checks_failed=1/3" in captured.err
    assert "skipped_rows=1" in captured.err
    assert "missing_source_text=1" in captured.err
    assert "failure=output_checks" in captured.err
    assert "summary=" in captured.err


def test_text_tail_bounds_long_stderr() -> None:
    assert smoke._text_tail("\n".join(f"line{i}" for i in range(50))).splitlines() == [
        f"line{i}" for i in range(30, 50)
    ]
    assert len(smoke._text_tail("x" * 5000)) == 4000


def test_raw_row_profile_counts_json_bundle_rows(tmp_path: Path) -> None:
    source = tmp_path / "bundle.json"
    source.write_text(
        json.dumps({"support_tickets": [{}, {}, {}], "reviews": [{}, {}]}) + "\n",
        encoding="utf-8",
    )

    assert smoke._raw_row_profile(source, "auto") == {
        "raw_row_count": 5,
        "raw_row_count_source": "json_bundle.support_tickets,reviews",
    }


def test_raw_row_profile_returns_none_for_unrecognized_shape(tmp_path: Path) -> None:
    source = tmp_path / "odd.json"
    source.write_text(json.dumps({"items": [{}, {}]}) + "\n", encoding="utf-8")

    assert smoke._raw_row_profile(source, "auto") == {
        "raw_row_count": None,
        "raw_row_count_source": None,
    }


def test_faq_scale_failure_examples_separate_density_from_output_checks() -> None:
    density = json.loads(FAILURE_EXAMPLES["density"].read_text(encoding="utf-8"))
    output_checks = json.loads(FAILURE_EXAMPLES["output_checks"].read_text(encoding="utf-8"))

    assert density["ok"] is False
    assert density["input_profile"]["raw_row_count"] == 1000
    assert density["input_profile"]["usable_source_count"] == 46
    assert density["input_profile"]["usable_source_ratio"] < 0.1
    assert density["input_profile"]["skipped_row_count"] == 954
    assert density["input_profile"]["warnings_by_code"]["missing_source_text"] == 954

    assert output_checks["ok"] is False
    assert output_checks["input_profile"]["raw_row_count"] == 1000
    assert output_checks["input_profile"]["usable_source_count"] == 1000
    assert output_checks["input_profile"]["usable_source_ratio"] == 1.0
    assert output_checks["input_profile"]["skipped_row_count"] == 0
    assert output_checks["failure"]["type"] == "output_checks"
    assert output_checks["failure"]["failed_output_checks"]


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
