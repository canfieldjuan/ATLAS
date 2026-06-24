from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_deflection_snapshot_example.py"
REPORT_EXAMPLE_PATH = (
    ROOT / "docs/frontend/content_ops_faq_deflection_report_example.json"
)
SNAPSHOT_EXAMPLE_PATH = (
    ROOT / "docs/frontend/content_ops_faq_deflection_snapshot_example.json"
)


CLI = SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def test_generated_examples_match_committed_files() -> None:
    assert CLI.render_report_example() == REPORT_EXAMPLE_PATH.read_text(
        encoding="utf-8"
    )
    assert CLI.render_snapshot_example() == SNAPSHOT_EXAMPLE_PATH.read_text(
        encoding="utf-8"
    )


def test_snapshot_example_is_derived_from_report_example() -> None:
    report_payload = CLI.build_report_example_payload()

    assert CLI.build_snapshot_example_payload(report_payload) == (
        CLI.build_snapshot_example_payload()
    )


def test_demo_records_are_synthetic() -> None:
    records = CLI.synthetic_deflection_demo_records()

    assert records
    assert all(record["source_id"].startswith("ticket-") for record in records)
    assert all(
        "@" not in str(value) for record in records for value in record.values()
    )
    assert all("example.com" not in str(record.get("text", "")) for record in records)


def test_cli_writes_snapshot_example_to_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"

    assert (
        CLI.main([
            "--report-output",
            str(report_output),
            "--snapshot-output",
            str(snapshot_output),
        ])
        == 0
    )

    assert report_output.read_text(encoding="utf-8") == CLI.render_report_example()
    assert snapshot_output.read_text(encoding="utf-8") == CLI.render_snapshot_example()
    captured = capsys.readouterr()
    assert captured.out == f"wrote {report_output}\nwrote {snapshot_output}\n"
    assert captured.err == ""


def test_check_passes_when_snapshot_example_is_current(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    report_output.write_text(CLI.render_report_example(), encoding="utf-8")
    snapshot_output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert (
        CLI.main([
            "--report-output",
            str(report_output),
            "--snapshot-output",
            str(snapshot_output),
            "--check",
        ])
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == (
        f"{report_output} is current\n{snapshot_output} is current\n"
    )
    assert captured.err == ""


def test_check_fails_when_report_example_is_stale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    report_output.write_text("{}\n", encoding="utf-8")
    snapshot_output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert (
        CLI.main([
            "--report-output",
            str(report_output),
            "--snapshot-output",
            str(snapshot_output),
            "--check",
        ])
        == 1
    )

    assert report_output.read_text(encoding="utf-8") == "{}\n"
    captured = capsys.readouterr()
    assert captured.out == f"{snapshot_output} is current\n"
    assert captured.err == (
        f"{report_output} is stale; run this generator to refresh it\n"
    )


def test_check_fails_when_snapshot_example_is_stale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    report_output.write_text(CLI.render_report_example(), encoding="utf-8")
    snapshot_output.write_text("{}\n", encoding="utf-8")

    assert (
        CLI.main([
            "--report-output",
            str(report_output),
            "--snapshot-output",
            str(snapshot_output),
            "--check",
        ])
        == 1
    )

    assert snapshot_output.read_text(encoding="utf-8") == "{}\n"
    captured = capsys.readouterr()
    assert captured.out == f"{report_output} is current\n"
    assert captured.err == (
        f"{snapshot_output} is stale; run this generator to refresh it\n"
    )


def test_check_fails_when_report_example_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "missing" / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    snapshot_output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert (
        CLI.main([
            "--report-output",
            str(report_output),
            "--snapshot-output",
            str(snapshot_output),
            "--check",
        ])
        == 1
    )

    assert not report_output.exists()
    captured = capsys.readouterr()
    assert captured.out == f"{snapshot_output} is current\n"
    assert captured.err == (
        f"{report_output} is missing; run this generator to create it\n"
    )


def test_check_fails_when_snapshot_example_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "missing" / "snapshot.json"
    report_output.write_text(CLI.render_report_example(), encoding="utf-8")

    assert (
        CLI.main([
            "--report-output",
            str(report_output),
            "--snapshot-output",
            str(snapshot_output),
            "--check",
        ])
        == 1
    )

    assert not snapshot_output.exists()
    captured = capsys.readouterr()
    assert captured.out == f"{report_output} is current\n"
    assert captured.err == (
        f"{snapshot_output} is missing; run this generator to create it\n"
    )
