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


def test_generated_snapshot_example_matches_committed_file() -> None:
    report_payload = CLI.producer_deflection_report_payload()
    snapshot_payload = CLI.build_snapshot_example_payload()

    assert CLI.render_report_example(report_payload) == REPORT_EXAMPLE_PATH.read_text(
        encoding="utf-8"
    )
    assert CLI.render_snapshot_example(
        snapshot_payload
    ) == SNAPSHOT_EXAMPLE_PATH.read_text(encoding="utf-8")
    assert snapshot_payload == CLI.build_deflection_snapshot(
        report_payload,
        top_n=CLI.SNAPSHOT_TOP_N,
    ).as_dict()


def test_synthetic_demo_input_is_moderate_volume_and_public_safe() -> None:
    rows = CLI.synthetic_support_ticket_rows()
    source_ids = [str(row["source_id"]) for row in rows]
    encoded_rows = repr(rows).lower()

    assert len(rows) == 450
    assert all("source_weight" not in row for row in rows)
    assert len(source_ids) == len(set(source_ids))
    assert all(source_id.startswith("synthetic-") for source_id in source_ids)
    assert {row["source_type"] for row in rows} == {"support_ticket"}
    assert "@" not in encoded_rows
    assert "555-" not in encoded_rows
    assert "account " not in encoded_rows


def test_generated_demo_example_carries_coherent_marketing_scale_volume() -> None:
    report_payload = CLI.producer_deflection_report_payload()
    snapshot_payload = CLI.build_deflection_snapshot(
        report_payload,
        top_n=CLI.SNAPSHOT_TOP_N,
    ).as_dict()
    sections = {
        section["id"]: section["data"]
        for section in report_payload["report_model"]["sections"]
    }

    assert snapshot_payload["summary"]["repeat_ticket_count"] >= 300
    assert snapshot_payload["top_questions"][0]["ticket_count"] >= 90
    assert snapshot_payload["summary"]["generated"] > CLI.SNAPSHOT_TOP_N
    assert len(snapshot_payload["top_questions"]) == CLI.SNAPSHOT_TOP_N
    assert snapshot_payload["locked_questions"]
    for question in snapshot_payload["locked_questions"]:
        assert set(question) == {"rank", "ticket_count"}
        assert "question" not in question
    assert snapshot_payload["top_blind_spots"]
    assert sections["support_tax"]["repeat_ticket_count"] == snapshot_payload[
        "summary"
    ]["repeat_ticket_count"]
    assert report_payload["summary"]["ticket_source_count"] == snapshot_payload[
        "summary"
    ]["repeat_ticket_count"]
    evidence_rows = report_payload["evidence_export"]["evidence_rows"]
    complete_evidence = sections["complete_evidence"]
    assert complete_evidence["evidence_row_count"] == snapshot_payload[
        "summary"
    ]["repeat_ticket_count"]
    assert complete_evidence["source_id_count"] == snapshot_payload[
        "summary"
    ]["repeat_ticket_count"]
    assert len(evidence_rows) == snapshot_payload["summary"]["repeat_ticket_count"]
    for item in sections["ranked_questions"]["rows"]:
        matching_sources = [
            row for row in evidence_rows
            if row["question"] == item["question"]
        ]
        assert len(matching_sources) == item["ticket_count"]
        assert item["source_proof"] == f"{item['ticket_count']} source tickets"
    assert sections["top_unresolved_repeats"]["top_item_count"] >= 3
    assert sections["drafted_resolutions"]["top_item_count"] >= 3
    assert sections["already_covered_still_recurring"]["top_item_count"] >= 1
    assert sections["already_covered_still_recurring"]["items"][0][
        "status"
    ] == "Already covered but still recurring"
    assert sections["priority_fix_queue"]["status_counts"] == {
        "Already covered but still recurring": 1,
        "Draft ready": 3,
        "Needs answer": 3,
    }
    assert len(sections["outcome_diagnostics"]["rows"]) == snapshot_payload[
        "summary"
    ]["generated"]


def test_cli_writes_snapshot_example_to_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"

    assert CLI.main([
        "--report-output",
        str(report_output),
        "--snapshot-output",
        str(snapshot_output),
    ]) == 0

    assert report_output.read_text(encoding="utf-8") == CLI.render_report_example()
    assert snapshot_output.read_text(encoding="utf-8") == CLI.render_snapshot_example()
    captured = capsys.readouterr()
    assert captured.out == f"wrote {report_output}\nwrote {snapshot_output}\n"
    assert captured.err == ""


def test_deprecated_output_alias_writes_sibling_report_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_output = tmp_path / "snapshot.json"
    report_output = tmp_path / "snapshot.report.json"

    assert CLI.main(["--output", str(snapshot_output)]) == 0

    assert report_output.read_text(encoding="utf-8") == CLI.render_report_example()
    assert snapshot_output.read_text(encoding="utf-8") == CLI.render_snapshot_example()
    captured = capsys.readouterr()
    assert captured.out == f"wrote {report_output}\nwrote {snapshot_output}\n"
    assert captured.err == ""


def test_snapshot_output_without_report_output_writes_sibling_report_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_output = tmp_path / "demo-snapshot.json"
    report_output = tmp_path / "demo-snapshot.report.json"

    assert CLI.main(["--snapshot-output", str(snapshot_output)]) == 0

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

    assert CLI.main([
        "--report-output",
        str(report_output),
        "--snapshot-output",
        str(snapshot_output),
        "--check",
    ]) == 0

    captured = capsys.readouterr()
    assert (
        captured.out
        == f"{report_output} is current\n{snapshot_output} is current\n"
    )
    assert captured.err == ""


def test_deprecated_output_alias_check_uses_sibling_report_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_output = tmp_path / "snapshot.json"
    report_output = tmp_path / "snapshot.report.json"
    report_output.write_text(CLI.render_report_example(), encoding="utf-8")
    snapshot_output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert CLI.main(["--output", str(snapshot_output), "--check"]) == 0

    captured = capsys.readouterr()
    assert (
        captured.out
        == f"{report_output} is current\n{snapshot_output} is current\n"
    )
    assert captured.err == ""


def test_check_fails_when_snapshot_example_is_stale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    report_output.write_text(CLI.render_report_example(), encoding="utf-8")
    snapshot_output.write_text("{}\n", encoding="utf-8")

    assert CLI.main([
        "--report-output",
        str(report_output),
        "--snapshot-output",
        str(snapshot_output),
        "--check",
    ]) == 1

    assert snapshot_output.read_text(encoding="utf-8") == "{}\n"
    captured = capsys.readouterr()
    assert captured.out == f"{report_output} is current\n"
    assert (
        captured.err
        == f"{snapshot_output} is stale; run this generator to refresh it\n"
    )


def test_check_fails_when_snapshot_example_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "missing" / "snapshot.json"
    report_output.write_text(CLI.render_report_example(), encoding="utf-8")

    assert CLI.main([
        "--report-output",
        str(report_output),
        "--snapshot-output",
        str(snapshot_output),
        "--check",
    ]) == 1

    assert not snapshot_output.exists()
    captured = capsys.readouterr()
    assert captured.out == f"{report_output} is current\n"
    assert (
        captured.err
        == f"{snapshot_output} is missing; run this generator to create it\n"
    )


def test_check_fails_when_report_example_is_stale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    report_output.write_text("{}\n", encoding="utf-8")
    snapshot_output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert CLI.main([
        "--report-output",
        str(report_output),
        "--snapshot-output",
        str(snapshot_output),
        "--check",
    ]) == 1

    assert report_output.read_text(encoding="utf-8") == "{}\n"
    captured = capsys.readouterr()
    assert captured.out == f"{snapshot_output} is current\n"
    assert (
        captured.err
        == f"{report_output} is stale; run this generator to refresh it\n"
    )


def test_check_fails_when_report_example_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_output = tmp_path / "missing" / "report.json"
    snapshot_output = tmp_path / "snapshot.json"
    snapshot_output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert CLI.main([
        "--report-output",
        str(report_output),
        "--snapshot-output",
        str(snapshot_output),
        "--check",
    ]) == 1

    assert not report_output.exists()
    captured = capsys.readouterr()
    assert captured.out == f"{snapshot_output} is current\n"
    assert (
        captured.err
        == f"{report_output} is missing; run this generator to create it\n"
    )
