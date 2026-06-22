from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_deflection_snapshot_example.py"
EXAMPLE_PATH = ROOT / "docs/frontend/content_ops_faq_deflection_snapshot_example.json"


CLI = SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def test_generated_snapshot_example_matches_committed_file() -> None:
    assert CLI.render_snapshot_example() == EXAMPLE_PATH.read_text(encoding="utf-8")


def test_cli_writes_snapshot_example_to_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "snapshot.json"

    assert CLI.main(["--output", str(output)]) == 0

    assert output.read_text(encoding="utf-8") == CLI.render_snapshot_example()
    captured = capsys.readouterr()
    assert captured.out == f"wrote {output}\n"
    assert captured.err == ""


def test_check_passes_when_snapshot_example_is_current(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "snapshot.json"
    output.write_text(CLI.render_snapshot_example(), encoding="utf-8")

    assert CLI.main(["--output", str(output), "--check"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{output} is current\n"
    assert captured.err == ""


def test_check_fails_when_snapshot_example_is_stale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "snapshot.json"
    output.write_text("{}\n", encoding="utf-8")

    assert CLI.main(["--output", str(output), "--check"]) == 1

    assert output.read_text(encoding="utf-8") == "{}\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{output} is stale; run this generator to refresh it\n"


def test_check_fails_when_snapshot_example_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "missing" / "snapshot.json"

    assert CLI.main(["--output", str(output), "--check"]) == 1

    assert not output.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{output} is missing; run this generator to create it\n"
