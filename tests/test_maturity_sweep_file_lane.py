from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT = SCRIPTS_DIR / "maturity_sweep_file_lane.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SPEC = importlib.util.spec_from_file_location("maturity_sweep_file_lane", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
lane = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lane)


def test_iter_python_files_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"

    with pytest.raises(SystemExit) as excinfo:
        list(lane._iter_python_files([str(missing)]))

    assert "invalid explicit lane path(s):" in str(excinfo.value)
    assert "missing.py: missing" in str(excinfo.value)


def test_iter_python_files_rejects_non_python_path(tmp_path: Path) -> None:
    text_file = tmp_path / "lane.txt"
    text_file.write_text("not python\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        list(lane._iter_python_files([str(text_file)]))

    assert "lane.txt: not a Python file" in str(excinfo.value)


def test_sweep_files_deduplicates_explicit_inputs(tmp_path: Path, monkeypatch) -> None:
    product_file = tmp_path / "prod.py"
    product_file.write_text("def run():\n    return 1\n", encoding="utf-8")
    analyzed = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lane.maturity_sweep, "is_test_path", lambda path: False)
    monkeypatch.setattr(lane.maturity_sweep, "index_tests", lambda tests_root: ({}, ""))

    def fake_analyze_file(path, test_sources, all_test_text):
        analyzed.append(path)
        return SimpleNamespace(path=path.as_posix(), score=0, counts={}, findings=[])

    monkeypatch.setattr(lane.maturity_sweep, "analyze_file", fake_analyze_file)

    results = lane.sweep_files([str(product_file), str(product_file)], "tests")

    assert len(results) == 1
    assert analyzed == [product_file]


def test_json_mode_returns_nonzero_when_ratchet_fails(monkeypatch, capsys) -> None:
    result = SimpleNamespace(path="prod.py", score=9, counts={}, findings=[])

    monkeypatch.setattr(lane, "sweep_files", lambda paths, tests_root: [result])
    monkeypatch.setattr(lane.maturity_sweep, "load_baseline", lambda baseline: {})
    monkeypatch.setattr(
        lane.maturity_sweep,
        "ratchet_failures",
        lambda results, baseline, min_score, sensitive_globs: [object()],
    )

    assert lane.main(["prod.py", "--baseline", "baseline.json", "--json"]) == 1
    assert '"path": "prod.py"' in capsys.readouterr().out


def test_text_mode_propagates_print_report_exit_code(monkeypatch) -> None:
    result = SimpleNamespace(path="prod.py", score=9, counts={}, findings=[])

    monkeypatch.setattr(lane, "sweep_files", lambda paths, tests_root: [result])
    monkeypatch.setattr(lane.maturity_sweep, "load_baseline", lambda baseline: {})
    monkeypatch.setattr(
        lane.maturity_sweep,
        "ratchet_failures",
        lambda results, baseline, min_score, sensitive_globs: [object()],
    )
    monkeypatch.setattr(
        lane.maturity_sweep,
        "print_report",
        lambda results, top, min_score, ratchet, baseline_path: 1,
    )

    assert lane.main(["prod.py", "--baseline", "baseline.json"]) == 1
