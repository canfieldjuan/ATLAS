"""Unit tests for the repo-wide unit gate ratchet (#2035 / G1.1).

Drives scripts/check_unit_gate.py via its pure functions and its --report-file
mode, so the gate's LOGIC is tested without running the 30-minute suite.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_unit_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PYTEST_OUTPUT = """\
...F..E..                                                          [100%]
=================================== FAILURES ===================================
short test summary info
FAILED tests/security/test_network_ids.py::TestX::test_arp - AssertionError: boom
FAILED tests/test_call_workflow.py::TestCall::test_route[make a call] - x
ERROR tests/test_competitive_intelligence.py - ImportError: no module named foo
1 failed, 100 passed, 1 error in 4.20s
"""


def test_parse_failing_nodes_handles_failed_error_and_spaced_params():
    nodes = gate.parse_failing_nodes(SAMPLE_PYTEST_OUTPUT)
    assert nodes == {
        "tests/security/test_network_ids.py::TestX::test_arp",
        "tests/test_call_workflow.py::TestCall::test_route[make a call]",  # space kept
        "tests/test_competitive_intelligence.py",  # bare node = collection error
    }


def test_parse_ignores_non_summary_lines():
    # a passing run has no FAILED/ERROR summary lines
    assert gate.parse_failing_nodes("...\n5 passed in 1.2s\n") == set()


def test_compare_subset_is_no_regression():
    baseline = {"a", "b", "c"}
    regressions, fixed = gate.compare({"a", "b"}, baseline)
    assert regressions == []           # failing subset of baseline -> clean
    assert fixed == ["c"]              # c no longer fails -> ratchet-shrink hint


def test_compare_new_failure_is_regression():
    regressions, fixed = gate.compare({"a", "NEW"}, {"a", "b"})
    assert regressions == ["NEW"]      # NEW not in baseline -> gate must fail
    assert fixed == ["b"]


def test_load_baseline_ignores_comments_and_blanks(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("# header\n\ntests/x.py::t1\n  tests/y.py::t2  \n# trailer\n")
    assert gate.load_baseline(p) == {"tests/x.py::t1", "tests/y.py::t2"}


def _run(args, tmp_path, report):
    rf = tmp_path / "report.txt"
    rf.write_text(report)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_unit_gate.py"),
         "--report-file", str(rf), *args],
        capture_output=True, text=True,
    )


def test_cli_exit0_when_failing_subset_of_baseline(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(
        "tests/security/test_network_ids.py::TestX::test_arp\n"
        "tests/test_call_workflow.py::TestCall::test_route[make a call]\n"
        "tests/test_competitive_intelligence.py\n"
        "tests/extra_known_fail.py::t\n"  # baseline superset -> the extra is "fixed"
    )
    r = _run(["--baseline", str(baseline)], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no regression" in r.stdout


def test_cli_exit1_on_regression(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("tests/security/test_network_ids.py::TestX::test_arp\n")
    r = _run(["--baseline", str(baseline)], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REGRESSION" in r.stdout
    assert "tests/test_competitive_intelligence.py" in r.stdout


def test_cli_exit2_on_missing_baseline(tmp_path):
    r = _run(["--baseline", str(tmp_path / "nope.txt")], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 2


def test_committed_baseline_parses_and_is_sorted_unique():
    baseline = gate.load_baseline(ROOT / "tests" / "unit_gate_baseline.txt")
    assert len(baseline) >= 150  # the large stable pre-existing set
    lines = [l.strip() for l in (ROOT / "tests" / "unit_gate_baseline.txt")
             .read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert lines == sorted(lines)          # committed sorted
    assert len(lines) == len(set(lines))   # no dupes
