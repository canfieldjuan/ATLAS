"""Unit tests for the repo-wide unit gate ratchet (#2035 / G1.1).

Drives scripts/check_unit_gate.py via its pure functions and its --report-file
mode, so the gate's LOGIC is tested without running the 30-minute suite.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_unit_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PYTEST_OUTPUT = """\
...F..E..                                                          [100%]
=================================== FAILURES ===================================
short test summary info
FAILED tests/security/test_network_ids.py::TestX::test_arp - AssertionError: boom
FAILED tests/test_call_workflow.py::TestCall::test_route[make a call] - x
FAILED tests/test_deflection.py::test_case[Credit card - Fees] - ValueError: bad
ERROR tests/test_competitive_intelligence.py - ImportError: no module named foo
1 failed, 100 passed, 1 error in 4.20s
"""


def test_parse_failing_nodes_keeps_params_with_spaces_and_dashes():
    nodes = gate.parse_failing_nodes(SAMPLE_PYTEST_OUTPUT)
    assert nodes == {
        "tests/security/test_network_ids.py::TestX::test_arp",
        "tests/test_call_workflow.py::TestCall::test_route[make a call]",  # space kept
        "tests/test_deflection.py::test_case[Credit card - Fees]",          # dash-in-param kept whole
        "tests/test_competitive_intelligence.py",  # bare node = collection error
    }


def test_parse_ignores_non_summary_lines():
    assert gate.parse_failing_nodes("...\n5 passed in 1.2s\n") == set()


def test_compare_subset_is_no_regression():
    regressions, fixed = gate.compare({"a", "b"}, {"a", "b", "c"})
    assert regressions == []
    assert fixed == ["c"]


def test_compare_new_failure_is_regression():
    regressions, fixed = gate.compare({"a", "NEW"}, {"a", "b"})
    assert regressions == ["NEW"]
    assert fixed == ["b"]


def test_load_baseline_ignores_comments_and_blanks(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("# header\n\ntests/x.py::t1\n  tests/y.py::t2  \n# trailer\n")
    assert gate.load_baseline(p) == {"tests/x.py::t1", "tests/y.py::t2"}


# --- integrity: pytest must have actually run (P1) --------------------------

def test_ensure_pytest_ran_raises_on_infrastructure_exit():
    with pytest.raises(RuntimeError):
        gate.ensure_pytest_ran(4)   # usage error
    with pytest.raises(RuntimeError):
        gate.ensure_pytest_ran(5)   # no tests collected
    gate.ensure_pytest_ran(0)       # all passed -> no raise
    gate.ensure_pytest_ran(1)       # tests failed (expected) -> no raise


# --- integrity: baseline may only shrink (P1) -------------------------------

def test_added_baseline_entries_detects_growth():
    assert gate.added_baseline_entries({"a", "b", "NEW"}, {"a", "b"}) == ["NEW"]
    assert gate.added_baseline_entries({"a"}, {"a", "b"}) == []   # shrink is fine


def test_removed_baseline_entries_detects_shrink():
    assert gate.removed_baseline_entries({"a"}, {"a", "b", "c"}) == ["b", "c"]
    assert gate.removed_baseline_entries({"a", "b", "NEW"}, {"a", "b"}) == []


def test_pytest_target_files_extracts_test_file_args():
    assert gate.pytest_target_files([
        "tests/test_a.py",
        "-m",
        "not integration",
        "./tests/test_b.py::test_case[param]",
        "--tb=no",
        "-q",
    ]) == {"tests/test_a.py", "tests/test_b.py"}


def _run(args, tmp_path, report):
    rf = tmp_path / "report.txt"
    rf.write_text(report)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_unit_gate.py"),
         "--report-file", str(rf), *args],
        capture_output=True, text=True,
    )


def _run_gate_with_pytest_report(args, report, monkeypatch):
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout=report, stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    return gate.main(args), calls


_EXACT_BASELINE = (
    "tests/security/test_network_ids.py::TestX::test_arp\n"
    "tests/test_call_workflow.py::TestCall::test_route[make a call]\n"
    "tests/test_deflection.py::test_case[Credit card - Fees]\n"
    "tests/test_competitive_intelligence.py\n"
)


def test_cli_exit0_when_baseline_exactly_matches(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(_EXACT_BASELINE)
    r = _run(["--baseline", str(baseline)], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "exactly matches" in r.stdout


def test_cli_exit1_on_stale_baseline_entry(tmp_path):
    # a node that now PASSES but is still in the baseline -> ratchet must shrink
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(_EXACT_BASELINE + "tests/now_passing.py::t\n")
    r = _run(["--baseline", str(baseline)], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STALE" in r.stdout
    assert "tests/now_passing.py::t" in r.stdout


def test_cli_exit1_on_regression(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("tests/security/test_network_ids.py::TestX::test_arp\n")
    r = _run(["--baseline", str(baseline)], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REGRESSION" in r.stdout


def test_cli_exit2_on_missing_baseline(tmp_path):
    r = _run(["--baseline", str(tmp_path / "nope.txt")], tmp_path, SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 2


def test_cli_exit3_on_baseline_growth_vs_base(tmp_path):
    # PR baseline adds a node the base baseline does not have -> ratchet violation
    base = tmp_path / "base.txt"
    base.write_text("tests/security/test_network_ids.py::TestX::test_arp\n")
    pr = tmp_path / "pr.txt"
    pr.write_text(
        "tests/security/test_network_ids.py::TestX::test_arp\n"
        "tests/sneaked_in_new_failure.py::t\n"
    )
    r = _run(["--baseline", str(pr), "--base-baseline", str(base)], tmp_path,
             SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "RATCHET VIOLATION" in r.stdout


def test_cli_exit2_when_growth_only_cannot_prove_baseline_shrink(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_unit_gate.py"),
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
            "--growth-only",
        ],
        capture_output=True,
        text=True,
    )

    assert r.returncode == 2, r.stdout + r.stderr
    assert "--growth-only has no pytest report" in r.stderr
    assert "tests/test_a.py::old_failure" in r.stderr


def test_cli_exit2_when_scoped_run_omits_removed_baseline_node_file(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")
    selected = tmp_path / "selected.txt"
    selected.write_text("tests/test_b.py\n")

    r = _run(
        [
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
            "--selected-files",
            str(selected),
        ],
        tmp_path,
        "FAILED tests/test_b.py::still_fails - boom\n",
    )

    assert r.returncode == 2, r.stdout + r.stderr
    assert "selected-files omitted removed baseline node file" in r.stderr
    assert "tests/test_a.py" in r.stderr


def test_cli_exit2_when_unscoped_report_claims_baseline_shrink(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")

    r = _run(
        [
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
        ],
        tmp_path,
        "FAILED tests/test_b.py::still_fails - boom\n",
    )

    assert r.returncode == 2, r.stdout + r.stderr
    assert "--report-file cannot prove a baseline shrink" in r.stderr
    assert "tests/test_a.py::old_failure" in r.stderr


def test_cli_exit2_when_scoped_report_claims_baseline_shrink(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")
    selected = tmp_path / "selected.txt"
    selected.write_text("tests/test_a.py\ntests/test_b.py\n")

    r = _run(
        [
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
            "--selected-files",
            str(selected),
        ],
        tmp_path,
        "FAILED tests/test_b.py::still_fails - boom\n",
    )

    assert r.returncode == 2, r.stdout + r.stderr
    assert "--report-file cannot prove a baseline shrink" in r.stderr
    assert "tests/test_a.py::old_failure" in r.stderr


def test_cli_exit0_when_scoped_run_proves_removed_node_passes(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")
    selected = tmp_path / "selected.txt"
    selected.write_text("tests/test_a.py\ntests/test_b.py\n")

    returncode, calls = _run_gate_with_pytest_report(
        [
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
            "--selected-files",
            str(selected),
            "--pytest-args",
            "tests/test_a.py",
            "tests/test_b.py",
            "-m",
            "not integration",
        ],
        "FAILED tests/test_b.py::still_fails - boom\n",
        monkeypatch,
    )
    captured = capsys.readouterr()

    assert returncode == 0, captured.out + captured.err
    assert calls
    assert "tests/test_a.py" in calls[0]
    assert "tests/test_b.py" in calls[0]
    assert "exactly matches" in captured.out


def test_cli_exit2_when_selected_scope_not_bound_to_pytest_args(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")
    selected = tmp_path / "selected.txt"
    selected.write_text("tests/test_a.py\ntests/test_b.py\n")

    returncode, calls = _run_gate_with_pytest_report(
        [
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
            "--selected-files",
            str(selected),
            "--pytest-args",
            "tests/test_b.py",
        ],
        "FAILED tests/test_b.py::still_fails - boom\n",
        monkeypatch,
    )
    captured = capsys.readouterr()

    assert returncode == 2, captured.out + captured.err
    assert calls == []
    assert "selected file(s) missing from --pytest-args" in captured.err
    assert "tests/test_a.py" in captured.err


def test_cli_exit1_when_removed_baseline_node_still_fails(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.txt"
    base.write_text(
        "tests/test_a.py::old_failure\n"
        "tests/test_b.py::still_fails\n"
    )
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/test_b.py::still_fails\n")
    selected = tmp_path / "selected.txt"
    selected.write_text("tests/test_a.py\ntests/test_b.py\n")

    returncode, calls = _run_gate_with_pytest_report(
        [
            "--baseline",
            str(pr),
            "--base-baseline",
            str(base),
            "--selected-files",
            str(selected),
            "--pytest-args",
            "tests/test_a.py",
            "tests/test_b.py",
        ],
        "FAILED tests/test_a.py::old_failure - boom\n"
        "FAILED tests/test_b.py::still_fails - boom\n",
        monkeypatch,
    )
    captured = capsys.readouterr()

    assert returncode == 1, captured.out + captured.err
    assert calls
    assert "tests/test_a.py" in calls[0]
    assert "tests/test_b.py" in calls[0]
    assert "REGRESSION" in captured.out
    assert "tests/test_a.py::old_failure" in captured.out


def test_cli_report_without_baseline_shrink_still_gates_fixture_output(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(_EXACT_BASELINE)

    r = _run(["--baseline", str(baseline)], tmp_path, SAMPLE_PYTEST_OUTPUT)

    assert r.returncode == 0, r.stdout + r.stderr
    assert "exactly matches" in r.stdout


def test_cli_empty_base_baseline_allows_initial_seed(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("")  # base has no baseline yet
    pr = tmp_path / "pr.txt"
    pr.write_text("tests/security/test_network_ids.py::TestX::test_arp\n"
                  "tests/test_call_workflow.py::TestCall::test_route[make a call]\n"
                  "tests/test_deflection.py::test_case[Credit card - Fees]\n"
                  "tests/test_competitive_intelligence.py\n")
    r = _run(["--baseline", str(pr), "--base-baseline", str(base)], tmp_path,
             SAMPLE_PYTEST_OUTPUT)
    assert r.returncode == 0, r.stdout + r.stderr  # seed allowed, no regression


def test_committed_baseline_parses_and_is_sorted_unique():
    path = ROOT / "tests" / "unit_gate_baseline.txt"
    baseline = gate.load_baseline(path)
    assert len(baseline) >= 150
    lines = [l.strip() for l in path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines == sorted(lines)
    assert len(lines) == len(set(lines))
