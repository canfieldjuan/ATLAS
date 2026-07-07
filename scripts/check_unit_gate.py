#!/usr/bin/env python3
"""Repo-wide unit gate with a known-failures baseline (ratchet).

The per-area CI checks are path-filtered and run hand-maintained test lists, so
a test file that no workflow enrolls can pass CI while never executing
(#2035 / G1.1, G2). This gate runs the WHOLE unit suite on every PR and fails
only on a REGRESSION -- a failing/errored node that is not already in the
committed baseline `tests/unit_gate_baseline.txt`. Pre-existing failures are
tolerated (the repo has a large, stable set of them); the baseline can only
shrink over time (a node that starts passing is reported for removal).

Usage:
  # run the suite and gate against the baseline (CI mode):
  python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt

  # gate a pre-captured pytest summary (used by the unit tests; no re-run):
  python scripts/check_unit_gate.py --baseline B --report-file pytest_out.txt

  # maintenance: overwrite the baseline with the current failing set:
  python scripts/check_unit_gate.py --baseline B --update-baseline

Exit codes: 0 = no regression (failing set is a subset of the baseline);
1 = regression (>=1 failing node not in the baseline); 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# pytest short-summary lines look like:
#   FAILED tests/foo.py::TestX::test_y - AssertionError: ...
#   ERROR tests/bar.py - ImportError: ...        (collection error, no ::)
# Capture the node id non-greedily, stopping before an optional " - <message>".
# `.+?` (not `\S+`) so parametrized ids that contain spaces (e.g. "test[a b]")
# are kept whole.
_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>.+?)(?:\s+-\s+.*)?$")

DEFAULT_PYTEST_ARGS = ["tests/", "-m", "not integration and not e2e",
                       "--continue-on-collection-errors", "-rfE", "--tb=no",
                       "-q", "-p", "no:cacheprovider"]


def parse_failing_nodes(pytest_output: str) -> set[str]:
    """Extract the set of failing/errored node ids from a pytest run's stdout."""
    nodes: set[str] = set()
    for line in pytest_output.splitlines():
        m = _SUMMARY_RE.match(line.rstrip())
        if m:
            nodes.add(m.group("node").strip())
    return nodes


def load_baseline(path: Path) -> set[str]:
    """Read baseline node ids, ignoring blank lines and `#` comments."""
    nodes: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            nodes.add(line)
    return nodes


def compare(failing: set[str], baseline: set[str]) -> tuple[list[str], list[str]]:
    """Return (regressions, fixed) as sorted lists.

    regressions = failing not accounted for by the baseline (gate FAILS on any).
    fixed       = baseline entries that no longer fail (ratchet-shrink advisory).
    """
    return sorted(failing - baseline), sorted(baseline - failing)


def run_pytest(pytest_args: list[str]) -> str:
    """Run pytest and return combined stdout+stderr. pytest's non-zero exit on
    (baseline) failures is expected and captured -- the gate verdict is this
    script's own exit code, decided by compare(), not by pytest's."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args],
        capture_output=True, text=True,
    )
    return proc.stdout + proc.stderr


def write_baseline(path: Path, failing: set[str], *, header: str) -> None:
    body = "\n".join(sorted(failing))
    path.write_text(header.rstrip() + "\n" + body + "\n", encoding="utf-8")


BASELINE_HEADER = """\
# Unit gate known-failures baseline (#2035 / G1.1). Sorted pytest node ids that
# fail or error in the repo-wide unit suite today. The gate
# (.github/workflows/unit_gate.yml) fails only on a node NOT in this list.
# This ledger may only shrink: remediate failures, then remove them here.
# Regenerate the failing set with: scripts/check_unit_gate.py --update-baseline
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Repo-wide unit gate (ratchet baseline).")
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--report-file", type=Path,
                    help="Gate a pre-captured pytest summary instead of running pytest.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Overwrite the baseline with the current failing set and exit 0.")
    ap.add_argument("--pytest-args", nargs=argparse.REMAINDER,
                    help="Override the default pytest args (everything after this flag).")
    args = ap.parse_args()

    if args.report_file is not None:
        if not args.report_file.exists():
            print(f"report file not found: {args.report_file}", file=sys.stderr)
            return 2
        output = args.report_file.read_text(encoding="utf-8")
    else:
        output = run_pytest(args.pytest_args or DEFAULT_PYTEST_ARGS)

    failing = parse_failing_nodes(output)

    if args.update_baseline:
        write_baseline(args.baseline, failing, header=BASELINE_HEADER)
        print(f"wrote {len(failing)} node(s) to {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 2
    baseline = load_baseline(args.baseline)
    regressions, fixed = compare(failing, baseline)

    print(f"unit gate: {len(failing)} failing/errored node(s); "
          f"baseline={len(baseline)}; regressions={len(regressions)}; "
          f"newly-passing={len(fixed)}")
    if fixed:
        print("\nbaseline entries that now PASS (remove them from the baseline):")
        for node in fixed:
            print(f"  - {node}")
    if regressions:
        print("\nREGRESSION -- failing node(s) not in the baseline:")
        for node in regressions:
            print(f"  {node}")
        print("\nFix the test, or (if intentional) justify and add to the "
              "baseline. The gate fails on any un-baselined failure.")
        return 1

    print("\nOK: no regression -- every failing node is accounted for by the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
