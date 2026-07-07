#!/usr/bin/env python3
"""Repo-wide unit gate with a known-failures baseline (ratchet).

The per-area CI checks are path-filtered and run hand-maintained test lists, so
a test file that no workflow enrolls can pass CI while never executing
(#2035 / G1.1, G2). This gate runs the WHOLE unit suite on every PR and fails
only on a REGRESSION -- a failing/errored node that is not already in the
committed baseline `tests/unit_gate_baseline.txt`. Pre-existing failures are
tolerated (the repo has a large, stable set of them); the baseline can only
shrink over time (a node that starts passing is reported for removal).

Integrity guards:
  * pytest exit status is checked -- an infrastructure/usage/internal error
    (exit not in {0,1}) fails the gate instead of parsing to an empty failing
    set and going silently green (a suite that never ran must not pass).
  * `--base-baseline` rejects baseline GROWTH: a PR cannot add its own new
    failure to the ledger to pass, once a base-branch baseline exists.

Usage:
  # run the suite and gate against the baseline (CI mode):
  python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt \
      --base-baseline base_baseline.txt

  # gate a pre-captured pytest summary (used by the unit tests; no re-run):
  python scripts/check_unit_gate.py --baseline B --report-file pytest_out.txt

  # maintenance: overwrite the baseline with the current failing set:
  python scripts/check_unit_gate.py --baseline B --update-baseline

Exit codes: 0 = no regression; 1 = regression (failing node not in baseline);
2 = infrastructure/usage/IO error; 3 = baseline grew (ratchet violation).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# pytest short-summary lines look like:
#   FAILED tests/foo.py::TestX::test_y - AssertionError: ...
#   FAILED tests/foo.py::test_p[Credit card - Fees] - ValueError: ...
#   ERROR tests/bar.py - ImportError: ...        (collection error, no ::)
# The node id is non-space chars, then an OPTIONAL [..params..] that MAY contain
# spaces and dashes; the " - <message>" tail is not part of the id. Matching the
# bracket group explicitly (rather than splitting on " - ") keeps parametrized
# ids whole even when the params contain " - ". (Nested [] inside params is not
# handled -- vanishingly rare for pytest ids.)
_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>[^\s\[]+(?:\[[^\]]*\])?)")

# Normal pytest exit statuses for a gate run: 0 = all passed, 1 = tests failed
# (expected -- the baseline exists). 2 interrupted / 3 internal / 4 usage /
# 5 no-tests-collected all mean the suite did not run cleanly.
_OK_PYTEST_EXIT = frozenset((0, 1))

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


def added_baseline_entries(pr_baseline: set[str], base_baseline: set[str]) -> list[str]:
    """Node ids the PR adds to the baseline vs the base branch (ratchet growth)."""
    return sorted(pr_baseline - base_baseline)


def ensure_pytest_ran(returncode: int) -> None:
    """Raise when pytest did not finish as a normal pass/fail run.

    Without this, an infrastructure/usage error (e.g. a plugin crash, a bad
    arg, or no tests collected) produces no FAILED/ERROR summary lines, parses
    to an empty failing set, and the gate would go GREEN having run nothing."""
    if returncode not in _OK_PYTEST_EXIT:
        raise RuntimeError(
            f"pytest exited {returncode}, not a normal pass/fail run "
            f"(expected 0 or 1). The suite did not run cleanly; the gate "
            f"cannot trust an empty failing set."
        )


def run_pytest(pytest_args: list[str]) -> tuple[str, int]:
    """Run pytest and return (combined stdout+stderr, returncode). pytest's
    non-zero exit on (baseline) failures is expected -- ensure_pytest_ran()
    distinguishes a real 1-failures run from an infrastructure error."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args],
        capture_output=True, text=True,
    )
    return proc.stdout + proc.stderr, proc.returncode


def write_baseline(path: Path, failing: set[str], *, header: str) -> None:
    body = "\n".join(sorted(failing))
    path.write_text(header.rstrip() + "\n" + body + "\n", encoding="utf-8")


BASELINE_HEADER = """\
# Unit gate known-failures baseline (#2035 / G1.1). Sorted pytest node ids that
# fail or error in the repo-wide unit suite today. The gate
# (.github/workflows/unit_gate.yml) fails only on a node NOT in this list, and
# rejects any PR that GROWS this list (ratchet: it may only shrink).
# Remediate failures, then remove them here. Regenerate the current failing set
# with: scripts/check_unit_gate.py --update-baseline
"""


def _fail_growth(added: list[str]) -> int:
    print(f"\nRATCHET VIOLATION -- this PR adds {len(added)} node(s) to the "
          f"baseline (it may only shrink):")
    for node in added:
        print(f"  + {node}")
    print("\nA new failure must be fixed, not baselined. Baseline additions "
          "belong in a dedicated, reviewed change, not a feature PR.")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Repo-wide unit gate (ratchet baseline).")
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--base-baseline", type=Path,
                    help="Base-branch baseline; reject additions vs it (ratchet). "
                         "An empty/absent file means no base baseline yet (initial seed).")
    ap.add_argument("--report-file", type=Path,
                    help="Gate a pre-captured pytest summary instead of running pytest.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Overwrite the baseline with the current failing set and exit 0.")
    ap.add_argument("--pytest-args", nargs=argparse.REMAINDER,
                    help="Override the default pytest args (everything after this flag).")
    args = ap.parse_args()

    if not args.baseline.exists() and not args.update_baseline:
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 2

    # Ratchet growth guard (fast, no pytest): a PR may not add baseline entries
    # once the base branch has a baseline. An empty base file = initial seed.
    if args.base_baseline is not None and args.base_baseline.exists():
        base = load_baseline(args.base_baseline)
        if base:
            added = added_baseline_entries(load_baseline(args.baseline), base)
            if added:
                return _fail_growth(added)

    if args.report_file is not None:
        if not args.report_file.exists():
            print(f"report file not found: {args.report_file}", file=sys.stderr)
            return 2
        output = args.report_file.read_text(encoding="utf-8")
    else:
        output, returncode = run_pytest(args.pytest_args or DEFAULT_PYTEST_ARGS)
        try:
            ensure_pytest_ran(returncode)
        except RuntimeError as exc:
            print(f"unit gate: {exc}", file=sys.stderr)
            return 2

    failing = parse_failing_nodes(output)

    if args.update_baseline:
        write_baseline(args.baseline, failing, header=BASELINE_HEADER)
        print(f"wrote {len(failing)} node(s) to {args.baseline}")
        return 0

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
        print("\nFix the test (do not baseline it). The gate fails on any "
              "un-baselined failure.")
        return 1

    print("\nOK: no regression -- every failing node is accounted for by the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
