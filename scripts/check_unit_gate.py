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

  # no tests are reachable; still enforce the baseline-growth ratchet:
  python scripts/check_unit_gate.py --baseline B --base-baseline base.txt \
      --growth-only

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
_NO_TESTS_COLLECTED = 5

DEFAULT_PYTEST_ARGS = ["tests/", "-m", "not integration and not e2e",
                       "--continue-on-collection-errors", "-rfE", "--tb=no",
                       "-q", "-p", "no:cacheprovider"]
_PYTEST_OPTIONS_WITH_VALUES = frozenset((
    "-k",
    "-m",
    "-p",
    "--basetemp",
    "--confcutdir",
    "--deselect",
    "--ignore",
    "--ignore-glob",
    "--rootdir",
    "--tb",
))


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


def removed_baseline_entries(pr_baseline: set[str], base_baseline: set[str]) -> list[str]:
    """Node ids the PR removes from the baseline vs the base branch."""
    return sorted(base_baseline - pr_baseline)


def node_file(node_id: str) -> str:
    """The test-file part of a pytest node id (``tests/t.py::k[x::y]`` -> ``tests/t.py``)."""
    return node_id.split("::", 1)[0]


def restrict_baseline(baseline: set[str], selected_files: set[str]) -> set[str]:
    """Baseline entries belonging to ``selected_files``.

    compare() demands the failing set EXACTLY equal the baseline: an entry that
    no longer fails is reported as a stale ratchet entry and fails the gate. On
    a scoped run the unselected baseline entries were never executed, so without
    this they would every one look "newly passing" and every scoped run would
    fail. Restricting the baseline to what actually ran keeps both directions of
    the ratchet meaningful -- regressions AND stale entries -- within that scope.
    """
    return {node for node in baseline if node_file(node) in selected_files}


def pytest_positional_targets(pytest_args: list[str]) -> list[str]:
    """Return pytest positional targets, skipping option values."""
    targets: list[str] = []
    index = 0
    while index < len(pytest_args):
        arg = pytest_args[index]
        if arg == "--":
            targets.extend(pytest_args[index + 1:])
            break
        option = arg.split("=", 1)[0]
        if arg.startswith("-"):
            index += 2 if option in _PYTEST_OPTIONS_WITH_VALUES and "=" not in arg else 1
            continue
        targets.append(arg)
        index += 1
    return targets


def pytest_target_files(pytest_args: list[str]) -> set[str]:
    """Test file targets named in pytest args, normalized like selected-files."""
    targets: set[str] = set()
    for arg in pytest_positional_targets(pytest_args):
        if arg.startswith("-"):
            continue
        normalized = arg.removeprefix("./").rstrip("/").split("::", 1)[0]
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            targets.add(normalized)
    return targets


def validate_selected_pytest_args(selected_files: set[str], pytest_args: list[str]) -> int:
    """Fail closed when scoped proof claims files the pytest invocation won't run."""
    targets = pytest_target_files(pytest_args)
    missing = sorted(selected_files - targets)
    extra = sorted(targets - selected_files)
    if not missing and not extra:
        return 0
    print(
        "unit gate: --selected-files must match the pytest file targets used "
        "for a scoped run.",
        file=sys.stderr,
    )
    if missing:
        print("unit gate: selected file(s) missing from --pytest-args:", file=sys.stderr)
        for path in missing[:20]:
            print(f"  {path}", file=sys.stderr)
    if extra:
        print("unit gate: pytest target(s) outside --selected-files:", file=sys.stderr)
        for path in extra[:20]:
            print(f"  {path}", file=sys.stderr)
    return 2


def validate_unscoped_shrink_pytest_args() -> int:
    """Fail closed when custom args claim unscoped baseline-shrink proof."""
    print(
        "unit gate: unscoped custom --pytest-args cannot prove a baseline "
        "shrink; use the default full-suite invocation or pass --selected-files "
        "for a scoped proof.",
        file=sys.stderr,
    )
    return 2


def ensure_pytest_ran(returncode: int, *, allow_no_tests: bool = False) -> None:
    """Raise when pytest did not finish as a normal pass/fail run.

    Without this, an infrastructure/usage error (e.g. a plugin crash, a bad
    arg, or no tests collected) produces no FAILED/ERROR summary lines, parses
    to an empty failing set, and the gate would go GREEN having run nothing.

    A scoped run is different: marker filtering can legitimately remove every
    selected test (for example an e2e-only selected file under the unit marker
    expression). In that case pytest returns 5, but the selector still produced
    a real file scope and the ratchet comparison is restricted to that scope.
    """
    if allow_no_tests and returncode == _NO_TESTS_COLLECTED:
        return
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


def _fail_unproven_shrink(
    removed: list[str],
    *,
    missing_files: list[str] | None,
    report_file: bool = False,
) -> int:
    print(
        f"unit gate: baseline shrink removes {len(removed)} node(s), but this "
        "run did not provide pytest evidence for every removed node.",
        file=sys.stderr,
    )
    if report_file:
        print(
            "unit gate: --report-file cannot prove a baseline shrink because "
            "captured output is not bound to a verified pytest execution scope; "
            "run pytest through this gate instead.",
            file=sys.stderr,
        )
    elif missing_files is None:
        print(
            "unit gate: --growth-only has no pytest report; run the full suite "
            "or a scoped run that includes every removed node's test file.",
            file=sys.stderr,
        )
    elif missing_files:
        print(
            "unit gate: selected-files omitted removed baseline node file(s):",
            file=sys.stderr,
        )
        for path in missing_files:
            print(f"  {path}", file=sys.stderr)
    print("unit gate: removed baseline node(s):", file=sys.stderr)
    for node in removed[:20]:
        print(f"  {node}", file=sys.stderr)
    if len(removed) > 20:
        print(f"  ... {len(removed) - 20} more", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repo-wide unit gate (ratchet baseline).")
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--base-baseline", type=Path,
                    help="Base-branch baseline; reject additions vs it (ratchet). "
                         "An empty/absent file means no base baseline yet (initial seed).")
    ap.add_argument("--report-file", type=Path,
                    help="Gate a pre-captured pytest summary instead of running pytest.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Overwrite the baseline with the current failing set and exit 0.")
    ap.add_argument("--growth-only", action="store_true",
                    help="Only enforce the baseline-growth guard. Use this when "
                         "selection found no reachable unit tests; do not pass a "
                         "pytest report or selected-file scope.")
    ap.add_argument("--selected-files", type=Path,
                    help="File listing the test paths this run executed (scoped "
                         "run). The baseline is restricted to entries in those "
                         "files, since unselected entries never ran and would "
                         "otherwise read as newly-passing.")
    ap.add_argument("--pytest-args", nargs=argparse.REMAINDER,
                    help="Override the default pytest args (everything after this flag).")
    args = ap.parse_args(argv)

    if args.selected_files is not None and args.update_baseline:
        print("--selected-files cannot be combined with --update-baseline: a "
              "scoped run must never rewrite the repo-wide baseline",
              file=sys.stderr)
        return 2
    if args.growth_only and (
        args.report_file is not None
        or args.selected_files is not None
        or args.update_baseline
        or args.pytest_args
    ):
        print("--growth-only cannot be combined with pytest reports, scoped "
              "selection, pytest args, or --update-baseline",
              file=sys.stderr)
        return 2

    if not args.baseline.exists() and not args.update_baseline:
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 2

    # Ratchet diff guards (fast, no pytest): a PR may not add baseline entries
    # once the base branch has a baseline. It may shrink, but only when the run
    # can prove every removed node by executing that node's test file.
    removed_baseline_nodes: list[str] = []
    if args.base_baseline is not None and args.base_baseline.exists():
        base = load_baseline(args.base_baseline)
        if base:
            pr_baseline = load_baseline(args.baseline)
            added = added_baseline_entries(pr_baseline, base)
            if added:
                return _fail_growth(added)
            removed_baseline_nodes = removed_baseline_entries(pr_baseline, base)

    if args.growth_only:
        if removed_baseline_nodes:
            return _fail_unproven_shrink(removed_baseline_nodes, missing_files=None)
        print("unit gate: growth guard passed; no reachable unit tests selected")
        return 0

    selected: set[str] | None = None
    if args.selected_files is not None:
        if not args.selected_files.exists():
            print(f"selected files not found: {args.selected_files}", file=sys.stderr)
            return 2
        selected = {
            line.strip()
            for line in args.selected_files.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not selected:
            print("--selected-files is empty; use --growth-only for the "
                  "zero-test path", file=sys.stderr)
            return 2
        if args.pytest_args:
            scope_status = validate_selected_pytest_args(selected, args.pytest_args)
            if scope_status:
                return scope_status
        if removed_baseline_nodes:
            removed_files = {node_file(node) for node in removed_baseline_nodes}
            missing_removed_files = sorted(removed_files - selected)
            if missing_removed_files:
                return _fail_unproven_shrink(
                    removed_baseline_nodes,
                    missing_files=missing_removed_files,
                )
    elif removed_baseline_nodes and args.pytest_args:
        scope_status = validate_unscoped_shrink_pytest_args()
        if scope_status:
            return scope_status

    if args.report_file is not None:
        if removed_baseline_nodes:
            return _fail_unproven_shrink(
                removed_baseline_nodes,
                missing_files=None,
                report_file=True,
            )
        if not args.report_file.exists():
            print(f"report file not found: {args.report_file}", file=sys.stderr)
            return 2
        output = args.report_file.read_text(encoding="utf-8")
    else:
        output, returncode = run_pytest(args.pytest_args or DEFAULT_PYTEST_ARGS)
        try:
            ensure_pytest_ran(returncode, allow_no_tests=selected is not None)
        except RuntimeError as exc:
            print(f"unit gate: {exc}", file=sys.stderr)
            return 2

    failing = parse_failing_nodes(output)

    if args.update_baseline:
        write_baseline(args.baseline, failing, header=BASELINE_HEADER)
        print(f"wrote {len(failing)} node(s) to {args.baseline}")
        return 0

    baseline = load_baseline(args.baseline)
    scope_note = ""
    if selected is not None:
        full_size = len(baseline)
        baseline = restrict_baseline(baseline, selected)
        # Any failing node from a file we did not select means pytest ran
        # something the selector did not claim. Comparing that against a
        # baseline restricted to the selection would score it as a regression
        # on evidence the run was never scoped to produce, so fail loudly
        # rather than guess.
        stray = sorted({node for node in failing if node_file(node) not in selected})
        if stray:
            print(f"unit gate: {len(stray)} failing node(s) outside the selected "
                  f"files; scope is inconsistent with the run:", file=sys.stderr)
            for node in stray[:20]:
                print(f"  {node}", file=sys.stderr)
            return 2
        scope_note = (f" [scoped: {len(selected)} test file(s); "
                      f"baseline {len(baseline)}/{full_size}]")
    regressions, fixed = compare(failing, baseline)

    print(f"unit gate: {len(failing)} failing/errored node(s); "
          f"baseline={len(baseline)}; regressions={len(regressions)}; "
          f"newly-passing={len(fixed)}{scope_note}")
    # The baseline must EXACTLY equal the current failing set: no un-baselined
    # failure (regression), AND no stale entry that now passes. A stale entry
    # is a live allow-list hole -- a later PR could reintroduce that failure and
    # still pass because it stays in the ledger. So the ratchet enforces both
    # sides; the ledger tracks reality and can only move by an explicit edit.
    if fixed:
        print("\nSTALE baseline entries -- these node(s) PASS now; remove them "
              "from the baseline (the ratchet must shrink):")
        for node in fixed:
            print(f"  - {node}")
    if regressions:
        print("\nREGRESSION -- failing node(s) not in the baseline:")
        for node in regressions:
            print(f"  {node}")
        print("\nFix the test (do not baseline it). The gate fails on any "
              "un-baselined failure.")
    if regressions or fixed:
        return 1

    print("\nOK: the baseline exactly matches the current failing set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
