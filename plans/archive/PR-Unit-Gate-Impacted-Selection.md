# PR-Unit-Gate-Impacted-Selection

## Why this slice exists

`unit-gate` runs the entire unit suite on every PR push. Measured on run
30181595014: **447s of pytest inside a 9.4-minute median job**, install only 98s.
Every push pays it, including pushes that cannot affect any test -- #2202 was two
Markdown edits and spent 9.7 minutes executing 755 test files.

The full-suite guarantee it provides ("a test file no workflow enrolls still
runs", #2035 / G1.1, G2) is **already held daily** by
`repo_wide_unit_backstop.yml` (06:00 UTC cron + `workflow_dispatch`). `unit-gate`
re-pays for that same guarantee on every push of every PR.

### Problem-derived contract

- Root cause: the gate's test selection is constant (`tests/`) and independent of
  the diff, so cost is a function of the repo's size rather than the change's
  blast radius.
- Correct fix must: pick tests from the change's transitive reverse-import
  closure, not a one-hop name match; preserve executable package initializer
  edges; escalate to the full suite whenever the input cannot be proven scoped
  (deleted paths, unknown Python roots, runtime/config assets, path-loaded
  scripts, or reachable `conftest.py` fixture fanout); and restrict the ratchet
  baseline to the executed files, because `compare()` demands the failing set
  *exactly* equal the baseline -- unselected entries would otherwise read as
  newly-passing and fail every scoped run.
- Must not change: the ratchet's meaning within its scope (regressions AND stale
  entries still gate), the growth guard (a text comparison that must run even on
  a zero-test change), the baseline file, or the daily backstop.

## Scope (this PR)

Ownership lane: ci-gates
Slice phase: vertical slice

1. `scripts/select_impacted_tests.py` (new): AST import graph over the
   first-party roots, transitive reverse reachability from the changed files to
   test files, package-initializer dependency edges, and `FULL` escalation on
   any input the selector cannot prove scoped.
2. `scripts/check_unit_gate.py`: `--selected-files` restricts the baseline to the
   executed files; a failing node outside the selection is a hard error, not a
   regression verdict; scoped marker-filtered no-test runs are allowed only with
   an explicit selected-file scope; `--growth-only` enforces the baseline growth
   guard without pretending `/dev/null` is a pytest report; `--selected-files`
   may not be combined with `--update-baseline`.
3. `.github/workflows/unit_gate.yml`: select, then run FULL / scoped /
   growth-guard-only.
4. `tests/test_select_impacted_tests.py` (new): 30 fixture tests / 36 executed
   cases, weighted to the under-selection direction.

### Review Contract

- Acceptance criteria (structural properties, per `AGENTS.md` 1a):
  1. A test reachable from a changed module through **any** number of
     first-party import hops appears in the selection (not only direct
     importers).
  2. A change to a package `__init__.py` is selected when tests import a module
     under that package, because importing the leaf executes the initializer.
  3. Every input the selector cannot prove scoped produces `FULL`; there is no
     code path from a deleted path, unknown Python root, runtime/config asset,
     path-loaded script, or reachable `conftest.py` to a non-empty partial
     selection.
  4. An empty selection is produced only for explicitly test-free text changes
     and only after the growth guard still runs.
  5. On a scoped run the baseline compared against contains exactly the entries
     whose node id file is in the executed set.
  6. Within the executed scope both ratchet directions still fail the gate: an
     un-baselined failure, and a baseline entry that now passes.
  7. A scoped run whose selected files are fully removed by the unit marker
     filter does not fail as infrastructure merely because pytest returned
     "no tests collected"; unscoped no-test collection still fails loudly.
  8. The growth guard runs on every path through the workflow, including the
     zero-test path, via `--growth-only` rather than a synthetic pytest report.
- Reachability proof: the workflow's `Select impacted tests` step writes
  the selected-tests temp file, echoed in the job log; `check_unit_gate.py` prints
  `[scoped: N test file(s); baseline M/188]` on a scoped run. Both are observable
  in the Actions log of this PR.
- Affected surfaces: CI only. No runtime, API, DB, or product surface.
- Risk areas: under-selection (a real regression not executed pre-merge);
  baseline mis-scoping; growth guard skipped on the zero-test path.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Files touched

- `.github/workflows/unit_gate.yml`
- `plans/PR-Unit-Gate-Impacted-Selection.md`
- `scripts/check_unit_gate.py`
- `scripts/select_impacted_tests.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

Build a module -> importers map by parsing every first-party Python module with `ast`
(not regex: a regex cannot resolve `from . import x`). Walk it breadth-first from
the changed modules; every `tests.*` node reached is selected. Relative imports
resolve against the file's **repo-relative** package -- resolving against the
absolute path silently prefixes every `from .x` with the checkout directory, so
the edge never matches and the importer looks independent. That bug existed in
the first draft of this script and was caught by
`test_relative_import_is_followed`, which is why that test is in the suite.
Package imports keep all real ancestor module edges, not just the longest leaf
edge, because Python executes package initializers while resolving submodules.

Before graph traversal, the selector classifies changed paths. Only Python
modules inside known graph roots and explicit test-free Markdown can stay
scoped. Deleted paths, unknown Python roots, non-Markdown assets/config/workflow
files, and scripts that tests may load by path all escalate to `FULL`. If graph
traversal reaches `tests.conftest` (or a nested conftest), it also escalates to
`FULL`, because fixture consumers are discovered by pytest collection rather
than import edges.

The gate side is one restriction: `restrict_baseline()` keeps baseline entries
whose node-id file part is in the executed set. `node_file()` splits on the first
`::` so a parametrized id containing `::` inside its brackets still maps to its
file.

## Intentional

- **Escalate, never guess.** Unparseable module, unmappable path, empty diff, or
  a git failure all yield `FULL`. The dangerous direction is running too few
  tests and reporting green, so every uncertainty resolves toward running more.
- **Scripts escalate.** Some tests load `scripts/*.py` through
  `importlib.util.spec_from_file_location`, `runpy.run_path`, or subprocess path
  strings. The selector does not claim a scoped proof for that surface.
- **Failing node outside the selection is exit 2, not a regression.** If pytest
  ran something the selector did not claim, the scope and the run disagree and
  the verdict would be scored against evidence it was not built for.
- **`--selected-files` + `--update-baseline` is refused.** A scoped run must
  never rewrite the repo-wide baseline; it would delete the 90%+ of entries it
  never executed.
- **Growth guard still runs on the zero-test path** via `--growth-only`, so a
  docs-only PR cannot add baseline entries and the normal ratchet comparison is
  not polluted by a synthetic empty report.
- **No `pytest-xdist`.** Parallelism is the other lever on the same 447s and is
  probably worth more, but it is a separate change with its own risk (session
  fixtures, ordering) and does not belong bundled here.

## Deferred

- `pytest-xdist` on the full-suite path and the daily backstop.
- Enrolling `unit-gate` in branch protection. It is still advisory; that decision
  should follow evidence from this change, not precede it.

Parked hardening: none.

## Verification

    python -m pytest tests/test_select_impacted_tests.py -q
    # -> 36 passed

    python scripts/select_impacted_tests.py --base origin/main
    # -> FULL

    python scripts/maturity_sweep.py scripts --tests-root tests \
      --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 \
      --sensitive-glob 'scripts/**'
    # -> ratchet gate passed: no new brittleness above baseline

Selector exercised against this repo:

| Changed input | Result |
|---|---|
| `AGENTS.md`, `docs/REVIEWER_RULES.md` | empty -> growth guard only |
| `atlas_brain/services/customer_context.py` | 64 of 755 test files |
| `tests/conftest.py` | `FULL` |
| this PR diff (`scripts/check_unit_gate.py`, `scripts/select_impacted_tests.py`, `.github/workflows/unit_gate.yml`) | `FULL` |

Node counts for the `customer_context.py` selection, via `--collect-only`:
**1,455 collected vs 20,320 full (7.2%)**; collection alone drops 22.6s -> 8.8s.

Not run: a live scoped CI run. This PR's own diff touches
`scripts/check_unit_gate.py` and `.github/workflows/unit_gate.yml`, both on the
`GLOBAL_FILES` list, so it deliberately escalates itself to `FULL` -- the scoped
path is proven by the fixture tests and the local selections above, and will be
exercised by the first PR that does not touch the gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/unit_gate.yml` | 40 |
| `plans/PR-Unit-Gate-Impacted-Selection.md` | 187 |
| `scripts/check_unit_gate.py` | 105 |
| `scripts/select_impacted_tests.py` | 307 |
| `tests/test_select_impacted_tests.py` | 338 |
| **Total** | **977** |
