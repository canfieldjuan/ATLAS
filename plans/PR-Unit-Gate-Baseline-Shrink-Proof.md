# PR-Unit-Gate-Baseline-Shrink-Proof

## Why this slice exists

The operator asked to keep attacking mechanical checks that were not respected
after #2258 exposed a unit-gate failure caused by `tests/unit_gate_baseline.txt`
shrinking on `main` while the removed node ids still failed in another slice.
That is a workflow/process defect: a baseline shrink is a claim that those node
ids are now passing, and the gate should mechanically require evidence from a
run that could have observed the removed nodes.

Diff-budget note: this is over the 400 LOC target because the indivisible
repair includes the checker behavior, adversarial CLI fixtures, baseline
correction against the pinned CI runner, and the plan evidence needed to
explain the failed review/CI rounds without opening separate symptom PRs.

### Problem-derived contract

- Root cause: `scripts/check_unit_gate.py` rejects baseline growth and reports
  stale entries from whatever scope ran, but it does not independently require a
  baseline shrink to be backed by a pytest run that included every removed
  baseline node's file. A scoped or growth-only path can therefore treat absence
  from the failure report as evidence even when the removed node was never in
  the executed scope.
- Correct fix must touch/change: add a baseline-shrink proof check in
  `scripts/check_unit_gate.py`, exercised through the CLI path used by
  `.github/workflows/unit_gate.yml`; add adversarial tests in
  `tests/test_check_unit_gate.py` for growth-only, unscoped-report,
  scoped-report, scoped-missing-proof, scoped-invocation mismatch,
  scoped-invocation filter rejection, unscoped custom-invocation rejection,
  scoped-proven-shrink, genuine-pass outcome proof, exact selected-file
  targets, bare-file collection-error proof, warning-text near misses, and
  removed-node-still-failing cases; isolate the order-dependent LLM registry
  state that made #2259-removed routing nodes fail only in the full suite;
  remove only the stale baseline entries the full gate proved passing.
- Must not change: do not change product behavior, reviewer-convergence policy
  in #2263, live-reconciliation policy from #2258, selected-test ownership
  mappings except where this proof requires them, or runtime LLM routing
  behavior.

## Scope (this PR)

Ownership lane: workflow/unit-gate-baseline-shrink-proof
Slice phase: Workflow/process

1. Require baseline-shrink evidence inside `scripts/check_unit_gate.py` before
   a removed baseline node can be treated as passing.
2. Add CLI-level tests that prove shrink-without-execution fails closed and
   shrink-with-execution behaves as the ratchet intends.
3. Make the existing reasoning graph routing tests independent of full-suite
   LLM registry state so the four #2259-removed nodes are genuinely proven.
4. Shrink `tests/unit_gate_baseline.txt` by the stale entries reported by the
   full gate before and after the routing isolation fix.

### Review Contract

- Acceptance criteria:
  - A PR baseline that removes node ids from the base baseline fails closed on
    `--growth-only`, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_growth_only_cannot_prove_baseline_shrink`.
  - A scoped run whose selected files omit any removed baseline node's file
    exits 2 before treating the shrink as passing, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_scoped_run_omits_removed_baseline_node_file`.
  - A captured report without a selected-file scope cannot prove a baseline
    shrink, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_unscoped_report_claims_baseline_shrink`.
  - A captured report with a selected-file scope still cannot prove a baseline
    shrink because the claimed scope is not bound to the captured output,
    settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_scoped_report_claims_baseline_shrink`.
  - A scoped run whose `--selected-files` claim is broader than the file targets
    in `--pytest-args` exits 2 before launching pytest, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_selected_scope_not_bound_to_pytest_args`.
  - A scoped run whose custom `--pytest-args` can filter, deselect,
    collect-only, short-circuit, or otherwise skip removed nodes exits 2 before
    launching pytest, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_scoped_custom_pytest_args_can_skip_removed_node`.
  - A scoped run cannot prove a removed node by selecting only the containing
    file; every removed node is run directly and exits 2 if pytest cannot
    collect it, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_removed_node_was_not_collected`.
  - A scoped shrink run that collects no tests exits 2 because no removed node
    was observed, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_scoped_shrink_run_collects_no_tests`.
  - A direct removed-node proof that exits 0 because the node was skipped or
    xfailed still exits 2 because baseline removal requires a genuine pass,
    settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_removed_node_did_not_genuinely_pass`.
  - Warning prose that contains words like "skipped" does not invalidate a
    genuine pass because outcome proof reads pytest outcome counts, settled by
    `tests/test_check_unit_gate.py::test_removed_node_pass_proof_ignores_warning_prose_near_misses`.
  - Scoped shrink proof rejects node-level targets and pytest `@args`
    indirection before subprocess launch, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_scoped_pytest_target_is_not_exact_file`.
  - A removed bare-file collection-error entry can be removed when the file now
    collects and exits cleanly even if legitimate tests skip, settled by
    `tests/test_check_unit_gate.py::test_cli_exit0_when_removed_file_collection_error_is_fixed_with_skip`.
  - An unscoped custom `--pytest-args` invocation cannot prove a baseline shrink;
    shrink proof must use the default full-suite invocation or a scoped
    `--selected-files` proof, settled by
    `tests/test_check_unit_gate.py::test_cli_exit2_when_unscoped_custom_pytest_args_cannot_prove_shrink`.
  - A scoped run launched by the gate that selected every removed node's file
    and whose report no longer contains those nodes can pass, settled by
    `tests/test_check_unit_gate.py::test_cli_exit0_when_scoped_run_proves_removed_node_passes`.
  - A scoped run launched by the gate where a removed node still fails reports
    a regression, settled by
    `tests/test_check_unit_gate.py::test_cli_exit1_when_removed_baseline_node_still_fails`.
  - Reasoning graph routing tests pass as a full file and no longer depend on
    prior-suite LLM registry state, settled by
    `tests/test_reasoning_graph_routing.py`.
  - The PR baseline shrink is itself proven by the full unit gate with the
    base baseline resolved from `origin/main`, settled by
    `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt --base-baseline /tmp/unit_gate_base_baseline.txt`.
- Reachability proof: `python scripts/check_unit_gate.py --baseline <pr>
  --base-baseline <base> --selected-files <scope> --pytest-args <files...>`
  exercises the same CLI entrypoint the unit-gate workflow calls; focused tests
  assert the observable exit code/stdout/stderr outcomes.
- Affected surfaces: `scripts/check_unit_gate.py`,
  `tests/test_check_unit_gate.py`, `tests/test_reasoning_graph_routing.py`,
  `tests/unit_gate_baseline.txt`, and this plan.
- Risk areas: fail-closed evidence requirement, scoped-run false positives,
  growth-only false positives, baseline ratchet compatibility.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/check_unit_gate.py` baseline ratchet/admission
  gate for PR baseline shrink claims.
- Replaced-path behaviors: baseline growth remains exit 3; unchanged exact
  baseline comparisons remain exit 0/1; new shrink-without-proof cases exit 2.
- Guard-relevant fields: base-baseline node ids, PR baseline node ids, removed
  node ids, selected test files, pytest positional targets, pytest-argument file
  targets, scoped-shrink pytest options, parsed pytest failing node ids.
- Caller x input shape: `.github/workflows/unit_gate.yml` calls
  `check_unit_gate.py` in FULL, scoped, and growth-only modes; tests drive the
  same CLI modes with fixture baselines/reports.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no env/config fallback change.
- Explicit value probe: N/A - no env/config fallback change.
- Absent value probe: N/A - no env/config fallback change.
- Default-session/default-context probe: N/A - no env/config fallback change.
- Side-effect ordering: N/A - local/CI checker only, no side effects.

### Files touched

- `plans/PR-Unit-Gate-Baseline-Shrink-Proof.md`
- `scripts/check_unit_gate.py`
- `tests/test_check_unit_gate.py`
- `tests/test_reasoning_graph_routing.py`
- `tests/unit_gate_baseline.txt`

## Mechanism

Compute `removed_baseline_entries = base_baseline - pr_baseline` when
`--base-baseline` is available. If that set is non-empty, require a proof path:
a real full run can prove the shrink from the pytest invocation it just ran;
captured `--report-file` evidence cannot prove a shrink because the captured
output is not bound to a verified execution scope; scoped runs launched by the
gate must include every removed node's test file in `--selected-files`, and
when `--pytest-args` is supplied the file targets in that invocation must match
the selected-file scope before pytest is launched. A scoped shrink proof also
rejects custom pytest options outside the unit-gate workflow's known execution
model because filters, deselectors, collect-only modes, and early-exit flags can
skip removed nodes while still targeting the selected file. Any live shrink
proof also runs each removed node id directly through pytest. Leaf node ids
must report genuine passed outcomes from pytest's numeric outcome summary, so
deleted, renamed, skipped, xfailed, xpassed, or deselected nodes fail closed
without matching unrelated warning prose. Bare-file collection-error baseline
entries are proved separately: the repaired file must collect and finish with
pytest exit 0, but legitimate skip/xfail outcomes inside the now-collecting
file do not block removing the obsolete collection-error entry. An unscoped
custom `--pytest-args` invocation cannot prove a shrink at all; use the default
full-suite invocation or a scoped `--selected-files` proof. `--growth-only` has
no pytest evidence and fails closed. The existing comparison then still catches
the other half: if a removed node still appears in the failing set, it is a
regression because it is no longer in the PR baseline.

Pytest inventory closure:

- `_PYTEST_OPTIONS_WITH_VALUES` is an open parser helper, not a permission
  list. Its source is the pytest option shapes this checker must skip while
  extracting positional file targets. Unlisted dash-prefixed options are treated
  as flags with no value for target parsing, then rejected by scoped-shrink
  proof unless separately admitted below; `@args` indirection and node-level
  positional targets are rejected before shrink proof.
- `_SCOPED_SHRINK_ALLOWED_FLAGS` is a closed allowlist for value-less options
  in scoped shrink proof. Membership comes from the unit-gate workflow's
  execution model, and any unlisted flag makes the shrink proof exit 2 before
  pytest runs.
- `_SCOPED_SHRINK_ALLOWED_OPTIONS_WITH_VALUES` is a closed allowlist for
  option/value pairs in scoped shrink proof. Membership comes from
  `UNIT_GATE_OPTION_ARGS`, and any unlisted option or listed option with an
  unlisted value makes the shrink proof exit 2 before pytest runs.

## Intentional

- No selector mapping change in this slice; the selector already treats
  `tests/unit_gate_baseline.txt` as global/FULL, and this PR hardens the gate
  against future scoped or growth-only callers.
- Baseline edits do not grow the ledger versus `origin/main`; the first CI run
  proved that 20 of the attempted removals still fail on the pinned runner, so
  this update restores those and leaves only the four reasoning-routing removals
  that CI did not report as regressions.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_unit_gate.py tests/test_select_impacted_tests.py tests/test_unit_gate_selector_fallback.py -q`
  - `109 passed in 0.72s`
- `python -m pytest tests/test_reasoning_graph_routing.py -q -rfE --tb=short`
  - `12 passed, 1 warning in 2.02s`
- `python -m py_compile scripts/check_unit_gate.py`
  - passed
- `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt --base-baseline /tmp/origin-main-unit-baseline.txt --selected-files /tmp/unit-gate-selected-reasoning.txt --pytest-args tests/test_reasoning_graph_routing.py -m 'not integration and not e2e' --continue-on-collection-errors -rfE --tb=no -q -p no:cacheprovider`
  - `OK: the baseline exactly matches the current failing set`; scoped baseline
    `0/175`, proving the four removed reasoning nodes by exact-node proof plus
    the selected file run.
- `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt --base-baseline /tmp/origin-main-unit-baseline.txt --growth-only; test $? -eq 2`
  - expected exit 2; names exactly the four remaining removed
    `tests/test_reasoning_graph_routing.py::*` routing entries.
- `git show origin/main:tests/unit_gate_baseline.txt > /tmp/origin-main-unit-baseline.txt && comm -23 <(rg -v '^#|^$' /tmp/origin-main-unit-baseline.txt | sort) <(rg -v '^#|^$' tests/unit_gate_baseline.txt | sort)`
  - remaining removed nodes are exactly the four
    `tests/test_reasoning_graph_routing.py::*` routing entries.
- GitHub unit-gate job `91580674681` on head
  `6f424d8b8578b02c1f16d4427e5f0c087a1f59c3` reported:
  `unit gate: 175 failing/errored node(s); baseline=155; regressions=20; newly-passing=0`.
  This update restores those 20 CI-proven failures, leaving a 175-node PR
  baseline and the four-node shrink above.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Unit-Gate-Baseline-Shrink-Proof.md` | 253 |
| `scripts/check_unit_gate.py` | 361 |
| `tests/test_check_unit_gate.py` | 589 |
| `tests/test_reasoning_graph_routing.py` | 28 |
| `tests/unit_gate_baseline.txt` | 4 |
| **Total** | **1235** |
