# PR-Unit-Gate-Baseline-Shrink-Proof

## Why this slice exists

The operator asked to keep attacking mechanical checks that were not respected
after #2258 exposed a unit-gate failure caused by `tests/unit_gate_baseline.txt`
shrinking on `main` while the removed node ids still failed in another slice.
That is a workflow/process defect: a baseline shrink is a claim that those node
ids are now passing, and the gate should mechanically require evidence from a
run that could have observed the removed nodes.

Diff-budget note: this is slightly over the 400 LOC target because the
indivisible repair includes the checker behavior, adversarial CLI fixtures,
baseline correction against the pinned CI runner, and the plan evidence needed
to explain the failed first run without opening a separate symptom PR.

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
  scoped-report, scoped-missing-proof, scoped-proven-shrink, and
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
  node ids, selected test files, parsed pytest failing node ids.
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
gate must include every removed node's test file in `--selected-files`;
`--growth-only` has no pytest evidence and fails closed. The existing comparison
then still catches the other half: if a removed node still appears in the
failing set, it is a regression because it is no longer in the PR baseline.

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
  - `75 passed in 0.66s`
- `python -m pytest tests/test_reasoning_graph_routing.py -q -rfE --tb=short`
  - `12 passed, 1 warning in 1.15s`
- `python -m py_compile scripts/check_unit_gate.py`
  - passed
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
| `plans/PR-Unit-Gate-Baseline-Shrink-Proof.md` | 180 |
| `scripts/check_unit_gate.py` | 70 |
| `tests/test_check_unit_gate.py` | 191 |
| `tests/test_reasoning_graph_routing.py` | 28 |
| `tests/unit_gate_baseline.txt` | 4 |
| **Total** | **473** |
