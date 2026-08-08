# PR-Unit-Gate-Baseline-Shrink

## Why this slice exists

While preparing the workflow/pr-enforcement hook slice after PR #2321, the full
local unit-gate mirror ran against `origin/main` and reported `149`
failing/errored nodes with a baseline of `169`. CI review then proved that 11
of those local passers depended on this workstation's seeded Postgres state and
still fail in GitHub's clean unit-gate runner. The truthful shrink is therefore
the 9 security monitor/network nodes that pass once the runner installs the
missing Scapy dependency; the invoice/live-DB nodes stay baselined until a
dedicated seeded database workflow can prove them.

### Problem-derived contract

- Root cause: `tests/unit_gate_baseline.txt` lists 9 security pytest nodes that
  should pass in the unit-gate runner, but the runner did not install Scapy; the
  local mirror also allowed a developer-local Postgres to make 11 live-DB invoice
  nodes look stale when the clean CI runner still fails them.
- Correct fix must touch/change: Declare Scapy only in the unit-gate test
  environment, keep the root/production dependency lock current, make the CI and
  local unit-gate environments explicitly avoid developer/runner Postgres state,
  remove only the 9 security node IDs from `tests/unit_gate_baseline.txt`, keep
  the invoice/live-DB nodes baselined, update the committed baseline integrity
  test so it permits the ratchet to shrink below the old arbitrary 150-entry
  floor, and verify the full unit gate reports zero regressions/newly-passing
  entries.
- Must not change: Do not change product behavior, the unit-gate checker,
  selector, invoice tests, security monitor code, or the pending delete-push hook
  slice.

## Scope (this PR)

Ownership lane: atlas-workflow/pr-enforcement
Slice phase: Workflow/process

1. Shrink `tests/unit_gate_baseline.txt` by removing only the 9 security nodes
   proven stale under the explicit unit-gate environment.
2. Keep the committed-baseline integrity test focused on non-empty/sorted/unique
   instead of an outdated minimum entry count.
3. Add the missing Scapy dependency only to the unit-gate workflow.
4. Keep the root/ASR constraints lock and digest current after leaving Scapy out
   of production requirements.
5. Make CI and local unit-gate runs ignore developer-local Postgres state.
6. Verify the updated baseline against the full unit-gate command.

### Review Contract

- Acceptance criteria:
  - The diff removes exactly the stale security monitor/network nodes that pass
    in the explicit unit-gate environment with Scapy installed.
  - The diff keeps the monthly invoice live-DB nodes baselined because they still
    fail in GitHub's clean unit-gate runner.
  - The diff does not add any baseline entries.
  - Scapy stays out of the production/root dependency set and is installed only
    by the unit-gate workflow as a test dependency.
  - CI and local unit-gate runs both use an explicit no-Postgres configuration
    so developer-local seeded DB state cannot decide baseline truth.
  - Committed regression coverage pins that no-Postgres environment in both the
    GitHub workflow and the local unit-gate mirror.
  - The updated baseline remains sorted within the existing file order.
  - The committed-baseline integrity test no longer fails solely because the
    baseline truthfully shrank from 169 to 160 entries; it still asserts a
    non-empty, sorted, unique baseline.
  - The full unit-gate command reports zero regressions and zero newly-passing
    stale entries against the updated baseline.
- Reachability proof: Real entrypoint is
  `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt
  --base-baseline <merge-base baseline>`; observable output is the unit-gate
  ratchet summary.
- Affected surfaces: `.github/workflows/unit_gate.yml`,
  `scripts/local_pr_review.sh`, `tests/unit_gate_baseline.txt`,
  `tests/test_check_unit_gate.py`, `tests/test_local_pr_review.py`,
  `tests/test_unit_gate_selector_fallback.py`, and this plan.
- Risk areas: ratchet correctness, accidental baseline growth, accidental removal
  of still-failing known failures.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: Unit-gate known-failure baseline.
- Closure declaration: the baseline is an enumerated closed set of exact pytest
  node IDs sourced from the canonical full unit-gate run output for this slice.
  `scripts/check_unit_gate.py` reads the committed list literally; it does not
  recompute membership from pytest output during enforcement. Maintenance is
  ratchet-based: future changes must commit reviewed additions/removals from a
  full-gate run. Membership in this set is the only tolerated failing-node
  exception; any failing node outside the set is an unbaselined regression and
  fails the gate, while any passing node inside the set is stale and fails the
  gate until removed.
- Replaced-path behaviors: Previously the baseline listed 169 known failures;
  after this shrink it should list 160 known failures matching the current full
  unit suite under the explicit no-Postgres unit-gate environment.
- Guard-relevant fields: pytest node IDs in `tests/unit_gate_baseline.txt`.
- Caller x input shape: `scripts/check_unit_gate.py` reads one sorted pytest node
  ID per non-comment line.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `.github/workflows/unit_gate.yml` and the
  local mirror read `tests/unit_gate_baseline.txt` by default and set
  `ATLAS_DB_CONNECTION_STRING=` plus `ATLAS_DB_HOST=127.0.0.1` /
  `ATLAS_DB_PORT=1` plus `ATLAS_DB_SOCKET_PATH=` for the unit-gate command.
- Explicit value probe: Full unit-gate verification uses the updated baseline as
  `--baseline`.
- Absent value probe: N/A; this slice does not change fallback behavior.
- Default-session/default-context probe: PRs that escalate to `FULL` should no
  longer fail solely because the 9 security nodes remain in the baseline or
  because local seeded Postgres state diverges from CI.
- Side-effect ordering: Baseline shrink lands before the delete-push hook PR is
  published.

### Files touched

- `.github/workflows/unit_gate.yml`
- `plans/PR-Unit-Gate-Baseline-Shrink.md`
- `requirements.unit_gate.txt`
- `scripts/local_pr_review.sh`
- `tests/test_check_unit_gate.py`
- `tests/test_local_pr_review.py`
- `tests/test_unit_gate_selector_fallback.py`
- `tests/unit_gate_baseline.txt`

## Mechanism

The unit gate treats `tests/unit_gate_baseline.txt` as the known-failure set.
When a baselined node passes, the checker reports it as a stale baseline entry
and fails so the ratchet can shrink. This PR makes the gate environment explicit:
the GitHub workflow installs Scapy as a test-only unit-gate dependency while the
root/production `requirements.txt` and `constraints.root-asr.txt` stay free of
Scapy. Both CI and the local mirror point Atlas DB settings at an unavailable
local port and clear any inherited Unix socket path so seeded developer
databases cannot create false stale entries. With that environment, the
known-failure set shrinks from 169 to 160 by removing only the 9 security nodes.
The committed-baseline integrity test keeps checking that the file is parseable,
sorted, unique, and non-empty, but no longer encodes a minimum count that
contradicts the ratchet's ability to shrink.

## Intentional

- This PR keeps the monthly invoice live-DB nodes baselined. They pass on this
  workstation only because local seeded Postgres state exists; GitHub's clean
  unit-gate runner proved they are not stale there.
- This PR does not include the pending delete-push hook fix; it only unblocks the
  required full unit-gate validation for workflow/process PRs.

## Deferred

- Resume the pending delete-push hook slice after this baseline shrink lands.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_unit_gate.py::test_committed_baseline_parses_and_is_sorted_unique -q` - 1 passed.
- `python scripts/compile_root_asr_constraints.py --check` - passed.
- `python -m pytest tests/test_compile_root_asr_constraints.py -q` - 10 passed.
- `bash -n scripts/local_pr_review.sh` - passed.
- `python -m pytest tests/test_local_pr_review.py::test_local_pr_review_runs_local_unit_gate_mirror -q` - 1 passed.
- `python -m pytest tests/test_unit_gate_selector_fallback.py::test_workflow_pins_the_no_postgres_unit_gate_environment tests/test_unit_gate_selector_fallback.py::test_workflow_installs_shared_unit_gate_test_requirements -q` - 2 passed.
- Full unit-gate command with the updated baseline and `origin/main` baseline as
  `--base-baseline`, plus explicit no-Postgres `ATLAS_DB_*` values - passed:
  160 failing/errored node(s), baseline=160,
  regressions=0, newly-passing=0.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/unit_gate.yml` | 15 |
| `plans/PR-Unit-Gate-Baseline-Shrink.md` | 185 |
| `requirements.unit_gate.txt` | 1 |
| `scripts/local_pr_review.sh` | 18 |
| `tests/test_check_unit_gate.py` | 2 |
| `tests/test_local_pr_review.py` | 42 |
| `tests/test_unit_gate_selector_fallback.py` | 25 |
| `tests/unit_gate_baseline.txt` | 9 |
| **Total** | **297** |
