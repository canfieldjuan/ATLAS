# PR-Unit-Gate-Workflow-Posture-Owner

## Why this slice exists

The operator reported that Unit Gate is still taking 8-10 minutes. Live Unit
Gate logs showed the selector escalating to `FULL` for PRs that touch
`scripts/audit_workflow_security_posture.py`:
`select_impacted_tests: scripts/audit_workflow_security_posture.py may be
loaded by filesystem path; escalating to FULL`. The matching PR also changed
`tests/test_audit_workflow_security_posture.py`, so this is a known governance
script with a focused owner test that the selector simply does not know about.

### Problem-derived contract

- Root cause: `scripts/select_impacted_tests.py` treats unknown Python files
  under `scripts/`
  paths as path-loaded runtime surfaces and correctly escalates them to `FULL`;
  `scripts/audit_workflow_security_posture.py` is a known CI-governance script
  but is missing from `EXPLICIT_TEST_OWNERS`.
- Correct fix must touch/change: Add the script-to-test mapping in
  `scripts/select_impacted_tests.py` and add matching fixture coverage in
  `tests/test_select_impacted_tests.py`.
- Must not change: Do not weaken the conservative `FULL` fallback for unknown
  scripts, deleted files, global config, unparseable modules, or unmapped
  runtime/config surfaces. Do not change Unit Gate workflow semantics or product
  behavior.

## Scope (this PR)

Ownership lane: ci/unit-gate-selector
Slice phase: Workflow/process

1. Register `scripts/audit_workflow_security_posture.py` as explicitly owned by
   `tests/test_audit_workflow_security_posture.py`.
2. Extend the existing explicit-owner selector fixture so that this path returns
   the focused owner test instead of `FULL`.

### Review Contract

- Acceptance criteria:
  1. `scripts/select_impacted_tests.py` maps
     `scripts/audit_workflow_security_posture.py` to
     `tests/test_audit_workflow_security_posture.py`.
  2. `tests/test_select_impacted_tests.py` includes that mapping in the existing
     explicit-owner fixture table, proving the selector returns the owner test.
  3. Existing tests for unknown scripts and missing owner files continue to
     exercise the `FULL` fallback.
- Reachability proof: `pytest tests/test_select_impacted_tests.py -q` executes
  the selector entrypoint used by Unit Gate's `Select impacted tests` step.
- Affected surfaces: Unit Gate impacted-test selector only.
- Risk areas: under-selection of tests, accidentally weakening unknown-script
  fallback, owner-map drift.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: Unit Gate explicit owner map in
  `scripts/select_impacted_tests.py`.
- Replaced-path behaviors: one known governance script moves from conservative
  `FULL` fallback to focused explicit owner selection.
- Guard-relevant fields: changed path
  `scripts/audit_workflow_security_posture.py`; owner test
  `tests/test_audit_workflow_security_posture.py`.
- Caller x input shape: Unit Gate passes changed paths from git diff; fixture
  calls `select([path], repo)`.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no deployed config.
- Explicit value probe: N/A - no deployed config.
- Absent value probe: N/A - no deployed config.
- Default-session/default-context probe: N/A - no deployed config.
- Side-effect ordering: N/A - pure selector mapping.

### Files touched

- `plans/PR-Unit-Gate-Workflow-Posture-Owner.md`
- `scripts/select_impacted_tests.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

`select_impacted_tests.py` checks `EXPLICIT_TEST_OWNERS` before applying the
unknown Python-file-under-`scripts/` fallback. Adding this one known script to the map lets the
selector return its focused governance test. Unknown scripts still hit the
existing path-loaded `FULL` branch.

## Intentional

- No workflow rewrite: the Unit Gate workflow already has selected-test support.
  The defect is a missing owner mapping.
- No broad script allowlist: only the script proven by the live slow run and its
  existing owner test is added.

## Deferred

None.

Parked hardening: none.

## Verification

- `pytest tests/test_select_impacted_tests.py -q` - PASS (62 tests).
- `python scripts/select_impacted_tests.py --changed-file <file containing scripts/audit_workflow_security_posture.py>` - PASS
  (`tests/test_audit_workflow_security_posture.py`).
- Pending before push: `python scripts/sync_pr_plan.py plans/PR-Unit-Gate-Workflow-Posture-Owner.md --check`.
- Pending before push: `bash scripts/local_pr_review.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Unit-Gate-Workflow-Posture-Owner.md` | 123 |
| `scripts/select_impacted_tests.py` | 3 |
| `tests/test_select_impacted_tests.py` | 4 |
| **Total** | **130** |
