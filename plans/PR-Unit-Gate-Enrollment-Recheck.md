# PR-Unit-Gate-Enrollment-Recheck

## Why this slice exists

Issue #2260 tracks the CI/CD enforcement arc. The 2026-08-04 required-workflow
enrollment audit deferred `unit-gate` branch protection because its selector let
governance-doc changes pass growth-only without running their owning tests.
#2290 closed that blocker by mapping the governance docs to concrete pytest
owners and by making the watcher governance owner audit the checked-out repo
docs, not only fixture docs.

### Problem-derived contract

- Root cause: `unit-gate` remained classified as `ci_blocking_not_required`
  after its named selector blocker was fixed, leaving the merge contract weaker
  than the repo's current mechanical-check intent.
- Correct fix must touch/change: promote only `unit-gate` to
  `branch_required`, update branch-protection docs and tests to include the
  context, update live `main` branch protection to require the GitHub Actions
  `unit-gate` check, and verify the required-status checker against a fresh live
  payload.
- Must not change: do not promote `pre-push-audit`, do not change branch
  protection `strict`, do not alter `unit_gate.yml` execution semantics, and do
  not weaken any existing branch-required context.

## Scope (this PR)

Ownership lane: dev-workflow/unit-gate-enrollment
Slice phase: Workflow/process

1. Promote `unit-gate` from `ci_blocking_not_required` to `branch_required` in
   `ci/gates.yml`.
2. Update required-status docs/tests/workflow-path auditing to include
   `.github/workflows/unit_gate.yml` and the `unit-gate` context.
3. Record that `pre-push-audit` remains visible CI, not branch-required, because
   its trusted-base PR-side docs/test consistency blocker is separate.
4. Update live branch protection for `main` to require the GitHub Actions
   `unit-gate` context and verify the fresh payload.

### Review Contract

- Acceptance criteria:
  - `ci/gates.yml` marks `unit-gate` as `branch_required`.
  - `ci/gates.yml` leaves `pre-push-audit` as `ci_blocking_not_required`.
  - `tests/test_security_guardrails_workflow.py::REQUIRED_STATUS_CONTEXTS`
    includes `unit-gate`.
  - `.github/workflows/branch_protection_required_checks.yml` watches
    `.github/workflows/unit_gate.yml` changes.
  - `docs/SECURITY_GUARDRAILS.md` and
    `docs/ci_cd_autonomous_coding_map.md` name `unit-gate` in the
    branch-required context set.
  - `docs/audits/required-workflow-enrollment-audit-2026-08-04.md` records the
    2026-08-05 recheck decision: promote `unit-gate`, keep `pre-push-audit`
    deferred.
  - `python3 scripts/check_required_status_checks.py --payload-file <fresh live
    payload>` passes after live branch protection is updated.
- Reachability proof: the real deployed config is GitHub `main` branch
  protection; the observable output is the required-status checker PASS against
  a fresh payload after the live update.
- Affected surfaces: `ci/gates.yml`, branch-protection audit workflow paths,
  security/CI docs, required-status tests, live `main` branch protection, and
  this plan.
- Risk areas: accidental `pre-push-audit` promotion, missing live branch
  protection update, stale docs/tests omitting `unit-gate`, requiring the wrong
  app/source for the `unit-gate` context, and changing branch `strict` outside
  this slice.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `ci/gates.yml` branch-required merge gate registry and
  live GitHub branch protection required-status set.
- Replaced-path behaviors: `unit-gate` moves from visible CI to hard
  branch-required merge gate; `pre-push-audit` remains visible CI.
- Guard-relevant fields: gate `context`, gate `enforcement`, GitHub Actions
  `app_id`, branch-protection `strict`, and required-status `checks`.
- Caller x input shape: PRs targeting `main` that publish a `unit-gate` GitHub
  Actions check.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: live
  `repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks`.
- Explicit value probe: fetch fresh live payload after update and run
  `python3 scripts/check_required_status_checks.py --payload-file
  /tmp/atlas-required-status-checks-unit-gate-recheck-after.json`.
- Absent value probe: `pre-push-audit` remains absent from the required set by
  design; docs name its remaining blocker.
- Default-session/default-context probe: keep `strict: false`; this slice does
  not decide stale-branch merge policy.
- Side-effect ordering: update repo registry/docs/tests first, then patch live
  branch protection to add only `unit-gate`, then verify the fresh payload.

### Files touched

- `.github/workflows/branch_protection_required_checks.yml`
- `.github/workflows/unit_gate.yml`
- `ci/gates.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`
- `docs/audits/required-workflow-enrollment-audit-2026-08-04.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/ci_cd_runtime_duplication_audit.md`
- `plans/PR-Unit-Gate-Enrollment-Recheck.md`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

The required-status checker derives its expected contexts from
`ci/gates.yml`. This slice changes only the `unit-gate` row to
`branch_required`, updates docs/tests to include the new required context, and
adds `.github/workflows/unit_gate.yml` to the branch-protection audit workflow's
trusted `main` push path filters.

Live branch protection is updated out-of-band through the GitHub API because
`ci/gates.yml` is policy intent while GitHub branch protection is deployed
config. The update preserves `strict: false` and the existing required GitHub
Actions checks, adding only `unit-gate` with app id `15368`.

## Intentional

- `pre-push-audit` remains `ci_blocking_not_required`; its trusted-base PR-side
  docs/test consistency blocker is not solved by #2290.
- Branch protection `strict` remains `false`; stale-branch merge policy is a
  separate decision.
- No `unit_gate.yml` runtime behavior changes are included. This is enrollment,
  not runtime optimization.

## Deferred

- Add a safe trusted-base `pre-push-audit` PR-side docs/test consistency probe,
  then re-evaluate whether `pre-push-audit` should become branch-required.
- Decide branch-protection `strict` behavior separately from required-context
  enrollment.

Parked hardening: none.

## Verification

- `python3 scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks-unit-gate-recheck-before.json`
  -- failed before live update with `unit-gate: missing required check`.
- `/tmp/atlas-pr2259-venv/bin/python -m pytest tests/test_security_guardrails_workflow.py tests/test_select_impacted_tests.py tests/test_unit_gate_selector_fallback.py -q`
  -- passed locally, 101 passed.
- `gh api --method PATCH repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks --input /tmp/atlas-required-status-checks-unit-gate-patch.json`
  -- applied live branch-protection update preserving `strict: false` and adding
  `unit-gate` with GitHub Actions app id `15368`.
- `gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks > /tmp/atlas-required-status-checks-unit-gate-recheck-after-fresh.json`
  -- fetched fresh live payload after the update.
- `python3 scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks-unit-gate-recheck-after-fresh.json`
  -- passed locally; required set includes `unit-gate`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/branch_protection_required_checks.yml` | 1 |
| `.github/workflows/unit_gate.yml` | 6 |
| `ci/gates.yml` | 2 |
| `docs/SECURITY_GUARDRAILS.md` | 7 |
| `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` | 22 |
| `docs/audits/required-workflow-enrollment-audit-2026-08-04.md` | 25 |
| `docs/ci_cd_autonomous_coding_map.md` | 12 |
| `docs/ci_cd_runtime_duplication_audit.md` | 8 |
| `plans/PR-Unit-Gate-Enrollment-Recheck.md` | 173 |
| `tests/test_security_guardrails_workflow.py` | 12 |
| **Total** | **268** |
