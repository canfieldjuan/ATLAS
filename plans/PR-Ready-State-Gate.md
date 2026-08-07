# PR-Ready-State-Gate

## Why this slice exists

The operator identified a repeated process failure: `AGENTS.md` says human PRs
open ready for review and several other rules are already written, but agents
still bypass the compliant path. This slice turns the first objective part of
that rule into mechanism: human full PR bodies prove they were published
through `scripts/open_pr.sh`. It also bootstraps the trusted-base workflow
posture allowlist needed before a future ready-state workflow can safely land.

Diff budget overage is accepted for this workflow/process slice because the
wrapper marker, trusted-base allowlist bootstrap, required PR-body workflow
enrollment, docs, and tests must land atomically; shipping any one without the
others would create a self-contradicting admission gate.

### Problem-derived contract

- Root cause: The open-wrapper rule is enforced only when builders use
  `scripts/open_pr.sh`; hand-rolled PR creation can still publish an unstamped
  full body, and the existing required PR-body gate had no direct proof that the
  wrapper path was used.
- Correct fix must touch/change: Stamp wrapper-published full PR bodies, make
  PR-body audit able to require that stamp through the existing
  branch-required PR-body context, and bootstrap the future `pr-ready-state`
  trusted-base allowlist before adding that workflow in a follow-up. The unit
  gate selector must also know the PR-body workflow's owning test so this slice
  can run the same pre-open unit gate locally without escalating to FULL.
- Must not change: Do not change product code, Codex review policy severity,
  `live-reconciliation`, docs-only admission semantics, or Dependabot
  exemptions.

## Scope (this PR)

Ownership lane: atlas-workflow/pr-enforcement
Slice phase: Workflow/process

1. Human full PR bodies must include an `atlas-open-pr-wrapper` marker when the
   PR-body audit is run in wrapper-required mode.
2. `scripts/open_pr.sh` adds that marker to the exact temporary body it audits
   and sends to GitHub.
3. The workflow posture allowlist recognizes the future `pr-ready-state`
   trusted-base job shape before that workflow is added in the next slice.
4. The unit-gate selector maps the PR-body workflow to its owning workflow test
   instead of escalating to FULL.

### Review Contract

- Acceptance criteria: `scripts/open_pr.sh` publishes a stamped full PR body
  without mutating the caller's body file; `scripts/audit_pr_body.py
  --require-wrapper-marker` rejects unstamped human full bodies and keeps
  docs-only/Dependabot exemptions; `scripts/audit_workflow_security_posture.py`
  recognizes the future ready-state trusted-base job shape without this PR
  adding that workflow.
- Reachability proof: The real entrypoints are `scripts/open_pr.sh` and
  `scripts/audit_pr_body.py --require-wrapper-marker`; observable effects are
  tests and the existing `pr-body-contract` context once this lands on `main`.
- Affected surfaces: PR-opening wrapper, PR-body audit, workflow posture
  allowlist, unit-gate selector, workflow/security docs.
- Risk areas: PR-body exactness, existing docs-only admission, Dependabot
  generated PRs, future pull_request_target workflow bootstrap, local-vs-CI
  unit-gate parity.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: PR body publication and future trusted-base workflow
  admission.
- Replaced-path behaviors: Hand-rolled full PR bodies become rejectable by
  audit; future ready-state workflow admission is bootstrapped before the
  workflow file exists.
- Guard-relevant fields: docs-only body marker, wrapper marker, future workflow
  file/job name.
- Caller x input shape: Local wrapper arguments/body file.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: No runtime config.
- Explicit value probe: Unit tests cover wrapper marker required/present.
- Absent value probe: Unit tests cover missing/empty title/body and missing
  wrapper marker.
- Default-session/default-context probe: Wrapper tests cover the default
  non-draft create path.
- Side-effect ordering: Wrapper tests prove local review runs before GitHub
  mutation and the body sent to GitHub is the stamped reviewed body.

### Files touched

- `.github/workflows/pr_body_contract.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `plans/PR-Ready-State-Gate.md`
- `scripts/audit_pr_body.py`
- `scripts/audit_workflow_security_posture.py`
- `scripts/open_pr.sh`
- `scripts/select_impacted_tests.py`
- `tests/test_audit_pr_body.py`
- `tests/test_audit_workflow_security_posture.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_pr_body_contract_workflow.py`
- `tests/test_security_guardrails_workflow.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

`open_pr.sh` derives a temporary publish body from the caller's body file. Full
plan-backed bodies receive `<!-- atlas-open-pr-wrapper: v1 -->`; docs-only
bodies are left untouched. The wrapper then runs the existing audit, local
review, hash checks, and `gh` stdin mutation against that temporary file.

`audit_pr_body.py` gains `--require-wrapper-marker`, used by trusted CI, which
rejects human full bodies missing the marker while preserving docs-only and
Dependabot paths. `audit_workflow_security_posture.py` gains the future
ready-state trusted-base allowlist entry first. That split is required because
trusted-base pre-push CI runs the allowlist from `origin/main`; adding a new
`pull_request_target` workflow before the allowlist is on `main` fails the
security posture gate before the workflow can land.

`select_impacted_tests.py` maps the PR-body workflow to
`tests/test_pr_body_contract_workflow.py`, keeping unit-gate local and CI runs
on the same selected-files path instead of escalating to the repo-wide suite for
this known governance workflow.

## Intentional

- The pause escape hatch is explicit text, not a new label, so the trusted-base
  workflow can decide from event metadata without extra API permission.
- Docs-only PRs are not wrapper-marker-gated; their first-line marker is already
  the narrow admission proof.
- Reviewer-scope heuristics stay advisory/reactive in this slice; this PR only
  mechanizes objective PR admission behavior.
- The ready-state workflow itself is deferred; a new `pull_request_target`
  workflow must be allowlisted on `main` before CI can evaluate it as trusted.

## Deferred

- Promote or retire heuristic review-scope checks after measuring false
  positives; do not make seam convergence, guard class closure, or boundary
  enumeration required in this slice.
- A complete AGENTS.md rule enforcement matrix can follow once this first
  targeted strict gate proves clean.
- Add the `pr-ready-state` workflow/checker, then promote it to
  `branch_required` and enroll it in live branch protection after the workflow
  emits the context from `main`.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_pr_body.py tests/test_open_pr_wrapper.py tests/test_pr_body_contract_workflow.py tests/test_security_guardrails_workflow.py tests/test_audit_workflow_security_posture.py tests/test_select_impacted_tests.py -q` - 308 passed.
- Local unit-gate mirror: selected 10 governance test files and `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt --base-baseline /tmp/base_baseline.txt --selected-files /tmp/selected.txt --pytest-args ...` reported 0 regressions.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pr_body_contract.yml` | 1 |
| `docs/SECURITY_GUARDRAILS.md` | 9 |
| `docs/ci_cd_autonomous_coding_map.md` | 10 |
| `plans/PR-Ready-State-Gate.md` | 178 |
| `scripts/audit_pr_body.py` | 29 |
| `scripts/audit_workflow_security_posture.py` | 1 |
| `scripts/open_pr.sh` | 50 |
| `scripts/select_impacted_tests.py` | 3 |
| `tests/test_audit_pr_body.py` | 18 |
| `tests/test_audit_workflow_security_posture.py` | 1 |
| `tests/test_open_pr_wrapper.py` | 28 |
| `tests/test_pr_body_contract_workflow.py` | 1 |
| `tests/test_security_guardrails_workflow.py` | 2 |
| `tests/test_select_impacted_tests.py` | 4 |
| **Total** | **335** |
