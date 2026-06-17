# PR-Workflow-Action-Pin-OIDC-Audit

## Why this slice exists

PR-Security-Guardrail-CI parked a security/workflow item for remaining mutable
GitHub Actions refs and the Claude workflow's OIDC posture. The root cause is
that Atlas introduced pinned security workflows but left older product/check
workflows without an executable inventory, while `.github/workflows/claude.yml`
allowed any `@claude` trigger surface to start an OIDC-enabled action.

This change fixes the highest-risk root inside the slice boundary by
owner-gating and SHA-pinning the OIDC-enabled Claude workflow, then adds a
CI-enrolled workflow posture audit that fails new unreviewed
`pull_request_target` or `id-token: write` usage and reports remaining mutable
action tags as warnings for the follow-up pin-drain work.

This slice is slightly over the 400 LOC target because it includes the required
#1638 plan archive housekeeping plus negative fixtures for the new workflow
posture detector. Splitting out the tests would leave a security audit without
CI-proof that its failure branches fire.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Restrict the Claude Code workflow's OIDC-enabled job to repository-owner
   invocations and pin its third-party actions by commit SHA.
2. Add a tested workflow security posture audit for `pull_request_target`,
   `id-token: write`, and mutable action refs.
3. Enroll the audit and workflow-shape tests in CI through `pre_push_audit`.
4. Archive the merged #1638 plan as folded housekeeping.

### Review Contract
- Acceptance criteria:
  - [ ] `.github/workflows/claude.yml` keeps `id-token: write` only behind an
        owner-gated `@claude` trigger and SHA-pinned actions.
  - [ ] The workflow posture audit errors on unapproved `pull_request_target`
        and unapproved `id-token: write` usage.
  - [ ] Mutable action refs are inventoried as warnings rather than failing
        this PR, so the existing fleet can be drained in follow-up slices.
  - [ ] Audit and workflow-shape tests are enrolled in CI.
- Affected surfaces: CI / GitHub Actions / workflow security docs.
- Risk areas: CI compromise / OIDC credential exposure / workflow churn.
- Reviewer rules triggered: R1, R2, R3, R10, R11, R12, R14.

### Files touched

- `.github/workflows/claude.yml`
- `.github/workflows/pre_push_audit.yml`
- `HARDENING.md`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Workflow-Action-Pin-OIDC-Audit.md`
- `plans/archive/PR-Gitleaks-Baseline-Rotation-Escape-Hatch.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`
- `tests/test_claude_workflow_security.py`
- `tests/test_pre_push_audit_workflow.py`

## Mechanism

`.github/workflows/claude.yml` now requires `github.actor == github.repository_owner`
before the OIDC-enabled Claude job can run. Its `actions/checkout` and
`anthropics/claude-code-action` steps are pinned to the commit SHAs currently
behind their existing tags, with comments retaining the human tag version.

`scripts/audit_workflow_security_posture.py` parses workflow YAML and emits:

- `ERROR` for unapproved `pull_request_target`.
- `ERROR` for unapproved `id-token: write`.
- `WARN` for mutable or missing action refs.

The current `.github/workflows/security_guardrails.yml` `pull_request_target`
and `.github/workflows/claude.yml` OIDC use are explicitly allowlisted with
rationale. The audit is wired into `.github/workflows/pre_push_audit.yml`, and
fixture tests prove the failure branches fire.

## Intentional

- This PR does not mass-pin every existing product workflow action. Dependabot
  already opened overlapping action-update PRs, and a full fleet pin would be a
  noisy cross-lane PR. The new audit inventories mutable refs as warnings so
  the dedicated security/dependency lane can drain them safely.
- `.github/workflows/claude.yml` keeps `id-token: write` because the third-party
  Claude action is OIDC-enabled, but the job is now owner-gated and retains
  read-only GitHub token permissions.

## Deferred

- Pin remaining mutable action refs across product/check workflows in a
  dedicated fleet-drain or Dependabot-triage slice.

Parked hardening: updated `Audit remaining workflow action pins and Claude OIDC
trigger` to leave only the remaining mutable-action pin drain.

## Verification

- `python -m pytest tests/test_audit_workflow_security_posture.py tests/test_pre_push_audit_workflow.py tests/test_claude_workflow_security.py -q` - 16 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` - passed with mutable-action warnings and the expected allowed Claude OIDC warning.
- YAML parse smoke for `.github/workflows/claude.yml` and `.github/workflows/pre_push_audit.yml` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Workflow-Action-Pin-OIDC-Audit.md --check` - passed.
- Pending before push: `bash scripts/push_pr.sh tmp/pr_body_workflow_action_pin_oidc_audit.md -u origin HEAD`, which runs the mechanical local review bundle once with the PR body context.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/claude.yml` | 16 |
| `.github/workflows/pre_push_audit.yml` | 5 |
| `HARDENING.md` | 10 |
| `docs/SECURITY_GUARDRAILS.md` | 12 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Workflow-Action-Pin-OIDC-Audit.md` | 120 |
| `plans/archive/PR-Gitleaks-Baseline-Rotation-Escape-Hatch.md` | 0 |
| `scripts/audit_workflow_security_posture.py` | 136 |
| `tests/test_audit_workflow_security_posture.py` | 162 |
| `tests/test_claude_workflow_security.py` | 20 |
| `tests/test_pre_push_audit_workflow.py` | 13 |
| **Total** | **497** |
