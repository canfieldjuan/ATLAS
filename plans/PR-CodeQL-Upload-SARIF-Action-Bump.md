# PR-CodeQL-Upload-SARIF-Action-Bump

## Why this slice exists

The security guardrail workflows now upload Trivy and Checkov SARIF through
`github/codeql-action/upload-sarif`, and the repo is intentionally SHA-pinning
third-party actions to avoid mutable-tag CI compromise. Dependabot opened this
slice to advance the pinned CodeQL action ref from
`411bbbe57033eedfc1a82d68c01345aa96c737d7` to
`8aad20d150bbac5944a9f9d289da16a4b0d87c1e` without changing the guardrail
scan behavior.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Update only the CodeQL upload-sarif action SHA used by security guardrail
   SARIF upload steps.
2. Preserve existing advisory scan behavior, permissions, paths, and SARIF
   filenames.

### Review Contract

Acceptance criteria:
- `.github/workflows/security_guardrails.yml` keeps both SARIF upload steps on
  the same pinned CodeQL action SHA.
- No guardrail trigger, permission, severity, exit-code, scan path, or
  soft-fail behavior changes in this slice.
- The PR body and plan doc satisfy the Atlas PR-shape contract for Dependabot's
  generated update.

Affected surfaces:
- `.github/workflows/security_guardrails.yml`

Risk areas:
- CI action supply-chain integrity.
- Accidental workflow behavior drift while updating the action ref.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R3 Security/auth
- R10 Workflow/process
- R14 Codebase verification

### Files touched

- `.github/workflows/security_guardrails.yml`
- `plans/PR-CodeQL-Upload-SARIF-Action-Bump.md`

## Mechanism

Dependabot rewrites the two `github/codeql-action/upload-sarif@...` references
in `.github/workflows/security_guardrails.yml` from the previous pinned commit
to the newer pinned commit. The workflow still uses the same upload-sarif
action family and the same `sarif_file` inputs for Trivy and Checkov outputs.

## Intentional

- Keep this as a SHA-only maintenance slice; no scan ratcheting, permission
  tightening, or Checkov/Trivy finding triage is bundled here.
- Preserve Dependabot's action-update intent rather than folding this into a
  broader workflow refactor.

## Deferred

- None.

Parked hardening: none.

## Verification

- Command: python scripts/audit_pr_body.py tmp/pr_body_1630.md - passed.
- Command: python scripts/sync_pr_plan.py --check plans/PR-CodeQL-Upload-SARIF-Action-Bump.md - passed after syncing this plan.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/security_guardrails.yml` | 8 |
| `plans/PR-CodeQL-Upload-SARIF-Action-Bump.md` | 83 |
| **Total** | **91** |
