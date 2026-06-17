# PR-Workflow-Setup-Python-Pin

## Why this slice exists

PR-Workflow-Action-Pin-OIDC-Audit added a CI-enrolled workflow posture audit
that inventories mutable workflow supply-chain refs. The root cause this slice
addresses is that Atlas still has a repo-wide fleet of `actions/setup-python@v5`
tag refs in product/check workflows, so those CI jobs still trust a mutable tag
for Python toolchain setup.

This change fixes one high-reuse class of mutable workflow action refs by
pinning the existing `actions/setup-python@v5` call sites to the current commit
behind the v5 tag: `a26af69be951a213d495a4c3e4e4022e16d87065`.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Replace existing workflow `actions/setup-python@v5` refs under
   `.github/workflows` with
   `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5`.
2. Add/adjust focused workflow-posture test coverage so mutable
   `setup-python@v5` refs stay detectable while the pinned form is clean.
3. Archive the merged #1659 plan as folded housekeeping.

### Review Contract
- Acceptance criteria:
  - [ ] No `actions/setup-python@v5` refs remain under `.github/workflows`.
  - [ ] The replacement SHA matches the commit currently behind the upstream
        `actions/setup-python` v5 tag.
  - [ ] `scripts/audit_workflow_security_posture.py .github/workflows` no
        longer emits setup-python mutable-ref warnings.
  - [ ] Existing non-setup-python mutable refs remain deferred rather than
        being silently widened into this slice.
- Affected surfaces: GitHub Actions / workflow supply-chain posture.
- Risk areas: CI supply-chain compromise / workflow churn.
- Reviewer rules triggered: R1, R2, R3, R10, R11, R12, R14.

### Files touched

- `.github/workflows/admin_costs_checks.yml`
- `.github/workflows/ai_reconciliation_live.yml`
- `.github/workflows/atlas_b2b_campaign_migration_checks.yml`
- `.github/workflows/atlas_blog_public_checks.yml`
- `.github/workflows/atlas_brand_voice_checks.yml`
- `.github/workflows/atlas_content_ops_auth_checks.yml`
- `.github/workflows/atlas_content_ops_claim_registry_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `.github/workflows/atlas_deflection_migration_apply_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `.github/workflows/atlas_main_voice_startup_checks.yml`
- `.github/workflows/atlas_migrations_runner_checks.yml`
- `.github/workflows/extracted_competitive_intelligence_checks.yml`
- `.github/workflows/extracted_llm_infrastructure_checks.yml`
- `.github/workflows/extracted_pipeline_checks.yml`
- `.github/workflows/extracted_umbrella_checks.yml`
- `.github/workflows/marketing_content_check.yml`
- `.github/workflows/maturity_sweep_advisory.yml`
- `.github/workflows/pre_push_audit.yml`
- `.github/workflows/semantic_diff_advisor.yml`
- `HARDENING.md`
- `plans/INDEX.md`
- `plans/PR-Workflow-Setup-Python-Pin.md`
- `plans/archive/PR-Workflow-Action-Pin-OIDC-Audit.md`
- `tests/test_audit_workflow_security_posture.py`

## Mechanism

This is a mechanical pin-drain of one action family. Each workflow keeps the
same action major version semantics via a trailing `# v5` comment, but the
runtime `uses:` ref becomes the immutable commit SHA currently returned by:

`gh api repos/actions/setup-python/git/ref/tags/v5 --jq '.object.sha'`

The workflow posture audit already treats 40-character SHA refs as pinned, so
the proof is the combination of a focused grep check, the existing audit, and
the workflow-posture tests.

## Intentional

- This PR does not pin `actions/checkout`, `actions/setup-node`, Trivy, ZAP,
  CodeQL, or container/service image refs. Several action-update Dependabot PRs
  are already open, and this slice stays narrowly on `setup-python@v5` to avoid
  crossing into those branches.
- This PR pins the current v5 tag commit rather than upgrading to v6. It is a
  supply-chain immutability slice, not a behavior-upgrade slice.

## Deferred

- Pin remaining mutable action, reusable workflow, setup-node, checkout, and
  container/service image refs in follow-up fleet-drain slices or through the
  already-open Dependabot action PRs.

Parked hardening: continue draining `Pin remaining mutable workflow
supply-chain refs`.

## Verification

- `rg -n "actions/setup-python@v5" .github/workflows` - no matches.
- `rg -n "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" .github/workflows | wc -l` - 26.
- `python -m pytest tests/test_audit_workflow_security_posture.py -q` - 13 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` - passed; no setup-python mutable-ref warnings remained.
- Workflow YAML parse smoke across the GitHub workflow directory - passed.
- Pending before push: push through `scripts/push_pr.sh` with the PR body file.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/admin_costs_checks.yml` | 2 |
| `.github/workflows/ai_reconciliation_live.yml` | 2 |
| `.github/workflows/atlas_b2b_campaign_migration_checks.yml` | 2 |
| `.github/workflows/atlas_blog_public_checks.yml` | 2 |
| `.github/workflows/atlas_brand_voice_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_auth_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_claim_registry_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_generated_assets_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_input_provider_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 2 |
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 2 |
| `.github/workflows/atlas_deflection_migration_apply_checks.yml` | 2 |
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `.github/workflows/atlas_main_voice_startup_checks.yml` | 2 |
| `.github/workflows/atlas_migrations_runner_checks.yml` | 2 |
| `.github/workflows/extracted_competitive_intelligence_checks.yml` | 2 |
| `.github/workflows/extracted_llm_infrastructure_checks.yml` | 2 |
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `.github/workflows/extracted_umbrella_checks.yml` | 2 |
| `.github/workflows/marketing_content_check.yml` | 2 |
| `.github/workflows/maturity_sweep_advisory.yml` | 2 |
| `.github/workflows/pre_push_audit.yml` | 2 |
| `.github/workflows/semantic_diff_advisor.yml` | 2 |
| `HARDENING.md` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Workflow-Setup-Python-Pin.md` | 148 |
| `plans/archive/PR-Workflow-Action-Pin-OIDC-Audit.md` | 0 |
| `tests/test_audit_workflow_security_posture.py` | 36 |
| **Total** | **241** |
