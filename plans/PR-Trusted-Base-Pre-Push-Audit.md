# PR-Trusted-Base-Pre-Push-Audit

## Why this slice exists

#1949 moved API-driven PR meta-gates to trusted-base execution, but deferred tree-dependent gates. The first remaining hot path is `.github/workflows/pre_push_audit.yml`, which still executes `scripts/local_pr_review.sh` from the PR tree.

Root cause: the workflow conflates executable audit code with the inspected tree, so a PR can alter the gate that judges it. This fixes that root for pre-push audit only; maturity sweeps are deferred because of their lane filters and baselines.

Diff-size note: this is over the 400 LOC soft cap because the root fix needs the workflow split, wrapper root plumbing, helper adoption, and safety tests in one PR.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Vertical slice

1. Let local review and pre-push wrappers execute from a trusted script root
   while inspecting a separate repo root.
2. Convert `.github/workflows/pre_push_audit.yml` so PR events run from the
   base SHA and materialize the PR head only as inspected data.
3. Add workflow and unit proof for the trusted split.

### Review Contract

- Acceptance criteria: PR events use `pull_request_target`, SHA-pinned
  `actions/checkout`, and `${{ github.event.pull_request.base.sha }}` for
  executable code; the PR head is a separate data worktree passed via
  `--repo-root`; push-to-main still runs without a PR body file.
- Affected surfaces: pre-push audit workflow, local review wrappers, and audit
  helpers that previously resolved the inspected tree from `__file__`.
- Risk areas: auditing main instead of PR data, executing PR-owned scripts
  under `pull_request_target`, or breaking normal local review usage.
- Reviewer rules triggered: R2, R10, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `extracted/_shared/scripts/check_ascii_python.sh`
- `plans/PR-Trusted-Base-Pre-Push-Audit.md`
- `scripts/_audit_repo_root.py`
- `scripts/audit_claude_md_claims.py`
- `scripts/audit_extracted_manifests.py`
- `scripts/audit_mcp_tool_names_match_docs.py`
- `scripts/audit_plan_code_consistency.py`
- `scripts/audit_review_rules_triggered.py`
- `scripts/audit_ui_test_enrollment.py`
- `scripts/audit_workflow_security_posture.py`
- `scripts/check_ascii_python.sh`
- `scripts/local_pr_review.sh`
- `scripts/pre_push_audit.sh`
- `tests/test_audit_workflow_security_posture.py`
- `tests/test_local_pr_review.py`
- `tests/test_pre_push_audit.py`
- `tests/test_pre_push_audit_workflow.py`

## Mechanism

The wrappers gain `--repo-root` for inspected files and `--script-root` for
executed scripts. Defaults keep local behavior unchanged. The PR workflow checks
out base as `$GITHUB_WORKSPACE`, fetches PR head into `$RUNNER_TEMP/pr-tree`,
then runs the base copy of `scripts/local_pr_review.sh` against that PR tree.

`scripts/_audit_repo_root.py` lets Python audits read `ATLAS_AUDIT_REPO_ROOT`
in trusted mode while preserving their old root when no override is set.

## Intentional

- Maturity sweeps are deferred; this is the first tree-dependent vertical.
- Permissions remain read-only.
- The PR checkout is data only; PR-owned scripts/tests are not executed here.

## Deferred

- Trusted-base treatment for `maturity_sweep_advisory.yml` and `maturity_sweep_deflection_content_ops.yml`.
- Scheduled trusted sweep for review-event suppression, carried forward from
  #1949.

Parked hardening: none.

## Verification

- python -m pytest tests/test_local_pr_review.py tests/test_pre_push_audit_workflow.py tests/test_audit_workflow_security_posture.py -q (37 passed)
- python -m pytest tests/test_pre_push_audit.py tests/test_audit_claude_md_claims.py tests/test_audit_mcp_tool_names_match_docs.py tests/test_audit_extracted_manifests.py tests/test_audit_ui_test_enrollment.py tests/test_audit_plan_code_consistency.py tests/test_audit_review_rules_triggered.py -q (56 passed)
- python -m py_compile scripts/_audit_repo_root.py scripts/audit_claude_md_claims.py scripts/audit_mcp_tool_names_match_docs.py scripts/audit_extracted_manifests.py scripts/audit_ui_test_enrollment.py scripts/audit_plan_code_consistency.py scripts/audit_review_rules_triggered.py scripts/audit_workflow_security_posture.py (passed)
- python scripts/audit_workflow_security_posture.py .github/workflows (passed with existing warnings only)
- bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file tmp/pr-body-trusted-base-pre-push-audit.md (passed)

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 77 |
| `extracted/_shared/scripts/check_ascii_python.sh` | 2 |
| `plans/PR-Trusted-Base-Pre-Push-Audit.md` | 109 |
| `scripts/_audit_repo_root.py` | 20 |
| `scripts/audit_claude_md_claims.py` | 5 |
| `scripts/audit_extracted_manifests.py` | 5 |
| `scripts/audit_mcp_tool_names_match_docs.py` | 5 |
| `scripts/audit_plan_code_consistency.py` | 5 |
| `scripts/audit_review_rules_triggered.py` | 5 |
| `scripts/audit_ui_test_enrollment.py` | 5 |
| `scripts/audit_workflow_security_posture.py` | 1 |
| `scripts/check_ascii_python.sh` | 5 |
| `scripts/local_pr_review.sh` | 69 |
| `scripts/pre_push_audit.sh` | 76 |
| `tests/test_audit_workflow_security_posture.py` | 1 |
| `tests/test_local_pr_review.py` | 38 |
| `tests/test_pre_push_audit.py` | 1 |
| `tests/test_pre_push_audit_workflow.py` | 26 |
| **Total** | **455** |
