# PR-Review-Contract-Producer

## Why this slice exists

#2104/#2035 call for process gates to move from advisory/hidden checks toward
independently visible branch-protection contexts. #2124 pre-admitted exactly one
canonical `.github/workflows/review_contract.yml` workflow shape in the base-owned workflow
security posture auditor, but the producer workflow itself does not exist yet.

This slice adds that actual producer workflow. It is the next bootstrap step
before any branch-protection enrollment: the workflow can burn in as a visible
advisory context, while the later enrollment slice can require it only after the
context is stable and existing required checks are preserved.

### Problem-derived contract

- Root cause: Review Contract discipline is still enforced only by local/pre-push
  audits and PR review convention. The base workflow-security auditor now
  permits exactly one future `.github/workflows/review_contract.yml` shape, but there is no
  GitHub Actions producer for that context yet, so the process gate remains
  invisible to branch protection and cannot burn in.
- Correct fix must touch/change: add `.github/workflows/review_contract.yml`
  matching the `REVIEW_CONTRACT_CANONICAL_WORKFLOW` shape from
  `scripts/audit_workflow_security_posture.py`; verify the base-owned workflow
  security posture audit accepts it; include this plan and PR body
  reconstruction.
- Must not change: do not mutate branch protection, do not mark Review Contract
  required, do not change `scripts/audit_plan_doc.py`,
  `scripts/audit_review_rules_triggered.py`, `plan-admission`, `session-lane`,
  `claude-review`, `live-reconciliation`, product behavior, protected
  S6/content/dependabot/local-model lanes, or the canonical posture-auditor
  workflow text beyond adding the actual workflow file.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Add `.github/workflows/review_contract.yml` as an advisory
   `pull_request_target` trusted-base producer.
2. Prove the workflow matches the #2124 canonical whole-workflow posture
   admission and can run the existing plan-doc/reviewer-rule audits against PR
   data.

### Review Contract

- Acceptance criteria:
  - [ ] The only runtime surface added is
        `.github/workflows/review_contract.yml`.
  - [ ] The workflow matches the canonical normalized YAML admitted by
        `scripts/audit_workflow_security_posture.py`.
  - [ ] The workflow uses trusted-base checkout, materializes PR head as data,
        and runs base-owned `audit_plan_doc.py` /
        `audit_review_rules_triggered.py` against that data worktree.
  - [ ] The workflow no-ops cleanly for zero branch-added plan docs by running
        reviewer-rule triggering without `--plan`.
  - [ ] This PR does not change branch protection or required-check settings.
- Reachability proof: `python scripts/audit_workflow_security_posture.py
  .github/workflows` exercises the real workflow file and emits the expected
  trusted-base allowlist warning rather than an error.
- Affected surfaces: GitHub Actions workflow producers and workflow-security
  posture verification.
- Risk areas: drifting from the canonical workflow text, auditing PR-owned code
  as trusted code, accepting write permissions or unsafe later steps, breaking
  no-plan/doc-only exemptions, or enrolling the context too early.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/review_contract.yml`
- `plans/PR-Review-Contract-Producer.md`

## Mechanism

The workflow runs on `pull_request_target` but checks out the trusted base SHA
first. It fetches the PR head as git data into `$RUNNER_TEMP/pr-tree`, points the
PR worktree's remote HEAD at the base ref, and then runs base-owned audit scripts
from `$GITHUB_WORKSPACE` against the PR data worktree. Plan discovery is limited
to branch-added plans/PR-*.md files. If no branch-added plan exists, the
workflow still runs reviewer-rule triggering without `--plan`; if more than one
branch-added plan exists, it fails closed.

## Intentional

- This PR intentionally adds the producer as advisory only. Branch-protection
  enrollment remains a later #2104 slice.
- This PR does not change the canonical posture-auditor workflow string. The
  producer must conform to the already-merged canonical shape rather than move
  the admission target in the same PR.

## Deferred

- #2104 enrollment slice: require Review Contract only after producer burn-in
  and a REST branch-protection patch that preserves every existing context.
- #2104 reviewer slice: replace or constrain the forgeable `claude-review`
  publisher before requiring it.

Parked hardening: none.

## Verification

- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed with existing mutable-action warnings and expected trusted-base allowlist warnings, including the Review Contract workflow.
- `python -m pytest tests/test_audit_workflow_security_posture.py -q` — 19 passed.
- `python scripts/sync_pr_plan.py plans/PR-Review-Contract-Producer.md --check` — passed after sync.
- `python scripts/audit_pr_body.py /tmp/atlas-review-contract-producer-pr-body.md` — passed.
- Plan/code consistency audit — passed.
- `git diff --check` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/review_contract.yml` | 58 |
| `plans/PR-Review-Contract-Producer.md` | 115 |
| **Total** | **173** |
