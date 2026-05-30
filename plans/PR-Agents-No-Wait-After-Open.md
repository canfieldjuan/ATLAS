# PR-Agents-No-Wait-After-Open

## Why this slice exists

The builder workflow needs to match the operator handoff contract: once the PR
is open, the builder should not keep polling checks or waiting for review
comments. The operator will tell the builder when review comments are up or
when checks are ready to inspect.

## Scope (this PR)

Ownership lane: workflow/agents-contract
Slice phase: Workflow/process

1. Add the no-wait post-open handoff rule to `AGENTS.md`.
2. Keep reviewer-side CI/LGTM requirements unchanged.

### Files touched

- `plans/PR-Agents-No-Wait-After-Open.md`
- `AGENTS.md`

## Mechanism

The builder workflow section now states that after opening or updating a PR,
the builder stops active polling and returns the PR URL/status. Follow-up
inspection happens only after the operator says comments/checks are ready.

## Intentional

- This does not weaken the reviewer CI gate. Reviewers still require green CI
  before LGTM.
- This does not change the local pre-push validation requirement before a PR is
  opened.

## Deferred

- Parked hardening: none.

## Verification

- python scripts/audit_plan_doc.py plans/PR-Agents-No-Wait-After-Open.md - passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Agents-No-Wait-After-Open.md - passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-agents-no-wait-open-pr.md - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 45 |
| AGENTS.md contract text | 15 |
| **Total** | **60** |
