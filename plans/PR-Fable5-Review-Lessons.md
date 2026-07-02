# PR-Fable5-Review-Lessons

## Why this slice exists

The operator asked to "Codefy" the analysis of Fable 5's autonomous PR
1935-1941 arc so the lessons persist locally and can be carried into the active
GitHub issue. Without a checked-in artifact, the review lessons remain only in
chat history and are easy for future autonomous builder sessions to miss.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Workflow/process

1. Add a local workflow/process note that captures the recurring review
   findings, the solutions Fable took, the deferrals that were acceptable, and
   the patterns to avoid.
2. Include an issue-ready Markdown block that can be copied into the active
   GitHub issue when direct issue mutation is unavailable from this checkout.

### Files touched

- `docs/fable5_pr_1935_1941_review_lessons.md`
- `plans/PR-Fable5-Review-Lessons.md`

## Mechanism

The new document turns the prior PR-review synthesis into a durable local
reference organized by actionable categories: keep-doing patterns, pre-push
checks, acceptable deferrals, deferrals to avoid, and an issue-ready comment.
No runtime code changes are made.

## Intentional

- This PR does not attempt to mutate GitHub directly because the local checkout
  has no configured remote and `gh` is not installed in the environment.
- The document keeps the source details concise instead of embedding full PR
  transcripts; it is meant to be a standing builder checklist, not a historical
  archive.

## Deferred

- Posting the issue-ready block into the active GitHub issue remains deferred to
  an environment with GitHub issue access or to the operator.

Parked hardening: none.

## Verification

- Sync plan helper completed against `HEAD~1`.
- Sync plan check completed against `HEAD~1`.
- Local PR review helper completed against `HEAD~1` with environment warnings noted below.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/fable5_pr_1935_1941_review_lessons.md` | 148 |
| `plans/PR-Fable5-Review-Lessons.md` | 60 |
| **Total** | **208** |
