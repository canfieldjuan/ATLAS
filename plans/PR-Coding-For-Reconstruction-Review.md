# PR-Coding-For-Reconstruction-Review

## Why this slice exists

#2005 codified how reviewers reconstruct a PR independently from the diff. The
missing builder-side rule is what to do while coding when you know that review
method will be used. Without that, builders can still treat reconstruction as a
post-hoc review format instead of a design constraint on the plan, diff, tests,
and PR body.

This slice fixes the process root: add builder rules for coding toward
reconstruction review, then reference them from the Atlas workflow contract and
Claude Code repo guidance.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md` with builder rules for coding
   when a PR will be reviewed by independent reconstruction.
2. Reference that doc from `AGENTS.md` §3a so builders apply it before opening
   or updating a PR.
3. Reference that doc from `CLAUDE.md` in the workflow summary and key
   conventions so fresh Claude Code sessions inherit the rule.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md` is builder-focused, not a
        duplicate reviewer protocol.
  - [ ] The rules tell builders to start from the problem/correct-fix shape,
        keep the diff self-explaining, keep scope visible, fix upstream, test
        behavior, and keep the PR body exact.
  - [ ] `AGENTS.md` references the builder rule from the builder workflow before
        PR mutation.
  - [ ] `CLAUDE.md` references the builder rule for fresh Claude Code sessions.
  - [ ] Docs-only change; no code, config, migration, workflow, or test touched.
- Reachability proof: N/A -- docs/process-only change with no runtime, UI,
  report, billing, or public-contract surface. Proof is the committed docs and
  references.
- Affected surfaces: builder workflow docs (`AGENTS.md`), Claude Code repo
  guidance (`CLAUDE.md`), and the new builder rule doc.
- Risk areas: process clarity; avoid confusing the new builder rules with the
  already-merged reviewer reconstruction protocol.
- Reviewer rules triggered: R1.

### Files touched

- `AGENTS.md`
- `CLAUDE.md`
- `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md`
- `plans/PR-Coding-For-Reconstruction-Review.md`

## Mechanism

The new doc states the builder operating rules:

- start from the problem, not the patch;
- make the diff explain itself;
- keep scope visible;
- fix the upstream cause;
- test behavior, not story;
- make the PR description a receipt;
- run the diff/correct-fix/description self-check before push.

`AGENTS.md` points builders to this doc immediately after the plan-first rule
and applies the self-check before PR opens and updates. `CLAUDE.md` references
the same doc in the workflow highlights and key conventions so a fresh Claude
Code session sees it before PR work.

## Intentional

- This does not change the reviewer protocol from #2005. That remains
  `docs/PR_RECONSTRUCTION_PROTOCOL.md`; this slice adds the builder-side
  companion.
- This does not add a mechanical CI gate. The rule is a coding discipline,
  not a detector or blocker.
- AI reconciliation: narrowed the mock allowance so storage is not named as a
  mockable external boundary, preserving the existing real-adapter rule.
- AI reconciliation: matched the AGENTS.md wiring to the rule's "before opening
  or updating a PR" self-check scope.
- AI reconciliation: clarified that shipped behavior changes belong in
  Scope/Mechanism and the PR body, while Deferred is only for split-out
  follow-up work.

## Deferred

- If the rule later needs enforcement, add a separate detector/audit slice for
  PR-body overclaims, plan/diff drift, or missing self-check evidence.

Parked hardening: none.

## Verification

- Passed: focused grep for the coding-rule doc name, "Code for reconstruction",
  and "correct-fix" across `AGENTS.md`, `CLAUDE.md`, and the new doc.
- Passed: plan shape audit on this plan.
- Passed: plan files-touched audit against `origin/main`.
- Passed: plan diff-size audit against `origin/main` -- actual diff is 184 LOC.
- Passed: reviewer-rules trigger audit against `origin/main`.
- Passed: local PR review with planned PR body.
- Passed: local PR review with updated PR body after AI reconciliation.
- Passed: focused docs grep after AI reconciliation wording update.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 9 |
| `CLAUDE.md` | 9 |
| `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md` | 62 |
| `plans/PR-Coding-For-Reconstruction-Review.md` | 114 |
| **Total** | **194** |
