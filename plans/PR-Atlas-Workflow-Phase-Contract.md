# PR-Atlas-Workflow-Phase-Contract

## Why this slice exists

The Atlas workflow already requires thin end-to-end slices and parks
non-blocking hardening work in `HARDENING.md`, but it does not require
each PR to name where it sits in the feature lifecycle. That leaves
parallel sessions deciding "by ear" whether a PR is feature work,
validation, robust testing, hardening, or polish, which makes scope drift
and cross-session collisions harder to catch before review.

This slice makes the lifecycle explicit in the PR contract so builders,
reviewers, and operators can tell what kind of work a PR is doing before
reading the diff.

## Scope (this PR)

Ownership lane: atlas-workflow

Slice phase: Workflow/process.

1. Require every plan doc, PR body, and commit message to name the slice
   phase.
2. Define the standard phase names and their intended order.
3. Extend thin-slice triage so robust-testing and production-hardening
   slices intentionally pull required parked work forward.
4. Extend reviewer checks so the verdict confirms phase/scope alignment.

### Files touched

- `AGENTS.md`
- `plans/PR-Atlas-Workflow-Phase-Contract.md`

## Mechanism

`AGENTS.md` gains a lightweight phase label instead of a new required
section. The existing `Ownership lane` line stays first, then the phase
line follows it, preserving the seven-section plan shape while making
phase visible in the highest-signal places:

```md
Ownership lane: content-ops/faq-generator
Slice phase: Vertical slice

Plan: plans/PR-<Slice-Name>.md
Slice phase: <Vertical slice | Functional validation | Robust testing |
Production hardening | Product polish | Workflow/process>
```

The builder workflow explains the lifecycle order and how the phase
affects triage. Reviewer checklist updates make the phase part of the
LGTM gate.

## Intentional

- No mechanical audit script update in this slice. The contract change is
  useful immediately, and automated enforcement can land separately after
  reviewers have used the wording once.
- The existing seven required plan sections stay unchanged. A phase line is
  less disruptive than introducing an eighth required heading across all
  plan-shape checks.
- `Workflow/process` is included so repo-contract changes like this one do
  not have to pretend they are product validation or production hardening.

## Deferred

- Future PR: add `scripts/local_pr_review.sh` enforcement that the plan doc
  and PR body include a recognized `Slice phase:` value.
- Parked hardening: none.

## Verification

- `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| AGENTS workflow contract | ~45 |
| **Total** | ~120 |
