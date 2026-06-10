# PR-Reviewer-Codebase-Verification-Rule

## Why this slice exists

The operator called out a reviewer failure mode: a reviewer can accept the PR
narrative without verifying the claim against the checked-out codebase. The
existing reviewer workflow says to reproduce commands and spot-check file
lines, but the rule pack does not make source-of-truth verification a named
rule and the review template does not force a reviewer to disclose unverified
claims.

This workflow slice makes that explicit for every reviewer: verdicts are based
on the current PR head plus code/caller/test/artifact evidence, not the PR
description, issue summary, or builder claims.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add a reviewer source-of-truth rule to `docs/REVIEWER_RULES.md`.
2. Update `AGENTS.md` reviewer templates/checklists/bootstrap language so the
   required review output includes checked head, codebase verification, and
   "not verified" disclosure.
3. Add the R14 provenance row to `REVIEW_MISSES.md` so the reviewer flywheel
   records the originating misses.
4. Keep this as a documentation/process contract only; no runtime behavior
   changes.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/REVIEWER_RULES.md` names a rule requiring verdicts to be based
        on checked-out PR-head evidence, not PR/body/builder claims.
  - [ ] `AGENTS.md` says a reviewer cannot issue LGTM from claims alone.
  - [ ] The reviewer template requires checked head, code/caller/test/artifact
        verification, and explicit "not verified" entries when anything is
        skipped.
  - [ ] The reviewer bootstrap points fresh reviewer sessions at the same rule.
  - [ ] `REVIEW_MISSES.md` records the source-of-truth review misses promoted
        into R14.
- Affected surfaces: docs / workflow / reviewer contract.
- Risk areas: stale reviewer instructions / rule-number drift / ambiguous LGTM
  requirements.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `AGENTS.md`
- `REVIEW_MISSES.md`
- `docs/REVIEWER_RULES.md`
- `plans/PR-Reviewer-Codebase-Verification-Rule.md`

## Mechanism

Add R14 as a universal reviewer rule, then thread that rule through the
reviewer-facing spots that actually shape behavior:

- the rule-pack introduction and verdict language;
- the AGENTS review template;
- the reviewer independent-verification checklist;
- the final audit checklist before LGTM;
- the fresh-reviewer bootstrap prompt;
- the reviewer-miss ledger row that records this as a promoted durable rule.

R14 stays prose-only rather than a path glob because it applies to every
review verdict, not only PRs touching certain files. The existing
`scripts/audit_review_rules_triggered.py` surfaces prose-only trigger rows as
advisory, so this preserves the current audit model.

## Intentional

- No new mechanical checker in this PR. This is a reviewer contract update; a
  future checker could lint review bodies, but the immediate gap is the
  instruction reviewers follow.
- No changes to product code, CI workflows, or generated artifacts.

## Deferred

- Optional future hardening: add a review-body linter that rejects LGTM reviews
  missing checked-head and codebase-verification sections.

Parked hardening: none.

## Verification

- Reviewer-rule trigger audit using `scripts/audit_review_rules_triggered.py`
  against `plans/PR-Reviewer-Codebase-Verification-Rule.md`
  - Passed; R14 surfaces as an advisory universal reviewer-verdict rule.
- Git whitespace check
  - Passed.
- R14 coverage grep across `AGENTS.md` and `docs/REVIEWER_RULES.md`
  - Passed; both reviewer entry points include the new requirement.
- Local PR review via `scripts/push_pr.sh`
  - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 50 |
| `REVIEW_MISSES.md` | 1 |
| `docs/REVIEWER_RULES.md` | 22 |
| `plans/PR-Reviewer-Codebase-Verification-Rule.md` | 105 |
| **Total** | **178** |
