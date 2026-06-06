# PR-Reviewer-Rules-Contract

## Why this slice exists

`AGENTS.md` section 4 (the reviewer workflow) is strong on the builder side and
honest that the reviewer side is the weak link: automated reviewers have no
severity contract, and the recurring-lapse flywheel in `docs/SESSION_BOOTSTRAP.md`
only captures *builder* mistakes -- nothing captures escaped defects or reviewer
patterns. (The same gap is written up as "gap (b)" in the AI dev operating-model
doc proposed in PR #1317, not yet merged to main.) This is slice S1 of the
review-workflow redesign (issue #1328): the docs/contract foundation that the
later mechanical gates (S2 AI-reconciliation audit, S3 path-trigger audit) cite.
Docs/process only, no code paths.

Diff budget: ~410 LOC, just over the 400 soft cap. The overage is the
irreducible rule pack itself (`docs/REVIEWER_RULES.md`, ~183 LOC) plus this plan
doc (~109 LOC of inherent planning overhead); the rule pack is indivisible
without splitting R1-R12 across PRs, which would defeat the point of a single
canonical pack. Net new prose excluding the plan doc is ~300 LOC.

## Scope (this PR)

Ownership lane: dev-workflow/review-contract
Slice phase: Workflow/process

1. Add `docs/REVIEWER_RULES.md`: the Reviewer Rules Pack v1 (R1-R12), the Review
   Contract shape, the path-to-rule trigger table, the AI-finding reconciliation
   rule, and the misses-into-mechanism rule.
2. Add `REVIEW_MISSES.md`: the reviewer-side flywheel ledger, mirroring
   `HARDENING.md` -- every escaped defect becomes a durable gate.
3. Edit `AGENTS.md`: reframe section 4 (prove-the-contract, challenger pass,
   mandatory AI reconciliation), extend the section 2a reviewer template (rule
   matrix + AI reconciliation), note the Review Contract block in section 1a,
   extend the section 4d checklist, and update the section 8 reviewer bootstrap
   prompt.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/REVIEWER_RULES.md` defines R1-R12, the Review Contract block, the
        path trigger table, and the reconciliation + misses rules.
  - [ ] `REVIEW_MISSES.md` exists with the rule, the pattern-surfacing note, the
        ledger table, and the lifecycle note.
  - [ ] `AGENTS.md` references the rules pack and the new gates, with the 7
        required plan sections still intact and in order.
  - [ ] Existing behavior unchanged: BLOCKER/MAJOR/NIT/LGTM ladder kept; no code
        paths or CI behavior altered.
- Affected surfaces: docs / workflow contract only. No API, DB, auth, frontend,
  jobs, config, or third-party surface.
- Risk areas: none material -- documentation. Only risk is process churn if the
  contract is heavier than reviewers adopt; mitigated by keeping S1 doc-only and
  deferring enforcement to S2/S3.
- Reviewer rules triggered: R1 (requirements match), R10 (maintainability of the
  docs). No code-path rules apply.

### Files touched

- `AGENTS.md`
- `REVIEW_MISSES.md`
- `docs/REVIEWER_RULES.md`
- `plans/PR-Reviewer-Rules-Contract.md`

## Mechanism

The redesign is entirely contract text. `docs/REVIEWER_RULES.md` becomes the
canonical rule pack the reviewer runs; `AGENTS.md` points to it and reframes the
reviewer's job from "review the code" to "prove the Review Contract holds and no
rule is violated." The reviewer template gains a rule matrix and an AI-
reconciliation line so reviews are structured, not vibe-based. The Review
Contract lives as a `### Review Contract` subsection inside the plan doc's Scope
section -- a level-3 heading, so the seven level-2 sections audited by
`scripts/audit_plan_doc.py` are unchanged. This plan doc carries its own Review
Contract above as the worked example. `REVIEW_MISSES.md` mirrors `HARDENING.md`:
every escaped defect is logged and converted into one durable gate, which is the
reviewer-side half of the flywheel that `docs/SESSION_BOOTSTRAP.md` runs for the
builder.

## Intentional

- Doc-only by design. The mechanical enforcement (an AI-reconciliation audit and
  a path-trigger audit, each with failure-proving fixtures) is deliberately
  split into later slices so S1 stays small and reviewable; until they land, the
  reconciliation and trigger rules are reviewer discipline, stated as such in
  `docs/REVIEWER_RULES.md`.
- The verdict ladder is kept as BLOCKER/MAJOR/NIT/LGTM rather than adopting the
  generic APPROVE/REQUEST-CHANGES taxonomy; rule IDs layer underneath it.
- The Review Contract is a level-3 subsection of Scope rather than a new
  top-level section, so it does not disturb the audited 7-section shape or
  require editing the `scripts/new_pr_plan.sh` scaffold in this slice.

## Deferred

- S2: a mechanical AI-finding reconciliation audit (under scripts/, with
  fixtures per AGENTS.md section 3h) wired into local_pr_review.sh -- closes
  operating-model gap (b) mechanically.
- S3: a path-to-rule trigger audit that derives required rule IDs from the diff
  and fails when the plan's triggered-rules line omits one.
- S4: a reviewer-metrics summarizer over `REVIEW_MISSES.md` (AI-findings-missed-
  by-human and escaped-defect tallies).
- Wiring the Review Contract block into the `scripts/new_pr_plan.sh` scaffold
  template, with its fixture tests updated -- left out to keep this slice
  doc-only.

Parked hardening: none.

## Verification

- `scripts/audit_plan_doc.py` run on this plan -- 7 sections present, in order.
- `scripts/audit_plan_code_consistency.py` run on this plan -- all backticked
  path claims resolve on disk.
- `scripts/check_ascii_python.sh` -- no non-ASCII in .py files (none touched,
  run for hygiene).
- `scripts/local_pr_review.sh` -- full mechanical bundle green before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 73 |
| `REVIEW_MISSES.md` | 45 |
| `docs/REVIEWER_RULES.md` | 184 |
| `plans/PR-Reviewer-Rules-Contract.md` | 123 |
| **Total** | **425** |
