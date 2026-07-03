# PR-Reachability-Proof-Review-Rule

## Why this slice exists

Issue #1951 captures a review gap from the #1950 run: a slice can add a
helper, fixture, workflow branch, report field, or UI component and still pass
tests without proving the new surface is wired into a real path. Stubs can wear
green when acceptance only says "tests pass."

Root cause: the Review Contract and rule pack require meaningful tests, but do
not explicitly require a reachability proof for newly introduced surfaces. This
change fixes the process root by making the proof explicit for builders and
reviewers.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add a calibrated reachability-proof requirement to the review contract and
   reviewer rules.
2. Front-load the same reminder in AGENTS.md and session bootstrap guidance.
3. Archive the merged #1950 plan as required teardown.

### Review Contract

- Acceptance criteria:
  - `docs/REVIEWER_RULES.md` requires a reachability proof for any new runtime,
    workflow, UI, report, billing, delivery, or public contract surface.
  - `AGENTS.md` tells builders to include real-entrypoint plus observable-effect
    proof in the plan/verification for those surfaces.
  - `docs/SESSION_BOOTSTRAP.md` carries the recurring-lapse reminder for
    restarted sessions.
  - The rule distinguishes new reachable surfaces from pure helpers/refactors,
    so docs do not force fake E2E where no new surface exists.
- Reachability proof: N/A - this docs/process slice introduces no runtime,
  workflow, UI, report, billing, delivery, or public contract surface.
- Affected surfaces: review contracts, reviewer rule pack, builder bootstrap,
  and merged-plan housekeeping.
- Risk areas: over-scoping into mandatory giant E2E, or leaving wording too
  vague to catch half-wired stubs.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `AGENTS.md`
- `docs/REVIEWER_RULES.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/INDEX.md`
- `plans/PR-Reachability-Proof-Review-Rule.md`
- `plans/archive/PR-Trusted-Base-Pre-Push-Audit.md`

## Mechanism

The docs introduce a named "reachability proof" concept:

- exercise the real entrypoint for a new surface; and
- assert an observable output, persisted state, rendered UI, emitted artifact,
  queued job, or gate result.

The wording is intentionally calibrated as a thin smoke proof, not a full E2E
mandate. Unit-only proof remains acceptable for PRs that introduce no new
reachable surface, or when the plan explicitly defers wiring with a reason.

## Intentional

- This slice is documentation/process only. Mechanical enforcement is deferred
  until we see builders skip the new proof in practice.
- No new rule ID; the requirement fits under R1/R2/R12/R14 rather than adding
  another verdict bucket.
- The #1950 plan archive is included as required merge teardown.

## Deferred

- Optional mechanical plan audit that requires a `Reachability proof:` line for
  vertical slices or new-surface PRs.

Parked hardening: none.

## Verification

- python scripts/archive_plans.py index (passed)
- python scripts/audit_plan_doc.py plans/PR-Reachability-Proof-Review-Rule.md (passed)
- bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file tmp/pr-body-reachability-proof-review-rule.md (passed)
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-reachability-proof-review-rule.md (passed)

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 15 |
| `docs/REVIEWER_RULES.md` | 17 |
| `docs/SESSION_BOOTSTRAP.md` | 1 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reachability-Proof-Review-Rule.md` | 97 |
| `plans/archive/PR-Trusted-Base-Pre-Push-Audit.md` | 0 |
| **Total** | **133** |
