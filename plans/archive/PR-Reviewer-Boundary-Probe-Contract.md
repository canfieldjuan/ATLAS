# PR-Reviewer-Boundary-Probe-Contract

## Why this slice exists

Several recent review misses shared the same shape: a guard, validator,
sanitizer, cap, or classifier was checked on the obvious side of its boundary,
but the opposite failure direction was not probed before LGTM. Existing rules
already require negative fixtures and codebase-backed review, but the reviewer
template does not force the reviewer to state the second-side boundary probe in
the verdict.

Root cause: the review contract names test evidence and R14 verification, but
does not make the two-sided guard boundary check a visible reviewer output.
This fixes the root at the reviewer-contract layer by requiring a
`boundary-probe:` line and defining the probe checklist for guard-shaped PRs.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add a boundary-probe review requirement for guard-shaped PRs in the shared
   reviewer rules pack.
2. Add the matching `boundary-probe:` line to the AGENTS reviewer verdict
   template and checklist.
3. Keep the change documentation-only; no product code, tests, or CI behavior
   changes ship in this slice.

### Files touched

- `AGENTS.md`
- `docs/REVIEWER_RULES.md`
- `plans/PR-Reviewer-Boundary-Probe-Contract.md`

### Review Contract

- Acceptance criteria:
  - [ ] `docs/REVIEWER_RULES.md` defines guard-shaped PRs and the required
        two-sided boundary-probe checklist.
  - [ ] `AGENTS.md` tells reviewers to include a `boundary-probe:` verdict
        line when that rule applies.
  - [ ] The rule distinguishes missing boundary proof severity for
        security/billing/data deletion/customer-visible/CI-release gates from
        lower-risk guards.
- Affected surfaces: reviewer workflow docs only.
- Risk areas: process friction / reviewer ambiguity / stale template wording.
- Reviewer rules triggered: R1, R10, R14.

## Mechanism

The reviewer rules pack gains a dedicated boundary-probe section after R14,
where reviewers already have the universal codebase-verification rule. The new
section describes when the probe applies, which input classes to check, and how
to classify missing proof.

The AGENTS reviewer template and audit checklist gain a `boundary-probe:` line
so the rule is visible in every LGTM-shaped review when it applies. That output
is intentionally explicit: if a guard-shaped PR review lacks the line, the
operator can see that the reviewer skipped the required step.

## Intentional

- No mechanical audit in this slice. The immediate miss pattern is reviewer
  judgment, so the first fix is the human-visible review contract. A future
  audit could check for a `boundary-probe:` line in posted reviews, but that is
  separate enforcement work.
- No new rule number. This sits under existing R2/R13/R14 review obligations
  rather than expanding the rule matrix.
- No product-code tests. This is process documentation; verification is by
  plan-shape, wording inspection, and local review.

## Deferred

- Optional mechanical enforcement: a future workflow/process slice can audit
  reviewer verdict text for the `boundary-probe:` line when the PR diff touches
  guard-shaped paths or declares R2/R10 gate risk.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py plans/PR-Reviewer-Boundary-Probe-Contract.md --check`
  - Passed.
- `scripts/local_pr_review.sh`
  - Passed with `tmp/pr-body-reviewer-boundary-probe-contract.md` supplied as
    the current PR body file.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 10 |
| `docs/REVIEWER_RULES.md` | 32 |
| `plans/PR-Reviewer-Boundary-Probe-Contract.md` | 95 |
| **Total** | **137** |
