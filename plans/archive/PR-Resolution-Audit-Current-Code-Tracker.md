# PR-Resolution-Audit-Current-Code-Tracker

## Why this slice exists

Issue #1993 is the active Resolution Audit CSV remediation issue, but its body
mixes historical audit claims, stale file paths, fixed findings, sample-derived
percentages, and current-code defects. The operator asked for a local and GitHub
place to track the arc and code changes before implementation starts.

Root cause: the issue body is acting as both evidence archive and current
remediation tracker, so stale claims keep looking like current requirements.
This slice fixes the tracking root, not the product defects: it creates a
current-code tracker doc and points #1993 at the verified code-grounded arc.
The tracker is deliberately living and non-exhaustive so S1 can start before
the ledger becomes another blocking project.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Workflow/process

1. Add a local current-code remediation tracker under the existing Resolution
   Audit CSV audit directory.
2. Update issue #1993 so future slices track confirmed current defects, stale
   or contradicted findings, and the implementation order separately.

### Review Contract

- No product code changes.
- The tracker distinguishes confirmed, contradicted, and partially-confirmed
  claims from the independent code audit.
- The tracker names concrete follow-up slices without changing user-facing
  report shape; any user-facing shape/copy changes remain operator-consent
  gated.
- The tracker cites current paths instead of stale `atlas_brain/...` paths.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `plans/PR-Resolution-Audit-Current-Code-Tracker.md`

## Mechanism

The new doc becomes the local source of truth for the arc:

- a "gaps first" section for stale, contradicted, or over-broad issue claims;
- a confirmed current-code fix queue grouped by root cause, including the
  review-corrected queue items for row-key normalization, full money
  reconciliation, runtime QA scorecard wiring, full-thread limits, token-set
  skip behavior, comment privacy, status normalization, missing-ID stability,
  and hosted annualized-field exposure;
- a slice plan that separates tracker-only, code-remediation, performance,
  and user-facing product-surface work;
- a rule that follow-up implementation PRs update the tracker checkbox and
  link the fixing PR.

Issue #1993 remains the GitHub tracker/history bucket. Its body is updated to
point at this local doc and carry a condensed checklist rather than the older
unverified finding dump.

## Intentional

- Keep #1993 instead of replacing it so the historical audit comments and PR
  links stay attached.
- Do not mark any remediation item complete in this slice; this PR only fixes
  the tracking surface.
- Do not change product copy, report shape, parser behavior, or limits here.

## Deferred

- Product-code remediation starts in follow-up vertical/functional slices named
  in the tracker.
- Any user-facing report/snapshot/copy shape changes require operator approval
  before implementation.

Parked hardening: none.

## Verification

- `gh issue view 1993 --json number,title,state,body,comments`
- `gh pr list --state open --json number,title,headRefName,author`
- `git log --oneline -15 origin/main`
- Plan sync helper: `scripts/sync_pr_plan.py`
- Local review helper, advisory dirty pass: `scripts/local_pr_review.sh`
- Clean local review through `scripts/push_pr.sh`: passed before opening #2008.
- Review-thread fetch through the Codex GitHub review-comment workflow.
- Runtime scorecard caller check with ripgrep: confirmed tests/standalone
  scripts exist, but no product/runtime report-generation caller.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 298 |
| `plans/PR-Resolution-Audit-Current-Code-Tracker.md` | 96 |
| **Total** | **394** |
