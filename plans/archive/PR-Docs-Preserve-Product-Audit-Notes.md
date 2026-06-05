# PR-Docs-Preserve-Product-Audit-Notes

## Why this slice exists

Worktree cleanup preserved several local archive artifacts rather than deleting
them blindly. A follow-up comparison showed most archived files are already on
`origin/main`, generated report outputs, or superseded plans. Three product and
audit notes remain useful historical context but are missing from main. This
slice preserves only those notes in-repo and marks them as archived context so
future readers do not mistake stale claims for current product state.

## Scope (this PR)

Ownership lane: docs/product-history-preservation
Slice phase: Workflow/process

1. Restore the two missing audit logs from the preserved markdown archive.
2. Restore the ticket-deflection GTM note from the preserved markdown archive.
3. Add a short archive-status note to each restored document, clarifying that
   current implementation truth remains the live code, plans, and merged PRs.
4. Do not restore generated demand reports, superseded plan drafts, local skill
   files, or PR-body temp files.

### Files touched

- `docs/audits/ai_content_ops_landing_page_user_workflow_gap_log_2026-05-22.md`
- `docs/audits/faq_generator_landing_blog_coupling_gap_log_2026-05-22.md`
- `docs/products/ticket-deflection-gtm.md`
- `plans/PR-Docs-Preserve-Product-Audit-Notes.md`

## Mechanism

The archived notes already exist locally from
`Atlas-primary-untracked-markdown`; this PR commits those three notes after
adding an explicit archived-context banner near the top of each file. The
banner points readers back to current code, merged plans, and PR history for
truth on built capabilities.

No code paths, generated reports, or product behavior change.

## Intentional

- The generated `reports/demand/*.md` outputs stay archived only. The generator
  and its tests are already on main, and committing generated snapshots would
  create maintenance churn without improving runtime behavior.
- `plans/PR-Stripe-Integration-Hardening.md` stays archived only because the
  actual hardening shipped as `PR-Stripe-Billing-Hardening` (#1205).
- The old FAQ deflection handoff PR body stays archived only because PR bodies
  are not source-of-truth repo artifacts.

## Deferred

- If these restored notes become product-owned docs, a future product-docs
  slice can refresh stale claims and move them out of archived context.

Parked hardening: none.

## Verification

- `git diff --check -- $(git diff --name-only origin/main)`
- `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Landing-page workflow audit log | 165 |
| FAQ coupling audit log | 83 |
| Ticket deflection GTM note | 158 |
| Plan doc | 76 |
| **Total** | **482** |

This exceeds the 400 LOC target because the three documents are preserved
together as one cleanup decision: restore useful archived notes, leave all other
archived artifacts out. Splitting would add process overhead without reducing
review risk because the change is docs-only and mechanical.
