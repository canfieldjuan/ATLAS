# PR-FAQ-Deflection-Bulk-Route

## Why this slice exists

`faq_markdown` has route-level 1000-row execution proof, but
`faq_deflection_report` only has a small hosted-route artifact test. The
customer-facing report wraps the FAQ generator and adds its own Markdown and
summary partitioning, so it needs the same scale proof before we call the report
path robust.

This slice demonstrates that the full deflection report route can process a
1000-row support-ticket corpus and still emit the expected report envelope,
counts, and proven/no-proven answer sections.

## Scope (this PR)

Ownership lane: content-ops/deflection-report

Slice phase: Robust testing

1. Add a hosted execute harness test for `faq_deflection_report` with 1000
   support-ticket source rows.
2. Verify the returned artifact includes top-level report Markdown, compact
   summary counts, nested FAQ result counts, and both drafted/no-proven sections.

### Files touched

| File | Purpose |
|---|---|
| `tests/test_extracted_content_ops_live_execute_harness.py` | Adds the 1000-row deflection report route proof. |
| `plans/PR-FAQ-Deflection-Bulk-Route.md` | Documents this slice contract. |

## Mechanism

The test builds a deterministic in-memory router with
`FAQDeflectionReportService`, submits 1000 inline `support_ticket` rows through
`/content-ops/execute`, and asserts the route returns a completed
`faq_deflection_report` step. The corpus mixes repeated export tickets with
resolution evidence and repeated SSO tickets without resolution evidence so the
report must produce both `Drafted Answers With Proven Solutions` and
`No Proven Answer Yet`.

## Intentional

- This is a route/harness proof, not a live deployed-host test.
- No new production behavior is introduced; the slice only pins the existing
  behavior under a larger representative input.
- The test uses deterministic synthetic SaaS support tickets so the robust
  proof is on-domain and does not depend on external data.

## Deferred

- Parked hardening: none.
- Future production-hardening slice: concurrent `faq_deflection_report` route
  latency and failure-budget proof once this single-request bulk path is pinned.

## Verification

- Command: python -m pytest tests/test_extracted_content_ops_live_execute_harness.py -q -k "bulk_faq_deflection"
  - Result: 1 passed, 9 deselected.
- Command: python -m py_compile tests/test_extracted_content_ops_live_execute_harness.py
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Bulk-Route.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Bulk-Route.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-bulk-route.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Route test | 76 |
| Plan doc | 78 |
| **Total** | **154** |
