# PR-FAQ-Macro-Writeback-Publish-Trigger

## Why this slice exists

The FAQ macro writeback lane can now preview approved FAQ items, persist
idempotency mappings, and publish macros through the Zendesk adapter. What is
still missing is the product-level trigger that ties those pieces together:
fetch an approved FAQ draft for one tenant, publish only the verified macro
items, and report any pending Zendesk reconciliation state instead of letting a
retry duplicate external macros.

This lands above the 400-LOC soft target because the service boundary and its
approval/skipped/pending/dry-run tests need to ship together for the trigger to
be reviewable as one real flow.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add a small FAQ macro publish service that fetches tenant-scoped FAQ drafts,
   runs the existing preview double gate, calls an injected macro publish
   provider, and returns a structured summary.
2. Mark a FAQ draft `published` only when the whole draft cleanly publishes:
   no skipped items, at least one publishable macro, and every provider result
   is `published` or `updated`.
3. Surface pending mapping reconciliation as first-class summary state so the
   create-then-mapping-failure path from the Zendesk adapter does not disappear
   behind a generic failure count.
4. Add focused service tests proving approval gating, tenant-scoped repository
   calls, clean publish status transition, skipped-item behavior, and pending
   reconciliation reporting.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Publish-Trigger.md` — plan for this slice.
- `extracted_content_pipeline/faq_macro_writeback_publish.py` — publish trigger service.
- `tests/test_extracted_ticket_faq_macro_writeback_publish.py` — service coverage.
- `scripts/run_extracted_pipeline_checks.sh` — extracted-package CI enrollment for the new service test.

## Mechanism

The new `FAQMacroWritebackPublishService` accepts the existing
`TicketFAQRepository` and `MacroPublishProvider` ports. Its `publish_faq_draft`
method:

1. Calls `faq_repository.get_draft(faq_id, scope=scope)`.
2. Builds a `MacroWritebackPreview` from the returned draft.
3. Calls `provider.publish(preview.macros, scope=scope)` only when the preview
   has publishable macros.
4. Counts successful, failed, skipped, and pending-reconcile outcomes.
5. Calls `faq_repository.update_status(faq_id, "published", scope=scope)` only
   when the draft had no skipped items and every provider result succeeded.

Pending reconciliation is detected by provider result errors ending in
`pending_reconcile`, including the Zendesk adapter's
`zendesk_macro_mapping_pending_reconcile` result.

## Intentional

- No API route in this slice. The service is the narrow product trigger
  boundary; routing, auth wiring, and UI controls belong in the next slice once
  the orchestration contract is proven.
- No live Zendesk lookup/reconcile yet. The adapter already prevents duplicate
  POSTs on pending mappings, and this service surfaces that state so the next
  publish-route slice can decide whether to run a reconcile command, show an
  operator action, or both.
- The draft status changes to `published` only on all-clean results. Partial
  publish, skipped unverified items, missing drafts, and pending reconciliation
  all leave the review status unchanged.

## Deferred

- `PR-FAQ-Macro-Writeback-Publish-Route`: expose this service through a scoped
  backend route / CLI entry point and wire the selected provider.
- `PR-FAQ-Macro-Writeback-Pending-Reconcile`: look up pending Zendesk macros by
  their reserved title/category metadata and backfill `external_id` when a prior
  create succeeded but mapping persistence failed.
- `PR-FAQ-Macro-Writeback-Publish-UI`: add the operator review action once the
  route exists.

Parked hardening: none

## Verification

- python -m pytest tests/test_extracted_ticket_faq_macro_writeback_publish.py — 6 passed.
- python -m py_compile extracted_content_pipeline/faq_macro_writeback_publish.py tests/test_extracted_ticket_faq_macro_writeback_publish.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- python scripts/check_extracted_imports.py — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.
- python scripts/smoke_extracted_pipeline_imports.py — passed.
- python scripts/smoke_extracted_pipeline_standalone.py — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-publish-trigger.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~105 |
| Service | ~165 |
| Tests | ~260 |
| CI enrollment | ~5 |
| Total | ~535 |
