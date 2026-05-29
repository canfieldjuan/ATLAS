# PR-FAQ-Macro-Writeback-Pending-Reconcile

## Why this slice exists

The Zendesk adapter now prevents duplicate macro creation by leaving an
idempotency row in `pending` when Zendesk creates a macro but mapping
persistence fails. The publish route surfaces that as
`zendesk_macro_mapping_pending_reconcile`, but the row currently has no
self-heal path. This slice adds the missing source-level fix: when a pending row
exists, search Zendesk for the reserved macro title, backfill the mapping, and
continue through the normal update path instead of staying stuck forever.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add pending mapping reconciliation inside `ZendeskMacroPublishProvider`.
2. Search Zendesk macros by the reserved title stored in pending mapping
   metadata, falling back to the current macro title.
3. Require an exact normalized title match before accepting a search result.
4. Backfill the mapping with the reconciled external macro id and continue the
   existing PUT update path.
5. Add focused tests for successful reconciliation, no-match behavior, and safe
   exact-match rejection.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Pending-Reconcile.md` — plan for this slice.
- `extracted_content_pipeline/faq_macro_writeback_zendesk.py` — pending mapping reconciliation.
- `tests/test_extracted_ticket_faq_macro_writeback_zendesk.py` — adapter coverage.

## Mechanism

When `_publish_one` finds an existing mapping with blank `external_id`, it calls
a new reconcile helper before returning `zendesk_macro_mapping_pending_reconcile`.
The helper:

1. Reads `metadata["title"]` from the pending mapping, falling back to
   `macro.title`.
2. Calls Zendesk's official macro search endpoint:
   `/api/v2/macros/search?query=<title>`.
3. Accepts only a result whose normalized `title` exactly matches the requested
   title.
4. Upserts the mapping with the found `external_id`, `external_url`, and the
   reserved metadata.
5. Returns the reconciled mapping so the existing PUT path updates the macro
   body and returns `updated`.

No match still returns the existing pending-reconcile failure, so retries remain
duplicate-safe.

## Intentional

- Reconcile is title-exact, not fuzzy. A fuzzy match could attach one FAQ item
  to the wrong Zendesk macro.
- Reconcile does not POST when a pending mapping exists. The pending row is the
  idempotency guard; reposting would reintroduce the duplicate-macro bug.
- Category is retained in metadata but not required for matching, because the
  Zendesk search response shape does not reliably include the same local
  category metadata we stored.

## Deferred

- `PR-FAQ-Macro-Writeback-Pending-Reconcile-CLI`: optional operator command to
  scan and reconcile all pending rows without waiting for a publish retry.
- `PR-FAQ-Macro-Writeback-Tenant-Credentials`: tenant-scoped encrypted
  credential storage for multi-customer live writeback.
- `PR-FAQ-Macro-Writeback-Publish-UI`: review UI action for the macro publish
  route.

Parked hardening: none

## Verification

- python -m pytest tests/test_extracted_ticket_faq_macro_writeback_zendesk.py -q — 9 passed.
- python -m py_compile extracted_content_pipeline/faq_macro_writeback_zendesk.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- python scripts/check_extracted_imports.py — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.
- python scripts/smoke_extracted_pipeline_imports.py — passed.
- python scripts/smoke_extracted_pipeline_standalone.py — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-pending-reconcile.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Adapter | ~101 |
| Tests | ~95 |
| Total | ~291 |
