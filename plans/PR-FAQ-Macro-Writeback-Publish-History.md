# PR-FAQ-Macro-Writeback-Publish-History

## Why this slice exists

The FAQ macro writeback path now works from the dashboard and has a functional
validation proving tenant credentials, idempotency mapping, and Zendesk publish
behavior. The remaining operational gap is that a publish response only exists
in the current request. If an operator needs to understand what happened after
the page refreshes, there is no durable record of whether a publish updated,
failed, skipped, or needed pending reconciliation.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add an append-only Postgres table for FAQ macro publish attempts.
2. Add a small publish-attempt repository port and Postgres adapter.
3. Have `FAQMacroWritebackPublishService` record the final publish summary when
   an attempt reaches a real FAQ draft.
4. Wire the generated asset publish route to the Postgres attempt repository.
5. Add focused service, repository, and route tests proving clean and
   non-clean summaries are persisted with tenant scope.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Publish-History.md` -- plan for this slice.
- `extracted_content_pipeline/api/generated_assets.py` -- route wiring for publish history.
- `extracted_content_pipeline/faq_macro_writeback_postgres.py` -- Postgres attempt persistence.
- `extracted_content_pipeline/faq_macro_writeback_publish.py` -- attempt repository port and service hook.
- `extracted_content_pipeline/storage/migrations/330_ticket_faq_macro_publish_attempts.sql` -- append-only attempt table.
- `tests/test_extracted_content_asset_api.py` -- route wiring coverage.
- `tests/test_extracted_ticket_faq_macro_writeback_postgres.py` -- attempt repository and migration coverage.
- `tests/test_extracted_ticket_faq_macro_writeback_publish.py` -- service history hook coverage.

## Mechanism

The publish service gains an optional `attempt_repository`. After it builds the
final summary for a found FAQ draft, and after any clean status transition, it
calls `record_attempt(summary, scope=scope)`.

The Postgres adapter inserts into `ticket_faq_macro_publish_attempts` with:

- tenant scope and FAQ draft id
- `ok`, draft status, counts, and status-transition flag
- skipped items and provider results as JSONB

The generated asset route constructs the service with both
`PostgresTicketFAQRepository(pool)` and
`PostgresFAQMacroPublishAttemptRepository(pool)`, so the hosted route persists
the same summary it returns.

## Intentional

- No history read API or UI in this slice. This closes the write-side audit gap;
  surfacing the attempts can be a separate product slice once the data exists.
- Missing FAQ drafts are not recorded. The route returns 404, and there is no
  resolved draft to audit.
- The attempt record is written after the final summary is known, so the stored
  row matches the response semantics exactly.

## Deferred

- `PR-FAQ-Macro-Writeback-Publish-History-Review-UI`: display recent attempts
  in Generated Asset Review if operators need the audit trail in the drawer.
- `PR-FAQ-Macro-Writeback-Live-Smoke`: optional sandbox Zendesk smoke once safe
  credentials and test data are available.

Parked hardening: none

## Verification

- python -m pytest tests/test_extracted_ticket_faq_macro_writeback_publish.py -q -- 8 passed.
- python -m pytest tests/test_extracted_ticket_faq_macro_writeback_publish.py tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_content_asset_api.py -k 'publish_macros or macro_writeback or publish_attempt or record_publish_attempt or ticket_faq_macro' -q -- 22 passed, 59 deselected.
- python -m py_compile extracted_content_pipeline/faq_macro_writeback_publish.py extracted_content_pipeline/faq_macro_writeback_postgres.py extracted_content_pipeline/api/generated_assets.py tests/test_extracted_ticket_faq_macro_writeback_publish.py tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_content_asset_api.py -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- passed; 135 matching tests enrolled.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- python scripts/check_extracted_imports.py -- passed.
- python scripts/smoke_extracted_pipeline_imports.py -- passed.
- python scripts/smoke_extracted_pipeline_standalone.py -- passed.
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline -- passed.
- git diff --check -- passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-publish-history.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~97 |
| Migration | ~36 |
| Service/adapter | ~80 |
| Route wiring | ~2 |
| Tests | ~199 |
| Total | ~414 |

This is above the 400 LOC soft cap after review because the fix adds
source-level failure isolation and regression coverage for both audit-write
exceptions and unscoped publish attempts.
