## Why this slice exists

PR-FAQ-Macro-Writeback-Publish-History added append-only publish attempt writes for FAQ macro writeback. Operators still cannot read that history through the generated asset API, so the UI cannot show whether the last macro publish succeeded, failed, or needs reconcile.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add a serializable publish-attempt DTO and repository contract for listing recent attempts.
2. Implement the Postgres list query against the existing `ticket_faq_macro_publish_attempts` table, scoped by account and FAQ draft id.
3. Add a generated asset API route for reading recent attempts for one FAQ draft.
4. Cover the repository and route behavior with focused tests.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Publish-History-Read-Route.md` — slice plan.
- `extracted_content_pipeline/faq_macro_writeback_publish.py` — read DTO and repository contract.
- `extracted_content_pipeline/faq_macro_writeback_postgres.py` — tenant-scoped attempt history query.
- `extracted_content_pipeline/api/generated_assets.py` — FAQ-only publish attempt read route.
- `tests/test_extracted_ticket_faq_macro_writeback_postgres.py` — Postgres adapter coverage.
- `tests/test_extracted_content_asset_api.py` — API route coverage.

## Mechanism

The write-side table already stores normalized publish-attempt summaries. This slice adds `FAQMacroPublishAttempt` as the read DTO and extends the attempt repository port with `list_attempts(faq_id, scope, limit)`. The Postgres adapter filters by `account_id` and `faq_draft_id`, orders newest-first, and caps results with the requested limit.

The API route is:

```text
GET /content-assets/faq_markdown/drafts/{draft_id}/publish-macro-attempts?limit=20
```

It rejects non-FAQ assets, validates the draft id as a UUID, confirms the FAQ draft exists under tenant scope, then returns recent attempts.

## Intentional

- No UI change in this slice. The route is the stable backend contract the review drawer can consume next.
- No new migration. The table and indexes landed in the prior publish-history slice.
- The route verifies the FAQ draft exists before listing history. An empty history for an existing FAQ returns `200` with no attempts; a missing FAQ returns `404`.

## Deferred

- PR-FAQ-Macro-Writeback-Publish-History-UI will surface this history in the Generated Asset Review drawer.
- Parked hardening: none.

## Verification

- `python -m py_compile extracted_content_pipeline/faq_macro_writeback_publish.py extracted_content_pipeline/faq_macro_writeback_postgres.py extracted_content_pipeline/api/generated_assets.py` — passed.
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_content_asset_api.py -k 'macro_publish_attempt or publish_attempt or publish_macros' -q` — 9 passed, 68 deselected.
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_content_asset_api.py -q` — 77 passed.
- `git diff --check` — passed.
- `bash scripts/validate_extracted_content_pipeline.sh` — passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` — passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` — passed.
- `bash scripts/check_ascii_python.sh` — passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` — passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-publish-history-read-route.md` — passed; cross-layer caller hints were advisory and covered by the focused route/repository tests above.

## Estimated diff size

| Area | Estimate |
|---|---:|
| Plan | ~65 |
| DTO / repository contract | ~40 |
| Postgres adapter | ~60 |
| API route | ~45 |
| Tests | ~130 |
| Total | ~340 |
