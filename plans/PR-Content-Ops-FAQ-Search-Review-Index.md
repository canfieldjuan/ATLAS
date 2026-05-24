# PR-Content-Ops-FAQ-Search-Review-Index

## Why this slice exists
The FAQ search route is mounted and contract-checked, but normal generated FAQ
drafts still only persist to `ticket_faq_markdown`. This slice wires review
actions to `ticket_faq_search_documents` so approved FAQ drafts feed the route.
The slice is over the usual 400 LOC budget because review validation exposed the
same write path in CLI fakes, lifecycle smoke fakes, and the real Postgres
composition. Keeping those fixes with the projection change prevents the PR from
shipping a route that works in one harness but fails in another.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a shared projection-key helper so empty FAQ items can clear stale rows.
2. Update single and batch FAQ status updates to replace search projections.
3. Keep non-FAQ asset review behavior unchanged.
4. Add focused tests proving FAQ review writes/deletes search projection rows.
5. Keep legacy/no-scope review updates from rolling back when they cannot build a
   search projection key.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Review-Index.md`
- `extracted_content_pipeline/ticket_faq_search.py`
- `extracted_content_pipeline/ticket_faq_postgres.py`
- `tests/test_extracted_ticket_faq_search.py`
- `tests/test_extracted_ticket_faq_postgres.py`
- `tests/test_extracted_ticket_faq_search_postgres.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_extracted_content_asset_review_cli.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism
`PostgresTicketFAQRepository.update_status(...)` and `update_statuses(...)`
switch to `UPDATE ... RETURNING` the same columns used by `list_drafts`.
Returned drafts are projected through `build_ticket_faq_search_documents(...)`
and written through `PostgresTicketFAQSearchRepository.replace_documents(...)`.
Every updated draft also supplies a `TicketFAQSearchProjectionKey`. If a draft
has no searchable FAQ items, the replace still deletes stale rows for that
`account_id` / `corpus_id` / `faq_id` group.
If a legacy/no-scope review update cannot supply `account_id`, `corpus_id`, or
`faq_id`, the status update still commits and skips search projection instead of
raising inside the review path.

## Intentional
- Search projection indexing stays in the FAQ repository, not the FastAPI route.
- Statuses other than `approved` are projected with their actual status. The
  search API defaults to `approved`, so rejected/draft rows stop appearing
  without requiring a special delete path.
- No-scope review updates keep the prior status-update behavior. They do not
  create search rows because the hosted route is tenant-scoped and the projection
  key is intentionally fail-closed.

## Deferred
- Backfilling already-approved FAQ drafts remains an ops/backfill slice.

## Verification
- `pytest tests/test_extracted_ticket_faq_search.py tests/test_extracted_ticket_faq_postgres.py tests/test_extracted_content_asset_api.py::test_generated_asset_router_reviews_ticket_faq_with_host_defined_status -q` passed with 16 tests.
- `pytest tests/test_extracted_content_asset_api.py -q` passed with 59 tests.
- `pytest tests/test_extracted_content_asset_review_cli.py::test_asset_review_cli_updates_ticket_faq_status tests/test_smoke_content_ops_faq_lifecycle.py::test_faq_lifecycle_smoke_generates_exports_reviews_and_reexports tests/test_smoke_content_ops_faq_lifecycle.py::test_faq_lifecycle_smoke_persists_1000_row_json_bundle -q` passed with 3 tests.
- `pytest tests/test_extracted_ticket_faq_postgres.py tests/test_extracted_ticket_faq_search.py tests/test_extracted_ticket_faq_search_postgres.py -q` passed with 26 tests and 2 skips when no database URL is configured.
- Full extracted pipeline check via `scripts/run_extracted_pipeline_checks.sh` passed with 2004 tests, 1 skip, and 1 third-party warning.
- Python compile check for changed FAQ/search modules and focused tests passed.
- Package guardrails and `bash scripts/local_pr_review.sh --allow-dirty origin/main` passed.

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 76 |
| Search helper | 18 |
| FAQ repository | 131 |
| Tests | 387 |
| **Total** | **612** |
