# PR-Content-Ops-FAQ-Search-Postgres-Projection

## Why this slice exists

`PR-Content-Ops-FAQ-Search-Projection-V1` defines the compact FAQ search
document and `{query, results, count}` envelope, but it still searches an
in-memory iterable. That proves the contract, not production retrieval. A live
demo with concurrent users needs a separate persisted search layer so reads do
not scan raw uploaded ticket rows or full generated Markdown blobs.

This slice adds the database projection needed before the hosted
`faq-deflection-search` route can be built.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a Postgres migration for `ticket_faq_search_documents`.
2. Add tenant/corpus/status and full-text search indexes for read latency.
3. Add a Postgres search repository that replaces projected rows for a FAQ draft
   and searches the projection using the #923 response envelope.
4. Preserve replace semantics when a regenerated FAQ has zero searchable rows.
5. Make each projection delete/insert replacement atomic.
6. Fail closed before projection writes when tenant/corpus/status invariants are
   missing.
7. Reject duplicate ranks per projection group before any database write.
8. Register the migration in the extracted package manifest.
9. Add focused tests for replace semantics, tenant/corpus/status SQL filters,
   invalid projection writes, duplicate ranks, real Postgres isolation/FTS, and
   row-to-envelope mapping.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Postgres-Projection.md`
- `atlas_brain/storage/migrations/327_ticket_faq_search_documents.sql`
- `extracted_content_pipeline/storage/migrations/327_ticket_faq_search_documents.sql`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/ticket_faq_search.py`
- `tests/test_extracted_ticket_faq_search_postgres.py`

## Mechanism

The migration creates one projected row per FAQ item:

- `account_id`, `corpus_id`, `status` for isolation and filtering.
- `faq_id`, `target_id`, `target_mode`, `rank` for traceability.
- `topic`, `question`, `answer_summary`, `source_ids`, `ticket_count`, and
  `search_text` for rendering and retrieval.
- a generated `search_vector` with a GIN index for Postgres full-text search.
- CHECK constraints for non-blank tenant/corpus/status plus positive rank and
  non-negative ticket counts.

`PostgresTicketFAQSearchRepository.replace_documents(...)` deletes existing
projection rows for each `(account_id, corpus_id, faq_id)` group and inserts the
new documents inside a transaction boundary. This replacement behavior prevents
stale FAQ item rows after a draft is regenerated with different item counts, and
rolls back to the prior projection if an insert fails after the delete.

Callers can pass explicit `TicketFAQSearchProjectionKey` values through
`replace_keys` so a regenerated FAQ with zero searchable rows still clears the
prior projection rows for that FAQ/corpus.

Before any delete/insert runs, `replace_documents(...)` validates the projection
document invariants that the table also enforces. Invalid projection rows and
duplicate ranks within a projection group raise `ValueError` without touching
the pool, preventing silent overwrite under the `(account_id, corpus_id, faq_id,
rank)` unique key.

`search(...)` uses `websearch_to_tsquery('english', query)` and scopes every read
by `account_id`, optional `corpus_id`, and optional `status`. It returns the same
`TicketFAQSearchResponse.as_dict()` envelope introduced in #923.

The integration test applies the parent FAQ and search projection migrations to
the configured Postgres database, seeds tenant-isolated rows, verifies FTS hit
and miss behavior through the repository, and uses a scoped trigger to force an
insert failure after delete so rollback is checked against the real table.

## Intentional

- The hosted FastAPI route is still deferred. This PR is only the durable search
  layer behind that future route.
- The repository replaces projection rows for a FAQ instead of patching per item;
  FAQ generation emits a complete item list, so replace semantics are simpler
  and avoid stale rows.
- No vector/embedding search is added. Deterministic Postgres full-text search is
  sufficient for the first production retrieval seam and easier to harden.

## Deferred

- `PR-Content-Ops-FAQ-Deflection-Search-Route`: expose the hosted route that
  calls this repository and returns `{query, results, count}`.
- `PR-Content-Ops-FAQ-Search-Concurrency-Smoke`: seed multiple corpora and run
  concurrent filtered searches with latency and isolation assertions.
- Search snippets/highlighting and synonym expansion remain deferred until the
  route exists.

## Verification

- pytest tests/test_extracted_ticket_faq_search_postgres.py tests/test_extracted_ticket_faq_search.py -q - 15 passed, 1 skipped without DB URL.
- pytest tests/test_extracted_ticket_faq_search_postgres.py::test_ticket_faq_search_contract_against_postgres -q - 1 passed with EXTRACTED_DATABASE_URL constructed from local `.env.backup` ATLAS_DB_* fields.
- python -m py_compile extracted_content_pipeline/ticket_faq_search.py tests/test_extracted_ticket_faq_search_postgres.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline - passed.
- bash scripts/local_pr_review.sh --allow-dirty - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 127 |
| Migration source + synced target | 88 |
| Search repository | 243 |
| Tests | 470 |
| Manifest mapping | 4 |
| **Total** | **932** |

This is above the 400 LOC target because the migration, durable repository, and
tests are indivisible for a real persisted retrieval seam. The additional
projection invariant checks and review-driven transactional replacement semantics
keep tenant isolation and stale-row correctness enforced at the source rather
than relying only on the future route. The hosted route and concurrency smoke
stay deferred to keep this slice bounded.
