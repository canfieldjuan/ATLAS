# PR-Content-Ops-FAQ-Search-Route-Smoke

## Why this slice exists
The FAQ search repository has concurrency and Postgres coverage, and the hosted
route has fake-pool contract tests. The demo handoff consumes the mounted
`/content-ops/faq-deflection-search` route shape, so this slice pins the route
composition against real Postgres instead of only the repository primitive.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a real-Postgres integration test for the FastAPI FAQ search router.
2. Seed one projected FAQ row through the canonical search repository writer.
3. Assert the route returns the documented `{ query, results, count }` envelope.
4. Assert tenant isolation and no-match behavior through the route.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Route-Smoke.md`
- `tests/test_extracted_ticket_faq_search_api.py`

## Mechanism
The integration test applies the FAQ markdown and search migrations, inserts a
parent FAQ markdown row, and writes one search projection row with
`PostgresTicketFAQSearchRepository.replace_documents(...)`. A FastAPI app mounts
`create_faq_deflection_search_router(...)` with a real asyncpg pool provider and
a mutable tenant scope provider. The test calls the route as tenant A and tenant
B, then verifies tenant A receives the seeded result and tenant B receives the
empty documented envelope.

## Intentional
- This is test-only. The existing route already has the required q cap, limit
  cap, timeout wrapper, and generic 503 catch.
- The test skips when no `EXTRACTED_DATABASE_URL` or `DATABASE_URL` is present,
  matching the existing integration-test pattern in this lane.
- The seeded account/corpus/FAQ ids are unique per run and cleaned up through the
  parent `ticket_faq_markdown` row so cascading deletes remove projection rows.

## Deferred
- External deployed-host smoke with bearer token remains covered by
  `scripts/check_content_ops_faq_search_route_contract.py`; this PR does not add
  deployed-environment orchestration.
- Concurrent HTTP route load remains deferred because repository-level
  concurrent retrieval already exists, and this slice only closes route/DB
  composition.

## Verification
- `pytest tests/test_extracted_ticket_faq_search_api.py tests/test_extracted_ticket_faq_search.py -q` passed with 14 tests and 1 skip because no database URL is configured in this checkout.
- Python compile check for the changed route API test passed.

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 58 |
| Integration test | 180 |
| **Total** | **238** |
