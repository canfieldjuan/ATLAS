# PR-Content-Ops-FAQ-Search-Concurrent-Scope-Integration

## Why this slice exists

#1081 proved mixed-corpus route reads for one authenticated account. The next
deferred gap is cross-account route safety under concurrent HTTP requests. The
production route relies on a per-request `ContextVar` bridge from the host auth
dependency into the extracted FAQ router, so a direct repository test is not
enough to prove that concurrent requests keep their tenant scope isolated.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Robust testing.

1. Add a focused Postgres/FastAPI integration test for concurrent FAQ search and
   detail requests across two accounts.
2. Exercise the real `build_content_ops_scope` / `set_current_auth_user`
   ContextVar bridge with a local fake auth dependency.
3. Seed both accounts with the same query and corpus so any request-scope leak
   returns the wrong FAQ.
4. Keep runtime behavior unchanged.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Concurrent-Scope-Integration.md`
- `tests/test_extracted_ticket_faq_search_api.py`

## Mechanism

The test mounts `create_faq_deflection_search_router(...)` with a dependency
that reads an `x-test-account-id` header and calls `set_current_auth_user(...)`.
The router's `scope_provider` is the same `build_content_ops_scope` bridge used
by the Atlas host route. It seeds two accounts into the real local Postgres
projection tables with the same corpus and query, then uses a thread pool to
issue concurrent search and detail requests for both accounts. Each response
must return only that request's expected account/FAQ/detail item.

The test performs one warm-up request before starting the executor so the
route's lazy asyncpg pool is initialized before concurrent requests fan out.

## Intentional

- The auth dependency is a local fixture, not a new production auth path.
- This does not mint real bearer tokens; deployed cross-account smoke remains a
  separate production-host validation step.
- The test skips when no `EXTRACTED_DATABASE_URL` or `DATABASE_URL` is present,
  matching the existing real-Postgres FAQ route integration test.

## Deferred

- Deployed-host cross-account route stress remains deferred until multiple real
  bearer tokens mapped to different accounts are available.
- Parked hardening: none unless the test exposes a new nonblocking issue.

## Verification

- `python -m py_compile` against `tests/test_extracted_ticket_faq_search_api.py` and `tests/test_atlas_content_ops_scope.py` - passed.
- `pytest -q` against `tests/test_atlas_content_ops_scope.py` and `tests/test_extracted_ticket_faq_search_api.py` - passed, 16 passed / 2 skipped.
- Focused real-DB run with local `.env.local` loaded through `python-dotenv` - passed, 2 integration tests.
- Post-review rerun after the warm-up request fix: `python -m py_compile` against
  `tests/test_extracted_ticket_faq_search_api.py` - passed; focused real-DB run
  with local `.env.local` loaded through `python-dotenv` passed the concurrent
  scope integration test.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~66 |
| Integration test | ~146 |
| **Total** | **~206** |
