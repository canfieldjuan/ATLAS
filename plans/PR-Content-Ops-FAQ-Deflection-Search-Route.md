# PR-Content-Ops-FAQ-Deflection-Search-Route

## Why this slice exists

`PR-Content-Ops-FAQ-Search-Postgres-Projection` adds a durable Postgres search
projection, but the demo and downstream client still need a hosted API seam.
The next thin end-to-end step is a FastAPI route that takes a user query,
resolves the host tenant scope, reads the projection, and returns the documented
`{query, results, count}` envelope.

This slice is stacked on the Postgres projection PR while that PR is in review.
After the projection lands, this branch should rebase onto `origin/main` before
opening the GitHub PR.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a host-mountable FastAPI router for `/content-ops/faq-deflection-search`.
2. Wire the route to `PostgresTicketFAQSearchRepository` through injected pool
   and tenant-scope providers.
3. Enforce the go-live request guards at the route boundary: required tenant,
   non-blank capped query, bounded limit, optional corpus/status filters, and a
   server-side search timeout.
4. Return the projection repository's `{query, results, count}` envelope without
   reshaping result rows.
5. Register the new API module in the extracted package manifest.
6. Add focused route tests for the envelope, SQL filter wiring, request guards,
   limit capping, and unavailable database handling.

### Files touched

- `plans/PR-Content-Ops-FAQ-Deflection-Search-Route.md`
- `extracted_content_pipeline/api/faq_search.py`
- `extracted_content_pipeline/manifest.json`
- `tests/test_extracted_ticket_faq_search_api.py`

## Mechanism

The new router factory follows the existing extracted API pattern:

- `pool_provider` is host-injected and resolved per request.
- `scope_provider` is host-injected and may return `TenantScope`, a mapping, or
  `None`.
- Auth remains host-owned through FastAPI `dependencies`; the package route only
  enforces that a non-blank `account_id` reaches the repository.

`GET /content-ops/faq-deflection-search` accepts:

- `q`: required search text, stripped and capped by config.
- `corpus_id`: optional corpus filter.
- `status`: optional status filter; omitted uses the configured default
  (`approved`), while an empty string searches all statuses.
- `limit`: optional result cap clamped to the configured max.

The route calls `PostgresTicketFAQSearchRepository.search(...)` under
`asyncio.wait_for(...)` when a timeout is configured, then returns
`TicketFAQSearchResponse.as_dict()` unchanged.

## Intentional

- No Atlas global app mount or deployed host URL is added here. This is the
  package-level router factory; host wiring and bearer-token deployment remain
  separate.
- No client/demo code changes land here. The client should point at this route
  only after the host mounts it.
- No query highlighting, synonyms, or embedding search are added. The route is
  deliberately thin over the deterministic Postgres projection.

## Deferred

- `PR-Content-Ops-FAQ-Deflection-Host-Mount`: mount this route in the Atlas host
  app, set auth dependencies, and document the deployed URL/token handoff.
- `PR-Content-Ops-FAQ-Search-Concurrency-Smoke`: run concurrent filtered
  retrieval against seeded corpora and capture latency/isolation output.
- Client-side abort timeout and query cap remain in the demo/client lane.

## Verification

- pytest tests/test_extracted_ticket_faq_search_api.py tests/test_extracted_ticket_faq_search_postgres.py tests/test_extracted_ticket_faq_search.py -q - 21 passed, 1 skipped without DB URL.
- python -m py_compile extracted_content_pipeline/api/faq_search.py tests/test_extracted_ticket_faq_search_api.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 102 |
| API router | 197 |
| Manifest entry | 3 |
| Route tests | 174 |
| **Total** | **476** |

This is above the 400 LOC target because the route, request guards, and tests
are the smallest useful hosted seam. The host mount and concurrency smoke stay
deferred.
