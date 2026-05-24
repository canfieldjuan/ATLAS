# PR-Content-Ops-FAQ-Search-Route-Contract

## Why this slice exists

`PR-Content-Ops-FAQ-Deflection-Host-Mount` mounted the FAQ search route but deferred hosted deployment handoff. The demo session now needs a way to verify that a provided Atlas API URL and bearer token return the documented `{ query, results, count }` envelope before page wiring. This PR exceeds 400 LOC because review identified the checker's negative contract/HTTP paths as under-tested; splitting that coverage out would leave the detector under-proven.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a live-route contract checker for the hosted FAQ search endpoint.
2. Validate the response envelope and first-result fields without logging tokens.
3. Add focused tests for URL construction, validation, auth, and failure paths.
4. Enroll the focused test in extracted pipeline CI.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Route-Contract.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/check_content_ops_faq_search_route_contract.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`

## Mechanism

The script accepts `--base-url`, `--token`, `--query`, optional filters, and `--limit`, then calls `GET <base-url>/api/v1/content-ops/faq-deflection-search` with `Authorization: Bearer <token>`.

The checker fails closed unless the response has string `query`, list `results`, and integer `count` matching `len(results)`. With `--require-results`, it also checks at least one result plus the demo-facing first-result fields: `question`, `answer_summary`, `topic`, `source_ids`, `ticket_count`, and `score`.

## Intentional

- No hosted URL or bearer token is committed; callers pass them via flags or env.
- The script validates the HTTP contract only. It does not seed FAQ documents.
- The script uses Python's standard library to avoid a new HTTP dependency.

## Deferred

- Seeded hosted demo data remains a deployment/ops task.
- Hosted HTTP concurrency remains deferred until a live demo URL/token exists.

## Verification

- `pytest tests/test_check_content_ops_faq_search_route_contract.py -q` passed with 24 tests.
- Python compile check for the route contract script and focused test passed.
- CI enrollment: `scripts/run_extracted_pipeline_checks.sh` includes the focused test, and `.github/workflows/extracted_pipeline_checks.yml` includes the script/test path filters.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 57 |
| CI enrollment | 5 |
| Contract checker script | 187 |
| Focused tests | 256 |
| **Total** | **505** |
