# PR-Content-Ops-FAQ-Search-Detail-Contract

## Why this slice exists
PR-Content-Ops-FAQ-Search-Detail added the tenant-scoped full FAQ hydration
route, but the deployed smoke checker still validates only the compact search
envelope. The demo handoff needs one command that can prove the hosted API can
return search results and hydrate the selected generated FAQ by `faq_id`.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Extend the hosted FAQ search contract checker with an optional detail
   hydration check.
2. Reuse the existing search route as the default detail route root:
   `/api/v1/content-ops/faq-deflection-search/{faq_id}`.
3. Validate the detail envelope shape without writing full Markdown into the
   result artifact.
4. Add fixture tests for URL construction, malformed search result ids, valid
   detail payloads, and detail contract failures while preserving the checker
   branch coverage merged in PR #954.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Detail-Contract.md`
- `scripts/check_content_ops_faq_search_route_contract.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`

## Mechanism
The checker gains `--require-detail` and an optional `--detail-route` override.
When detail is required, the search response must include a first result object
with a string `faq_id`. The checker calls the detail URL with the same bearer
token and timeout, validates the full FAQ artifact fields, and writes only
compact proof fields (`detail_checked`, `detail_route`, `detail_faq_id`) into
the JSON result. Markdown and item bodies stay out of the result file.

## Intentional
- Detail validation is opt-in so the existing lightweight search smoke remains
  unchanged for environments that only need the compact route.
- The result artifact does not duplicate Markdown content; it records proof that
  the detail payload was fetched and validated.
- The script does not seed data. `--require-results --require-detail` remains a
  hosted-environment check that expects an already indexed approved FAQ.

## Deferred
- A public/redacted detail contract is deferred until a public detail surface
  exists. This checker targets the authenticated content-ops API.
- Latency thresholds for search plus detail are deferred; this slice verifies
  contract shape, not performance budgets.

## Verification
- pytest tests/test_check_content_ops_faq_search_route_contract.py -q passed with 50 tests after rebasing over PR #954.
- python -m compileall scripts/check_content_ops_faq_search_route_contract.py tests/test_check_content_ops_faq_search_route_contract.py passed.
- Pending local run: bash scripts/local_pr_review.sh

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 60 |
| Checker script | 120 |
| Tests | 160 |
| **Total** | **340** |
