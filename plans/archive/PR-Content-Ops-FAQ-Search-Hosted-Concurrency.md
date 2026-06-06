# PR-Content-Ops-FAQ-Search-Hosted-Concurrency

## Why this slice exists

PR-Content-Ops-FAQ-Search-Contract-Latency added single-request timing budgets
to the hosted FAQ search contract checker. That still does not exercise the
read-path failure Claude called out: several demo users querying the hosted
route at the same time can fail differently than one well-formed request.

This slice adds the thinnest hosted concurrency smoke for the existing
`/api/v1/content-ops/faq-deflection-search` route. It does not seed data or
change the route; it gives operators a repeatable way to pressure the deployed
read envelope with a real URL and token.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Robust testing.

1. Add a hosted FAQ search route concurrency smoke script.
2. Reuse the existing route URL builder, fetcher, and envelope validator.
3. Run a caller-specified number of concurrent search requests.
4. Emit compact JSON with request totals, error rate, p50/p95/max latency, and
   budget failures.
5. Add focused unit tests for argument validation, latency/error summaries, and
   mixed success/failure concurrent execution.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Hosted-Concurrency.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The new smoke reads the same base URL, token, query, corpus/status filters,
limit, route, and timeout inputs used by
`check_content_ops_faq_search_route_contract.py`. It uses a
`ThreadPoolExecutor` to issue blocking hosted HTTP reads concurrently, validates
each response with the existing search envelope validator, and records each
request as a compact result row.

The summary fails when any request raises, any response violates the envelope,
the observed error rate exceeds `--max-error-rate`, or optional p95/max latency
budgets are exceeded. By default the error-rate budget is zero; latency budgets
remain opt-in because hosted environments own their own thresholds.

## Intentional

- No data seeding lands here. Hosted route concurrency needs a deployed route
  and token; test data preparation remains an operator responsibility.
- No detail-route hydration lands here. This smoke focuses on concurrent compact
  search reads.
- No default latency SLO lands here. Operators pass environment-specific p95/max
  budgets when they have a baseline.

## Deferred

- A hosted seeded end-to-end load test that creates isolated corpora, searches
  them over HTTP, and cleans them up remains a later production-hardening slice.
- Multi-query mixes are deferred until we have representative demo queries from
  the deployed data set.

## Verification

- pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py -q passed with 7 tests.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . passed with 115 matching tests enrolled.
- python scripts/smoke_content_ops_faq_search_route_concurrency.py --base-url '' --token token-123 --output-result tmp/faq_search_hosted_concurrency_preflight.json --json returned exit 2 and wrote the preflight JSON artifact.
- Pending local run: bash scripts/local_pr_review.sh

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 86 |
| CI runner | 1 |
| Smoke script | 254 |
| Tests | 182 |
| **Total** | **523** |

This is over the 400 LOC target because the slice needs an operator script,
result artifact, validation, budget accounting, and tests to be usable end to
end. Route/data seeding stays deferred rather than expanding the PR further.
