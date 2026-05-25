# PR-Content-Ops-FAQ-Search-Concurrency-Review-Fixes

## Why this slice exists

The post-merge review for PR #960 found that the hosted FAQ search concurrency
smoke was structurally useful but too easy to run as a liveness-only check. The
default invocation allowed empty `200 OK` responses, the test suite did not
prove the real worker ran under the executor, and the exit-1 guard path was not
covered end to end.

This slice fixes those review findings at the source while keeping the smoke
thin and operator-focused.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Production hardening.

1. Make hosted concurrency smoke correctness-checking by default.
2. Add an explicit opt-out flag for liveness-only empty-result probes.
3. Keep URL construction inside the worker failure boundary so result artifacts
   still capture malformed invocation failures.
4. Narrow the worker catch to expected route/invocation failures instead of
   swallowing every code bug.
5. Add tests that prove the real executor path runs concurrently and `main()`
   returns exit 1 with a JSON artifact when route responses fail contract checks.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Concurrency-Review-Fixes.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

`--require-results` remains accepted, but the parser now defaults
`require_results=True`. Operators who intentionally want liveness-only behavior
must pass `--allow-empty-results`, which sets `require_results=False` and is
visible in the JSON artifact.

The worker now builds the URL inside its guarded route-call block and catches
only expected runtime/value/type invocation failures. It still records those
failures in the per-request result row so `main()` can write a compact artifact
and return exit 1 instead of crashing before output.

Tests add a barrier-backed `_run_concurrent(...)` case that lets two real worker
threads enter the fetch path at once, plus a `main(...)` route-contract failure
case that writes an output artifact and exits 1.

## Intentional

- No hosted data seeding lands here. This only tightens the smoke behavior added
  in PR #960.
- No latency default is introduced; the review finding was about correctness
  defaults and coverage, not hosted SLO selection.
- `--allow-empty-results` keeps the previous liveness-only mode available, but
  makes it explicit instead of the default.

## Deferred

- Hosted seeded end-to-end load testing with isolated corpora and cleanup
  remains a later production-hardening slice.
- Multi-query mixes remain deferred until representative demo queries are
  available.

## Verification

- pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py -q passed with 13 tests.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . passed with 115 matching tests enrolled.
- python scripts/smoke_content_ops_faq_search_route_concurrency.py --base-url '' --token token-123 --output-result tmp/faq_search_concurrency_review_preflight.json --json returned exit 2 and wrote the preflight JSON artifact with `require_results: true`.
- Transport-boundary tests cover malformed JSON, non-object JSON, bad `results`,
  bad `count`, empty results, fetch failure, and raw timeout detection without
  replacing the checker fetch helper.
- bash scripts/local_pr_review.sh passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| Smoke script | 20 |
| Tests | 166 |
| **Total** | **271** |
