# PR-Content-Ops-FAQ-Search-Contract-Latency

## Why this slice exists
PR-Content-Ops-FAQ-Search-Detail-Contract added hosted shape validation for the
compact search route plus optional full FAQ detail hydration, but explicitly
deferred latency thresholds. The demo handoff now has a contract checker that can
prove response shape; this slice lets the same checker fail when a hosted
environment exceeds caller-provided latency budgets.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Robust testing.

1. Time the hosted search request and optional detail request.
2. Add opt-in maximum search, detail, and total elapsed-millisecond budget flags.
3. Record compact timing and budget fields in the JSON result artifact.
4. Fail the checker when any supplied latency budget is exceeded.
5. Add focused tests for passing and failing latency budgets.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Contract-Latency.md`
- `scripts/check_content_ops_faq_search_route_contract.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`

## Mechanism
The checker wraps `_fetch_json(...)` with a small monotonic timer and records
search, detail, and total elapsed milliseconds. Three optional flags are added:
`--max-search-ms`, `--max-detail-ms`, and `--max-total-ms`, with matching
environment variables. When omitted, the checker remains contract-only and never
fails on timing. When present, the checker compares the measured durations to
the supplied budgets, appends deterministic contract errors for exceeded
budgets, and writes budget values plus elapsed timings to `--output-result`.

## Intentional
- No default latency threshold is introduced. Hosted environments own their
  budgets.
- This remains a single-request contract checker, not a concurrent load test.
- Timing fields are compact numbers only; response bodies and Markdown remain out
  of the result artifact.

## Deferred
- Concurrent hosted route load testing remains a separate production-hardening
  slice because it needs hosted URL/token coordination and a different execution
  shape than this single-request checker.
- Environment-specific SLO defaults remain outside the repo.

## Verification
- pytest tests/test_check_content_ops_faq_search_route_contract.py -q passed with 56 tests.
- python -m compileall scripts/check_content_ops_faq_search_route_contract.py tests/test_check_content_ops_faq_search_route_contract.py passed.
- ATLAS_FAQ_SEARCH_MAX_SEARCH_MS=bad python scripts/check_content_ops_faq_search_route_contract.py --base-url https://atlas.example.invalid --token token returns argparse exit 2 with no traceback.
- Pending local run: bash scripts/local_pr_review.sh

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 61 |
| Checker script | 174 |
| Tests | 132 |
| **Total** | **367** |
