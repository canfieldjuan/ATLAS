# PR-Content-Ops-FAQ-Search-Seeded-Route-Cases

## Why this slice exists
The hosted FAQ search concurrency smoke can now run mixed query/corpus cases,
but it only validates the generic response envelope. It cannot prove that seeded
corpora return expected rows or that miss cases stay empty.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Add expected count and first-result fields to hosted route case validation.
2. Add account override plus hosted case-file output to the seeded DB smoke.
3. Require kept seed data when writing a hosted route case file.
4. Add negative fixtures for every new checker branch.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Seeded-Route-Cases.md`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`
## Mechanism
The seeded DB smoke can write a hosted route `--case-file` JSON artifact. Hit
cases include seeded corpus, expected count, and expected first result
account/corpus/FAQ IDs. Miss cases set `require_results=false` and
`expected_count=0`.
The hosted route smoke keeps existing no-case-file behavior. When expectations
are present, each request validates actual count and first-result fields after
the envelope check; mismatches become request errors in the JSON summary.
## Intentional
- The DB smoke does not launch hosted HTTP; operators pass the emitted case file
  to the hosted smoke with the correct base URL and token.
- Route case output requires `--keep-data`; cleanup remains explicit so seeded
  rows are not removed before the hosted route reads them.
- Account override is limited to one seeded account because one token maps to
  one tenant.
## Deferred
- One-command seed, hosted HTTP load, and cleanup orchestration.
- Hosted detail-route expectation checks.
## Verification
- pytest tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py -q - 52 passed in 0.13s.
- python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py; git diff --check - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 49 |
| Seeded DB smoke | 68 |
| Hosted route smoke | 79 |
| Tests | 203 |
| **Total** | **399** |
