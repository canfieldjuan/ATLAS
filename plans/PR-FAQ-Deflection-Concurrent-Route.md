# PR-FAQ-Deflection-Concurrent-Route

## Why this slice exists

PR-FAQ-Deflection-Bulk-Route proved one 1000-row `faq_deflection_report` request
through the hosted execute harness. The next survivability question is whether
multiple report requests can run through the same route concurrently without
dropping the customer-facing report envelope or mixing counts between requests.

This slice pins the concurrent route behavior with a deterministic in-process
failure-budget test.

## Scope (this PR)

Ownership lane: content-ops/deflection-report

Slice phase: Production hardening

1. Add a hosted execute harness test for concurrent `faq_deflection_report`
   requests.
2. Verify every concurrent response completes, keeps per-request counts, and
   still emits both proven and no-proven report sections.

### Files touched

| File | Purpose |
|---|---|
| `tests/test_extracted_content_ops_live_execute_harness.py` | Adds the concurrent deflection report route proof. |
| `plans/PR-FAQ-Deflection-Concurrent-Route.md` | Documents this slice contract. |

## Mechanism

The test builds one deterministic in-memory router with an execute concurrency
limit of four, then launches four concurrent `faq_deflection_report` execute
requests through a barriered async service wrapper. The wrapper waits until all
four requests have entered service execution before letting any request render a
report, so the test proves actual overlap rather than only scheduling four
coroutines.

Each request carries 250 support-ticket rows with request-specific source IDs:
175 export tickets with resolution evidence and 75 SSO tickets without
resolution evidence.

The failure budget is zero request errors. The barrier must observe all four
requests waiting at once. Each response must complete with 250 source rows, one
drafted answer bucket, one no-proven bucket, and the customer-facing deflection
report sections intact.

## Intentional

- This is an in-process route harness, not a deployed-host load test.
- No strict wall-clock latency assertion is added because that would make the
  unit test host-sensitive; this slice proves correctness under concurrency.
- The test uses deterministic synthetic SaaS support tickets so the proof stays
  on-domain and external-data-free.

## Deferred

- Parked hardening: none.
- Future production-hardening slice: deployed-host `faq_deflection_report`
  concurrency latency and timeout budgets once the operator-provisioned host is
  available.

## Verification

- Command: python -m pytest tests/test_extracted_content_ops_live_execute_harness.py -q -k "concurrent_faq_deflection"
  - Result: 1 passed, 10 deselected.
- Command: python -m py_compile tests/test_extracted_content_ops_live_execute_harness.py
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Concurrent-Route.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Concurrent-Route.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-concurrent-route.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Route test | 100 |
| Plan doc | 87 |
| **Total** | **187** |
