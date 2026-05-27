# PR-Content-Ops-FAQ-Route-Concurrency-Runbook

## Why this slice exists

The hosted FAQ search route concurrency smoke now has mixed-case visibility,
detail hydration checks, detail latency budgets, and opt-in per-case error
budgets. Those controls are only discoverable from the CLI and recent PRs. A
demo or go-live operator needs one stable runbook that shows the safe invocation
shape, required environment variables, output artifacts, and how to interpret
the new per-case signals.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add a hosted FAQ route concurrency runbook under `docs/extraction/validation/`.
2. Document the recommended detail-required hit smoke command.
3. Document separate miss/liveness coverage that omits detail hydration checks.
4. Name the key fail-closed budget flags and result fields operators should
   inspect.
5. Add a focused test that keeps the runbook flags aligned with the CLI parser.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Runbook.md`
- `docs/extraction/validation/content_ops_faq_route_concurrency_runbook.md`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The runbook is a static Markdown operator note. The test reads the Markdown,
asserts that the documented command references the current hardening flags,
uses the documented bearer-token fallback, separates detail-required hit cases
from no-detail miss cases, and then parses those flags through
`smoke_content_ops_faq_search_route_concurrency` so future flag renames or
removals break locally.

## Intentional

- No runtime behavior changes. The prior slices already implemented the checks.
- No live host or token is checked into docs. The runbook uses environment
  placeholders and keeps credentials operator-owned.
- The runbook documents opt-in budgets, not default SLO values.

## Deferred

Parked hardening: none.

Live threshold recommendations remain deferred until hosted runs provide stable
latency and error-rate evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py
  (62 passed)
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Runbook.md
  (passed)
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  (122 matching tests enrolled)
- bash scripts/run_extracted_pipeline_checks.sh
  (2561 passed, 7 skipped)
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-runbook.md
  (passed)

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 75 |
| Runbook | 101 |
| Test | 52 |
| **Total** | **228** |
