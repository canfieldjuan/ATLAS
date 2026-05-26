# PR-Content-Ops-FAQ-Search-Seeded-Detail-E2E

## Why this slice exists
#967 gave us a one-command seeded hosted FAQ search e2e, and #969/#971 made
cleanup safer and more visible. The remaining deferred gap is that the e2e
proves search results but does not prove a returned FAQ ID can be dereferenced
through the hosted detail route into the full generated FAQ shape.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Add a detail-contract step to the seeded hosted FAQ search e2e runner.
2. Reuse the existing `check_content_ops_faq_search_route_contract.py`
   checker with `--require-results --require-detail`.
3. Select the detail probe from the seeded hit cases already written for the
   hosted search smoke.
4. Keep teardown unchanged so seeded rows are still cleaned up after route or
   detail failures.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Seeded-Detail-E2E.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
## Mechanism
After the DB seed smoke writes the route case artifact, the e2e runner reads the
first case with `require_results: true` and builds a call to the existing
contract checker. The checker re-runs the hosted search for that case, follows
`results[0].faq_id` through the detail route, and validates the full FAQ detail
envelope. The e2e summary gains a `detail` section next to `seed`, `route`, and
`cleanup`.
## Intentional
- No new detail-route validator is introduced; this composes the existing
  contract checker so the hosted route shape stays single-sourced.
- The detail check runs only after seed and route success to keep failures
  attributable. Cleanup still runs regardless of route/detail outcome.
- A `--skip-detail-check` escape hatch is included for local liveness/debug
  runs, but the default e2e path exercises the real search-to-detail flow.
## Deferred
- Latency budgets for the detail contract checker remain a standalone checker
  concern; this slice wires correctness into the seeded e2e.
- Rowcount mismatch gating remains deferred from #971.
## Verification
- pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q - 42 passed in 0.09s.
- python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Seeded-Detail-E2E.md - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 47 |
| E2E runner | 108 |
| Tests | 180 |
| **Total** | **340** |
