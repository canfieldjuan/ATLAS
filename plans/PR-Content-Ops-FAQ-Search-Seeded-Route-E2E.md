# PR-Content-Ops-FAQ-Search-Seeded-Route-E2E

## Why this slice exists
The seeded DB smoke can now emit expectation-bearing hosted route cases, but an
operator still has to run seed, hosted HTTP, and cleanup manually. That leaves
the highest-risk path unproven as one repeatable command.

This slice intentionally exceeds the 400 LOC target because the runner is a new
checker/gate and ships with negative fixtures for its parser, preflight, phase,
and cleanup failure branches.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Add a thin seeded-hosted FAQ search e2e runner.
2. Reuse the existing seeded DB smoke and hosted route concurrency smoke.
3. Always clean up seeded FAQ drafts by emitted FAQ IDs unless explicitly kept.
4. Write one compact JSON result for seed, route, cleanup, and artifacts.
5. Add focused negative fixtures for the new detector branches.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Seeded-Route-E2E.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
## Mechanism
The new runner creates a temporary artifact directory, invokes the DB smoke with
`--keep-data --route-case-file-output`, invokes the hosted route smoke with that
case file, then reads seeded FAQ IDs from the case file and deletes those drafts.
The route token's tenant must be supplied as `--account-id`, so seeded rows match
the hosted auth scope.

Each phase is summarized separately. Seed or route failures still attempt cleanup
when a case file exists. Cleanup failure makes the run fail closed unless
`--keep-data` is set.
## Intentional
- No new database seeding implementation is added; the existing DB smoke remains
  the producer.
- No token decoding is attempted. The operator supplies the account ID that
  matches the bearer token.
- Cleanup deletes only emitted FAQ IDs, not all drafts for an account.
## Deferred
- Hosted detail-route expectation checks.
- CI enrollment for this live e2e runner.
## Verification
- pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_extracted_pipeline_route_ci_contract.py tests/test_audit_extracted_pipeline_ci_enrollment.py -q - 27 passed in 0.18s.
- python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py - Passed.
- git diff --check - Passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . - Passed; 116 matching tests enrolled.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Seeded-Route-E2E.md - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 57 |
| CI runner enrollment | 1 |
| Runner | 325 |
| Tests | 308 |
| **Total** | **691** |
