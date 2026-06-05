# PR-Content-Ops-FAQ-SaaS-Demo-Route-E2E-Smoke

## Why this slice exists

The SaaS demo route-case runbook now documents the real seed-to-hosted-route
flow, and the host runbook links to it. The remaining operator gap is execution
friction: proving the checked SaaS demo through the deployed route still takes
multiple commands plus manual cleanup of the emitted FAQ id.

This slice adds the thinnest one-command smoke for that exact flow. It composes
the already-tested seeder and hosted route concurrency smoke instead of adding a
new persistence or HTTP implementation.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Vertical slice

1. Add a SaaS demo hosted route e2e smoke script that seeds the checked SaaS FAQ,
   writes a route case file, runs hosted route/detail validation against it, and
   cleans up the seeded FAQ unless `--keep-data` is set.
2. Preserve compact child result artifacts in one top-level JSON summary.
3. Add focused subprocess-boundary tests for success, preflight, route failure,
   and cleanup command wiring.
4. Keep seeder, route smoke, API, repository, and runbook behavior unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-E2E-Smoke.md` | Plan contract for this one-command SaaS demo validation slice. |
| `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` | Compose SaaS demo seed, hosted route/detail smoke, and cleanup. |
| `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py` | Prove command composition and fail-closed result behavior. |
| `scripts/run_extracted_pipeline_checks.sh` | Enroll the new smoke test in the local CI mirror. |
| `.github/workflows/extracted_pipeline_checks.yml` | Trigger extracted checks when the new smoke script changes. |

## Mechanism

The new smoke runs three existing commands:

```text
seed_content_ops_faq_saas_demo.py --route-case-file-output <case-file>
smoke_content_ops_faq_search_route_concurrency.py --case-file <case-file> --require-detail
seed_content_ops_faq_saas_demo.py --cleanup-faq-id <seeded-faq-id>
```

The wrapper reads `faq_id` from the seed result artifact for cleanup and embeds
compact seed, route, and cleanup result summaries into the top-level result. If
preflight fails, it writes a result and exits `2`. If any child command fails or
cleanup fails, it writes the result and exits `1`.

## Intentional

- No direct DB writes or hosted HTTP calls in the wrapper. The existing seeder
  and route smoke remain the source of truth for those behaviors.
- No migration step in the wrapper. Operators still run the existing migration
  command before hosted validation, matching the runbook.
- No live environment test in this slice. Required hosted inputs are not present
  in this checkout, so tests mock the subprocess boundary.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this wrapper slice.
- Future robust-testing slice: run this smoke against a deployed host and commit
  the validation report once `DATABASE_URL`, API base URL, token, and account id
  are available in the test environment.

## Verification

- `python -m py_compile scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py` - passed.
- `python -m pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q` - 6 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-E2E-Smoke.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-E2E-Smoke.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 123 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2597 passed, 7 skipped.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 96 |
| Smoke script | 376 |
| Tests | 215 |
| CI enrollment | 1 |
| Workflow filters | 2 |
| **Total** | **690** |

The estimate is over the 400 LOC soft cap because the slice is only useful if
the wrapper, its subprocess-boundary regression tests, and CI enrollment ship
together.
