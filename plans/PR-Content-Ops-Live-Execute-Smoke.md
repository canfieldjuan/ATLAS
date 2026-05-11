# PR-Content-Ops-Live-Execute-Smoke

## Why this slice exists

The current AI Content Ops backlog puts live execute persistence smoke coverage
first. Existing tests cover the lower-level executor, route contract, offline
CLI, and service wiring separately, but no single test proves the hosted
`POST /content-ops/execute` route can run all generated outputs through real
generation services and persistence ports.

## Scope (this PR)

1. Add a route-level smoke test that mounts the real Content Ops control-surface
   router.
2. Inject real generated-asset services with in-memory host ports.
3. Assert all selected outputs execute, persist drafts under tenant scope, and
   report reasoning consumption when a host reasoning provider is wired.
4. Add the smoke test to the extracted pipeline gauntlet.

### Files touched

- `tests/test_extracted_content_ops_live_execute_smoke.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Live-Execute-Smoke.md`

## Mechanism

The test uses FastAPI `TestClient` against `create_content_ops_control_surface_router`.
It injects `ContentOpsExecutionServices` containing the real campaign, blog,
report, landing-page, sales-brief, and signal-extraction services. Storage,
intelligence, skills, LLM, and reasoning are host ports implemented in memory
inside the test, so the smoke exercises the product runtime without DB,
network, or provider credentials.

## Intentional

- In-memory host ports instead of Postgres. This is a route/service persistence
  smoke, not a database migration or live-DSN test.
- Quality gates are disabled in the request so the smoke focuses on route,
  dispatch, generation, persistence, and reasoning payload plumbing.
- One broad smoke test instead of one test per asset. The purpose is the
  cross-output route contract.

## Deferred

- Live Postgres/provider smoke for customer environments.
- Browser/UI validation of the execution response.

## Verification

- `pytest tests/test_extracted_content_ops_live_execute_smoke.py`
- `pytest tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_ops_live_execute_smoke.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `git diff --check`

## Estimated diff size

4 files, about 440 LOC. This is slightly over the 400 LOC soft cap because the
single route smoke needs in-memory host ports for every generated asset
repository plus LLM, skill, intelligence, blueprint, and reasoning providers.
Splitting those fakes away from the route smoke would leave the cross-output
persistence contract only partially covered.
