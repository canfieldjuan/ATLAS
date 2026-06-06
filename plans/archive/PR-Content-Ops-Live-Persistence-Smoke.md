# PR-Content-Ops-Live-Persistence-Smoke

## Why this slice exists

`docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` lists live execute
persistence smoke coverage as the top AI Content Ops deferral. Existing tests
already prove the in-memory execution harness persists every generated asset,
but the direct harness test bypasses the hosted `/content-ops/execute` route.

This slice closes that route-level gap without adding new runtime behavior.

## Scope

1. Add a route-level smoke to `tests/test_extracted_content_ops_live_execute_harness.py`.
2. Reuse `build_content_ops_live_execute_harness()` and
   `default_content_ops_execute_payload()` instead of creating a parallel fixture.
3. Assert the hosted `POST /content-ops/execute` route:
   - enters the route endpoint,
   - applies the provided tenant scope,
   - uses host-injected services,
   - persists every LLM-backed generated asset,
   - reports saved ids and consumed reasoning audits.
4. Claim this slice in `docs/extraction/coordination/inflight.md`.

### Files touched

- `plans/PR-Content-Ops-Live-Persistence-Smoke.md`
- `tests/test_extracted_content_ops_live_execute_harness.py`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The test builds a router with the Content Ops control-surface router factory,
injects the existing harness services and reasoning provider, resolves the
`/content-ops/execute` route, and calls the route endpoint with the existing
payload fixture.

Persistence is verified through the harness repositories, not by trusting the
response alone. The response is still checked for generated ids and reasoning
audit fields because those are the customer-facing route contract.

## Intentional

- No production code changes are expected. This is a smoke-contract slice.
- The test stays in-memory. It validates the host adapter seam without opening
  a real database or provider connection.
- The existing direct harness test remains in place because it gives a smaller
  failure surface when the executor changes.

## Deferred

- A real Postgres-backed live smoke can follow once test infrastructure has a
  stable database fixture for all generated asset repositories.
- Blog blueprint population remains the next AI Content Ops backlog item after
  this smoke contract lands.

## Verification

```bash
python -m pytest tests/test_extracted_content_ops_live_execute_harness.py
python -m pytest tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_execution.py
python -m py_compile tests/test_extracted_content_ops_live_execute_harness.py
bash scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Live-Persistence-Smoke.md` | 66 |
| `tests/test_extracted_content_ops_live_execute_harness.py` | 77 |
| `docs/extraction/coordination/inflight.md` | 2 |
| **Total** | **~145** |
