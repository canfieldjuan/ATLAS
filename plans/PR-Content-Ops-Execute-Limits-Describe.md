# PR-Content-Ops-Execute-Limits-Describe

## Why this slice exists

The FAQ generator has now been proved offline at 50,000 rows and the hosted
execute route is being proved at the current 1,000-row synchronous cap. The cap
exists in request validation, but clients only discover it by failing a request
or reading code/tests. That creates room for drift across sessions and for
frontends to imply large synchronous uploads are safe when the production-safe
path is bounded execution or background work.

This slice makes the existing execute-route limits visible through
`/content-ops/control-surfaces`. It does not raise limits or add background job
execution.

## Scope (this PR)

Ownership lane: content-ops/control-surfaces

Slice phase: Production hardening.

1. Add an `execution.limits` object to the control-surfaces describe response.
2. Surface the current synchronous execute concurrency and source-material row
   cap from existing constants/config.
3. Add focused API tests that lock the response shape and prove the values do
   not alias the cached static catalog payload.

### Files touched

- `plans/PR-Content-Ops-Execute-Limits-Describe.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`

## Mechanism

The static catalog payload already carries ingestion limits. This slice adds a
small static `execute` limits block for `max_source_material_rows`, then
projects it into the existing per-request `execution` response along with
`ContentOpsControlSurfaceApiConfig.execute_max_concurrency`.

Expected response shape:

```json
"execution": {
  "configured": true,
  "configured_outputs": ["faq_markdown"],
  "limits": {
    "max_concurrency": 8,
    "max_source_material_rows": 1000,
    "large_upload_strategy": "background_or_offline"
  }
}
```

## Intentional

- No execute-route behavior changes. This only publishes limits that already
  exist.
- No background queue in this PR. That remains the production solution for
  uploads larger than the synchronous cap.
- No frontend work. Clients can consume this contract in a separate lane.

## Deferred

- Background/durable execution for large FAQ uploads remains tracked by
  `FAQSCALE-1` in `HARDENING.md`.
- Route-level persisted FAQ execution proof remains a separate functional
  validation slice after PR #916 review.

## Verification

- Focused API pytest passed: 98 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` passed.
- `bash scripts/check_ascii_python.sh` passed.
- `bash scripts/local_pr_review.sh --allow-dirty` passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Control-surface response | 15 |
| API tests | 44 |
| Plan doc | 81 |
| **Total** | 140 |
