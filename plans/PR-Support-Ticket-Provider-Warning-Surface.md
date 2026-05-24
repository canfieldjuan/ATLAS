# PR-Support-Ticket-Provider-Warning-Surface

## Why this slice exists

The support-ticket package now reports skipped and truncated rows, and the
package-smoke CLI exposes those warnings. The normal Content Ops API path still
drops `ContentOpsInputPackage.warnings` when it merges provider inputs into a
request payload, so preview/plan/execute callers cannot see that an uploaded
ticket export was truncated before generation.

This slice surfaces provider diagnostics without changing generation behavior.
It keeps the package warnings available to host/UI layers while preserving the
existing request contract for generators.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Production hardening

1. Preserve input-provider diagnostics when converting a
   `ContentOpsInputPackage` into a request payload.
2. Include provider diagnostics in preview, plan, and execute API responses when
   an input provider is configured.
3. Add focused tests proving provider warnings/metadata survive the package
   merge and both preflight and execute routes.

### Files touched

- `extracted_content_pipeline/content_ops_input_provider.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_ops_input_provider.py`
- `tests/test_extracted_content_control_surface_api.py`
- `plans/PR-Support-Ticket-Provider-Warning-Surface.md`

## Mechanism

`content_ops_payload_from_input_package(...)` will attach an `input_provider`
diagnostic object alongside the existing request keys:

```json
{
  "input_provider": {
    "provider": "support_ticket_upload",
    "metadata": {},
    "warnings": []
  }
}
```

`request_from_mapping(...)` already ignores unknown top-level keys, so generation
services continue to receive the same normalized request. The API routes can
copy this diagnostic object onto their response payloads after preview/plan or
execute completes.

## Intentional

- No warning text is folded into `inputs`; generators should not consume
  operational warnings as content context.
- API responses only expose the provider name, warnings, and allowlisted
  operational metadata fields. Host-injected metadata remains available inside
  the package but is not echoed wholesale to callers.
- No hosted upload UI changes. This only makes the diagnostic available to the
  API response layer.
- No blocking behavior for truncation warnings. The existing package cap stays
  non-fatal; product policy can decide later whether to reject, warn, or queue
  oversized uploads.
- Cross-layer caller hints flag the router factory and payload merge helper
  because they are shared. This slice adds focused preview/plan/execute route
  assertions and ran the full extracted suite; existing callers that ignore the
  optional `input_provider` response field remain compatible.

## Deferred

- Future PR: hosted upload/intake UI can display `input_provider.warnings` to
  users or operators.
- Future PR: product policy can decide whether files above the synchronous
  1,000-row package cap should be rejected or moved to a background job.
- Parked hardening: none.

## Verification

- python -m py_compile for `extracted_content_pipeline/content_ops_input_provider.py`,
  `extracted_content_pipeline/api/control_surfaces.py`, and focused tests -
  passed.
- pytest for `tests/test_extracted_content_ops_input_provider.py` plus focused
  preview/plan/execute API tests - 8 passed.
- validate_extracted_content_pipeline.sh - passed.
- forbid_atlas_reasoning_imports.py for `extracted_content_pipeline` - passed.
- audit_extracted_standalone.py with fail-on-debt - passed.
- check_ascii_python.sh - passed.
- sync_extracted.sh for `extracted_content_pipeline` - passed.
- run_extracted_pipeline_checks.sh - 1959 passed, 1 skipped.
- local PR review - passed; caller-hint warning reviewed and covered by focused
  route tests plus the full extracted suite.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Payload/API code | ~45 |
| Tests | ~85 |
| **Total** | **~210** |
