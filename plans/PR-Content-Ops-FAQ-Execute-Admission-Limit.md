# PR-Content-Ops-FAQ-Execute-Admission-Limit

## Why this slice exists
`HARDENING.md` records `FAQSCALE-1`: 50,000-row deterministic FAQ generation
works offline, but hosted synchronous execution must not accept unbounded large
uploads. The execute route already has generic request-shape caps, but the FAQ
sync row limit is implicit and static. This slice makes the FAQ admission rule
an explicit host config and exposes the configured value through the existing
control-surface metadata.
## Scope (this PR)
Ownership lane: content-ops/faq-generation-scale
Slice phase: Production hardening.
1. Add a host-configured maximum source-material row count for synchronous FAQ
   Markdown execution.
2. Enforce that limit only when the execute request resolves to `faq_markdown`.
3. Surface the configured FAQ sync limit in the existing execute metadata.
4. Add focused tests for under-limit pass, over-limit fail-closed behavior,
   non-FAQ requests, invalid config, and metadata exposure.
### Files touched
- `plans/PR-Content-Ops-FAQ-Execute-Admission-Limit.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
## Mechanism
`ContentOpsControlSurfaceApiConfig` gains
`faq_execute_max_source_material_rows`, defaulting to the existing inline
source-material limit. The `/execute` route calls a small admission helper after
input-provider merge and before reasoning/service execution. The helper resolves
the requested outputs, counts top-level and known bundled `source_material`
rows, and raises an HTTP 413 with the configured max when a synchronous
`faq_markdown` request exceeds the cap.

`GET /content-ops/control-surfaces` continues to expose the generic
`max_source_material_rows` value for compatibility and adds the explicit
`faq_max_source_material_rows` value for operators and clients.
## Intentional
- The generic Pydantic request-shape cap stays in place. This PR adds a
  product-specific admission rule rather than loosening broad request safety.
- The FAQ guard runs after input-provider merge so hosted uploaded-ticket
  packages are admitted or rejected based on the actual request handed to the
  generator.
- The guard is FAQ-specific. Other outputs are still governed by the existing
  generic shape limits, but not by this configurable FAQ cap.
## Deferred
- Background jobs, durable queues, retryable job status, and cross-process
  backpressure remain the larger follow-up for `FAQSCALE-1`.
- Raising the hosted synchronous FAQ cap above the current inline safety limit
  remains deferred until the background/offline execution path is ready.
## Verification
- pytest tests/test_extracted_content_control_surface_api.py -q - 105 passed in 2.90s.
- python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_control_surface_api.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Execute-Admission-Limit.md - Passed after plan verification formatting was corrected.
- bash scripts/local_pr_review.sh - Passed.
- bash scripts/validate_extracted_content_pipeline.sh - Passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - Passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - Passed.
- bash scripts/check_ascii_python.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 57 |
| Control-surface API | 55 |
| Tests | 185 |
| **Total** | **297** |
