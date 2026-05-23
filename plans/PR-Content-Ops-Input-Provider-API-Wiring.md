# PR: Content Ops Input Provider API Wiring

## Why this slice exists

PR #899 added the pure input-provider contract, but hosted preview, plan, and
execute routes still use the raw request body directly. That means ingestion
providers can build a normalized input package, but the API layer has no place
to apply it before the existing control surface runs.

This slice wires the configured input provider into the hosted API path without
creating a new content layer. The provider only supplies the same
`ContentOpsRequest.inputs` shape the existing preview/plan/execute code already
understands.

## Scope (this PR)

Ownership lane: content-ops/input-provider-api-wiring

1. Add an optional `input_provider` argument to
   `create_content_ops_control_surface_router`.
2. Apply the provider package before `/preview`, `/plan`, and `/execute` run the
   existing control surface.
3. Resolve tenant scope once for execute and reuse it for provider handoff and
   generation execution.
4. Support both sync and async provider implementations.
5. Keep request/operator inputs authoritative over provider defaults.
6. Return a bounded `503` if the provider fails or returns an invalid package.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Input-Provider-API-Wiring.md` | Plan doc for this slice. |
| `extracted_content_pipeline/api/control_surfaces.py` | Wire optional input provider into preview/plan/execute. |
| `tests/test_extracted_content_control_surface_api.py` | Route-level coverage for provider merge, sync/async providers, scope, and failure handling. |

## Mechanism

The router factory accepts `input_provider`. When configured, each request body
is converted to a mapping, the tenant scope is resolved, and the provider's
`build_content_ops_input_package(scope=..., request=...)` result is merged under
the request via `merge_content_ops_input_package(...)`.

The provider merge uses only explicitly supplied request fields. This matters
because the API model expands omitted fields to defaults such as `outputs=[]`;
those defaults must not erase provider-selected outputs.

The merge happens before:

- `preview_from_mapping(...)`;
- `build_generation_plan_from_mapping(...)`;
- structured reasoning selection and `execute_content_ops_from_mapping(...)`.

This keeps the existing control surface, plan builder, and generators as the
source of truth.

## Intentional

- No new public route. This only wires the existing hosted routes.
- No FAQ implementation. FAQ remains owned by the existing FAQ generator lane.
- No automatic provider discovery. Hosts configure the provider explicitly.
- Provider failures are treated as service unavailability, not user validation
  errors, because the provider is host-owned infrastructure.
- Explicitly empty request values such as `outputs=[]` and `inputs={}` are
  treated like omitted values by the existing request-normalization path; this
  slice preserves that behavior so hosts use non-empty request fields when they
  intend to override provider defaults.

## Deferred

- Future PR: `PR-Content-Ops-Ticket-Input-Package` can implement a concrete
  provider that maps uploaded support-ticket imports into `source_material`,
  landing-page context, and blog topic/filter inputs.
- Future PR owned by the FAQ session: standalone FAQ article output contract.
- Parked hardening: none.

## Verification

- `py_compile` for `extracted_content_pipeline/api/control_surfaces.py` and
  `tests/test_extracted_content_control_surface_api.py` -> passed.
- `pytest` for `tests/test_extracted_content_control_surface_api.py`,
  `tests/test_extracted_content_ops_input_provider.py`,
  `tests/test_extracted_content_control_surfaces.py`, and
  `tests/test_extracted_content_generation_plan.py` -> 173 passed.
- `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` -> passed.
- `scripts/audit_extracted_standalone.py` with `--fail-on-debt` -> passed.
- `scripts/check_ascii_python.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~100 |
| API wiring | ~70 |
| Tests | ~155 |
| **Total** | **~325** |
