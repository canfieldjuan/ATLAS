# PR: Content Ops Cache Policy Request

## Why this slice exists

Content Ops exact-cache policy and adapter wiring now exist, and the usage UI
shows cache savings once they are recorded. The remaining backend gap is the
request contract: operators and upstream ingestion layers cannot explicitly ask
for Content Ops exact cache or no-store behavior through the normal
preview/plan/execute payload.

This slice adds the first-class request field and routes it into the existing
policy decision path. It does not add UI controls yet; the backend contract
should be stable and tested before the UI exposes it.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a normalized `content_ops_cache_policy` request field to
   `ContentOpsRequest`.
2. Validate and normalize accepted policy values at the control-surface and API
   boundary.
3. Preserve the policy field through input-provider request merges.
4. Thread the normalized policy into Content Ops LLM trace metadata so the
   existing `ContentOpsExactCachePolicy` and adapter see the operator request.
5. Keep support-ticket/customer-data no-store behavior unchanged.
6. Add focused tests for request normalization, API response shaping,
   input-provider preservation, and execution trace-policy handoff.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Policy-Request.md` | Plan doc for the backend cache policy request contract. |
| `extracted_content_pipeline/content_ops_cache_policy.py` | Expose a canonical cache-policy request normalizer used by the request boundary and policy decision. |
| `extracted_content_pipeline/control_surfaces.py` | Add the normalized request field and include it in preview/plan normalized request payloads. |
| `extracted_content_pipeline/api/control_surfaces.py` | Validate and normalize the API request model field. |
| `extracted_content_pipeline/content_ops_input_provider.py` | Preserve the cache policy field through provider request merges. |
| `extracted_content_pipeline/content_ops_execution.py` | Add the normalized policy to shared LLM trace context before generation calls. |
| `tests/test_extracted_content_ops_cache_policy.py` | Cover canonical request-policy normalization. |
| `tests/test_extracted_content_control_surfaces.py` | Cover request normalization and rejection of unsupported values. |
| `tests/test_extracted_content_control_surface_api.py` | Cover API normalized-request response shaping for the cache policy field. |
| `tests/test_extracted_content_ops_input_provider.py` | Cover cache policy preservation through input-provider merge. |
| `tests/test_extracted_content_ops_execution.py` | Cover exact-cache request metadata reaching the existing policy path while support-ticket data remains no-store. |

## Mechanism

`content_ops_cache_policy.py` becomes the single source for normalizing request
policy text. Accepted exact-cache spellings normalize to `exact`; accepted
no-store spellings normalize to `no_store`; blank values remain `None`; unknown
values raise `ValueError` at request boundaries.

`ContentOpsRequest` carries `content_ops_cache_policy`. `request_from_mapping`
normalizes it, `ContentOpsRequestModel` validates it for HTTP routes, and
`ContentOpsInputProvider` keeps it in the request-level allowlist so upstream
packages do not drop the operator's cache choice.

Execution adds the normalized policy to the existing shared LLM trace context.
`PipelineLLMClient` already merges scoped metadata into
`ContentOpsExactCachePolicy.decide`, so this keeps the cache decision in one
existing source of truth instead of adding a separate cache rule layer.

## Intentional

- This does not add frontend controls. The backend request field lands first so
  a UI slice can call a stable contract.
- This does not loosen support-ticket/customer-data cache safety. Even when an
  operator requests `exact`, support-ticket markers still produce
  `customer_data_no_store`.
- This does not add a new cache policy implementation. The new request field is
  routed into the existing `ContentOpsExactCachePolicy`.
- This does not persist a tenant-level default cache setting. The field is
  explicit per request.

## Deferred

- Future PR: Content Ops UI cache controls once the backend request contract is
  merged and review-approved.
- Future PR: persisted tenant defaults if operators need always-on cache
  posture instead of per-run choices.
- Future PR: redacted/digest-only cache envelopes for support-ticket/customer
  source material if customer-data caching becomes a product requirement.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_content_ops_cache_policy.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py::test_preview_generation_route_normalizes_cache_policy_field tests/test_extracted_content_control_surface_api.py::test_preview_generation_route_rejects_unknown_cache_policy_field tests/test_extracted_content_ops_input_provider.py::test_merge_input_package_preserves_explicit_cache_policy tests/test_extracted_content_ops_execution.py::test_execute_marks_support_ticket_trace_context_for_cache_policy -q — 50 passed.
- python -m compileall -q extracted_content_pipeline/content_ops_cache_policy.py extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/api/control_surfaces.py extracted_content_pipeline/content_ops_input_provider.py extracted_content_pipeline/content_ops_execution.py tests/test_extracted_content_ops_cache_policy.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_input_provider.py tests/test_extracted_content_ops_execution.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2515 passed, 7 skipped, 1 warning.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas_pr_cache_policy_request_body.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~105 |
| Request/policy/API/provider/execution wiring | ~95 |
| Focused tests | ~150 |
| **Total** | **~350** |
