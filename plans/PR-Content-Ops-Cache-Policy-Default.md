# PR: Content Ops Cache Policy Default

## Why this slice exists

Content Ops can now request exact cache or no-store per run, and the usage
surface can show savings and cache diagnostics. The remaining operator friction
is that every run has to carry the cache choice explicitly. Hosts need a
source-level way to apply an account/tenant default while preserving the
existing per-run override and customer-data no-store protections.

This slice adds that default at the request boundary. It does not create a new
cache rule layer: the default is normalized into the existing
`content_ops_cache_policy` request field before preview, plan, or execute.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add an optional cache-policy default provider to the Content Ops control
   surface router.
2. Resolve the default with tenant scope and apply it only when the merged
   request has no explicit cache policy.
3. Normalize provider values through the existing cache-policy normalizer.
4. Preserve explicit per-run cache-policy overrides through input-provider merges.
5. Wire the Atlas host mount to an env-backed default provider so the seam is
   active in production without adding DB settings in this slice.
6. Add focused route tests for preview/execute defaulting, override precedence,
   and invalid provider values.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Policy-Default.md` | Plan doc for the cache-policy default provider slice. |
| `atlas_brain/api/__init__.py` | Pass the hosted Content Ops cache-policy default provider into the control-surface router. |
| `atlas_brain/config.py` | Add the env-backed hosted Content Ops cache-policy default setting. |
| `extracted_content_pipeline/api/control_surfaces.py` | Add and apply the optional tenant-scoped cache-policy default provider. |
| `tests/test_atlas_content_ops_generated_assets_api.py` | Pin the Atlas host route wiring for the cache-policy default provider. |
| `tests/test_extracted_content_control_surface_api.py` | Cover default application, override precedence, and invalid provider values. |

## Mechanism

`create_content_ops_control_surface_router(...)` accepts an optional
`cache_policy_default_provider`. When preview, plan, or execute builds the
request payload, the route first lets the input provider merge data into the
payload. Then it applies the cache default only if `content_ops_cache_policy`
is still blank. The provider receives the resolved tenant scope and returns a
policy value such as `exact` or `no_store`; unsupported values fail closed with
an HTTP error instead of silently running with a surprising cache posture.

Because the default is written to the existing request field, all downstream
integration points stay source-aligned: `request_from_mapping`, preview/plan
normalized output, execute trace context, and `ContentOpsExactCachePolicy`
consume the same field they already use today.

The Atlas host passes a provider backed by
`ATLAS_B2B_CAMPAIGN_CONTENT_OPS_CACHE_POLICY_DEFAULT`. Blank keeps the existing
per-run behavior. Nonblank values still flow through the extracted router's
normalizer and customer-data cache safety checks before any execution path sees
them.

## Intentional

- This does not add persistence or a settings table. The hosted env default
  makes the seam live now, and the provider seam still lets the host plug in
  DB-backed tenant settings later without changing the product request contract
  again.
- This does not let defaults bypass explicit no-store requests, including when
  the request also flows through an input provider.
- This does not loosen support-ticket/customer-data cache safety; existing
  source markers still make the exact-cache policy return no-store.
- This does not add frontend controls. The current UI per-run control remains
  the explicit override.

## Deferred

- Future PR: DB-backed tenant cache settings if operators need durable in-app
  defaults.
- Future PR: UI surface for editing that tenant default.
- Parked hardening: none. Root `HARDENING.md` was scanned and has no active
  cost-surfacing parked items.

## Verification

- python -m pytest tests/test_extracted_content_control_surface_api.py::test_preview_generation_route_applies_cache_policy_default_from_scope tests/test_extracted_content_control_surface_api.py::test_preview_generation_route_preserves_explicit_cache_policy_over_default tests/test_extracted_content_control_surface_api.py::test_preview_generation_route_rejects_invalid_cache_policy_default tests/test_extracted_content_control_surface_api.py::test_execute_route_applies_cache_policy_default_to_trace_context -q — 4 passed.
- python -m compileall -q extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_control_surface_api.py — passed.
- python -m pytest tests/test_atlas_content_ops_generated_assets_api.py::test_content_ops_preview_route_uses_host_cache_policy_default tests/test_atlas_content_ops_generated_assets_api.py::test_content_ops_cache_policy_default_provider_keeps_blank_unset -q — 2 passed, 1 warning.
- python -m compileall -q atlas_brain/api/__init__.py atlas_brain/config.py tests/test_atlas_content_ops_generated_assets_api.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2545 passed, 7 skipped, 1 warning.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <git-path>/codex-pr-bodies/cache-policy-default.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~105 |
| Atlas host config and route wiring | ~65 |
| Router default provider seam | ~65 |
| Tests | ~165 |
| **Total** | **~400** |

This sits at the 400 LOC soft cap because the slice now includes the host
integration point that makes the extracted provider live in production.
