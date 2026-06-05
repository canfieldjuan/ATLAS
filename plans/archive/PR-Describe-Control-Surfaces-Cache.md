# PR: cache the static portion of `describe_control_surfaces`

## Why this slice exists

`GET /content-ops/control-surfaces` is the UI's catalog endpoint — hit
on every page load that surfaces output choices. Today it iterates
`OUTPUT_CATALOG` and `PRESETS` (both immutable `MappingProxyType`s)
on every request, building 6 + 5 dicts per call. The contents are
purely a function of module-level globals; only `execution_configured`,
`can_execute`, and `execution.{configured,configured_outputs}` change
per-request based on the host-injected services.

This PR factors the static portion into a module-level cached payload
computed once at import, leaving only the dynamic per-request flags
to be merged in.

## Scope (this PR)

- One file: `extracted_content_pipeline/api/control_surfaces.py`.
- One test file: `tests/test_extracted_content_control_surface_api.py`
  (new mutation-safety test; existing behavior tests stay unchanged).
- No public-API change. `describe_control_surfaces` returns the same
  shape; only the implementation is rearranged.

### Files touched

1. `extracted_content_pipeline/api/control_surfaces.py`
   - Add `_build_static_catalog_payload()` helper that computes the
     static `outputs` / `presets` / `ingestion_profiles` portion from
     `OUTPUT_CATALOG` and `PRESETS`.
   - Add module-level `_STATIC_CATALOG_PAYLOAD` constant computed
     once at import via the helper.
   - Add `_compose_describe_response(*, static, configured_outputs,
     execution_configured)` helper that merges the static cache with
     the per-request flags into a fresh dict (no shared mutable
     references with the cache).
   - Rewrite the `describe_control_surfaces` route body to call
     `_compose_describe_response` with the resolved
     `configured_outputs` frozenset.

2. `tests/test_extracted_content_control_surface_api.py`
   - Add `test_describe_control_surfaces_returns_independent_dict_per_call`:
     two sequential calls return mutually independent dicts; mutating
     one does not affect the next call.
   - Add `test_describe_control_surfaces_static_cache_is_built_once`:
     spy / assert that `_build_static_catalog_payload` is not invoked
     per request.

## Mechanism

The cache is just a module-level constant plus a per-request shallow
re-projection. No `functools.lru_cache`, no TTL, no invalidation —
the static portion is genuinely immutable for the process lifetime
because `OUTPUT_CATALOG` and `PRESETS` are `MappingProxyType` over
module-level globals (`control_surfaces.py:96-194`). If a future PR
adds a hot-reload mechanism for these catalogs, this cache becomes a
reset target — but until then, the constant is correct.

The per-request merge produces a fresh outer dict and fresh per-output
dicts so callers can mutate the response without bleeding into the
next request. The cached entries themselves are read-only (we copy
out, never hand them back).

## Intentional

- Static-only caching: the `execution_configured`, `can_execute`, and
  `execution` block stay per-request because they depend on the
  host-injected services.
- Computed at import: `_STATIC_CATALOG_PAYLOAD` is materialized once
  at module load. No first-request lazy fill, no double-checked
  locking, no concurrency complexity.
- Fresh-dict-per-call invariant: protects callers (and FastAPI's
  serialization layer) from accidentally mutating the cache. Verified
  by a new mutation-safety test.

## Deferred

- TTL / invalidation hooks (catalogs are process-immutable; if that
  changes a future slice can add a `clear_static_cache()` helper).
- `etag` / `If-None-Match` HTTP-level caching at the endpoint
  boundary — separate concern from the in-process compute cache.
- `Cache-Control` headers — out of scope; routing concern, not
  computation.
- Metrics / counters for cache reuse — premature without a hot-path
  signal.

## Verification

- `pytest tests/test_extracted_content_control_surface_api.py` — the
  existing 4 describe tests stay green; 2 new tests pass.
- `pytest tests/test_extracted_content_ops_execution.py
   tests/test_extracted_blog_generation.py
   tests/test_extracted_content_control_surfaces.py
   tests/test_extracted_content_generation_plan.py` — no regressions.
- `bash scripts/validate_extracted_content_pipeline.sh` — clean.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
   extracted_content_pipeline` — clean.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` —
  Atlas runtime import findings: 0.
- `bash scripts/check_ascii_python.sh` — clean.

## Estimated diff size

~50 LOC + ~30 LOC tests + ~110 LOC plan = ~190 LOC total. Well under
the 400 LOC target.
