# PR: wire `execution_services_provider` for `signal_extraction` (E1 of N)

## Why this slice exists

PR #406 mounted the `extracted_content_pipeline` content-ops
router into the host's aggregate `api_router` with **no**
`execution_services_provider`. That correctly returned `503
"Content Ops execution services are not configured."` for
`POST /content-ops/execute`, with the explicit Deferred:

> **Wiring `execution_services_provider` into the host mount.**
> v0 mounts the content-ops router with no provider, so
> `/execute` returns 503 by design. A follow-up slice wires the
> host's existing repositories (...) into a
> `ContentOpsExecutionServices` factory and passes it through.

Now that Screens 1 / 2 / 3 have all landed (PRs #408-#424 across
parallel sessions), the Execute button is the last functional
gap: it submits, but the backend always 503s.

This PR closes the gap for **`signal_extraction` only** -- the
deterministic generator with **zero dependencies** (no
`IntelligenceRepository`, no `LLMClient`, no `SkillStore`). It
proves the provider-wiring pattern; the remaining 5 generators
(campaign / blog / report / landing_page / sales_brief) ship
in follow-up slices once their host-side repository factories
are in place.

## Scope (this PR)

Tightly bounded:

1. **New module `atlas_brain/_content_ops_services.py`**
   that builds a `ContentOpsExecutionServices` bundle with
   `signal_extraction` populated. Exposes a single function
   `build_content_ops_execution_services()` returning the
   bundle. Other slots in the bundle stay `None`; the executor
   falls back to `service_not_configured` for them, which the
   route layer already maps to a clean per-step error.
2. **Modify `atlas_brain/api/__init__.py`** to import the
   factory and pass it to
   `create_content_ops_control_surface_router(
   execution_services_provider=lambda: build_content_ops_execution_services())`.
3. **One regression test** in
   `tests/test_atlas_content_ops_execution_services.py` pinning
   that the bundle round-trips through the executor for
   `signal_extraction` and continues to return
   `service_not_configured` for the other outputs.

### Files touched

1. `atlas_brain/_content_ops_services.py` (new) -- ~30 LOC
   factory + module-level singleton + docstring.
2. `atlas_brain/api/__init__.py` -- ~5 LOC delta to thread the
   provider into the existing
   `create_content_ops_control_surface_router(...)` call.
3. `tests/test_atlas_content_ops_execution_services.py` (new)
   -- ~100 LOC. Three tests:
   - `signal_extraction` runs through the executor.
   - The other 5 outputs still return
     `service_not_configured`.
   - `test_bundle_only_advertises_wired_outputs`: pins that
     `configured_outputs()` advertises only the wired output
     (canary for future slot population).
4. `plans/PR-Content-Ops-Execution-Services-Wire-1.md` (this
   file).

### What's NOT in this slice

- **Wiring `campaign` / `blog_post` / `report` /
  `landing_page` / `sales_brief`.** Each needs a
  Postgres-backed repository, an `LLMClient`, and a
  `SkillStore` -- those host-side factories are themselves
  follow-up work. Once any one of them lands, an E2 / E3 slice
  plugs it into the bundle in the same shape.
- **Tenant scope plumbing.** The route already accepts a
  `scope_provider`; that is a separate slice and explicitly
  not introduced here.
- **Reasoning context provider.** Already wired via PR
  #ControlSurfaces-Reasoning-Provider (PR #402). This slice
  doesn't touch it.
- **Frontend changes.** None. The Execute button on Screen 3
  is already implemented; it currently 503s. After this slice,
  hitting Execute with `signal_extraction` selected returns a
  real result.

## Mechanism

The factory returns a singleton-shaped bundle:

```python
# atlas_brain/_content_ops_services.py
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
)
from extracted_content_pipeline.signal_extraction import (
    SignalExtractionService,
)

_SIGNAL_EXTRACTION_SERVICE = SignalExtractionService()

def build_content_ops_execution_services() -> ContentOpsExecutionServices:
    return ContentOpsExecutionServices(
        signal_extraction=_SIGNAL_EXTRACTION_SERVICE,
    )
```

`SignalExtractionService` is stateless (no DB, no LLM, no
skills) so a module-level singleton is safe. The other slots
stay `None`; the executor's per-step dispatcher already returns
`service_not_configured` for unset slots, and the route layer's
`_sanitize_execution_result` keeps that error legible to the
UI.

The router-construction call gains `execution_services_provider`:

```python
# atlas_brain/api/__init__.py
from .._content_ops_services import build_content_ops_execution_services

content_ops_router = create_content_ops_control_surface_router(
    dependencies=[Depends(require_b2b_plan("b2b_growth"))],
    execution_services_provider=build_content_ops_execution_services,
)
```

The factory is passed as a bare function reference (not wrapped
in a lambda) -- `_resolve_execution_services` accepts any
`Callable[[], ContentOpsExecutionServices]` shape.

The provider is a plain `Callable[[], ContentOpsExecutionServices]`;
the route's `_resolve_execution_services` handles both sync and
async returns.

## Intentional

- **Module-level singleton, not per-request factory.**
  `SignalExtractionService` has no per-request state; recreating
  it every call would just churn allocations. When future
  services need per-request state (DB sessions, etc.) the
  factory becomes per-request without touching this signature.
- **Bundle leaves other slots `None`.** The executor handles
  unset slots cleanly; populating them with stubs would mask
  legitimate "not configured" signals from the UI's "Execute"
  button enable-state.
- **No env-var gate.** The factory always builds the bundle.
  Hosts that want to disable Content Ops execution can omit the
  router mount itself (the existing try/except already handles
  the disabled case).
- **Factory lives in `atlas_brain/_content_ops_services.py`,
  not directly in `__init__.py`.** Keeps `__init__.py`'s router
  aggregation lean; lets the factory grow as more services
  arrive without bloating the package init.

## Deferred

- **`campaign` service**: needs `IntelligenceRepository`
  (Postgres), `CampaignRepository` (Postgres), `LLMClient`,
  `SkillStore`. Host has these primitives; wiring is a separate
  slice.
- **`blog_post` service**: needs `BlogBlueprintRepository`
  (Postgres), `BlogPostRepository` (Postgres), `LLMClient`,
  `SkillStore`.
- **`report` / `landing_page` / `sales_brief`**: same shape as
  campaign / blog (intelligence + repo + LLM + skills).
- **`scope_provider` wiring** to thread tenant scope from auth
  into the executor.
- **Per-output e2e smoke tests** that POST to /execute and
  confirm a real draft lands. Today's tests stop at the bundle
  layer.

## Verification

- `python -m pytest tests/test_atlas_content_ops_execution_services.py`
  -- 3 passed.
- `bash scripts/check_ascii_python.sh` -- clean.
- `python -c "from atlas_brain._content_ops_services import
   build_content_ops_execution_services as f; b = f(); print(
   b.configured_outputs())"` -- `('signal_extraction',)`.
- AST-parse + ASCII-clean spot-check on
  `atlas_brain/api/__init__.py` and
  `atlas_brain/_content_ops_services.py`.

## Estimated diff size

- `_content_ops_services.py`: ~30 LOC.
- `__init__.py`: ~5 LOC delta.
- Test: ~100 LOC.
- Plan doc: ~140 LOC.

Total: ~275 LOC. Within the 400 LOC PR target. Tightly scoped.
