# PR: route-level `reasoning_context_provider` seam for `api.control_surfaces /execute`

## Why this slice exists

`api.campaign_operations` (the legacy route) accepts a
`reasoning_context_provider` and resolves it per-request, wiring
`MultiPassCampaignReasoningProvider` into campaign generation. The
new `api.control_surfaces /execute` route has no such seam --
hosts can only bake reasoning into services at construction time
inside `execution_services_provider`. There's no per-request
reasoning override and no documented pattern for hosts who want
the canonical multi-pass setup.

This is the last reasoning-wiring gap on the AI Content Ops
backend. After PR #399 brought `BlogPostGenerationService` to
constructor parity, all 5 generators accept
`CampaignReasoningContextProvider` at `__init__`. This PR closes
the loop at the route layer so a host wires one provider at
router-construction time and all five outputs use it.

## Scope (this PR)

The slice has three layers, each tightly bounded:

1. **Service-class seam**: each of the 5 generators gains a
   `with_reasoning_context(provider)` method returning a new
   instance of itself with `_reasoning_context` rebound.
2. **Bundle-level seam**: `ContentOpsExecutionServices` gains a
   matching `with_reasoning_context(provider)` that returns a
   derived bundle by calling each non-None service's helper.
3. **Route-level seam**: `create_content_ops_control_surface_router`
   accepts a new optional `reasoning_context_provider` callable.
   When set, `/execute` resolves it per-request and derives a
   reasoning-aware bundle before calling the executor.

### Files touched

1. `extracted_content_pipeline/blog_generation.py`
2. `extracted_content_pipeline/campaign_generation.py`
3. `extracted_content_pipeline/report_generation.py`
4. `extracted_content_pipeline/landing_page_generation.py`
5. `extracted_content_pipeline/sales_brief_generation.py`
   - Each: add `with_reasoning_context(provider)` method that
     returns a new instance with the same ports / config but a
     rebound `_reasoning_context`. ~7 LOC per service.

6. `extracted_content_pipeline/content_ops_execution.py`
   - `ContentOpsExecutionServices` gains
     `with_reasoning_context(provider)` returning a derived
     `ContentOpsExecutionServices` with each non-None service
     replaced by `service.with_reasoning_context(provider)`. Skips
     services that don't expose the helper (signal_extraction
     doesn't consume reasoning context, so it stays as-is).

7. `extracted_content_pipeline/api/control_surfaces.py`
   - Add `ReasoningContextProvider` type alias.
   - Add `reasoning_context_provider` parameter to
     `create_content_ops_control_surface_router`.
   - In `/execute` route: after resolving services, also resolve the
     reasoning provider; if non-None, derive
     `services.with_reasoning_context(provider)` and use the derived
     bundle for the executor call.

8. `tests/test_extracted_content_control_surface_api.py`
   - 2 new tests:
     - `test_execute_route_threads_reasoning_provider_into_services`:
       a configured `reasoning_context_provider` reaches each
       service's `_reasoning_context` for the duration of the
       request.
     - `test_execute_route_without_reasoning_provider_passes_services_unchanged`:
       no provider -> executor receives the original bundle, no
       wrapping, no surprises.

9. `tests/test_extracted_content_ops_execution.py`
   - 1 new test:
     `test_services_with_reasoning_context_derives_new_bundle_with_provider_attached`
     -- direct unit coverage of the bundle-level helper without
     spinning up the route.

## Mechanism

The route resolves the provider per-request:

```python
explicit_reasoning = await _resolve_provider(reasoning_context_provider)
services = await _resolve_execution_services(execution_services_provider)
if explicit_reasoning is not None and services is not None:
    services = services.with_reasoning_context(explicit_reasoning)
```

`ContentOpsExecutionServices.with_reasoning_context` derives a fresh
bundle:

```python
def with_reasoning_context(
    self,
    provider: CampaignReasoningContextProvider,
) -> "ContentOpsExecutionServices":
    return replace(
        self,
        campaign=_rebind(self.campaign, provider),
        blog_post=_rebind(self.blog_post, provider),
        report=_rebind(self.report, provider),
        landing_page=_rebind(self.landing_page, provider),
        sales_brief=_rebind(self.sales_brief, provider),
        # signal_extraction left unchanged; doesn't consume reasoning.
    )

def _rebind(service, provider):
    if service is None:
        return None
    helper = getattr(service, "with_reasoning_context", None)
    if helper is None:
        return service  # service didn't opt in; leave as-is
    return helper(provider)
```

Each service's `with_reasoning_context` is a one-line copy
constructor:

```python
def with_reasoning_context(
    self,
    provider: CampaignReasoningContextProvider | None,
) -> "BlogPostGenerationService":
    return BlogPostGenerationService(
        blueprints=self._blueprints,
        blog_posts=self._blog_posts,
        llm=self._llm,
        skills=self._skills,
        reasoning_context=provider,
        config=self._config,
    )
```

This pattern preserves immutability (no setter on
`_reasoning_context`), is concurrency-safe (each request derives
its own bundle), and doesn't disturb the existing constructor
arity of any service.

## Intentional

- **No per-call kwarg on `service.generate()`.** Could've added
  `reasoning_context=` to every `generate()` signature (mirroring
  the OptionA per-field-kwarg pattern), but the route-derived
  bundle approach keeps the per-call surface unchanged and
  concentrates the reasoning seam in one place. Less surface area
  to maintain.
- **`signal_extraction` stays untouched.** That generator doesn't
  consume `CampaignReasoningContext`; rebinding would be a no-op
  with extra surface area.
- **`with_reasoning_context(None)` is supported.** If a host
  explicitly wires a reasoning provider that resolves to None for
  some requests (e.g. a tenant policy that disables reasoning for
  certain accounts), the bundle is derived with each service's
  reasoning rebound to None -- predictable, no dropped state.
- **No new helper for *constructing* a default
  `MultiPassCampaignReasoningProvider`.** That's a host concern;
  `api.campaign_operations._generation_reasoning_context` already
  shows the canonical construction. This slice is the seam, not
  the default factory.

## Deferred

- A host-side convenience that constructs
  `MultiPassCampaignReasoningProvider` from API config + LLM/skills
  providers (mirroring `_generation_reasoning_context` in
  `api/campaign_operations.py`). Useful but separable.
- Per-output reasoning provider overrides (e.g. blog uses one
  provider, campaign uses another). YAGNI today; the same provider
  is used across all 5 outputs.
- Enriching `describe_control_surfaces` with a `reasoning`
  block that exposes whether the provider is configured. Small
  follow-up if needed.
- Test coverage that walks the full multi-pass reasoning chain
  end-to-end. The existing
  `tests/test_extracted_campaign_multi_pass_reasoning_provider.py`
  + per-service reasoning tests already cover the chain
  components; this slice only has to prove that the seam wires.

## Verification

- `pytest tests/test_extracted_content_control_surface_api.py
   tests/test_extracted_content_ops_execution.py
   tests/test_extracted_blog_generation.py
   tests/test_extracted_landing_page_generation.py
   tests/test_extracted_campaign_generation.py
   tests/test_extracted_report_generation.py
   tests/test_extracted_sales_brief_generation.py` -- existing tests
  stay green; 3 new tests pass.
- `bash scripts/validate_extracted_content_pipeline.sh` -- clean.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
   extracted_content_pipeline` -- clean.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` --
  Atlas runtime import findings: 0.
- `bash scripts/check_ascii_python.sh` -- clean.

## Estimated diff size

- 5 service files × ~10 LOC each = 50 LOC.
- `content_ops_execution.py`: ~30 LOC.
- `api/control_surfaces.py`: ~25 LOC.
- Tests: ~200 LOC (3 tests + a fake reasoning provider in the API
  test file).
- Plan doc: ~150 LOC.

Total: ~450 LOC. Slightly over the 400 LOC budget; justified
because the seam is structurally a 3-layer change (service +
bundle + route) and splitting at any layer leaves the slice
half-finished. If reviewer flags, the bundle-level + route-level
layers could ship as PR-A and the service-level
`with_reasoning_context` helpers as PR-B, but the seam is only
useful end-to-end.
