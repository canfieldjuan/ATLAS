# PR: bring `BlogPostGenerationService` to reasoning-parity with the other 4 generators

## Why this slice exists

Of the five Content Ops generators, four (`campaign`, `report`,
`landing_page`, `sales_brief`) accept a uniform constructor kwarg:

```python
reasoning_context: CampaignReasoningContextProvider | None = None,
```

and consume it inside their `generate()` flow via an
`_opportunity_with_reasoning_context` (or `_payload_with_reasoning_context`
on `landing_page`) helper. Blog is the only generator that has neither
the kwarg nor the consumption path -- `grep -c "reasoning_context"
extracted_content_pipeline/blog_generation.py` -> 0 vs. 15-22 for the
others.

The route slice that comes next (PR-ControlSurfaces-Reasoning-Provider)
will pass a single `reasoning_context_provider` through the new
`/execute` route into all five services. That's only useful if all
five services accept it. Blog needs to land first.

## Scope (this PR)

Two concerns, both narrow:

1. **Constructor surface parity**: blog accepts the same kwarg shape
   as the other four.
2. **Generation-time consumption**: when configured, blog merges
   reasoning context into the blueprint payload before sending it to
   the LLM, mirroring `landing_page_generation.py`'s
   `_payload_with_reasoning_context` shape (which is itself the
   non-opportunity-shaped variant of campaign / report / sales_brief's
   pattern).

Blog operates on **blueprints**, not opportunities, so the lookup
key is the blueprint id (existing `_blueprint_id` helper) plus a
fixed `target_mode="blog_blueprint"` -- mirroring landing_page's
`target_mode="marketing_campaign"` constant.

### Files touched

1. `extracted_content_pipeline/blog_generation.py`
   - Import `CampaignReasoningContextProvider` from `campaign_ports`,
     and `campaign_reasoning_context_metadata`,
     `campaign_reasoning_context_payload`,
     `normalize_campaign_reasoning_context` from
     `services.campaign_reasoning_context`.
   - Add `_BLOG_REASONING_TARGET_MODE = "blog_blueprint"` constant.
   - Add `reasoning_context: CampaignReasoningContextProvider | None
     = None` to `BlogPostGenerationService.__init__`.
   - Add `_blueprint_with_reasoning_context(scope, blueprint)`
     helper: fetches context keyed by `_blueprint_id(blueprint)` +
     the fixed target mode, normalizes via
     `normalize_campaign_reasoning_context`, and (on non-empty
     content) returns a copy of the blueprint with merged
     `reasoning_context` payload + metadata fields.
   - In `generate()`, before calling `_generate_one`, invoke the
     helper so the blueprint passed to the LLM already carries
     reasoning context (same prompt template; the JSON gets richer).
   - Pass per-draft reasoning metadata through to
     `BlogPostDraft.metadata` so downstream consumers can see whether
     reasoning was applied.

2. `tests/test_extracted_blog_generation.py`
   - 3 new tests:
     - `test_generate_no_reasoning_provider_passes_blueprint_unchanged`:
       the existing happy path stays unchanged when no provider is
       wired.
     - `test_generate_with_reasoning_provider_merges_context_into_blueprint`:
       provider returns a non-empty context; blueprint passed to the
       LLM carries `reasoning_context` payload; draft metadata
       reports `reasoning_provider`.
     - `test_generate_with_reasoning_provider_returning_empty_is_noop`:
       provider returns empty / no-content; blueprint stays
       unchanged; no reasoning metadata fields on the draft.

## Mechanism

The pattern is lifted from `landing_page_generation.py`:

```python
async def _blueprint_with_reasoning_context(
    self,
    *,
    scope: TenantScope,
    blueprint: Mapping[str, Any],
) -> Mapping[str, Any]:
    if self._reasoning_context is None:
        return blueprint
    provided = await self._reasoning_context.read_campaign_reasoning_context(
        scope=scope,
        target_id=_blueprint_id(blueprint),
        target_mode=_BLOG_REASONING_TARGET_MODE,
        opportunity=blueprint,
    )
    provided_context = normalize_campaign_reasoning_context(provided)
    if not provided_context.has_content():
        return blueprint
    enriched = dict(blueprint)
    enriched["reasoning_context"] = campaign_reasoning_context_payload(
        provided_context
    )
    enriched.update(campaign_reasoning_context_metadata(provided_context))
    return enriched
```

The LLM sees the merged blueprint as JSON; the `{topic}` placeholder
substitution remains untouched. Reasoning context becomes
**additive** information the LLM can weave into the post -- it does
not replace the blueprint's own structure.

Blueprint id resolution falls through `_blueprint_id` (existing
helper at line 383): id -> slug -> topic -> suggested_title -> "".
Empty id short-circuits the lookup (provider returns nothing), so
empty-id blueprints continue to behave exactly as today.

## Intentional

- **Same port shape (`CampaignReasoningContextProvider`)** as the
  other 4 generators. The "campaign" naming is a historical artifact
  of where the port was first introduced; the port is now the
  package-wide reasoning-context contract regardless of output
  type. Don't rename in this slice.
- **Fixed `target_mode` for the reasoning lookup**, mirroring
  landing_page. The route-level `target_mode` (e.g.
  "vendor_retention") is an unrelated tenant-scope concept.
- **Blueprint mutation is a fresh dict per draft.** No shared
  mutable state with the source blueprint mapping; keeps callers
  insulated.
- **Metadata-only signal in `BlogPostDraft.metadata`** so downstream
  consumers can audit whether reasoning was applied without
  inspecting the prompt. Same pattern as the other generators.

## Deferred

- Per-call reasoning-context override on `generate()`. The other
  generators don't expose this either; if needed, lift it across
  all five in a follow-up.
- Blueprint-level lookup keys richer than `_blueprint_id`
  (e.g. brand-specific or category-specific reasoning context).
  Possible future slice; today the blueprint id is the right grain.
- Wiring `MultiPassCampaignReasoningProvider` into the new
  `api.control_surfaces /execute` route. That's the next slice
  (PR-ControlSurfaces-Reasoning-Provider) -- can't land until this
  one closes the service-layer gap.

## Verification

- `pytest tests/test_extracted_blog_generation.py` -- existing tests
  stay green; 3 new tests pass.
- `pytest tests/test_extracted_content_ops_execution.py
   tests/test_extracted_content_control_surfaces.py
   tests/test_extracted_content_generation_plan.py
   tests/test_extracted_content_ops_execution_smoke.py
   tests/test_extracted_content_control_surface_api.py` -- no
  regressions in the orchestration layer.
- `bash scripts/validate_extracted_content_pipeline.sh` -- clean.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
   extracted_content_pipeline` -- clean.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` --
  Atlas runtime import findings: 0.
- `bash scripts/check_ascii_python.sh` -- clean.
- `grep -c reasoning_context extracted_content_pipeline/blog_generation.py`
  -> non-zero (at least 5; was 0).

## Estimated diff size

- `blog_generation.py`: +~50 LOC (imports + constructor kwarg +
  helper + integration call + metadata threading).
- `test_extracted_blog_generation.py`: +~120 LOC (3 new tests with
  fakes for the reasoning provider).
- Plan doc: ~150 LOC.

Total: ~320 LOC. Within the <400 LOC PR target.
