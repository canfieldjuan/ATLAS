# PR: wire `blog_post` service into bundle (E4 of N)

## Why this slice exists

After PR #456 (E3, IntelligenceRepository-dependent
generators) and PR #458 (`PostgresBlogBlueprintRepository`
storage layer), every dependency required to wire
`blog_post` is now in place:

- `BlogPostGenerationService` (extracted package, already
  written).
- `PostgresBlogPostRepository` (extracted package, already
  written).
- `PostgresBlogBlueprintRepository` (PR #458, just landed).
- `LLMClient` + `SkillStore` adapters (PR #453).
- Pool factory + scope_provider wiring (PRs #455 / #456).

This slice plugs the last unwired Content Ops slot into the
host bundle factory. After E4, `configured_outputs()`
advertises 6 of 6 outputs when LLM and DB services are both
enabled.

## Scope (this PR)

The bundle factory and the test file. No frontend or route
changes; the production mount from PR #455 already passes
`enable_db_services=True`.

### Files touched

1. `atlas_brain/_content_ops_services.py`:
   - Add `_build_blog_post_service` helper following the
     established `_build_landing_page_service` template --
     short-circuit to `None` when LLM or pool is absent;
     otherwise construct `BlogPostGenerationService` with
     `PostgresBlogBlueprintRepository(pool=pool)` +
     `PostgresBlogPostRepository(pool=pool)` + LLM/Skill
     adapters.
   - Update `build_content_ops_execution_services()` to call
     the new helper when `enable_db_services=True` and pass
     the result through to the `ContentOpsExecutionServices`
     constructor's `blog_post` slot.
   - Update the module docstring's "currently wired" inventory
     so it lists `blog_post` alongside the other 5 outputs;
     drop the "follow-up slice (E4)" note since this is E4.
   - ~30 LOC delta.

2. `tests/test_atlas_content_ops_execution_services.py`:
   - Add 2 new "wired-when-LLM-active" canaries:
     - `test_blog_post_wired_when_llm_active_and_db_enabled`
       -- verifies `services.blog_post is not None` and
       `services.for_output("blog_post") is services.blog_post`.
     - `test_blog_post_slot_stays_none_when_no_active_llm`
       -- mirror of the existing landing_page / campaign
       fallback canaries. Pool-None canary is already
       covered by `test_e3_services_skip_together_when_pool_is_none`
       extended to assert the blog_post slot too.
   - Replace the now-obsolete
     `test_unwired_blog_post_still_returns_service_not_configured`
     -- with all 6 outputs wired, the unwired set is empty.
     Replace it with a canary that pins the new bundle shape
     (`for_output("blog_post")` returns the service).
   - Update the existing `configured_outputs()` expected
     tuples to include `blog_post`. The upstream iteration
     order is `(email_campaign, blog_post, report,
     landing_page, sales_brief, signal_extraction)` --
     verified at `extracted_content_pipeline/content_ops_execution.py:54-61`.
   - Update the docstring inventory from 13 to 14 tests
     (one swapped, two added).
   - ~80 LOC delta.

3. `plans/PR-Content-Ops-Execution-Services-Wire-4.md`
   (this file).

### What's NOT in this slice

- **Host autonomous task changes to populate
  `blog_blueprints`.** PR #458's storage layer adds the
  table; a separate slice (or operator-driven ETL) lands
  blueprints in it. The wired `blog_post` generator reads
  whatever's there; an empty table just means
  `BlogPostGenerationService.generate()` returns zero
  drafts -- not an error.
- **Reasoning context provider host wiring.** The bundle's
  `with_reasoning_context()` derivation (PR #402) handles
  per-request rebinding when the host wires a provider. Out
  of scope.
- **End-to-end smoke test posting to
  `/api/v1/content-ops/execute`** with `outputs=["blog_post"]`
  and asserting a real draft persists. Today's tests stop at
  the bundle/service layer.

## Mechanism

The helper follows the established
`_build_landing_page_service` template -- short-circuit on
missing dependency, otherwise construct the service:

```python
def _build_blog_post_service(
    *,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> BlogPostGenerationService | None:
    """E4: blog-post drafts. Same short-circuit shape as
    `_build_landing_page_service`."""

    if llm is None or pool is None:
        return None
    return BlogPostGenerationService(
        blueprints=PostgresBlogBlueprintRepository(pool=pool),
        blog_posts=PostgresBlogPostRepository(pool=pool),
        llm=llm,
        skills=skills,
    )
```

`build_content_ops_execution_services()` calls the helper in
the same `if enable_db_services:` block as the other
DB-backed generators, then passes the result to the
`ContentOpsExecutionServices(blog_post=...)` constructor.

## Intentional

- **Same helper shape as `_build_landing_page_service`.**
  Two repo args (blueprints + blog_posts) instead of one,
  but otherwise identical -- short-circuit on
  `llm is None or pool is None`, no IntelligenceRepository
  needed (blog_post takes blueprints, not opportunities).
- **No `intelligence is None` guard.** Unlike campaign /
  report / sales_brief, `BlogPostGenerationService` does not
  take an `IntelligenceRepository` -- its data source is the
  blueprint store, not the campaign-opportunity table.
- **`PostgresBlogBlueprintRepository` constructed with default
  `table="blog_blueprints"`.** The dataclass exposes a
  `table` override for tests; production accepts the default.
- **Bundle advertises `blog_post` even with an empty
  `blog_blueprints` table.** The slot wiring is independent
  of data presence -- an empty table just means the
  generator returns zero drafts. Surfacing
  `service_not_configured` for "no blueprints yet" would
  conflate "wiring missing" with "no data" and break the
  catalog endpoint's `execution.configured_outputs` contract.
- **`enable_db_services=False` production default
  preserved.** Same Codex P1 safety pin from PR #454; PR
  #455 flipped it on for the production mount. No change.

## Deferred

- Host autonomous task changes to populate
  `blog_blueprints` (separate slice or operator ETL).
- End-to-end smoke test posting to
  `/api/v1/content-ops/execute` with each generator and
  asserting real drafts persist.
- Multi-pass reasoning provider host wiring.
- Per-service config tuning.

## Verification

- `pytest tests/test_atlas_content_ops_execution_services.py`
  -- updated tests pass.
- AST + ASCII checks on the modified module + test file.

## Estimated diff size

- `_content_ops_services.py`: ~30 LOC delta.
- Test: ~80 LOC delta.
- Plan doc: ~155 LOC.

Total: ~265 LOC. Comfortably within the 400 LOC PR target.
The wiring is the smallest of the four execution-services
slices because the helper is single-shape (no shared
IntelligenceRepository instance) and the test extensions
are minimal -- the existing fixtures cover the new slot.
