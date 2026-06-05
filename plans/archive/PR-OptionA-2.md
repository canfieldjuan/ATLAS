# PR-OptionA-2: Promote LLM-tuning fields to load-bearing (temperature, max_tokens, parse_retry_attempts)

## Why this slice exists

PR-OptionA-1 (#368) closed the trust contract for the smoking-gun
fields the audit specifically called out (`channels`, `report_type`,
`brief_type`). The plan doc for that PR explicitly deferred the
remaining `step.config` fields:

> "Lower-stakes per-call fields (`skill_name`, `temperature`,
> `max_tokens`, `parse_retry_attempts`, `topic`,
> `quality_revalidation_enabled`, `quality_gates_enabled`) stay
> informational in `step.config` for now and get promoted in
> PR-OptionA-2."

This PR closes the next layer: the three LLM-tuning knobs that every
generated-asset service uses identically. After this lands, an
operator picking "use a cheaper model" or "lower retry budget" in
the control surface actually changes the LLM call -- not just the
plan preview.

## Scope (this PR)

Promotes ONLY these three fields, across ALL five services:

- `temperature` (float)
- `max_tokens` (int)
- `parse_retry_attempts` (int)

Why these three and not the rest:
- All five services have the same dataclass field shape and same
  call-site pattern (`self._config.X` referenced inside the inner
  `_generate_one` helper). One uniform fix.
- They map directly to provider cost / quality tradeoffs operators
  care about.
- The boolean flags (`quality_revalidation_enabled`,
  `quality_gates_enabled`) are gates rather than tuning knobs --
  different semantic, deferred to PR-OptionA-3.
- `topic` is content input not tuning, deferred to PR-OptionA-3.
- `skill_name` is a hard structural choice (which prompt to use);
  exposing it per-call is a bigger UX question.

## Mechanism

Same per-field-kwarg pattern as PR-OptionA-1, applied uniformly:

```python
async def generate(
    self, *, scope, target_mode, limit=None, filters=None,
    # ... existing OptionA-1 kwargs ...
    temperature: float | None = None,
    max_tokens: int | None = None,
    parse_retry_attempts: int | None = None,
) -> Result:
    resolved_temperature = (
        self._config.temperature if temperature is None else float(temperature)
    )
    resolved_max_tokens = (
        self._config.max_tokens if max_tokens is None else int(max_tokens)
    )
    resolved_parse_retry_attempts = (
        self._config.parse_retry_attempts
        if parse_retry_attempts is None
        else int(parse_retry_attempts)
    )
    # ... pass resolved values to _generate_one ...
```

Each service's `_generate_one` accepts the three new params and uses
them instead of `self._config.X` at the existing LLM call sites and
retry-loop construction.

## Dispatcher changes

`content_ops_execution.py` adds two helpers
(`_step_config_int`, `_step_config_float`) and threads the three
fields from `step.config` into each per-output dispatcher. The
default dispatcher (used by blog_post for now) also threads them.
The existing `_step_config_text` / `_step_config_sequence` helpers
from PR-OptionA-1 are reused.

The `_DISPATCH` table from PR-OptionA-1 now needs `blog_post` -- it
gets a per-output handler so the three new kwargs reach
`BlogPostGenerationService.generate(...)`. (Dispatch entry was
deferred in OptionA-1 because there were no per-call kwargs to
thread; now there are.)

## Files touched

1. `extracted_content_pipeline/campaign_generation.py` --
   `generate(temperature=, max_tokens=, parse_retry_attempts=)` +
   threading through `_generate_one_for_channel`
2. `extracted_content_pipeline/report_generation.py` -- same shape
3. `extracted_content_pipeline/sales_brief_generation.py` -- same
4. `extracted_content_pipeline/landing_page_generation.py` -- same
5. `extracted_content_pipeline/blog_generation.py` -- same
6. `extracted_content_pipeline/content_ops_execution.py` -- add
   `_step_config_int` / `_step_config_float` helpers; thread the 3
   fields into all 5 dispatchers; register `blog_post` in `_DISPATCH`
7. Test fakes in `tests/test_extracted_content_ops_execution.py`
   gain `temperature` / `max_tokens` / `parse_retry_attempts` named
   kwargs so the existing `extras == {}` assertions stay accurate
8. New regression tests in
   `tests/test_extracted_content_ops_execution.py` -- one
   end-to-end test per output covering all three fields flowing
   from plan to service
9. New per-service override tests in (one per service):
   - `tests/test_extracted_campaign_generation.py`
   - `tests/test_extracted_report_generation.py`
   - `tests/test_extracted_sales_brief_generation.py`
   - `tests/test_extracted_landing_page_generation.py`
   - `tests/test_extracted_blog_generation.py`

## Intentional (looks wrong but is deliberate)

- **Same per-field-kwarg pattern as OptionA-1, not a new abstraction.**
  Keeping the surface uniform: `channels`, `default_report_type`,
  `default_brief_type` (OptionA-1) + `temperature`, `max_tokens`,
  `parse_retry_attempts` (this PR) all use the same
  `kwarg_name=None falls back to self._config.kwarg_name` pattern.
  No `ConfigOverride` dataclass, no `apply_overrides_to(config)`
  helper. Six fields with consistent shape isn't enough surface to
  justify a shared abstraction yet.
- **Each service's `_generate_one` (or equivalent inner helper)
  gains 3 new params**, threaded down from `generate()`. Could be
  done with a "use this effective config" pattern via
  `dataclasses.replace`, but that adds a layer that costs more than
  it saves -- the inner helpers already explicitly read individual
  config fields, so passing them as params is a strictly local
  change.
- **Blog_post promoted from `_dispatch_default` to its own
  dispatcher.** The default dispatcher exists for future asset types
  with no per-call config; with this PR every existing asset has at
  least 3 per-call kwargs, so blog_post deserves its own handler.
  The default dispatcher stays for genuinely no-config future
  outputs.
- **No cap-validation on incoming integer overrides** (e.g.,
  `parse_retry_attempts=999`). The audit didn't ask for that; the
  values flow into the existing `max(1, int(...) + 1)` clamp at the
  call site. If hosts want guardrails they add them at the
  preview-layer threshold check, not at the executor.

## Deferred (looks missing but is on purpose)

- **`quality_revalidation_enabled` / `quality_gates_enabled`
  booleans.** Different semantic (gates, not tuning), deferred to
  PR-OptionA-3.
- **`topic` for blog_post.** Content input not tuning. Deferred to
  PR-OptionA-3.
- **`skill_name` per-call.** Bigger UX question (which prompt
  template to use). Deferred indefinitely.
- **`MarketingCampaign.context` leak** -- still on PR-OptionA-3's
  list.
- **`channel`/`channels` legacy dual-field cleanup.**
- **The 9 MINOR + 2 NIT findings from the audit.**
- **`PR-ContentAssets-Consistency-2`** -- still owed from PR #356.

## Verification

- `pytest` on the 6 touched test files plus
  `test_extracted_content_generation_plan.py` /
  `test_extracted_content_control_surfaces.py` /
  `test_extracted_content_control_surface_api.py` -> all green
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline` -> clean
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> 0
- `bash scripts/check_ascii_python.sh` -> passed

## Sibling references

- PR-OptionA-1 plan doc: `plans/PR-OptionA-1.md`
- Audit doc: `docs/audits/ai_content_ops_post_merge_audit_2026-05.md`
- UI contract:
  `extracted_content_pipeline/docs/control_surface_preview_api.md`
