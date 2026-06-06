# PR-OptionA-1: Thread plan step config to execution (channels, report_type, brief_type)

## Why this slice exists

Closes the BLOCKER from the AI Content Ops post-merge audit
(`docs/audits/ai_content_ops_post_merge_audit_2026-05.md`,
`content_ops_execution.py` section): `_run_step` ignores `step.config`
entirely. The control surface preview promises the user "channels:
[email_cold]" and the plan records it in `step.config`, but the
executor calls `service.generate(scope, target_mode, limit, filters)`
with no per-call channel override. The service uses whatever was
injected at construction time (default `("email_cold",
"email_followup")`), so the user gets both an `email_cold` draft AND
an `email_followup` draft despite asking for cold-only in the UI.

The user picked **Option A** from the audit's decision request:
plan-as-execution-contract -- step config is load-bearing; the
executor reads it and passes per-call config to services. The control
surface becomes a real control surface.

## Scope (this PR)

Tight first-cut. Promotes ONLY the smoking-gun fields the audit
specifically called out as breaking the trust contract:

- `channels` (email_campaign) -- the audit's specific failure-mode
  example
- `default_report_type` (report) -- operator-visible enum, drives
  `report_type` column on persisted drafts
- `default_brief_type` (sales_brief) -- operator-visible enum, drives
  `brief_type` column on persisted drafts

Plus: refactor `_run_step` to a per-output dispatch table, which kills
the `landing_page` hard-coded special-case MAJOR (same surface, not
drive-by creep -- the dispatcher rewrite IS the fix shape).

Lower-stakes per-call fields (`skill_name`, `temperature`,
`max_tokens`, `parse_retry_attempts`, `topic`,
`quality_revalidation_enabled`, `quality_gates_enabled`) stay
informational in `step.config` for now and get promoted in
PR-OptionA-2.

## Mechanism

Each affected service signature gains an optional kwarg matching the
plan's `step.config` field name. When the executor passes it, the
service uses the override; when it doesn't, the service falls back to
`self._config` (its construction-time default). No breaking change to
existing callers.

```python
# Before
async def generate(*, scope, target_mode, limit=None, filters=None):
    channels = self._channels()  # reads self._config.channels

# After
async def generate(*, scope, target_mode, limit=None, filters=None,
                   channels=None):  # NEW
    resolved_channels = self._channels(override=channels)
```

`_run_step` becomes a per-output dispatch table:

```python
_DISPATCH = {
    "email_campaign": _dispatch_campaign,
    "report": _dispatch_report,
    "sales_brief": _dispatch_sales_brief,
    "landing_page": _dispatch_landing_page,
    "blog_post": _dispatch_blog_post,
}

async def _run_step(step, *, request, service, scope, filters):
    handler = _DISPATCH.get(step.output, _dispatch_default)
    return await handler(step=step, service=service, request=request,
                         scope=scope, filters=filters)
```

Each handler reads the relevant `step.config` field and threads it
through to its service's `generate()` call. The `landing_page`
handler keeps its existing campaign-from-inputs shape; new outputs
get a one-line registry entry instead of a new `if` branch.

## Files touched

1. `extracted_content_pipeline/campaign_generation.py` --
   `generate(channels=None)` kwarg, threaded through `_channels()`
2. `extracted_content_pipeline/report_generation.py` --
   `generate(default_report_type=None)` kwarg, threaded through
   `_build_draft()`
3. `extracted_content_pipeline/sales_brief_generation.py` --
   `generate(default_brief_type=None)` kwarg, threaded through
   `_build_draft()`
4. `extracted_content_pipeline/content_ops_execution.py` --
   per-output dispatch table; threads the three new kwargs from
   `step.config`
5. `tests/test_extracted_campaign_generation.py` -- tests for
   per-call `channels` override
6. `tests/test_extracted_report_generation.py` -- tests for per-call
   `default_report_type` override
7. `tests/test_extracted_sales_brief_generation.py` -- tests for
   per-call `default_brief_type` override
8. `tests/test_extracted_content_ops_execution.py` -- tests for the
   dispatch-table refactor + plan->service config threading

## Intentional (looks wrong but is deliberate)

- **Per-field kwargs, not a `config_override: Mapping[str, Any]`
  dict.** A dict-shaped override would be more general but makes the
  contract opaque -- callers can't statically see which fields are
  load-bearing vs informational. Per-field kwargs make the surface
  type-checkable and the load-bearing fields explicit at the
  signature level.
- **`landing_page` and `blog_post` get no per-call overrides this
  PR.** Their plan configs are still emitted but stay informational
  for now. Promoting them to load-bearing in PR-OptionA-1 would
  double the LOC budget and pull in two more
  audit-flagged-but-out-of-scope MAJORs (MarketingCampaign.context
  leak, blog topic threading). The dispatch-table makes adding them
  later a one-handler-edit.
- **`default_report_type` / `default_brief_type` keep their
  ``default_`` prefix.** That naming captures the "fallback if LLM
  omits the type from JSON" semantic; the LLM's output still wins
  when present. The plan and the service agree on this semantic. UI
  copy should call them "Default report type" / "Default brief type"
  (not "Report type" / "Brief type" -- those imply the value is
  forced regardless of LLM output).
- **No new abstraction layer for "config override."** The dispatch
  table is the only new structure. Each service reads one or two
  optional kwargs; no `ConfigOverride` dataclass, no
  `apply_overrides_to(config)` helper. Three services with three
  one-field overrides isn't enough surface to justify a shared
  abstraction.

## Deferred (looks missing but is on purpose)

- **Promoting the rest of the plan's `step.config` fields to
  load-bearing.** PR-OptionA-2 will tackle `temperature`,
  `max_tokens`, `parse_retry_attempts`, and the
  `quality_revalidation_enabled` / `quality_gates_enabled` boolean
  flags. Same per-field-kwarg pattern. ~5 services x 2-3 fields each.
- **`MarketingCampaign.context` leak.** The audit flagged that
  `_marketing_campaign_from_inputs` dumps every non-{name, persona,
  value_prop, vendors, categories, tags} input field into
  `MarketingCampaign.context`. PR-OptionA-3.
- **`channel`/`channels` legacy dual-field on
  `CampaignGenerationConfig`.** The audit flagged the legacy
  `channel: str` field still exists alongside `channels: tuple[str,
  ...]`. Removing it is a separate cleanup PR -- not in scope here.
- **The 9 MINOR + 2 NIT findings.** All catalogued in the audit doc;
  each gets a focused follow-up.
- **PR-ContentAssets-Consistency-2.** Still owed from the PR-#356
  review threads (3 more adapters with byte-identical
  `_jsonb`/`_row_dict`).

## Verification

- `pytest` on the four touched test files + the existing
  `test_extracted_content_control_surfaces.py` /
  `test_extracted_content_generation_plan.py` to confirm no
  regression on the upstream layers.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline` -> clean
- `python scripts/audit_extracted_standalone.py --fail-on-debt` ->
  Atlas runtime import findings: 0
- `bash scripts/check_ascii_python.sh` -> passed
- AST scan of the three touched generators: no new `atlas_brain`
  references.

End-to-end smoke (manual, against fake services): a plan with
`channels: ["email_cold"]` results in `service.generate` receiving
`channels=("email_cold",)`, not the construction-time default.

## Sibling references

- Audit doc:
  `docs/audits/ai_content_ops_post_merge_audit_2026-05.md` (lives on
  PR #367)
- UI contract:
  `extracted_content_pipeline/docs/control_surface_preview_api.md`
- Source PR for the control-surface scaffold being fixed: #353
- Audit doc PR: #367 (will close after the BLOCKER + dispatch-table
  MAJOR are fixed by this PR)
