# PR-OptionA-4: Close audit MAJORs (quality_gates_enabled, MarketingCampaign.context leak)

## Why this slice exists

PR-OptionA-1/2/3 closed the per-field plan-as-execution-contract
trust gap for the fields that had a clean dataclass mapping. This
PR closes the remaining audit MAJORs that don't fit that pattern:

1. **`quality_gates_enabled`** (sales_brief, landing_page) is a
   phantom field -- the plan emits it in `step.config` but neither
   service has a config field for it, so the executor can't route an
   override even if it tried. The audit flagged this as a trust-
   contract break: operator picks "skip quality gates" in the
   control surface, plan records it, executor drops it on the
   floor. Fix: add `quality_gates_enabled: bool = True` to
   `LandingPageGenerationConfig` and `SalesBriefGenerationConfig`,
   teach `_quality_check` to short-circuit when False, thread per-
   call override through `generate()`, route through the dispatcher.

2. **`MarketingCampaign.context` leak** -- the audit flagged that
   `_marketing_campaign_from_inputs` in `content_ops_execution.py`
   dumps every non-{name, persona, value_prop, vendors, categories,
   tags} input field into `MarketingCampaign.context`. So if the
   request inputs include `target_account`, `offer`,
   `opportunity_id`, `topic`, `audience`, etc. (the standard
   control-surface input bag), all of them leak into the landing-
   page service's `campaign.context` -- which the prompt template
   then sees as if it were intentional context. Fix: replace the
   "everything except known fields" inversion with an explicit
   allowlist of context-shaped fields (`industry`, `pain_points`,
   `differentiators`, etc.) so unrelated inputs stay out of the
   campaign payload.

## Scope (this PR)

Only the two items above, plus a smoke-CLI mock fix that became
necessary mid-PR (see "CI fix piggybacked" below). The remaining
audit findings (`topic` for blog_post -- no service-side landing
surface; `channel`/`channels` legacy dual-field on
`CampaignGenerationConfig`; the 9 MINOR + 2 NIT findings) defer to
follow-ups.

### CI fix piggybacked

PR #366 (codex/content-ops-execution-smoke) added
``scripts/smoke_extracted_content_ops_execution.py`` plus
``tests/test_extracted_content_ops_execution_smoke.py`` to the
extracted-pipeline CI gate. The smoke fakes
(``_OpportunityAssetService``, ``_LandingPageAssetService``) had
strict signatures that pre-dated PR-OptionA-1/2/3, so they reject
the per-call kwargs the dispatcher now threads through
(``channels``, ``default_report_type``, ``default_brief_type``,
temperature/max_tokens/parse-retry knobs, the quality-gate flags).
#366 merged after #368/#369/#370 but did not update the fakes; this
PR (#371) is the first PR running CI after that merge, so the
breakage surfaced here.

Fix: the smoke fakes accept ``**extras`` and discard them. The
smoke CLI exists to verify the seam end-to-end -- strict per-kwarg
contract assertions live in
``tests/test_extracted_content_ops_execution.py``. Folding this
fix into #371 instead of opening a separate PR avoids a serialized
3-PR train (#371-fix -> #371-rebase -> #371-merge) for what is
ultimately a 4-line patch.

### `quality_gates_enabled`

Both services already have an optional `quality_policy: QualityPolicy
| None`. Currently the gate runs unconditionally when `_quality_check`
is called (which happens after every parse). The new field gates
that call: when False, skip evaluation entirely and treat the parse
as passed.

Per-call override follows the established OptionA shape:
`generate(quality_gates_enabled: bool | None = None)`, resolved at
the top of `generate()` against `self._config.quality_gates_enabled`.

The campaign service has its own `quality_revalidation_enabled`
field which already covers the equivalent gate; this PR doesn't
touch it.

### MarketingCampaign.context

**Backwards-compat note:** this is a breaking change for hosts whose
customized landing-page prompts reference context fields outside the
new allowlist. The packaged prompt uses ``{campaign_json}`` as a
generic JSON-dump substitution, so the LLM sees whatever is in
``campaign.context``; if a host customized the prompt to reference,
e.g., ``{campaign_json.context.target_account}``, post-fix that
reference resolves to empty. Migration path is in the Migration
section below.

Current `_marketing_campaign_from_inputs`:

```python
context={
    str(key): value
    for key, value in inputs.items()
    if key not in {"campaign_name", "offer", "audience", "vendors",
                   "categories", "tags"}
}
```

This is a "negative-list inversion" -- everything not explicitly
excluded leaks. Standard control-surface inputs (`target_account`,
`opportunity_id`, `topic`, `filters`, `report_type`, `brief_type`,
`channels`, etc.) all flow into `campaign.context`.

New shape: explicit allowlist of context-meaningful fields. Anything
not on the allowlist stays out. The allowlist is "fields the LLM is
allowed to see in ``campaign.context``" (the packaged prompt uses a
generic JSON dump, not field-by-name reference) -- not "fields the
prompt consumes." It starts conservative and grows by explicit
additions.

## Migration

Hosts on the prior shape who customized their landing-page prompt
template to reference context fields not in
``_MARKETING_CAMPAIGN_CONTEXT_FIELDS`` will see empty values for
those references post-fix. To restore visibility:

1. Identify the field name your custom prompt references (e.g.,
   ``target_account``).
2. Add it to ``_MARKETING_CAMPAIGN_CONTEXT_FIELDS`` in
   ``content_ops_execution.py``. The list is meant to grow.
3. Optionally update
   ``extracted_content_pipeline/docs/control_surface_preview_api.md``
   to advertise the field as part of the context contract.

Hosts using the packaged prompt unchanged are unaffected -- the
packaged prompt does not reference context fields by name.

## Intentional (looks wrong but is deliberate)

- **`quality_gates_enabled` defaults to True**, not False.
  Backwards-compat: existing hosts that don't set the field get the
  current always-on behavior. The override is the new affordance,
  not a behavior change for default callers.
- **The MarketingCampaign.context allowlist starts small.**
  `industry`, `pain_points`, `differentiators`, `customer_segments`,
  `key_metrics`. The audit's leak example was overly broad; the fix
  is to narrow rather than to fix the leak field-by-field. If hosts
  need additional context fields they're added here, not by
  whitelisting them at the call site.
- **`topic` is not promoted to per-call.** The plan emits it in
  `step.config["topic"]` but
  `BlogPostGenerationService._generate_one` doesn't reference
  `topic` anywhere -- it reads from `blueprint`. Adding a service
  kwarg with no landing surface would be plumbing for nothing.
  Closing this gap requires the blog service to actually consume
  `topic` in its prompt template, which is content-shape work, not
  trust-contract work.
- **`_step_config_bool` already handles `quality_gates_enabled`
  defensively.** Helper added in PR-OptionA-3; reused here.
- **`channel`/`channels` legacy dual-field cleanup is deferred.**
  Real cleanup but doesn't fit this slice's "close audit MAJORs"
  framing -- it's a private-implementation cleanup, not a trust-
  contract break.

## Deferred (looks missing but is on purpose)

- **`quality_gates_enabled` for ``report`` and ``blog_post``.** This
  PR adds it to ``sales_brief`` and ``landing_page`` only, leaving
  three different per-call gate-skip mechanisms across the five
  services: ``email_campaign`` uses ``quality_revalidation_enabled``
  (PR-OptionA-3), ``sales_brief`` / ``landing_page`` use
  ``quality_gates_enabled`` (this PR), ``report`` / ``blog_post``
  have no per-call skip. Operators picking "skip quality gates" in
  the control surface get inconsistent behavior depending on output.
  **PR-OptionA-5** will add ``quality_gates_enabled`` to
  ``ReportGenerationConfig`` and ``BlogPostGenerationConfig`` for
  symmetry. (Could be folded into this PR, but the structural fixes
  here are tightly scoped; symmetry-completion belongs in its own
  slice.)
- `topic` for blog_post (see Intentional).
- `channel`/`channels` legacy dual-field cleanup -- separate slice.
- 9 MINOR + 2 NIT findings from the audit -- batch cleanup PR.
- `PR-ContentAssets-Consistency-2` -- still owed.

## Verification

- `pytest` across all touched suites
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline` -> clean
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> 0
- `bash scripts/check_ascii_python.sh` -> passed
