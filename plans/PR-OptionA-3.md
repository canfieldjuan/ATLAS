# PR-OptionA-3: Promote quality + retry-excerpt knobs to load-bearing

## Why this slice exists

PR-OptionA-2 (#369) closed the LLM-tuning trio (`temperature`,
`max_tokens`, `parse_retry_attempts`) across all five generated-asset
services. That PR's plan doc deferred the remaining `step.config`
fields to "PR-OptionA-3."

This PR closes the next clean subset: the per-field knobs that have a
direct dataclass field in their service config and are referenced
inside the generation flow. Concretely:

- `quality_revalidation_enabled` (campaign) -- gate on whether the
  campaign service runs quality revalidation after parse
- `quality_prompt_proof_term_limit` (campaign) -- cap on proof-term
  count in revalidation prompts
- `parse_retry_response_excerpt_chars` (all five services) -- max
  characters of the prior invalid response to include in retry user
  prompts

After this lands, an operator picking "skip revalidation for this
batch" or "show longer error excerpts in retry prompts" actually
changes behavior. Before, those fields were emitted in
`step.config` but silently ignored.

## Scope (this PR)

Three fields. Same per-field-kwarg pattern as PR-OptionA-1 / -2:
optional kwarg on `generate()`, resolved at the top, threaded down
to the inner helper that uses it. `None` falls through to
`self._config.X`.

| Service | New `generate()` kwargs |
|---|---|
| `CampaignGenerationService` | `quality_revalidation_enabled`, `quality_prompt_proof_term_limit`, `parse_retry_response_excerpt_chars` |
| `ReportGenerationService` | `parse_retry_response_excerpt_chars` |
| `SalesBriefGenerationService` | `parse_retry_response_excerpt_chars` |
| `LandingPageGenerationService` | `parse_retry_response_excerpt_chars` |
| `BlogPostGenerationService` | `parse_retry_response_excerpt_chars` |

`content_ops_execution.py` adds `_step_config_bool` helper for
the boolean flag and threads the three fields from `step.config`
into the `email_campaign` dispatcher (which owns all three). Other
dispatchers thread only `parse_retry_response_excerpt_chars`.

## Intentional (looks wrong but is deliberate)

- **Same per-field-kwarg pattern as the rest of the OptionA series.**
  Nine load-bearing per-call fields total now (`channels`,
  `default_report_type`, `default_brief_type`, `temperature`,
  `max_tokens`, `parse_retry_attempts` from prior PRs +
  `quality_revalidation_enabled`, `quality_prompt_proof_term_limit`,
  `parse_retry_response_excerpt_chars` from this PR). Still using the
  same `kwarg=None falls back to self._config.kwarg` pattern. No
  shared abstraction; no dataclass override surface. The pattern
  compresses well by repetition, and a future "config override"
  abstraction would now have at least nine fields to wrap if it ever
  becomes worth introducing.
- **`quality_gates_enabled` (sales_brief / landing_page) deferred to
  PR-OptionA-4.** The plan layer emits `quality_gates_enabled` in
  step.config but neither service has a config field for it -- it's
  a phantom plan field. Implementing it per-call requires actually
  adding a "skip quality gate" path to the service generate flow,
  which is a behavior change beyond pure plumbing. Different shape;
  separate slice.
- **`topic` (blog_post) deferred to PR-OptionA-4.** The plan emits
  `topic` from `request.inputs`, but the blog service reads topic
  from `blueprint.get("topic")` -- it's content input, not a config
  knob. The override semantic is "force this topic for this call,
  ignoring the blueprint" which is conceptually different from the
  config-knob pattern of every other OptionA field.
- **Boolean kwarg defaults to None, not False.** A `False` default
  would be ambiguous: did the operator pick False, or did they leave
  it alone? `None` makes the override-vs-fallthrough distinction
  explicit at the call site, matching the pattern from the other
  OptionA kwargs.
- **`_step_config_bool` rejects non-bool values rather than coercing
  with `bool(...)`.** A mis-typed `step.config["quality_revalidation_enabled"]
  = "yes"` would coerce truthy under `bool(...)` and silently land
  the wrong value at the service. Defensive: returns None on anything
  that isn't actually a bool, lets the service fall through to its
  config default.

## Deferred (looks missing but is on purpose)

- **`quality_gates_enabled`** (phantom field) -- PR-OptionA-4.
- **`topic` for blog_post** (content-input shape) -- PR-OptionA-4.
- **`MarketingCampaign.context` leak** -- audit MAJOR; PR-OptionA-4.
- **`channel`/`channels` legacy dual-field cleanup** -- PR-OptionA-4
  or separate cleanup PR.
- **`skill_name` per-call** -- bigger UX question (which prompt
  template to use), deferred indefinitely.
- **The 9 MINOR + 2 NIT findings from the audit** -- separate batch
  cleanup PR.
- **`PR-ContentAssets-Consistency-2`** -- still owed; migrate
  `campaign_postgres_review.py` / `campaign_postgres_import.py` /
  `podcast_postgres.py` to the shared `_jsonb_helpers.py`.

## Verification

- `pytest` across all touched suites + the upstream control-surface /
  generation-plan tests
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline` -> clean
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> 0
- `bash scripts/check_ascii_python.sh` -> passed

## Sibling references

- PR-OptionA-1 plan: `plans/PR-OptionA-1.md`
- PR-OptionA-2 plan: `plans/PR-OptionA-2.md`
- Audit doc:
  `docs/audits/ai_content_ops_post_merge_audit_2026-05.md`
- UI contract:
  `extracted_content_pipeline/docs/control_surface_preview_api.md`
