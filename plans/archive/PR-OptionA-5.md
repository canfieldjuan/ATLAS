# PR-OptionA-5: quality_gates_enabled symmetry for report + blog_post

## Why this slice exists

PR-OptionA-4 (#371) added ``quality_gates_enabled`` to
``LandingPageGenerationConfig`` and ``SalesBriefGenerationConfig``,
leaving three different per-call gate-skip mechanisms across the
five services:

- ``email_campaign``: ``quality_revalidation_enabled``
  (PR-OptionA-3) -- campaign-specific revalidation gate
- ``sales_brief`` / ``landing_page``: ``quality_gates_enabled``
  (PR-OptionA-4) -- generic quality-gate opt-out
- ``report`` / ``blog_post``: nothing -- no per-call skip path

Operators picking "skip quality gates" in the control surface get
inconsistent behavior depending on which output they target. A
report or blog draft cannot opt out of its quality gate per call.

This PR closes the asymmetry by adding ``quality_gates_enabled`` to
the report and blog services using the same pattern as PR-OptionA-4.

## Scope (this PR)

Three files in the per-asset pipeline + one in the executor + plan
emission:

1. ``extracted_content_pipeline/report_generation.py`` -- add
   ``quality_gates_enabled: bool = True`` to
   ``ReportGenerationConfig``; teach ``_quality_check`` to short-
   circuit when False; thread per-call override through
   ``generate()``.
2. ``extracted_content_pipeline/blog_generation.py`` -- same shape.
3. ``extracted_content_pipeline/generation_plan.py`` -- emit
   ``quality_gates_enabled`` in the report and blog_post step
   configs (mirrors what landing_page and sales_brief steps already
   emit).
4. ``extracted_content_pipeline/content_ops_execution.py`` -- thread
   the field from step.config into the report and blog dispatchers
   via the existing ``_step_config_bool`` helper.

After this lands, four of the five services use the same
``quality_gates_enabled`` mechanism. ``email_campaign`` keeps its
domain-specific ``quality_revalidation_enabled`` because the
revalidation is a different gate than the parsed-output check;
unifying those two would conflate distinct quality concerns and
isn't in scope here.

## Intentional (looks wrong but is deliberate)

- **Same per-field-kwarg pattern as PR-OptionA-4.** Continues the
  uniform OptionA shape; no new abstraction.
- **``email_campaign`` keeps its own
  ``quality_revalidation_enabled``.** The revalidation gate is
  semantically different (post-parse re-prompting against quality
  criteria) from the standard ``_quality_check`` (deterministic
  validator on parsed structure). Operators picking "skip quality
  gates" in the control surface get a sensible default behavior on
  campaign without entangling the two gate concepts.
- **Default ``True`` preserves current always-on behavior.** Same
  rationale as PR-OptionA-4: the override is the new affordance,
  not a behavior change for default callers.
- **No service-level test for the blog ``quality_gates_enabled=False``
  skip path beyond the dispatcher test + the smoking-gun pattern
  used for sales_brief / landing_page.** The blog service's
  ``_quality_check`` shape mirrors the others -- adding a fifth
  service-level smoking-gun test (after the four already in place)
  is more coverage repetition than risk reduction. Dispatcher-level
  threading test + visual diff is sufficient.

## Deferred (looks missing but is on purpose)

- ``topic`` for blog_post (still no service-side landing surface).
- ``channel``/``channels`` legacy dual-field cleanup -- separate
  slice.
- 9 MINOR + 2 NIT findings from the audit -- batch cleanup PR.
- ``PR-ContentAssets-Consistency-2`` -- still owed.

## Verification

- ``pytest`` across the four touched suites + the upstream control-
  surface / generation-plan tests
- ``python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`` -> clean
- ``python scripts/audit_extracted_standalone.py --fail-on-debt`` -> 0
- ``bash scripts/check_ascii_python.sh`` -> passed

## Sibling references

- PR-OptionA-1 plan: ``plans/PR-OptionA-1.md``
- PR-OptionA-2 plan: ``plans/PR-OptionA-2.md``
- PR-OptionA-3 plan: ``plans/PR-OptionA-3.md``
- PR-OptionA-4 plan: ``plans/PR-OptionA-4.md``
