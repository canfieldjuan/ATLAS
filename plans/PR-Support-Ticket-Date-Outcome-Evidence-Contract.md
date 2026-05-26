# PR-Support-Ticket-Date-Outcome-Evidence-Contract

## Why this slice exists

PR #977 closed the immediate fabricated procedural-answer gap by adding explicit
support-ticket resolution evidence. The deferred portion of the broader
evidence-or-placeholder contract is still missing: generators should know
whether the uploaded ticket export has a real date window and whether it
contains measured outcome evidence before they write timeframe or impact claims.

Today the package has `source_period` and optional `faq_window_days`, but the
prompt-facing contract does not expose a direct `has_dated_window` flag. It also
does not expose measured outcome evidence, so prompts still rely on negative
rules instead of a source-truth signal for outcome claims.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-date-outcome-evidence

Slice phase: Production hardening

1. Add explicit date-window and measured-outcome evidence fields to the
   support-ticket package.
2. Preserve numeric zero outcome values as valid evidence.
3. Thread the new fields into landing-page `MarketingCampaign.context` and blog
   `data_context`.
4. Update support-ticket prompt rules so timeframe and outcome claims only use
   the new evidence fields when present.
5. Extend smoke summaries and focused route/package tests.

### Files touched

- `atlas_brain/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/landing_page_input_contract.py`
- `extracted_content_pipeline/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Support-Ticket-Date-Outcome-Evidence-Contract.md`
- `scripts/smoke_content_ops_live_generation.py`
- `scripts/smoke_content_ops_support_ticket_package.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_landing_page_generation.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`
- `tests/test_support_ticket_provider_landing_blog_execute.py`

## Mechanism

The package will expose these fields:

```python
has_dated_window: bool
has_measured_outcomes: bool
measured_outcome_count: int
measured_outcome_examples: list[dict[str, str]]
```

`has_dated_window` mirrors the existing date-window truth used to decide whether
`faq_window_days` is safe to emit. Measured outcome evidence is detected only
from explicit top-level outcome/impact/result fields. The extraction helper must
preserve numeric zero (`0`) as evidence, because zero can be a real measured
outcome.

Execution will pass the same fields into landing-page context and blog
`data_context`. Prompt contracts will then say:

- if `has_dated_window` is false or missing, do not invent calendar windows or
  recurring cadences;
- if `has_measured_outcomes` is false or missing, do not write support-volume,
  churn, retention, capacity, ROI, or time-savings impact claims;
- if `has_measured_outcomes` is true, only cite the exact supplied
  `measured_outcome_examples`.

## Intentional

- This slice does not add new outcome-claim detector patterns. The source fix is
  to provide explicit outcome evidence before generation; existing detectors
  remain as regression backstops.
- This slice does not parse outcome claims from free-form customer complaint
  text. Measured outcomes must come from explicit host/export fields so customer
  frustration is not mistaken for measured impact.
- This keeps the #977 resolution-evidence names unchanged to avoid forcing a
  second review of that already-merged contract.

## Deferred

Parked hardening: none.

Future PR: role-aware transcript parsing can add richer resolution or outcome
evidence into these same fields if support platforms export staff/customer roles
and measured deflection data.

## Verification

- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_content_ops_execution.py::test_marketing_campaign_context_does_not_leak_unrelated_inputs tests/test_extracted_landing_page_generation.py::test_generate_threads_seo_geo_aeo_context_into_system_prompt_payload tests/test_extracted_landing_page_generation.py::test_build_draft_preserves_support_ticket_source_context_metadata tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_csv_counts tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_date_window_when_dates_validate tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_threads_measured_outcomes -q` - 44 passed.
- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_landing_page_generation.py tests/test_extracted_content_ops_execution.py tests/test_smoke_content_ops_live_generation.py -q` - 162 passed.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q` - 32 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` - passed.
- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_threads_measured_outcomes -q` - 32 passed after sync.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Evidence extraction and context wiring | ~110 |
| Prompt contract updates | ~10 |
| Tests and smoke summary coverage | ~160 |
| Plan doc | ~110 |
| **Total** | **~390** |
