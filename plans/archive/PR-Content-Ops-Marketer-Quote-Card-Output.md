# PR: Content Ops Marketer Quote Card Output

## Why this slice exists

The marketer reviews-as-input lane has completed the social-post and ad-copy
generated asset paths. The remaining deferred marketer asset family from the
sales-brief handoff is stat/quote cards: short visual-card source material that
marketers can later export into designed creative. Operators can now produce
short social copy and paid-media copy from reviews, but they still cannot turn a
strong source quote into a bounded quote-card draft through the same Content Ops
execution path.

This slice adds the first quote-card vertical slice as a deterministic,
source-evidence-only generator. It mirrors the first `social_post` and
`ad_copy` output slices: register an executable output, generate bounded
evidence-backed quote-card drafts from `source_material`, wire host execution,
and prove preview/plan/run behavior. It deliberately does not touch #1268's
plan-only output-variations lane.

This PR may exceed the 400 LOC soft cap for the same indivisible first-output
reason as the prior marketer-output slices: the service, manifest entry,
catalog exposure, generation plan, executor dispatch, host wiring, reasoning
no-op policy, UI fixture, and tests need to land together or `quote_card` is
either invisible or not runnable.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input

Slice phase: Vertical slice

1. Add a package-owned deterministic `QuoteCardGenerationService` that turns
   source material into bounded quote-card drafts with source ids, quote text,
   attribution, theme metadata, and non-fatal source-material warnings.
2. Register `quote_card` in the control-surface catalog, generation plan,
   executor service bundle, host service factory, and no-op reasoning policy.
3. Keep quote-card generation source-evidence-only: no LLM prompt, persistence,
   generated-asset review API/UI branch, package-default enrollment, or #1268
   variant fan-out in this slice.
4. Add focused tests for service output/warnings/limits, preview/plan mapping,
   executor dispatch, host factory wiring, and catalog readiness.

### Files touched

- `plans/PR-Content-Ops-Marketer-Quote-Card-Output.md`
- `extracted_content_pipeline/quote_card_generation.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_quote_card_generation.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_atlas_content_ops_execution_services.py`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`

## Mechanism

`QuoteCardGenerationService.generate(...)` accepts `source_material`, expands
the same source-material bundles as the shared Content Ops source adapter via
`source_material_to_source_rows(...)`, normalizes each row through the existing
`source_row_to_campaign_opportunity(...)` adapter, and emits one quote-card
draft per usable opportunity up to `limit`. Each card has a stable
evidence-linked shape:

```python
{
    "id": source_id,
    "quote": "Pricing became hard to justify after renewal.",
    "attribution": "Acme Logistics",
    "headline": "Customer proof for HubSpot",
    "supporting_text": "Use this quote to frame pricing pressure.",
    "theme": "customer_proof",
    "source_id": source_id,
    "source_type": "review",
    "company_name": "...",
    "vendor_name": "...",
    "pain_points": [...],
}
```

The generator is deterministic and no-cost. `source_max_text_chars` controls the
existing source-row adapter text cap, matching `signal_extraction`,
`social_post`, and `ad_copy`. `ContentOpsExecutionServices` gets a `quote_card`
slot and dispatch handler that passes `scope`, `target_mode`,
`source_material`, `limit`, and `max_text_chars` to the service.

## Intentional

- No persistence or generated-asset review branch. This first slice proves the
  output can run from source evidence; durable review/export handoff should
  follow only after the reviewer accepts the output shape.
- No package-default enrollment yet. Review/competitive package defaults remain
  unchanged until the `quote_card` output is executable and tested.
- No #1268 output-variations work. `variant_count` and prompt-angle fan-out are
  a separate plan-only lane and remain untouched.
- No LLM design/creative prompt. This slice keeps quote cards bounded to real
  source evidence and avoids visual-asset hallucination risk.
- No `stat_card` yet. Quote cards are the narrower first card surface; stat-card
  extraction needs numeric-claim selection and validation and should be its own
  slice.

## Deferred

- Add `quote_card` to review and competitive input-package defaults in the next
  quote-card handoff slice.
- Persist generated quote-card drafts into the generated-assets review queue
  after the output shape is accepted.
- Add `stat_card` as a follow-up output with numeric-claim validation.
- LLM/style/channel variants and #1268-style output variations remain future
  product polish.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_quote_card_generation.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_reasoning_policy.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_execution_services.py -q` (326 passed, 1 skipped)
- `python -m json.tool atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json >/dev/null` (passed)
- `python -m py_compile extracted_content_pipeline/quote_card_generation.py extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/reasoning_policy.py atlas_brain/_content_ops_services.py tests/test_extracted_quote_card_generation.py` (passed)
- `git diff --check` (passed)
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (passed; 144 matching tests enrolled)
- `bash scripts/validate_extracted_content_pipeline.sh` (passed)
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` (passed)
- `python scripts/audit_extracted_standalone.py --fail-on-debt` (passed; findings: 0)
- `bash scripts/check_ascii_python.sh` (passed)
- `cd atlas-intel-ui && npm run lint` (passed after `npm ci`)
- `cd atlas-intel-ui && npm run build` (passed after `npm ci`)
- `bash scripts/run_extracted_pipeline_checks.sh` (3009 passed, 10 skipped)
- `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>` (passed on review-fix commit; caller hints reviewed and covered by focused host/API tests in this slice)

## Estimated diff size

Actual: 17 files, +834 / -8. This is above the 400 LOC soft cap for the
indivisible first-output wiring reason described in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Quote-card service | ~236 |
| Catalog/plan/executor/host/reasoning/docs/fixture wiring | ~93 |
| Focused tests | ~364 |
| Plan doc | ~148 |
| **Total** | **~840** |
