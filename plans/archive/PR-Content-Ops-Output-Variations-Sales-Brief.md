# PR-Content-Ops-Output-Variations-Sales-Brief

## Why this slice exists

The merged `PR-Content-Ops-Output-Variations.md` plan called for deterministic
output variations across blog posts, landing pages, and sales briefs. PR #1347
landed blog variants, PR #1349 landed landing-page variants, and PR #1351
removed their duplicated fan-out loop before adding a third generator.

This PR adds the remaining generator parity slice: sales-brief preview/plan and
execution now honor `variant_count`, and `SalesBriefGenerationService.generate`
gets the same optional `variant_angle` prompt/metadata threading already used by
blog and landing-page generation. It reuses the shared fan-out helper rather
than copying per-angle aggregation logic again.

The full diff may exceed the 400 LOC soft cap because this is the final
vertical slice across preview, plan, execution, generator prompt metadata, and
focused R2/R12 tests. The implementation itself should stay narrow; the bulk is
plan and regression coverage needed to prove sales briefs match the already
merged output-variation contract.

## Scope (this PR)

Ownership lane: content-ops/output-variations/sales-brief
Slice phase: Vertical slice

1. Make sales-brief preview cost and generation-plan metadata scale by the
   selected deterministic variant angles when `variant_count > 1`.
2. Fan out sales-brief execution once per selected angle through the shared
   `_dispatch_output_variants` helper, preserving per-item failure isolation and
   all-raising step/run failure behavior.
3. Add `variant_angle: str | None = None` to
   `SalesBriefGenerationService.generate(...)`, thread it into the LLM prompt,
   LLM metadata, parsed metadata, and saved draft metadata.
4. Keep `variant_count == 1` as the existing single sales-brief call and keep
   blog/landing-page behavior unchanged in this slice.

### Review Contract
- Acceptance criteria:
  - [ ] Sales-brief preview cost multiplies by the normalized variant count.
  - [ ] Sales-brief generation-plan steps expose `variant_count` and
        `variant_angles` when variants are requested.
  - [ ] `execute_content_ops_request` calls the sales-brief service once per
        selected angle, passes each angle instruction as `variant_angle`, and
        aggregates `variant_results`, counts, saved ids, errors, warnings, and
        consumed reasoning context.
  - [ ] One failed sales-brief variant does not abort sibling variants.
  - [ ] If every sales-brief variant raises an exception, the step/run fails
        while preserving the aggregate diagnostic result.
  - [ ] `SalesBriefGenerationService.generate(..., variant_angle=...)`
        injects the angle into the user prompt and preserves it in LLM call
        metadata and saved draft metadata; `None` is a no-op.
  - [ ] Existing blog and landing-page variant behavior is not changed.
- Affected surfaces: extracted package preview cost, generation-plan metadata,
  sales-brief execution dispatcher, sales-brief prompt/metadata, tests.
- Risk areas: cost accounting, execution result contract, prompt truthfulness,
  backward compatibility for existing sales-brief callers.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `plans/PR-Content-Ops-Output-Variations-Sales-Brief.md`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_sales_brief_generation.py`

## Mechanism

Preview cost expands the existing variant multiplier from blog/landing-page to
sales briefs. Generation-plan config reuses `_variant_config_for_request`, so
the selected deterministic angle metadata is visible before execution.

`_dispatch_sales_brief` keeps its current single service call when
`variant_count == 1`. For multi-variant requests it builds the same kwargs as
today, then passes a sales-brief callback into `_dispatch_output_variants`. The
callback calls `SalesBriefGenerationService.generate(...,
variant_angle=angle.instruction, **kwargs)`. The shared helper owns aggregate
counts, `variant_results`, saved ids, errors, warnings, consumed reasoning
contexts, partial failure isolation, and all-raising step failure status.

The sales-brief generator accepts an optional `variant_angle` string. When
present, `_sales_brief_user_prompt` adds a short "Variant angle" instruction
that tells the model to change framing while preserving the same opportunity
facts and supplied evidence. The same angle is sent in LLM metadata, carried on
the parsed payload, and written into `SalesBriefDraft.metadata`. Empty or
`None` values are ignored so existing callers keep current behavior.

## Intentional

- This is sales-brief parity, not ranking, A/B serving, or variant persistence.
  Those were explicitly deferred in the parent output-variations plan.
- The executor does not duplicate sales-brief quality-gate logic. Each per-angle
  service call still owns parse and quality-gate decisions; the executor only
  aggregates the public result contract.
- The prompt angle changes framing only. It must preserve the same opportunity
  facts and supplied evidence so variants do not invent new claims.

## Deferred

- Persistent variant grouping (`variant_group_id`) and a past-run variants view.
- Auto-ranking / recommended winner and A/B serving/analytics.
- Product UI affordances for comparing all generated variants after execution.

Parked hardening: none.

## Verification

- Command: `pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_sales_brief_generation.py -q` -- PASS, 211 tests.
- Command: `scripts/validate_extracted_content_pipeline.sh` via bash -- PASS.
- Command: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- PASS.
- Command: `python scripts/audit_extracted_standalone.py --fail-on-debt` -- PASS.
- Command: `scripts/check_ascii_python.sh` via bash -- PASS.
- Command: `scripts/run_extracted_pipeline_checks.sh` via bash -- PASS, 3,242 passed / 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/content_ops_execution.py` | 30 |
| `extracted_content_pipeline/control_surfaces.py` | 2 |
| `extracted_content_pipeline/generation_plan.py` | 1 |
| `extracted_content_pipeline/sales_brief_generation.py` | 53 |
| `plans/PR-Content-Ops-Output-Variations-Sales-Brief.md` | 133 |
| `tests/test_extracted_content_control_surfaces.py` | 4 |
| `tests/test_extracted_content_generation_plan.py` | 20 |
| `tests/test_extracted_content_ops_execution.py` | 159 |
| `tests/test_extracted_sales_brief_generation.py` | 26 |
| **Total** | **428** |
