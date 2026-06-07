# PR-Content-Ops-Output-Variations-Blog-Only

## Why this slice exists

The merged `PR-Content-Ops-Output-Variations.md` plan records the product ask:
marketers need several alternative takes from one run spec, not only one
take-it-or-leave-it draft. That plan also names a generator-parity fallback if
the full blog + landing page + sales brief slice exceeds the 400 LOC budget.
This PR takes that fallback deliberately: it proves deterministic
output-variation fan-out for `blog_post` first, through the real Content Ops
request -> preview -> plan -> execute -> blog prompt path, without touching the
adjacent landing-page and sales-brief generator contracts yet.

The full diff is over the 400 LOC soft cap because the complete vertical slice
crosses five package surfaces (request, preview, plan, execution, generator) and
ships focused regression tests for each contract. The implementation code stays
under 400 LOC; the remaining bulk is the required plan plus R2/R12 test
coverage.

## Scope (this PR)

Ownership lane: content-ops/output-variations/blog-only
Slice phase: Vertical slice

1. Add a normalized `variant_count` request knob for Content Ops, capped to the
   deterministic blog-angle catalogue.
2. Thread blog-only selected variant angles into preview cost, generation-plan
   metadata, execution fan-out, blog service prompts, LLM metadata, draft
   metadata, and the step result aggregate.
3. Keep `variant_count == 1` behavior as the existing single blog generation
   call so current callers do not see a new optional kwarg unless they request
   variants.
4. Add focused extracted-package tests for request normalization/capping, blog
   cost scaling, plan metadata, executor fan-out/failure isolation, and prompt
   injection.

### Review Contract
- Acceptance criteria:
  - [ ] `variant_count` defaults to 1, rejects values below 1, and caps values
        above the angle catalogue size.
  - [ ] Blog preview/plan cost and metadata scale by selected blog variants;
        non-blog outputs do not get variant fan-out in this slice.
  - [ ] `execute_content_ops_request` runs one blog generation per selected
        angle when `variant_count > 1`, tags per-angle results, aggregates
        saved ids/errors, and isolates one failed variant from the others.
  - [ ] If every requested blog variant raises an exception, the step/run is
        failed while preserving the aggregate warning and per-angle diagnostics.
  - [ ] `BlogPostGenerationService.generate(..., variant_angle=...)` injects
        the angle into the prompt and preserves it in generation/draft metadata.
  - [ ] Existing single-variant blog execution remains backward-compatible.
- Affected surfaces: extracted package API request model, preview/plan,
  execution result shape, blog-generation prompt metadata, tests.
- Risk areas: backcompat, cost accounting, result contract, prompt truthfulness.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/output_variations.py`
- `plans/PR-Content-Ops-Output-Variations-Blog-Only.md`
- `tests/test_extracted_blog_generation.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`

## Mechanism

`extracted_content_pipeline/output_variations.py` owns the small deterministic
angle catalogue and normalization helpers. `ContentOpsRequest.variant_count`
stores the capped request value and preview cost multiplies `blog_post` cost by
that value only. The generation plan exposes the selected angle metadata in the
blog step config so reviewers and UI callers can see the exact fan-out before
execution.

Runtime fan-out stays blog-local in this thin slice. `_dispatch_blog_post`
preserves today's single call for `variant_count == 1`; when variants are
requested, it calls the blog service once per selected angle, passes the angle
instruction as `variant_angle`, catches per-angle failures, and returns an
aggregate with `variant_results`, total generated/skipped/requested counts,
combined `saved_ids`, combined `errors`, and a warning when every requested
variant produces zero survivors. If the zero-survivor case came from caught
provider/execution exceptions, the dispatcher raises a step-result failure so
`ContentOpsExecutionResult.status` is `failed` while the failed step still
includes the aggregate warning and per-angle diagnostics.

The blog service accepts `variant_angle: str | None`. Non-empty angles are added
to the user prompt as framing guidance while preserving the blueprint facts and
evidence, included in LLM call metadata, carried through parsed result metadata,
and written to draft metadata. `None` / empty angle is a no-op.

## Intentional

- Blog-only first. Landing-page and sales-brief parity are deferred exactly as
  the merged full plan's fallback allowed.
- Fan-out is in the blog dispatcher rather than centralized in the generic step
  loop for this slice. Central fan-out becomes valuable once a second generator
  supports `variant_angle`; before then it would add abstraction without reuse.
- Quality gates remain inside `BlogPostGenerationService`. This slice does not
  duplicate gate logic in the executor; blocked/unparseable variants surface via
  the existing per-call skipped/errors fields and the aggregate all-zero warning.
- The older `ATLAS-HARDENING.md` deep-dive source/corpus items were scanned and
  left parked because they touch published deep-dive truthfulness, not the new
  blog output-variation fan-out path.

## Deferred

- Landing-page `variant_angle` parity.
- Sales-brief `variant_angle` parity.
- Centralized multi-generator fan-out after at least one additional generator
  supports variants.
- Persistent variant grouping (`variant_group_id`) and a past-run variants view.
- Auto-ranking / recommended winner and A/B serving/analytics.

Parked hardening: none.

## Verification

- Command: `pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_blog_generation.py -q` -- PASS, 250 tests.
- Command: `scripts/validate_extracted_content_pipeline.sh` via bash -- PASS.
- Command: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- PASS.
- Command: `python scripts/audit_extracted_standalone.py --fail-on-debt` -- PASS.
- Command: `scripts/check_ascii_python.sh` via bash -- PASS.
- Command: `scripts/run_extracted_pipeline_checks.sh` via bash -- PASS, 3,231 passed / 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 1 |
| `extracted_content_pipeline/blog_generation.py` | 27 |
| `extracted_content_pipeline/content_ops_execution.py` | 151 |
| `extracted_content_pipeline/control_surfaces.py` | 23 |
| `extracted_content_pipeline/generation_plan.py` | 12 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/output_variations.py` | 89 |
| `plans/PR-Content-Ops-Output-Variations-Blog-Only.md` | 146 |
| `tests/test_extracted_blog_generation.py` | 21 |
| `tests/test_extracted_content_control_surfaces.py` | 36 |
| `tests/test_extracted_content_generation_plan.py` | 20 |
| `tests/test_extracted_content_ops_execution.py` | 143 |
| **Total** | **672** |
