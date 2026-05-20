# PR-Blog-SEO-AEO-Quality-Gate

## Why this slice exists

PR-Blog-SEO-AEO-Readiness-Summary made generated blog rows show whether the
basic SEO/AEO contract is present after generation. That is useful for review,
but it still allows an incomplete blog draft to be saved first.

This slice moves the same contract into the blog quality gate for the extracted
AI Content Ops blog generator. The goal is narrow: generated blog drafts should
not save when the SEO/AEO fields that support the product claim are missing or
obviously out of bounds.

## Scope (this PR)

1. Add opt-in SEO/AEO findings to `extracted_quality_gate.blog_pack`.
2. Pass parsed blog SEO/AEO fields from `BlogPostGenerationService` into the
   quality context.
3. Add tests proving missing SEO/AEO fields block generated blog saves.
4. Add quality-pack unit tests for the new SEO/AEO findings.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-SEO-AEO-Quality-Gate.md` | Plan doc for this slice. |
| `extracted_quality_gate/blog_pack.py` | Add opt-in SEO/AEO validators. |
| `extracted_content_pipeline/blog_generation.py` | Opt blog generation into the SEO/AEO gate. |
| `tests/test_extracted_quality_gate_blog_pack.py` | Cover SEO/AEO quality findings. |
| `tests/test_extracted_blog_generation.py` | Prove incomplete SEO/AEO drafts do not save. |

## Mechanism

The quality pack stays pure and reusable. It only runs the new SEO/AEO checks
when the caller sets `require_seo_aeo` in the quality context. The extracted
blog generator sets that flag and passes the parsed SEO metadata into the
context before save.

The gate validates:

- SEO title is present and within the configured title character limit.
- SEO description is present and within the configured description character
  limit.
- Target keyword is present.
- At least one secondary keyword is present.
- The FAQ list meets the configured minimum count.
- The body has a simple answer-engine-friendly structure: either a
  question-style H2 or an answer-first H2 section opening.

## Intentional

- No GEO validator. GEO remains undefined as a separate product contract.
- No prompt changes. This slice enforces the current prompt contract instead of
  rewriting generation instructions.
- No frontend work. The prior readiness-summary slice already surfaces the
  review/export state after save.
- The SEO/AEO contract is opt-in at the quality-pack layer so other direct
  quality-pack consumers are not forced to pass metadata unless they choose to.

## Deferred

- Add a first-class GEO definition if the product wants to claim GEO separately.
- Share the readiness helper between export and quality-gate code if the
  heuristic grows beyond this small contract.
- Add repair-loop instructions that specifically mention missing SEO/AEO
  fields if parse retry expands to quality-repair retry.

## Verification

- Focused blog generation and blog quality-pack tests -> 61 passed.
- Python compile check over edited modules/tests -> passed.
- Diff whitespace check -> passed.
- Full extracted pipeline checks -> 1530 passed, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Quality gate | ~140 |
| Blog generator context | ~15 |
| Tests | ~160 |
| **Total** | **~390** |
