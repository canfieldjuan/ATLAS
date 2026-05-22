# PR-Landing-Page-AEO-GEO-Section-Source

## Why this slice exists

The landing-page generator already has SEO/AEO/GEO readiness checks, prompt
guardrails, quality repair, UI visibility, and export telemetry. The remaining
source-level gap is that generated sections are still only generic markdown
blocks with optional `order` metadata.

That makes AEO/GEO quality depend too much on the model happening to write
answer-first prose. This slice tightens the generation source contract so each
landing-page section can carry buyer-question and answer-summary metadata while
still rendering normal visible copy.

## Scope (this PR)

1. Extend the bundled landing-page prompt's section metadata contract.
2. Ask for answer-first section bodies that match the metadata summary.
3. Preserve the no-fake-proof and no-forced-FAQ policy.
4. Add prompt-regression coverage for the AEO/GEO source language.
5. Add a generation regression proving the section metadata survives parsing
   and draft construction.

### Files touched

- `plans/PR-Landing-Page-AEO-GEO-Section-Source.md`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `tests/test_extracted_campaign_skill_registry.py`
- `tests/test_extracted_landing_page_generation.py`

## Mechanism

The landing-page prompt already allows arbitrary per-section metadata, and
`LandingPageSection` already persists that metadata through draft construction.
This slice uses that existing contract instead of adding a new schema.

The prompt now asks each section for:

- `kind`: the role of the section, such as problem, solution, how-it-works,
  proof, pricing, FAQ, objection, or conversion.
- `primary_question`: the buyer/search question the section answers when one is
  natural.
- `answer_summary`: a short direct answer that must also appear in the first
  paragraph of `body_markdown`.

This gives future renderers and reviewers a stable source signal without hiding
important answer copy outside the visible page.

## Intentional

- No storage change: section metadata is already part of the persisted landing
  page shape.
- No parser change: the parser already accepts mapped section metadata.
- No quality-gate hard block: this prompt slice asks for better source
  structure without making older generated drafts invalid.
- No forced FAQ: FAQ/objection content should appear only when campaign context
  supports it.

## Deferred

- `PR-Landing-Page-AEO-GEO-Section-Readiness` can decide whether readiness
  helpers should score this metadata directly.
- `PR-Landing-Page-Publish-Structured-Data` can map FAQ/objection sections into
  public JSON-LD once a generated landing-page renderer exists.

## Verification

- `pytest tests/test_extracted_campaign_skill_registry.py tests/test_extracted_landing_page_generation.py -q` - 41 passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Prompt | ~18 |
| Tests | ~30 |
| **Total** | **~123** |
