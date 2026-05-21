# AI Content Ops Blog SEO/AEO/GEO Discovery

**Date:** 2026-05-20
**Scope:** Automated blog post generation in the Atlas AI Content Ops Station.

## Short Answer

Atlas can generate, persist, review, and publish blog drafts with SEO/AEO/GEO
readiness checks on the current AI Content Ops path.

This audit originally found that the extracted AI Content Ops blog path could
generate SEO fields but did not prove they survived into public/publishable
surfaces. That gap is now closed by the merged PR chain listed below.

The safest current claim is:

> Blog drafts include SEO metadata, FAQ answers, answer-engine-friendly
> structure, and GEO readiness checks. Published blog pages are covered by
> crawler-visible SEO/GEO verification.

Still avoid claiming guaranteed AI-engine placement or "fully optimized" SEO,
AEO, and GEO outcomes.

## Closeout Status

The follow-up chain that closed this audit:

- PR #665: persisted extracted blog SEO fields into first-class `blog_posts`
  columns and hydrated them back into `BlogPostDraft.metadata`.
- PR #669: added SEO/AEO readiness to generated blog draft export rows.
- PR #671: made missing SEO/AEO fields block generated blog saves.
- PR #672: defined the first-class GEO draft/publish contract.
- PR #674: added GEO readiness to generated blog draft export rows.
- PR #676: made missing GEO draft structure block generated blog saves.
- PR #678: added a targeted GEO repair loop.
- PR #682 and PR #683: added local and CI publish-level GEO verification.
- PR #688 and PR #690: surfaced readiness in the Atlas Intel review UI.
- PR #691 and PR #693: verified publish metadata, FAQ schema, and JSON-LD.
- PR #698: prerendered crawler-visible article bodies.

## What Exists

### SEO Generation

The blog prompts request SEO fields directly:

- `seo_title`
- `seo_description`
- `target_keyword`
- `secondary_keywords`
- `faq`

Relevant files:

- `extracted_content_pipeline/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/b2b_blog_post_generation.md`
- `atlas_brain/skills/digest/blog_post_generation.md`
- `atlas_brain/skills/digest/b2b_blog_post_generation.md`

The prompt rules include title length, meta description length, target keyword placement, secondary keywords, internal links, outbound authority links, FAQ answers, and featured-snippet-friendly structure.

### AEO Generation

The consumer blog prompt has an explicit `AEO (Answer Engine Optimization)` section. It tells the model to structure posts for ChatGPT, Perplexity, and Google AI Overviews by using:

- direct answers near the start of each section
- self-contained H2 sections
- question-format H2 headings
- concrete numbers and date anchoring
- clear entity names
- structured comparison tables

The B2B prompt does not use a separate AEO heading, but it asks for many of the same patterns:

- direct 40-60 word answer after question-like H2s
- self-contained sections
- full vendor names
- date anchoring
- FAQ answers backed by data

### Public Rendering

The public frontend can render several SEO/AEO artifacts when the fields are present:

- SEO title and description
- canonical blog URLs
- `BlogPosting` JSON-LD
- `BreadcrumbList` JSON-LD
- `FAQPage` JSON-LD when FAQ items exist
- keyword metadata from `target_keyword` and `secondary_keywords`

Relevant files:

- `atlas-churn-ui/vite.config.ts`
- `atlas-churn-ui/src/pages/BlogPost.tsx`
- `atlas-churn-ui/src/components/BlogArticleView.tsx`
- `atlas-intel-ui/src/components/SeoHead.tsx`

### Legacy B2B Blog Path

The legacy/autonomous B2B blog generation path writes SEO fields into first-class database columns:

- `seo_title`
- `seo_description`
- `target_keyword`
- `secondary_keywords`
- `faq`

Relevant file:

- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`

This path also has fallback logic for missing SEO title, SEO description, and target keyword.

## Main Gaps

### Gap 1: GEO Is Not a First-Class Contract

Status: closed for the current product contract.

The prompts mention AI answer engines and include AEO patterns that overlap with GEO. A first-class product definition now exists in:

- `docs/audits/ai_content_ops_blog_geo_contract_2026-05-20.md`

That contract defines GEO as Generative Engine Optimization: structuring a blog post so AI answer engines can understand the topic, extract a useful answer, identify the entities involved, and cite or summarize the page without needing hidden context.

The repo now exposes GEO readiness in generated-asset export/review output,
blocks generated drafts that miss the draft-level GEO contract, and verifies
publish-level crawler-visible output in the Atlas Intel UI build path.

Implemented:

- `geo_readiness` in generated-asset review/export output
- draft-level GEO checks in the blog quality gate
- publish-level checks for crawler-visible HTML, SEO metadata, JSON-LD, and
  article body content

### Gap 2: AI Content Ops Blog Persistence May Bury SEO Fields

Status: closed by PR #665.

The newer extracted AI Content Ops blog service builds a `BlogPostDraft` with SEO fields in `draft.metadata`.

Relevant file:

- `extracted_content_pipeline/blog_generation.py`

`PostgresBlogPostRepository.save_drafts()` now writes the generated SEO fields
into the first-class `blog_posts` columns while still preserving the metadata
bag under `data_context["_metadata"]`.

Relevant file:

- `extracted_content_pipeline/blog_post_postgres.py`

The public blog API reads first-class columns instead:

- `seo_title`
- `seo_description`
- `target_keyword`
- `secondary_keywords`
- `faq`
- `related_slugs`

Relevant file:

- `atlas_brain/api/blog_public.py`

Those columns exist in:

- `atlas_brain/storage/migrations/120_blog_seo.sql`

Risk status: closed for the extracted Content Ops blog-post repository.

### Gap 3: Quality Gate Does Not Validate SEO/AEO/GEO Metadata

Status: closed for the extracted Content Ops blog generator.

`extracted_quality_gate.blog_pack.evaluate_blog_post()` now validates long-form blog quality, word count, chart placeholders, internal links, quote use, unsupported data claims, SEO/AEO metadata, and GEO draft structure when the caller opts into those checks.

The extracted blog generator opts into checks for:

- `seo_title` presence and length
- `seo_description` presence and length
- target keyword presence
- secondary keyword presence
- FAQ count
- answer-first section format
- question-style H2 usage
- GEO/AEO citable-section requirements as a named score

Relevant files:

- `extracted_quality_gate/blog_pack.py`
- `extracted_content_pipeline/blog_generation.py`

Risk status: closed for save-time draft validation. Publish-time schema and
crawler-visible checks are handled by the Atlas Intel publish verifier.

### Gap 4: Two Blog Generation Paths Are Easy to Confuse

Status: mitigated for the current AI Content Ops path.

There are at least two relevant paths:

1. Legacy/autonomous blog generation, which writes SEO fields directly.
2. Extracted AI Content Ops `BlogPostGenerationService`, which stores SEO fields in draft metadata and relies on a repository that does not write the first-class SEO columns.

Risk status: the extracted AI Content Ops path now has its own persistence,
quality, export, review UI, and publish verification contracts. The legacy
autonomous path remains separate and should not be used as evidence for future
extracted-path claims unless a PR explicitly wires or tests both.

### Gap 5: Current Product Claim Needs Careful Wording

The repo now supports this claim:

> Generates data-backed blog drafts with SEO metadata, FAQ output,
> answer-engine-friendly structure, GEO readiness checks, and publish-surface
> verification.

The repo still should not claim:

> Automatically generates, validates, and publishes fully SEO/GEO/AEO optimized blog posts.

## Resolved Discovery Questions

1. The AI Content Ops Station path is `extracted_content_pipeline/blog_generation.py`
   with host-injected blueprint, LLM, skill, and repository ports.
2. Generated blog drafts persist through `PostgresBlogPostRepository`; public
   static/frontend publishing remains a separate route with its own verifier.
3. SEO/AEO and draft-level GEO blockers run before save on the extracted blog
   generator. Export/review rows also show readiness summaries for operators.
4. Review/export output shows the raw generated article plus SEO/AEO and GEO
   readiness summaries. The Atlas Intel UI renders both compact labels and a
   readiness breakdown.
5. The current proof is test and CI coverage for first-class SEO persistence,
   save-time quality gates, review/export readiness, review UI visibility,
   and publish-level crawler-visible verification.

## Recommended Next Slice

No active follow-up remains from this audit. The next Content Ops blog
SEO/GEO slice should require a new concrete trigger, such as:

1. A live generated blog draft fails a readiness check for a reason operators
   cannot understand or fix.
2. A public publish verifier misses a crawler-visible SEO/GEO regression.
3. The product changes the public claim beyond the safe wording below.

## Suggested Customer-Facing Language For Now

Use:

> Blog drafts include SEO metadata, FAQ answers, answer-engine-friendly article
> structure, and GEO readiness checks.

Avoid until validated:

> Fully optimized for SEO, GEO, and AEO.
