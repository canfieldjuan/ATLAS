# AI Content Ops Blog SEO/AEO/GEO Discovery

**Date:** 2026-05-20
**Scope:** Automated blog post generation in the Atlas AI Content Ops Station.

## Short Answer

Atlas partially generates SEO and AEO-ready blog drafts today.

It now has a first-pass GEO contract in
`docs/audits/ai_content_ops_blog_geo_contract_2026-05-20.md`, but the contract
is not fully implemented as a validator yet. At the time of the original audit,
at least one AI Content Ops persistence path did not appear to write the
generated SEO fields into the public `blog_posts` columns that the public API
and frontend read.

The safest current claim is:

> The blog generator can produce SEO and answer-engine-ready drafts when the run uses the SEO/AEO prompt and the publishing path preserves the generated metadata.

Avoid claiming fully automated SEO, AEO, and GEO optimization until the gaps below are closed.

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

Status: definition added, implementation still pending.

The prompts mention AI answer engines and include AEO patterns that overlap with GEO. A first-class product definition now exists in:

- `docs/audits/ai_content_ops_blog_geo_contract_2026-05-20.md`

That contract defines GEO as Generative Engine Optimization: structuring a blog post so AI answer engines can understand the topic, extract a useful answer, identify the entities involved, and cite or summarize the page without needing hidden context.

The system can produce GEO-friendly content, but the repo does not yet fully
prove or enforce "GEO-ready" as a distinct output.

Needed:

- add `geo_readiness` to generated-asset review/export output
- add draft-level GEO checks to the blog quality gate
- add publish-level checks for crawler-visible HTML and structured data

### Gap 2: AI Content Ops Blog Persistence May Bury SEO Fields

The newer extracted AI Content Ops blog service builds a `BlogPostDraft` with SEO fields in `draft.metadata`.

Relevant file:

- `extracted_content_pipeline/blog_generation.py`

But `PostgresBlogPostRepository.save_drafts()` writes only core draft fields and stores metadata under `data_context["_metadata"]`.

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

Risk:

Generated SEO/AEO metadata from the extracted AI Content Ops Station path may be saved in `data_context["_metadata"]` but not exposed to the public API or frontend as SEO fields.

### Gap 3: Quality Gate Does Not Validate SEO/AEO/GEO Metadata

`extracted_quality_gate.blog_pack.evaluate_blog_post()` validates long-form blog quality, word count, chart placeholders, internal links, quote use, unsupported data claims, and related content quality issues.

It does not appear to validate:

- `seo_title` length or keyword placement
- `seo_description` length or keyword inclusion
- target keyword presence
- FAQ count or answer quality
- answer-first section format
- question-style H2 usage
- BlogPosting/FAQ schema readiness
- GEO/AEO citable-section requirements as a named score

Relevant files:

- `extracted_quality_gate/blog_pack.py`
- `extracted_content_pipeline/blog_generation.py`

Risk:

The prompt asks for SEO/AEO, but the quality gate does not independently enforce it before a draft is saved.

### Gap 4: Two Blog Generation Paths Are Easy to Confuse

There are at least two relevant paths:

1. Legacy/autonomous blog generation, which writes SEO fields directly.
2. Extracted AI Content Ops `BlogPostGenerationService`, which stores SEO fields in draft metadata and relies on a repository that does not write the first-class SEO columns.

Risk:

We may be looking at a working SEO implementation in one path while the actual AI Content Ops Station path used by the product has a weaker contract.

### Gap 5: Current Product Claim Needs Careful Wording

The repo supports this claim:

> Generates data-backed blog drafts with SEO fields, FAQ output, and AEO-style structure.

The repo does not yet fully support this stronger claim:

> Automatically generates, validates, and publishes fully SEO/GEO/AEO optimized blog posts.

## Discovery Questions

1. Which blog path powers the AI Content Ops Station experience we want to sell: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` or `extracted_content_pipeline/blog_generation.py`?
2. When a customer generates a blog post from the Station, does the draft get published through the DB/API path or exported to static frontend `.ts` content?
3. Which GEO checks should block draft save versus only show in review output?
4. What should the report show the customer: raw generated article, SEO fields, AEO checklist, GEO checklist, or all of those?
5. What proof do we need before saying the output is SEO/GEO/AEO-ready?

## Recommended Next Slice

Before changing copy or selling this as SEO/GEO/AEO generation, tighten the product contract:

1. Add first-class SEO fields to the extracted AI Content Ops `BlogPostDraft` persistence path, or map `metadata` into the existing public SEO columns.
2. Add a small GEO readiness summary to blog draft export/review output.
3. Add tests proving that a generated blog draft persists `seo_title`, `seo_description`, `target_keyword`, `secondary_keywords`, and `faq` into the fields read by the public API.
4. Add publish-level GEO checks before claiming fully SEO/GEO/AEO-ready pages.

## Suggested Customer-Facing Language For Now

Use:

> Blog drafts include SEO metadata, FAQ answers, and answer-engine-friendly article structure.

Avoid until validated:

> Fully optimized for SEO, GEO, and AEO.
