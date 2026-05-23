# AI Content Ops Landing Page SEO/AEO/GEO Contract

**Date:** 2026-05-21
**Scope:** AI Content Ops generated landing-page drafts and future published
landing-page surfaces.

## Short Definition

For Atlas landing pages, SEO/AEO/GEO readiness means the generated page is
structured so search engines, human buyers, and AI answer engines can understand:

- what offer the page makes
- who the page is for
- what problem the offer solves
- why the reader should trust the page
- what action the reader should take next

This is not the same contract as a blog post. Blog GEO emphasizes evidence-rich
article sections that can be cited or summarized. Landing-page GEO emphasizes
offer clarity, audience fit, answer-first positioning, objection handling, trust
signals, and a visible conversion path.

GEO is not a placement guarantee. It does not mean ChatGPT, Perplexity, Claude,
Copilot, or Google AI Overviews will show the page. It means the generated draft
and future published page meet the conditions Atlas can control.

## Current Implementation Baseline

Atlas now has draft-level and publish-surface readiness coverage for generated
landing pages:

- `extracted_content_pipeline/landing_page_generation.py` generates one
  `LandingPageDraft` from a `MarketingCampaign`, uses quality repair attempts,
  and carries operator-supplied landing-page SEO/GEO/AEO inputs through campaign
  context.
- `extracted_content_pipeline/skills/digest/landing_page_generation.md` asks for
  a structured JSON page with `title`, `slug`, `hero`, `sections`, `cta`,
  `meta`, `reference_ids`, and section metadata used for AEO/GEO checks.
- `extracted_quality_gate/landing_page_pack.py` blocks malformed title, slug,
  CTA, sections, unresolved placeholders, unsafe phrasing, and selected
  landing-page readiness failures before save.
- `extracted_content_pipeline/landing_page_readiness.py` computes
  `seo_aeo_readiness`, `geo_readiness`, and repair issue names.
- `extracted_content_pipeline/landing_page_export.py` exposes readiness fields,
  structured data, public renderer rows, robots policy, and public CTA index
  policy.
- `extracted_content_pipeline/api/generated_assets.py` exposes tenant-scoped
  edit and repair routes for landing-page drafts plus unauthenticated public
  routes for approved generated landing pages and their sitemap.
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` renders landing-page
  readiness panels, structured-data preview, edit controls, and repair controls.
- `atlas-intel-ui/src/pages/PublicLandingPage.tsx` renders approved public
  generated landing pages at `/lp/:id/:slug`.
- `atlas-intel-ui/vite.config.ts` imports approved public landing-page sitemap
  entries at build time and prerenders crawler-visible `/lp/.../index.html`
  files when `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL` is configured.
- `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.mjs` verifies built
  public landing-page HTML for canonical metadata, robots, JSON-LD, visible
  body copy, H1, and CTA.

The current proof is deterministic test and CI coverage, not live search or AI
answer-engine placement. Continue to describe this as readiness and verification,
not guaranteed ranking or guaranteed AI answer inclusion.

## Relationship To SEO, AEO, And GEO

### SEO

Landing-page SEO helps search engines understand the offer and decide how to
index the page.

For generated landing pages, SEO readiness should cover:

1. `title_tag`
   - `meta.title_tag` is present.
   - The title tag is distinct enough from the H1 when possible.
   - It names the offer, audience, category, or problem.
   - It avoids date stamps unless the campaign explicitly requires time-bound
     positioning.

2. `meta_description`
   - `meta.description` is present.
   - The description is long enough to explain the offer in search snippets.
   - It names the audience or problem and includes the primary action or outcome.

3. `slug_quality`
   - `slug` is lowercase, hyphenated, ASCII, and derived from the campaign or
     offer.
   - It avoids generic slugs such as `landing-page`, `demo`, or `campaign`.

4. `metadata_consistency`
   - Hero, title, slug, and metadata all describe the same offer.
   - The metadata does not introduce claims that are absent from the visible
     page.

### AEO

Landing-page AEO helps a reader or answer engine answer the buyer's immediate
questions without digging.

For generated landing pages, AEO readiness should cover:

1. `answer_first_hero`
   - The hero headline or subheadline makes the offer clear in plain language.
   - A reader can answer "what is this?" and "who is this for?" from the first
     viewport.

2. `problem_solution_clarity`
   - The page includes a clear problem statement and a clear solution statement.
   - The solution statement connects directly to the campaign `value_prop`.

3. `audience_specificity`
   - The copy names or strongly implies the intended persona, segment, use case,
     or buyer role.
   - It avoids generic audience language such as "businesses", "teams", or
     "companies" unless that is the real campaign audience.

4. `objection_coverage`
   - The page answers likely buyer objections or questions.
   - This can be a FAQ section, comparison section, how-it-works section, proof
     section, or risk-reversal section.

### GEO

Landing-page GEO builds on SEO and AEO. A page is GEO-ready when an AI answer
engine can summarize the offer and buyer fit without hidden context.

For generated landing pages, GEO readiness should cover:

1. `offer_entity_clarity`
   - The page identifies the offer, product, service, or campaign clearly.
   - The offer is not only implied by CTA text.

2. `audience_entity_clarity`
   - The page identifies the audience, persona, market, or customer type.
   - The audience appears in visible copy, not only metadata.

3. `answer_extractability`
   - The hero or first content section gives a direct answer to:
     "What does this page offer, and why should the target reader care?"
   - The answer stands alone without requiring the full page.

4. `section_semantics`
   - Sections have specific, meaningful headings.
   - Headings avoid vague labels like "Overview", "Features", "Benefits", or
     "Conclusion" unless the section body and surrounding copy make the offer
     and audience clear.

5. `trust_signal_visibility`
   - The page includes at least one visible trust cue when the campaign provides
     evidence.
   - Trust cues can include reference IDs, case studies, customer logos,
     testimonial excerpts, source-row evidence, time windows, or concrete
     outcomes.
   - If no proof exists, the page should avoid pretending proof exists.

6. `conversion_path_clarity`
   - The page has a visible primary CTA.
   - Hero CTA and page-level CTA do not conflict.
   - CTA language matches the offer and does not create a different promise.

7. `claim_safety`
   - The page does not include unsupported numeric claims, fake social proof,
     placeholder URLs, placeholder customer names, or vague superlatives that
     imply proof.
   - Comparative claims require campaign-provided competitor or market context.

## Readiness Levels

### Draft Ready

A landing-page draft is readiness checked when the generated JSON payload itself
passes deterministic checks before review or persistence.

This level belongs in:

- `extracted_quality_gate/landing_page_pack.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `extracted_content_pipeline/landing_page_export.py`
- generated-asset API rows
- generated-asset review UI

Draft-ready output shape should mirror the blog pattern:

```json
{
  "seo_aeo_readiness": {
    "status": "ready",
    "passed": 8,
    "total": 8,
    "missing": [],
    "checks": {
      "title_tag": true,
      "meta_description": true,
      "slug_quality": true,
      "metadata_consistency": true,
      "answer_first_hero": true,
      "problem_solution_clarity": true,
      "audience_specificity": true,
      "objection_coverage": true
    }
  },
  "geo_readiness": {
    "status": "ready",
    "passed": 7,
    "total": 7,
    "missing": [],
    "checks": {
      "offer_entity_clarity": true,
      "audience_entity_clarity": true,
      "answer_extractability": true,
      "section_semantics": true,
      "trust_signal_visibility": true,
      "conversion_path_clarity": true,
      "claim_safety": true
    }
  }
}
```

### Publish Ready

A landing page is publish ready when the public rendered URL exposes the draft in
a way search engines and AI crawlers can inspect.

This level does not belong in the generator alone. It belongs in whichever host
or frontend renders approved landing-page drafts publicly.

Minimum publish checks:

1. `crawler_visible_html`
   - The hero, core sections, and CTA are visible in returned HTML or through the
     approved crawler-rendering strategy.

2. `canonical_url`
   - The page has a canonical URL.

3. `seo_social_metadata`
   - The page outputs title, meta description, Open Graph title/description, and
     usable social image metadata.

4. `structured_data`
   - The page emits appropriate structured data for the rendered page type.
   - Likely candidates: `WebPage`, `Organization`, `Product`, `Service`,
     `Offer`, `FAQPage`, or `BreadcrumbList`, depending on the page.

5. `visible_conversion_path`
   - The CTA is visible and points to a real URL.
   - Placeholder URLs such as `#`, `/demo` without a host-defined route, or fake
     external links should fail the publish check.

6. `faq_schema_when_present`
   - If the page includes FAQ/objection content, the public page emits matching
     FAQPage JSON-LD.

7. `indexable_response`
   - The page returns a successful indexable response and is not blocked by
     robots metadata for the target deployment.

## Implementation Status

The original roadmap has been implemented through draft and publish readiness:

1. Contract definition
   - Landed in this document and the associated plan doc.

2. Draft readiness helper
   - `landing_page_readiness.py` computes landing-page `seo_aeo_readiness` and
     `geo_readiness`.
   - `landing_page_export.py` includes readiness fields in generated-asset rows
     and CSV output.

3. Quality gate enforcement
   - `landing_page_pack.py` blocks the save path on malformed structure,
     placeholder CTAs, unresolved placeholders, unsafe claims, and selected
     readiness failures.
   - Operator-visible readiness checks remain more detailed than blockers.

4. Prompt alignment
   - `digest/landing_page_generation.md` asks for the structures the validators
     expect, including answer-first hero copy, section metadata, FAQ/objection
     handling, trust cues, and CTA consistency.

5. Review UI visibility
   - `ContentOpsAssetsReview.tsx` shows landing-page readiness breakdowns,
     structured data, edit controls, repair controls, and repair history.

6. Publish verification
   - `PublicLandingPage.tsx` renders approved public generated pages.
   - The public sitemap includes only approved, readiness-passing, indexable
     generated landing pages.
   - The frontend build prerenders public landing pages from the backend sitemap
     when configured.
   - `verify-landing-page-geo-prerender.mjs` checks the built public HTML for
     crawler-visible metadata, JSON-LD, body copy, H1, and CTA.

## Safe Customer-Facing Language

> Landing pages are generated and published with SEO, AEO, and GEO readiness
> checks for metadata, answer clarity, structured data, and crawler-visible page
> content.

Use this narrower draft-only version when discussing generated drafts before
approval or frontend publication:

> Landing-page drafts are checked for SEO metadata, answer clarity, audience
> fit, CTA consistency, and GEO readiness before review.

Avoid:

> Fully optimized landing pages for SEO, AEO, and GEO.

Avoid:

> Guaranteed to appear in AI answer engines.

## Decisions Now Reflected In Code

1. Structured data defaults to `WebPage`, with `FAQPage` added when
   question/answer section metadata is present.
2. FAQ/objection handling is part of draft readiness, but the structured-data
   builder does not invent FAQ schema when question metadata is absent.
3. Obvious placeholder CTA URLs such as `#` block the draft quality gate.
   Publish/index policy also keeps `/demo` and other placeholder destinations
   `noindex,follow`.
4. `geo_readiness` accepts either `reference_ids` or visible evidence language
   as the trust-signal source. If no proof exists, generated pages should avoid
   pretending proof exists.
5. The review UI reuses the generated-asset readiness panel pattern with
   landing-page-specific checks and labels.
