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

The landing-page generator already has the foundation needed for readiness work:

- `extracted_content_pipeline/landing_page_generation.py` generates one
  `LandingPageDraft` from a `MarketingCampaign`.
- `extracted_content_pipeline/skills/digest/landing_page_generation.md` asks for
  a structured JSON page with `title`, `slug`, `hero`, `sections`, `cta`,
  `meta`, and `reference_ids`.
- `extracted_quality_gate/landing_page_pack.py` validates basic title, slug,
  hero, CTA, sections, blocked phrasing, and optional meta-description length.
- `extracted_content_pipeline/landing_page_export.py` exports generated drafts
  for review.
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` already renders landing
  page previews and basic facts.

The missing piece is a landing-page-specific readiness contract. Today landing
pages do not expose `seo_aeo_readiness` or `geo_readiness`, and the quality gate
does not yet validate answer clarity, audience clarity, offer clarity, trust
signals, FAQ/objection handling, or conversion-path continuity.

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

## Implementation Roadmap

Recommended sequence:

1. Contract definition
   - Land this document and a plan doc.
   - No runtime code changes.

2. Draft readiness helper
   - Add landing-page `seo_aeo_readiness` and `geo_readiness` helpers.
   - Wire them into `landing_page_export.py` rows and CSV output.
   - Add focused tests for ready and incomplete pages.

3. Quality gate enforcement
   - Extend `landing_page_pack.py` with the checks that should block save.
   - Keep operator-visible readiness checks more detailed than blockers.

4. Prompt alignment
   - Update `digest/landing_page_generation.md` so the LLM is asked for the
     structures the validators expect.
   - Add FAQ/objection guidance without forcing fake FAQ content when the
     campaign lacks enough context.

5. Review UI visibility
   - Show landing-page readiness labels and breakdown panels in
     `ContentOpsAssetsReview.tsx`.

6. Publish verification
   - Only after a concrete public rendering path exists for generated landing
     pages.
   - Add crawler-visible page checks at that boundary, not inside the generator.

## Safe Customer-Facing Language

Before implementation:

> Landing pages include campaign metadata, hero copy, body sections, and CTA
> structure for review.

After draft-level checks:

> Landing pages are checked for SEO metadata, answer clarity, audience fit, and
> GEO readiness before review.

After draft and publish checks:

> Landing pages are generated and published with SEO, AEO, and GEO readiness
> checks for metadata, answer clarity, structured data, and crawler-visible page
> content.

Avoid:

> Fully optimized landing pages for SEO, AEO, and GEO.

Avoid:

> Guaranteed to appear in AI answer engines.

## Open Decisions

1. Which structured data type should the first public landing-page renderer use
   by default: `WebPage`, `Service`, `Product`, or `Offer`?
2. Should FAQ/objection handling be required for all landing pages, or only when
   the campaign provides enough context?
3. Should placeholder CTA URLs block draft save, or only publish approval?
4. Should landing-page `geo_readiness` require evidence/trust signals when
   `reference_ids` is empty?
5. Should landing pages inherit the blog UI readiness panel component or use a
   landing-page-specific panel label and checks?
