# AI Content Ops Blog GEO Contract

**Date:** 2026-05-20
**Scope:** AI Content Ops blog drafts and published blog pages.

## Short Definition

For Atlas, GEO means **Generative Engine Optimization**: structuring a blog post
so AI answer engines can understand the topic, extract a useful answer, identify
the entities involved, and cite or summarize the page without needing hidden
context.

GEO is not a ranking guarantee. It does not mean ChatGPT, Perplexity, Claude,
Copilot, or Google AI Overviews will show the page. It means the generated draft
and the published page meet the conditions we can control.

## Product Claim

Safe claim after draft-level implementation:

> Generates GEO-ready blog drafts with clear entities, answer-first sections,
> concrete evidence, FAQ coverage, and safe citation structure.

Safe claim after draft-level and publish-level implementation:

> Publishes GEO-ready blog pages with clear entities, answer-first sections,
> concrete evidence, FAQ schema, BlogPosting schema, canonical URLs, and
> crawler-visible article content.

Avoid:

> Guarantees placement in ChatGPT, Perplexity, Claude, Copilot, or Google AI
> Overviews.

Avoid until both draft and publish checks exist:

> Fully optimized for SEO, AEO, and GEO.

## Relationship To SEO And AEO

SEO helps search engines index and understand the page. The current Atlas SEO
contract covers metadata, keywords, links, canonical URLs, schema, and public
rendering.

AEO helps a page answer specific questions cleanly. The current Atlas AEO
contract covers question-style sections, direct answers, FAQ output, and
answer-first structure.

GEO builds on both, but adds a stricter requirement: an AI answer engine should
be able to lift one section from the page and still know:

- what question is being answered
- which entity or product the answer is about
- what evidence supports the answer
- how current the evidence is
- whether the answer is safe to cite without the rest of the article

## Readiness Levels

### GEO Draft Ready

A blog draft is GEO draft ready when the generated content itself is fit for AI
answer extraction.

This is the level that belongs in the extracted content pipeline quality gate
and generated-asset review/export rows.

Minimum checks:

1. `entity_clarity`
   - The title or opening section names the primary entity, product, vendor, or
     category.
   - H2 sections avoid vague headings like "Overview", "Conclusion",
     "Key Takeaways", or "Final Thoughts" unless the heading also names the
     entity or question.

2. `answer_first_sections`
   - At least one H2 is written as a question, or at least one H2 section opens
     with a direct 40-80 word answer.
   - The opening paragraph after a qualifying H2 can stand alone without
     requiring the previous section.

3. `citable_section_structure`
   - At least two H2 sections are self-contained.
   - A self-contained section names the topic/entity and gives a complete answer
     inside the first paragraph.

4. `evidence_specificity`
   - The article includes concrete evidence such as review counts, percentages,
     time windows, source names, customer wording, or quoted source material.
   - Evidence must be visible in the article body, not only hidden in metadata.

5. `freshness_context`
   - The article states a time window, publication date, review period, or other
     recency cue when making data-backed claims.

6. `faq_coverage`
   - The draft includes at least three FAQ entries.
   - FAQ questions use natural customer/search language rather than internal
     shorthand.

7. `citation_safety`
   - The quality gate does not find unsupported data claims, fake internal links,
     unresolved placeholders, or placeholder `href="#"` links.
   - If the article uses source quotes, they appear in the body and are not
     only referenced in metadata.

Draft-ready output shape:

```json
{
  "geo_readiness": {
    "status": "ready",
    "passed": 7,
    "total": 7,
    "missing": [],
    "checks": {
      "entity_clarity": true,
      "answer_first_sections": true,
      "citable_section_structure": true,
      "evidence_specificity": true,
      "freshness_context": true,
      "faq_coverage": true,
      "citation_safety": true
    }
  }
}
```

### GEO Publish Ready

A blog page is GEO publish ready when the public URL exposes the draft in a way
AI crawlers and search engines can inspect.

This level belongs in frontend/public-route verification, not just in the
generation pipeline.

Minimum checks:

1. `crawler_visible_html`
   - The article body is present in the returned HTML or reliably rendered for
     crawlers through the current frontend strategy.

2. `canonical_url`
   - The page has a canonical URL.

3. `blogposting_schema`
   - The page outputs `BlogPosting` JSON-LD with headline, description,
     published date, modified date when available, author or organization, and
     URL.

4. `faq_schema`
   - When FAQ entries exist, the page outputs `FAQPage` JSON-LD.

5. `breadcrumb_schema`
   - The page outputs `BreadcrumbList` JSON-LD or equivalent breadcrumb
     structure.

6. `open_graph_image`
   - The page has a usable `og:image`, either per-post or a good default.

7. `indexable_response`
   - The page returns a successful indexable response and is not blocked by
     robots metadata for the target deployment.

## Implementation Guidance

Do not make `geo_readiness` a single opaque score. Use named checks so operators
can see what failed and fix the draft.

Implementation status as of this audit refresh:

1. `geo_readiness` is exposed in blog generated-asset rows and CSV export.
2. Content-level SEO/AEO/GEO checks are wired into the extracted blog quality
   gate and the extracted blog generation service.
3. Public-route verification covers canonical URLs, SEO/social metadata,
   BlogPosting JSON-LD, BreadcrumbList JSON-LD, sitemap inclusion, source-date
   `lastmod`, indexability, and crawler-visible article bodies.

Remaining or active implementation work:

1. Add or preserve static chart evidence fallbacks for no-JavaScript crawlers.
2. Add FAQPage schema and static FAQ-body publish verification when source posts
   include FAQ entries.
3. Decide whether frontend prerendering and the verifier should share a
   generated source metadata manifest.
4. Only then update customer-facing copy from "checked for readiness" to
   broader fully SEO/AEO/GEO language.

## Customer-Facing Language

Before checks:

> Blog drafts include SEO metadata, FAQ answers, and answer-engine-friendly
> article structure.

After draft-level checks:

> Blog drafts are checked for GEO readiness: clear entities, answer-first
> sections, concrete evidence, FAQ coverage, and safe citation structure.

After draft and current publish checks:

> Blog posts are generated and published with SEO, AEO, and GEO readiness checks
> for metadata, answer extraction, structured data, and crawler-visible article
> pages.

Avoid until chart and FAQ publish fallbacks are complete:

> Fully optimized for SEO, AEO, and GEO.
