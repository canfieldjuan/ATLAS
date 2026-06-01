# AI Content Ops Landing Page User Workflow Gap Log - 2026-05-22

> Archived context: this note was preserved from a local worktree cleanup.
> Treat it as a historical gap log, not current implementation truth. Current
> capability truth lives in the codebase, merged plan docs, and PR history.

## Why this exists

This log captures the product workflow questions raised after reviewing the
blog SEO/AEO/GEO contract:

- How do users provide their own SEO, AEO, and GEO inputs?
- How do users review generated drafts?
- How do users edit or repair drafts before publishing?
- What gaps remain before landing pages can be sold as a clean product flow?

The landing-page backend is ahead of the public/product workflow. Several
implementation slices already exist or are partially implemented, but the
operator experience still needs a clear end-to-end path.

## Current Landing Page Capability

Landing page generation currently works from a `MarketingCampaign` input with:

- `campaign_name`
- `offer`
- `audience`
- `vendors`
- `categories`
- `tags`
- selected context fields such as `industry`, `pain_points`,
  `differentiators`, `customer_segments`, `key_metrics`, `proof_points`, and
  `competitive_alternatives`

The generated draft shape includes:

- page title
- slug
- hero block
- ordered sections
- CTA
- metadata (`title_tag`, `description`, `og_image_url`, etc.)
- reference IDs

Landing-page review/export rows already surface:

- `seo_aeo_readiness`
- `geo_readiness`
- generation token usage
- parse attempts
- reasoning usage summary

The quality gate also blocks or warns on several readiness-adjacent defects,
including invalid slugs, placeholder CTA URLs, unresolved placeholders, generic
section titles, missing/long metadata title tags, and inconsistent metadata.

## User Input Gap

Users do not yet have a clean landing-page form for SEO/AEO/GEO strategy.
Today, most advanced inputs must be provided through raw JSON or embedded in
the campaign/context payload.

Needed first-class fields:

- target keyword
- secondary keywords
- search intent
- primary offer/entity
- audience/entity
- buyer pain or problem statement
- objections or FAQ questions
- proof points and allowed reference IDs
- source period or freshness context
- internal links
- CTA label and URL
- forbidden claims or phrasing

These should map into the existing `MarketingCampaign.context` contract rather
than inventing a parallel request shape.

## Review Gap

Generated Asset Review can list landing-page drafts, show previews, expose
facts, open details, approve, reject, and export CSV.

The important gap is not visibility. The important gap is editability.

Users need to be able to review:

- hero headline/subheadline
- CTA label and URL
- page sections
- SEO title tag
- meta description
- proof/reference IDs
- SEO/AEO readiness
- GEO readiness

Then they need to change those fields in the app.

## Editing And Recheck Gap

The current review API is status-oriented. It can approve or reject drafts, but
it does not provide a product editing flow for generated landing pages.

Needed backend shape:

- `PATCH /content-assets/landing_page/drafts/{id}`
- tenant-scoped update
- structured patch for title, slug, hero, sections, CTA, meta, and reference IDs
- re-run landing-page readiness checks after update
- return the same review/export row shape the UI already knows how to render

Needed UI shape:

- edit mode in the asset detail drawer
- structured fields for hero, CTA, and meta
- section editor for ordered body sections
- reference/proof ID editor
- save and recheck action
- clear display of missing checks after save

## Repair Gap

Landing-page quality repair exists as a generation-time mechanism and a UI
control for repair attempts, but users still need an explicit product action:

- Fix missing readiness checks
- Regenerate with the same strategy inputs
- Regenerate after changing SEO/AEO/GEO fields

The first product version can repair the full landing page. Section-level repair
can be deferred until draft editing is stable.

## Product Boundary

Safe claim after the current/backend-complete layer:

> Landing page drafts include SEO metadata, answer-first page structure,
> objection/proof coverage, CTA consistency, and SEO/AEO/GEO readiness checks
> for review before publishing.

Safe claim after edit/recheck ships:

> Users can generate, review, edit, and recheck landing page drafts against
> SEO/AEO/GEO readiness before publishing.

Still avoid:

> Guaranteed rankings, guaranteed AI citations, or guaranteed conversion lift.

## Recommended Next Product Slice

The next slice should not be another prompt-only change. The highest-value gap
is the user-facing workflow:

1. Add a structured landing-page SEO/AEO/GEO input panel on New Run.
2. Persist those fields through `MarketingCampaign.context`.
3. Show landing-page readiness panels in review if not already present in the
   active branch.
4. Add landing-page draft edit and recheck.

The first implementation PR should be the input panel if the goal is better
generation quality. It should be edit/recheck if the goal is making the product
usable by a non-technical reviewer.
