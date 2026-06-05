# PR: Marketer Reviews-As-Input (content-ops source provider)

> Scoping plan for a future vertical slice. This PR lands the plan only.
> A code trace (2026-06-01) showed the generation engine already consumes
> review/competitive data, but the operator-facing content-ops input is
> hardcoded to support tickets. This plan scopes the smallest unlock for the
> marketer buyer and marks the already-built pieces as do-not-modify.

## Why this slice exists

The deflection-report buyer is a support-team buyer. The marketer buyer has a
different, often richer feed: customer reviews (voice-of-customer), which are
public, persuasive, and comparative. A trace of `origin/main` shows the
generation side already handles this:

- `extracted_content_pipeline/campaign_source_adapters.py` already recognizes
  `reviews` / `complaints` source types and converts review-shaped rows to
  campaign opportunities (`source_rows_to_campaign_opportunities`).
- The B2B autonomous generators already produce blog/campaign content grounded
  in review + competitive/displacement signal (shipped, but scheduled/batch).

The gap is the **operator-facing input**: the host content-ops provider
(`atlas_brain/_content_ops_input_provider.py`) is hardcoded to support tickets
(`_AtlasSupportTicketInputProvider`, gated on `_is_support_ticket_material`),
and there is no review input provider or source-type selection. So a marketer
cannot feed reviews through the productized New Run path the way a support buyer
feeds tickets. The seam itself is provider-agnostic
(`extracted_content_pipeline/content_ops_input_provider.py` defines the
`ContentOpsInputProvider` Protocol), so this is an input slice on a built
engine, not a new pipeline.

Validation precondition: confirm a marketer buyer wants review-grounded content
before implementing. The generation is built; the risk is selling an unwanted
offer, not engineering.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Workflow/process

This PR lands the scoping plan only. The implementing slice (separate PR) will:

1. Add a review/market-signal content-ops input provider that packages
   review-shaped source rows through the existing review ingestion, behind the
   existing `ContentOpsInputProvider` Protocol.
2. Add explicit source-type selection at the host so a request routes to the
   review provider or the support-ticket provider (instead of the current
   support-ticket-only path).
3. Surface a "reviews" source option in the New Run UI.
4. Tests + same-PR CI enrollment (the AGENTS section 3e rule).

### Explicitly NOT in scope (already built -- do not modify)

`campaign_source_adapters.py` review ingestion, the blog/landing/faq generators,
public render + SEO/sitemap. Re-implementing them is the failure mode this plan
prevents.

### Files touched

- `plans/PR-Marketer-Reviews-As-Input.md`

## Mechanism

The implementing slice adds a sibling provider behind the
`ContentOpsInputProvider` Protocol that recognizes review-shaped source material
and packages it via the existing review-to-opportunity ingestion, targeting the
same outputs (`blog_post`, `landing_page`, `faq_markdown`). Host selection keys
on an explicit request source-type rather than sniffing material shape.

## Intentional

- **Mirror, do not fork-weaker.** Build the review provider by extending a
  shared base with `_AtlasSupportTicketInputProvider` (or reusing its
  tenant-scope + fail-closed + ambiguity guards), not a hand-rolled copy. New
  sibling providers have drifted weaker before; share the resilient core.
- Tenant scoping + verified-content discipline are inherited from the existing
  providers; do not add a second scoping path.
- Reviews stay grounded: content is generated from real review language, the
  differentiator vs commodity LLM copy.

## Deferred

- Competitive/displacement as a content-ops input -- bigger; port the
  autonomous B2B generation logic (`b2b_blog_post_generation.py`) behind a
  content-ops provider in a later slice.
- New output skills (social copy, ad copy, stat/quote cards) -- orthogonal
  output work, separate slices.
- Pricing/packaging of the marketer offer.

## Parked hardening

None.

## Verification

- A content-ops run with review-shaped source produces a `blog_post` and a
  `landing_page` draft grounded in the review text -- demonstrated end to end
  (prove the review-source to blog/landing combination executes, like the
  upload-source route test, not assumed from components).
- Tenant isolation: a run cannot read another account's review rows.
- Fail-closed: empty or non-review source produces no run / a clear warning.
- Frontend + provider tests enrolled in their workflows in the same PR.

## Estimated diff size

| Area | LOC |
|---|---:|
| This PR (plan doc) | ~110 |
| **Total** | ~110 |

This PR is plan-only (~110 LOC). The implementing slice is estimated at
~250-400 LOC (review provider + host source-type selection + UI option +
tests), reusing the existing ingestion and generators; split provider and UI
into two PRs if it runs over budget.
