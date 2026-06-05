# PR: Deflection Report -> Content Upsell (offer surface)

> Scoping plan for a future vertical slice. This PR lands the plan only; the
> implementation is a follow-up. The point of writing it now is that a code
> trace showed the verified-FAQ -> content pipeline is already built and
> quality-gated, so the slice is a thin offer surface, not a pipeline build --
> and this plan exists to keep the implementer from re-building what is done.

## Why this slice exists

The deflection-report buyer has already uploaded support tickets and received
verified FAQ answers (`answer_evidence_status == resolution_evidence`). The
pipeline that turns those verified answers into blog posts and landing pages is
already built and quality-gated end to end (traced 2026-06-01):

- `extracted_content_pipeline/campaign_source_adapters.py` auto-detects a
  FAQ-output bundle passed as a run's `source_material` and converts it.
- `extracted_content_pipeline/faq_output_ingestion.py` carries an answer into
  the content seed only for `resolution_evidence` items; unverified drafts
  contribute no answer (the verified-only gate is already enforced here).
- `atlas_brain/_content_ops_input_provider.py` already loads persisted FAQ
  drafts by tenant scope via the `source_faq_draft_ids` input contract (#1231),
  and the support-ticket package's default outputs are `faq_markdown`,
  `landing_page`, and `blog_post`.
- Generated blog/landing pages already render publicly with SEO/GEO prerender
  and sitemap (#1224, #1227) behind the escape-first sanitizer.

What is missing is only the offer surface: there is no flow that takes a
finished deflection report and runs blog/landing generation from its verified
FAQ drafts. This slice adds that handoff and nothing else.

Validation precondition: confirm a deflection buyer wants content built from
their ticket data (one or two prospect conversations) before implementing.
The build is done, so the risk is selling an offer no one asked for, not
engineering.

## Scope (this PR)

Ownership lane: content-ops/deflection-to-content-upsell
Slice phase: Workflow/process

This PR lands the scoping plan only. The implementing slice (separate PR) will:

1. Add a "Generate content from this report" action on the deflection-report
   result surface, with an output choice (`blog_post` / `landing_page`).
2. Trigger a content-ops run seeded with the report's verified FAQ drafts via
   the existing `source_faq_draft_ids` input contract (not inline rows) and the
   chosen outputs.
3. Hand off to the generated-asset review queue (reuse the #1226 review-link
   pattern) so the operator approves before anything goes public.
4. Ship the frontend test enrolled in the workflow in the same PR (the §3e
   CI-enrollment rule).

### Explicitly NOT in scope (already built -- do not modify)

`faq_output_ingestion.py`, `campaign_source_adapters.py`, the generators, the
verified-only gate, the tenant-scoped FAQ-draft load, and public rendering are
done and tested. Re-implementing any of them is the failure mode this plan
exists to prevent.

### Files touched

- `plans/PR-Deflection-To-Content-Upsell.md`

## Mechanism

The deflection report persists FAQ drafts with stable IDs. The implementing
slice collects the report's verified draft IDs and posts a content-ops run:

```json
{
  "outputs": ["blog_post"],
  "inputs": { "source_faq_draft_ids": ["<verified-draft-id>", "..."] }
}
```

The existing input provider loads those drafts tenant-scoped, the support-ticket
package converts them (verified resolutions only), and the generator produces a
draft into the review queue. No new backend route or generation logic.

## Intentional

- Verified-only is inherited, not re-added -- the resolution gate already keys
  on `resolution_evidence`. The UI should only offer the action when the report
  has at least one verified draft, and fail closed otherwise.
- Tenant scoping is inherited -- `source_faq_draft_ids` loads are already
  account-scoped (#1231); do not add a second scoping path.
- Approve-before-public -- generated content lands as a review-queue draft;
  public rendering stays approved-only (#1224). No auto-publish.
- Pass draft IDs, not the full FAQ-output bundle -- lighter request that reuses
  the persisted-load path.

## Deferred

- Recurring/scheduled content regeneration as tickets evolve (the recurring
  upsell cadence) -- separate slice, after the one-shot offer is validated.
- Pricing/billing of the upsell and any gating behind a paid tier.
- Multi-output batch (blog and landing in one click).
- The live end-to-end demo against a real report (an operator/validation step,
  not code).

## Parked hardening

None.

## Verification

- A content-ops run built from a report's verified FAQ draft IDs with
  `outputs: ["blog_post"]` produces a blog draft in the review queue --
  demonstrated end to end, proving the FAQ-drafts to blog_post combination
  actually executes rather than assuming it from the components.
- Repeat with `landing_page`.
- Fail-closed: a report with zero `resolution_evidence` drafts does not offer
  the action and produces no run.
- Tenant isolation: the run cannot seed from another account's FAQ drafts
  (inherited from #1231; assert it still holds through the handoff).
- Frontend test enrolled in the Atlas Intel UI workflow in the same PR.

## Estimated diff size

| Area | LOC |
|---|---:|
| This PR (plan doc) | ~130 |
| **Total** | ~130 |

This PR is plan-only (~130 LOC). The implementing slice is estimated at
~150-250 LOC (report-result CTA + run-trigger handoff + tests), with no
generator or pipeline changes; split CTA and trigger into two PRs if the
report-result surface needs new run-trigger plumbing.
