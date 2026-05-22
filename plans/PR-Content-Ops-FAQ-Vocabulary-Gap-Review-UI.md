# PR-Content-Ops-FAQ-Vocabulary-Gap-Review-UI

## Why this slice exists

The FAQ generator now detects vocabulary gaps end to end in CLI/library output,
and the compact run summary exposes the signal for large runs. Reviewers in the
hosted Content Ops asset queue can inspect FAQ items, action steps, and sources,
but vocabulary-gap mappings are still only visible in the raw JSON dump.

This slice surfaces the existing per-item term mappings in the FAQ review drawer
so an operator can see customer phrasing versus documentation phrasing before
approving the generated FAQ.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-review-ui

1. Parse existing `items[].term_mappings` on FAQ Markdown drafts.
2. Show a compact vocabulary-gap count in FAQ preview/facts.
3. Render per-item vocabulary-gap mappings in the FAQ item drawer section.
4. Avoid backend/schema changes; use the row shape already exported by the FAQ
   draft API.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-Review-UI.md` | Plan doc for this review UI slice. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Displays FAQ vocabulary-gap mappings from existing item data. |

## Mechanism

`faqItemList(...)` already normalizes FAQ item records for the drawer. This
slice extends that normalized item shape with `termMappings`, sourced from
`term_mappings` on each item. Each mapping keeps only the review-visible fields:
customer term, documentation term, source count, zero-result count, and score.

The drawer renders those mappings under each FAQ item. The list view adds a
small `vocab gaps: N` fact/meta label so reviewers know whether opening the
drawer is useful.

## Intentional

- UI-only. No generated output, API, repository, or database change.
- No new hosted upload controls in this slice; this displays mappings already
  produced by CLI/library/service paths.
- No chart or table abstraction. The existing FAQ drawer cards are sufficient
  for this thin review surface.

## Deferred

- Hosted upload controls for documentation terms and vocabulary rules remain a
  separate product slice.
- Bulk filtering/sorting by vocabulary-gap count remains deferred until a real
  review queue needs it.
- Parked hardening considered: current `HARDENING.md` entries are landing-page
  repair items and do not touch this FAQ lane.

## Verification

- `npm run lint` from `atlas-intel-ui` - passed.
- `npm run build` from `atlas-intel-ui` - passed.
- `git diff --check` - passed.
- Local Vite dev server started at `http://127.0.0.1:5175/` for manual review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Review UI parser/rendering | ~90 |
| **Total** | ~170 |
