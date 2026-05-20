# PR-Blog-Readiness-Breakdown-UI

## Why this slice exists

PR-Blog-Readiness-Review-UI made SEO/AEO and GEO readiness visible as compact
labels on blog generated-asset rows. That is enough for scanning, but operators
still need the raw JSON when they want to see every passing and missing check.

This slice adds a focused readiness breakdown inside the generated asset detail
drawer.

## Scope (this PR)

1. Add a blog-only readiness section to `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`.
2. Show SEO/AEO and GEO status, pass counts, missing checks, and individual
   pass/fail checks in the detail drawer.
3. Reuse the existing generated-asset readiness fields from the API adapter.
4. Keep backend, generation, approval, and rejection behavior unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Readiness-Breakdown-UI.md` | Plan doc for this slice. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Add blog readiness breakdown cards to the detail drawer. |

## Mechanism

The detail drawer now builds blog readiness panels from the existing
`seo_aeo_readiness` and `geo_readiness` row fields. Each panel renders the
readiness status, count, missing checks, and boolean checks from the readiness
payload.

The compact row and preview labels stay unchanged; this is only the deeper view
for operators who need to inspect why a draft needs review.

## Intentional

- No API shape change.
- No backend change.
- No new quality gate behavior.
- No generic readiness UI for every asset type yet.

## Deferred

- Promote the readiness panel to a shared component if other generated asset
  types get first-class readiness contracts.
- Add UI component tests if the Atlas Intel UI gets a test runner.

## Verification

- Atlas UI lint passed.
- Atlas UI production build passed.
- Blog GEO prerender verification passed.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| Review UI breakdown | ~90 |
| Total | ~145 |
