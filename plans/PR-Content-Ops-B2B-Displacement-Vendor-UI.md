# PR: Content Ops B2B Displacement Vendor UI

## Why this slice exists

PR #1258 made canonical B2B displacement dynamics selectable through request
JSON for competitive Content Ops runs, but the operator-facing New Run screen
still has no control for choosing the tracked vendors. That leaves the product
flow technically wired but manually keyed, which is easy to mistype and hard to
use during live landing-page/blog generation.

This slice closes the first deferred gap from #1258 by adding a thin New Run UI
selector over the existing tenant tracked-vendor client. It does not add a new
backend endpoint: `fetchTrackedVendors()` already reads the tenant's tracked
vendors, and #1258 already consumes `b2b_displacement_vendors` in the request
inputs.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Fetch tenant tracked vendors on the New Run page through the existing B2B
   client.
2. Show a competitive-mode vendor selector that writes selected vendor names to
   `inputs.b2b_displacement_vendors`.
3. Preserve already-selected vendor names that are not in the current fetched
   list until the operator unchecks them.
4. Add focused frontend helper/page tests by extending the already-enrolled
   source-selection test script.

### Files touched

- `plans/PR-Content-Ops-B2B-Displacement-Vendor-UI.md`
- `atlas-intel-ui/src/pages/contentOpsSourceMode.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-review-source-selection.test.mjs`

## Mechanism

The source-mode helper will own the JSON contract helpers for
`b2b_displacement_vendors`. `ContentOpsNewRun` will import
`fetchTrackedVendors()` from the existing B2B client, render a selector when
`sourceMode === 'competitive'`, and call the helper whenever the operator
toggles a vendor. The selector mirrors the existing FAQ source selector shape:
loading, refresh, unavailable, empty, and missing-selected rows are all local UI
states.

The existing `test:content-ops-review-source-selection` script is already run by
the UI checks workflow, so this slice extends it instead of adding an unenrolled
test script.

## Intentional

- No backend route or SQL changes. #1258 already added the tenant-scoped B2B
  displacement loader; this PR only makes the request key easy to populate.
- No search/add tracked-vendor workflow inside New Run. Operators manage tracked
  vendors through the B2B tenant surface; this selector only chooses from that
  tenant-owned list.
- No new workflow step. The changed test lives in the already-enrolled
  `test:content-ops-review-source-selection` step.

## Deferred

- Competitive-specific output skills beyond blog and landing page, such as ad
  copy, social posts, or stat/quote cards.
- Product packaging/pricing for the marketer competitive offer.
- Inline tracked-vendor add/search from New Run, if operators need that instead
  of using the B2B tenant page.

## Parked hardening

None.

## Verification

- Command: `cd atlas-intel-ui && npm run test:content-ops-review-source-selection` -- 8 passed.
- Command: `cd atlas-intel-ui && npm run lint` -- passed.
- Command: `cd atlas-intel-ui && npm run build` -- passed.
- Command: `git diff --check` -- passed.
- Command: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-b2b-displacement-vendor-ui-pr-body.md` -- passed.

## Estimated diff size

Estimated: 399 LOC actual (4 files, +399 / -0).

| Area | Estimated LOC |
|---|---:|
| New Run selector + state | ~210 |
| Source-mode JSON helpers | ~45 |
| Frontend tests | ~52 |
| Plan doc | ~92 |
| **Total** | **~399** |
