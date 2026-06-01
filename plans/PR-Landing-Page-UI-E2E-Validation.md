# PR: Landing Page UI E2E Validation

## Why this slice exists

The landing-page generation stack is mostly productized: `/content-ops/new`
can execute `landing_page`, the host input provider can package support-ticket
and saved FAQ report source material, asset review can edit/approve generated
landing pages, and `/lp/:id/:slug` can render approved pages. The remaining UI
gap is the handoff between those pieces: after execution, the new-run screen
shows raw saved IDs but does not take the operator directly to the matching
landing-page review queue. That makes the path feel unfinished and easy to
mis-operate when testing live generation.

This slice finishes that handoff and locks it with focused UI/API contract
tests.

## Scope (this PR)

Ownership lane: content-ops/landing-pages-productization

Slice phase: Vertical slice

1. Add review links to generated-asset execution summaries so a completed
   `landing_page` run points directly at the review screen.
2. Teach the generated-asset review screen to honor `asset`, `status`, and
   repeated `id` query parameters on initial load.
3. Preserve the backend's existing `id` query contract by serializing array
   params as repeated keys from the frontend API helper.
4. Add a focused frontend script test for the new-run -> review deep link and
   review-page query handling.
5. Enroll that focused test in `atlas-intel-ui-checks` so the PR's regression
   lock runs in CI.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/scripts/content-ops-landing-page-e2e-ui.test.mjs`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `plans/PR-Landing-Page-UI-E2E-Validation.md`

## Mechanism

`GeneratedAssetSummary` will receive the step `output`, derive a review href
for reviewable asset outputs, and include saved landing-page IDs as repeated
`id` query parameters:

```text
/content-ops/assets?asset=landing_page&status=draft&id=<saved-id>
```

`ContentOpsAssetsReview` will read those query parameters on load, initialize
the asset/status filters, and pass the `id` array through
`fetchGeneratedAssetDrafts(...)`. The API query serializer will append arrays
with repeated keys so FastAPI receives `id: list[str]` instead of one comma
joined string.

For `landing_page` and `blog_post`, the review link is suppressed unless the
run returned at least one saved ID, preventing an ID-scoped output from opening
the general review queue after a skipped or failed persistence result.

## Intentional

- This is not another generation, quality-gate, or public-rendering rewrite.
  Those pieces already exist; the missing surface is the UI transition from
  generated draft IDs to review/approval/public URL.
- The review page uses query params as an initial deep-link/filter affordance.
  It does not attempt to keep every later tab/status change perfectly synced
  back to the URL in this slice.
- `id` filters are attached only where the backend supports them
  (`landing_page` and `blog_post`). Other reviewable assets still get an asset
  + status link without ID filtering because the link cannot accidentally imply
  a specific saved draft.

## Deferred

- Blog productization remains a later lane. This PR may keep the shared
  generated-asset helper compatible with `blog_post`, but it does not build the
  public `/blog` route.
- Hosted browser validation against a live Vercel/API deployment remains a
  follow-up once this UI contract lands.
- Parked hardening: none. `HARDENING.md` was scanned; no landing-page entries
  touch this slice.

## Verification

- `cd atlas-intel-ui && npm ci` - passed; npm reports the existing 6 audit
  findings already parked in `HARDENING.md`.
- `cd atlas-intel-ui && npm run test:content-ops-landing-page-e2e-ui` - 4
  passed.
- `cd atlas-intel-ui && npm run test:landing-page-prerender` - 4 passed.
- `cd atlas-intel-ui && npm run build` - passed; TypeScript and Vite build
  completed.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /home/juan-canfield/Desktop/landing-page-ui-e2e-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| New-run review link | ~55 |
| Asset-review query handling | ~65 |
| API query serialization | ~15 |
| Frontend script test + package script | ~75 |
| CI enrollment | ~5 |
| **Total** | **~295** |
