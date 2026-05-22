# PR-Landing-Page-Draft-Edit-UI

## Why this slice exists

PR #777 added the tenant-scoped landing-page edit API, but operators still have
no in-app way to use it. Generated landing pages can now be corrected safely on
the backend, but the review drawer remains read-only.

This slice adds the first landing-page-only edit mode in Generated Asset Review
so an operator can correct common draft fields, save them through the real API,
and immediately see the refreshed readiness result.

This is slightly above the 400 LOC target because the edit state, form, payload
mapping, and save/update wiring need to land together for the UI to be usable.
Splitting the form from the save path would create a visual shell that cannot
exercise the backend primitive from PR #777.

## Scope (this PR)

Ownership lane: content-ops/landing-page-draft-edit-ui

1. Add an Edit button to the generated-asset detail drawer for non-approved
   landing pages.
2. Add a structured edit form for title, slug, hero, CTA, meta, sections, and
   reference IDs.
3. Save through `updateGeneratedLandingPageDraft` and replace the updated row in
   local review state.
4. Keep reports, blogs, sales briefs, FAQ Markdown, and approved landing pages
   read-only.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Draft-Edit-UI.md` | Plan doc for this UI slice. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Add landing-page edit mode and save wiring in Generated Asset Review. |

## Mechanism

The page owns a `handleSaveLandingPageDraft` callback. The drawer receives that
callback, builds a `GeneratedLandingPageDraftUpdate` payload from the edit form,
and calls it when the operator saves. On success, the page replaces the matching
row in `data.rows` and updates `detailRow` with the API response. Because the
API returns the normal generated-asset review row, the existing Facts,
Readiness, Structured Data, Preview, and Raw Row sections update without a
second fetch.

The form edits only the fields #777 allows: `title`, `slug`, `hero`,
`sections`, `cta`, `meta`, and `reference_ids`. Section metadata and unknown
hero/CTA/meta keys are preserved by merging edits onto the existing row values.

## Intentional

- No editor for approved landing pages. Live public content should not be
  mutated through this manual draft editor.
- No raw JSON editor. The first UI should keep operators on the safe structured
  fields backed by the API allowlist.
- No edit UI for other asset types until they have comparable backend edit
  primitives.
- No autosave. Manual save keeps review-state changes explicit.

## Deferred

- `PR-Landing-Page-Saved-Draft-Repair` should add LLM-assisted repair for
  missing readiness checks.
- `PR-Landing-Page-Edit-Audit-Trail` can expose editor/user history once the
  backend stores it.
- `PR-Generated-Asset-Edit-Component` can extract a shared editor if other
  generated asset types gain edit APIs.

## Verification

- `npm run lint` in `atlas-intel-ui` -> passed.
- `npm run build` in `atlas-intel-ui` -> passed.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Review UI | ~490 |
| Total | ~575 |
