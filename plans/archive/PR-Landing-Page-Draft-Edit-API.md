# PR-Landing-Page-Draft-Edit-API

## Why this slice exists

The landing-page SEO/AEO/GEO work now has input contracts, prompt alignment,
save-time readiness checks, and review UI visibility. The next gap is that an
operator still cannot make a small correction to a generated landing page
without leaving the app or regenerating the whole asset.

This slice adds the backend edit contract first so the later UI editor can call
a real tenant-scoped API instead of inventing client-only state.

This exceeds the 400 LOC soft cap because the repository contract, Postgres
adapter, API route, frontend client wrapper, and tenant/readiness regression
tests need to land together. Splitting the route from the repository would leave
no callable edit primitive; splitting the tests would leave the write path
under-specified.

## Scope (this PR)

Ownership lane: content-ops/landing-page-draft-edit-api

1. Add tenant-scoped landing-page draft read/update methods to the repository
   contract and Postgres adapter.
2. Add a landing-page-only `PATCH /content-assets/landing_page/drafts/{id}`
   route for editable draft fields.
3. Re-run landing-page export/readiness shaping after an edit and return the
   updated review row.
4. Add a typed frontend API wrapper for the future review-drawer editor.
5. Cover repository and API behavior with focused tests.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Draft-Edit-API.md` | Plan doc for this API slice. |
| `extracted_content_pipeline/landing_page_ports.py` | Add draft read/update methods to the landing-page repository contract. |
| `extracted_content_pipeline/landing_page_postgres.py` | Implement tenant-scoped draft read/update persistence. |
| `extracted_content_pipeline/api/generated_assets.py` | Add the landing-page edit route and payload validation. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add the typed client wrapper for the edit endpoint. |
| `tests/test_extracted_landing_page_postgres.py` | Add repository read/update tests. |
| `tests/test_extracted_content_asset_api.py` | Add generated-asset edit API tests. |

## Mechanism

The route accepts only the fields the generated landing-page draft already
owns: `title`, `slug`, `hero`, `sections`, `cta`, `meta`, and
`reference_ids`. It loads the existing row through tenant scope, rejects missing
or approved drafts, applies the patch, persists the updated content, and returns
`landing_page_draft_export_row(updated_draft)`.

Returning the export row is the key integration point: the UI receives the same
shape it already knows how to render, including `seo_aeo_readiness`,
`geo_readiness`, structured data, section counts, and metadata summaries.

## Intentional

- Approved landing pages are not editable through this endpoint. Editing an
  approved public page would silently mutate live content and bypass review.
- A successful edit resets the row status to `draft` so rejected or blocked
  drafts re-enter the existing review queue.
- The route does not call an LLM or auto-repair copy. This is a manual edit
  primitive, not the saved-draft repair loop.
- No database migration is needed; the editable fields already exist on
  `landing_pages`.

## Deferred

- PR-Landing-Page-Draft-Edit-UI should add the drawer/editor controls that call
  this endpoint.
- PR-Landing-Page-Saved-Draft-Repair should add an LLM repair action for
  missing readiness checks.
- PR-Landing-Page-Edit-Audit-Trail can add richer editor/user audit metadata
  once the host exposes a stable user identity to this router.

## Verification

- Python compile check for the changed backend modules -> passed.
- Focused generated-asset API and landing-page Postgres pytest suite -> 60 passed.
- Atlas Intel UI lint -> passed.
- Atlas Intel UI production build -> passed.
- Git whitespace check -> passed.
- Extracted content pipeline validation wrapper -> passed.
- Extracted content pipeline reasoning-import guard -> passed.
- Extracted standalone audit -> passed.
- Extracted content pipeline ASCII check -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Backend API/repository | ~244 |
| Frontend API wrapper | ~43 |
| Tests | ~300 |
| Total | ~685 |

This is above the soft cap for the reason named in "Why this slice exists."
