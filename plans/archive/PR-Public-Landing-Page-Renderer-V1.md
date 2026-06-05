# PR-Public-Landing-Page-Renderer-V1

## Why this slice exists

Generated landing pages can now be reviewed, approved, exported, and inspected,
but there is no public renderer for approved pages. This slice adds a v1 public
route without turning pages into indexable SEO assets yet.

## Scope (this PR)

Ownership lane: content-ops/public-landing-page-renderer-v1

1. Add an approved-only public landing-page API lookup by id.
2. Render public landing pages at `/lp/:id/:slug`.
3. Redirect slug mismatches to the canonical slug returned by the API.
4. Use frontend-derived canonical URLs and existing structured data.
5. Mark v1 public generated landing pages `noindex,follow`.
6. Show/copy/open the approved public URL in the landing-page review drawer.
7. Keep the unauthenticated payload on a public allowlist and sanitize public
   markdown rendering.

### Files touched

- `plans/PR-Public-Landing-Page-Renderer-V1.md`
- `extracted_content_pipeline/landing_page_ports.py`
- `extracted_content_pipeline/landing_page_postgres.py`
- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `atlas_brain/api/__init__.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/components/SeoHead.tsx`
- `atlas-intel-ui/src/pages/PublicLandingPage.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/src/App.tsx`
- `tests/test_extracted_landing_page_postgres.py`
- `tests/test_extracted_landing_page_export.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`

## Mechanism

The backend adds `GET /content-assets/landing_page/public/{id}`. The route
validates the id as a UUID, fetches only rows with `status = 'approved'`, and
returns a dedicated public allowlist projection for renderer fields and
`structured_data`. It does not return review/export metadata, tenant scope,
reference ids, generation telemetry, reasoning fields, readiness fields, or
raw status.

The frontend adds the public route `/lp/:id/:slug`. It fetches by id without
auth headers, redirects when the returned slug differs from the URL slug, and
renders hero, sections, CTA, `SeoHead`, and JSON-LD. Section markdown disables
raw HTML before parsing and removes unsafe attributes/URLs from rendered HTML.
`SeoHead` gets optional robots meta support and the page uses `noindex,follow`.

The review drawer shows the public URL for approved landing pages and lets
operators open or copy it. Draft and rejected landing pages do not get an
active public link because the backend route is approved-only.

## Intentional

- No sitemap or prerender entry in v1.
- No slug-only lookup.
- No account handles, custom domains, or tenant scoped public route.
- No migration.
- Draft, queued, rejected, expired, missing, and invalid ids all behave as
  not found.

## Deferred

- `PR-Landing-Page-Indexable-Publish-Workflow` can add sitemap and index policy
  once public pages are ready to be discoverable.

## Verification

- Focused backend tests for the public route, allowlist projection, repository
  lookup, and export helpers - 54 passed.
- Frontend lint in `atlas-intel-ui` - passed.
- Frontend production build in `atlas-intel-ui` - passed.
- Git whitespace check - passed.
- Full extracted pipeline checks through `scripts/run_extracted_pipeline_checks.sh`
  - 1644 passed.
- Local PR review wrapper - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Backend public API | ~60 |
| Frontend renderer and review URL UI | ~370 |
| Tests | ~330 |
| **Total** | **~835** |
