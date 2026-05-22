# PR-Landing-Page-Public-Sitemap

## Why this slice exists

Generated landing pages now have a backend-owned robots policy, but there is
no public indexable URL feed for crawlers or deployment tooling. This slice
adds a dynamic sitemap endpoint that includes only approved generated landing
pages whose backend robots policy is `index,follow`.

## Scope (this PR)

Ownership lane: content-ops/landing-page-public-sitemap

1. Add a narrowed repository projection for approved public sitemap candidates.
2. Add a public sitemap XML route for generated landing pages.
3. Gate sitemap inclusion on `public_landing_page_robots(draft) == "index,follow"`.
4. Keep non-ready approved pages out of the sitemap.
5. Keep static Vite sitemap/prerender unchanged in this slice.

### Files touched

- `plans/PR-Landing-Page-Public-Sitemap.md`
- `extracted_content_pipeline/landing_page_ports.py`
- `extracted_content_pipeline/landing_page_postgres.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_landing_page_postgres.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`

## Mechanism

The public router exposes `GET /content-assets/landing_page/public/sitemap.xml`.
It fetches approved landing-page sitemap candidates, derives each candidate's
robots policy with the same helper used by the public renderer payload, and
emits XML only for candidates whose policy is `index,follow`. The configured
URL cap is applied after readiness filtering so approved-but-not-ready rows
cannot crowd out indexable pages.

The route uses a configured public frontend base URL when supplied and falls
back to the request origin otherwise. The response contains only absolute page
URLs, `lastmod`, `changefreq`, and `priority`; it does not expose readiness
details, metadata, tenant scope, reasoning, or reference ids.

The repository projection intentionally omits the `metadata` column so tenant
scope, generation usage, and review-only context never travel through the
public sitemap path.

## Intentional

- No static Vite sitemap changes.
- No static prerender of database-backed pages.
- No sitemap inclusion for approved-but-not-ready pages.
- No tenant-scoped public sitemap.

## Deferred

- `PR-Landing-Page-Static-Sitemap-Bridge` can wire the frontend/deployment
  layer to this dynamic sitemap feed if the production routing layer needs one
  canonical `/sitemap.xml`.

## Verification

- Focused backend tests for public sitemap, repository lookup, and host mount
  - 52 passed.
- Git whitespace check - passed.
- Full extracted pipeline checks through `scripts/run_extracted_pipeline_checks.sh`
  - 1655 passed.
- Local PR review wrapper - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Backend sitemap route | ~85 |
| Repository port/adapter | ~90 |
| Tests | ~130 |
| **Total** | **~380** |
