# PR-Content-Ops-Blog-Post-Asset-Review

## Why this slice exists

`blog_post` is an implemented AI Content Ops output with `BlogPostRepository`
list/review methods, but the host-facing generated-asset review path only
supports reports, landing pages, and sales briefs. Hosts can generate blog
drafts through the extracted service, but cannot use the same packaged
Postgres adapter, export CLI, review CLI, or generated-asset API to inspect and
approve them.

## Scope (this PR)

1. Add a Postgres `BlogPostRepository` adapter for the existing `blog_posts`
   table.
2. Add a read-only blog-post export helper parallel to report/landing/sales
   export helpers.
3. Add `blog_post` to the generated-asset CLI and API switchboards.
4. Add the minimal blog-post account-scope migration needed by the host review
   path.
5. Add focused tests and wire them into the extracted pipeline check script.

Files touched:

- `extracted_content_pipeline/blog_post_postgres.py`
- `extracted_content_pipeline/blog_post_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `scripts/export_extracted_content_assets.py`
- `scripts/review_extracted_content_assets.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/storage/migrations/276_blog_post_account_scope.sql`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `tests/test_extracted_blog_post_postgres.py`
- `tests/test_extracted_blog_post_export.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_extracted_content_asset_export_cli.py`
- `tests/test_extracted_content_asset_review_cli.py`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The new adapter stores generated blog drafts in `blog_posts`, preserving
generation metadata inside `data_context["_metadata"]` because the legacy table
does not have a standalone metadata JSONB column. A new `account_id` column
gives hosted installs the same scope filter shape as the other generated assets
without changing the legacy `slug` uniqueness or autonomous task conflict path.

The export helper flattens blog drafts into JSON/CSV review rows with generation
usage, parse-attempt, and reasoning-context summary fields. Existing generated
asset CLIs and API routes route `asset=blog_post` to the new adapter/helper.

## Intentional

- The migration adds `account_id` but does not remove the legacy unique slug
  constraint. Removing it would break existing `ON CONFLICT (slug)` writers.
- Blog metadata is stored under `data_context["_metadata"]` rather than adding a
  second metadata column in this slice. This keeps the adapter compatible with
  the existing `BlogPostDraft` shape and current `blog_posts` schema.
- Existing legacy blog rows default to `account_id=''`, so unscoped host review
  paths can still see them.

## Deferred

- Per-tenant duplicate slugs remain a schema limitation until legacy blog writers
  migrate away from `ON CONFLICT (slug)`.
- The generated-asset frontend can add a blog-post tab after the API exposes the
  asset path; this PR only ships backend/CLI review surfaces.

## Verification

Local checks run:

- `pytest tests/test_extracted_blog_post_postgres.py tests/test_extracted_blog_post_export.py tests/test_extracted_content_asset_api.py tests/test_extracted_content_asset_export_cli.py tests/test_extracted_content_asset_review_cli.py`
  - 38 passed
- `bash scripts/check_ascii_python.sh`
  - passed
- `python -m py_compile extracted_content_pipeline/blog_post_postgres.py extracted_content_pipeline/blog_post_export.py scripts/export_extracted_content_assets.py scripts/review_extracted_content_assets.py`
  - passed
- `python scripts/check_extracted_imports.py`
  - passed for 89 modules
- `bash scripts/validate_extracted_content_pipeline.sh`
  - passed
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - passed with 0 findings
- `pytest tests/test_extracted_campaign_manifest.py tests/test_extracted_storage_jsonb_helpers.py`
  - 24 passed
- `bash scripts/run_extracted_pipeline_checks.sh`
  - 1431 passed, 1 existing torch/pynvml warning

## Estimated diff size

17 files, +1055/-17. This is above the soft target because the slice needs one
adapter, one export helper, three switchboard updates, migration/docs/manifest
wiring, and parity tests at repository/export/API/CLI boundaries. The runtime
change is still one logical asset-review seam.
