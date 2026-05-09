# PR-Content-Ops-Blog-Post-Repository

## Why this slice exists

`blog_post` is an implemented AI Content Ops output, but it only has a
persistence protocol. Hosts cannot inject a packaged Postgres repository for
the existing `blog_posts` table, so blog generation is behind the other asset
services.

This is the first split from oversized draft PR #445. It only adds the
repository/storage foundation; export/API/CLI wiring stays out of this PR.

## Scope (this PR)

1. Add a scoped Postgres adapter for `BlogPostRepository`.
2. Add the minimal `account_id` migration needed for host-scoped blog review.
3. Add repository tests for save/list/update behavior.

Files touched: repository adapter, account-scope migration, repository tests,
plan, and coordination row.

## Mechanism

The adapter persists `BlogPostDraft` rows into `blog_posts`, storing draft
metadata under `data_context["_metadata"]` because the legacy table has no
standalone metadata column. The migration adds `account_id` and scoped indexes
for list/review calls.

## Intentional

- Preserve legacy unique `slug`; copied blog writers still use `ON CONFLICT (slug)`.
- Store malformed `source_report_date` strings as `NULL`.
- Default legacy rows to `account_id=''`, matching the unscoped review path.

## Deferred

- Blog-post export helpers stay in the next stacked PR.
- Generated-asset API/CLI switchboards and docs/status/check-script cleanup
  stay in later stacked PRs.
- Per-tenant duplicate blog slugs need a later schema/writer migration.

## Verification

- `pytest tests/test_extracted_blog_post_postgres.py`
  - 4 passed
- `python -m py_compile extracted_content_pipeline/blog_post_postgres.py`
  - passed
- `bash scripts/check_ascii_python.sh`
  - passed

## Estimated diff size

5 files, about +390/-1. Export/API/CLI/docs stay in separate stacked PRs.
