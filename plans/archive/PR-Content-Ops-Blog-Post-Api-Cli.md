# PR-Content-Ops-Blog-Post-Api-Cli

## Why this slice exists

PR #447 added the blog-post Postgres repository and PR #448 added the read-only
export helper. The existing generated-asset API and CLIs still only route
reports, landing pages, and sales briefs. Hosts need `blog_post` to use the
same review/export surface.

## Scope (this PR)

1. Add `blog_post` to generated-asset API list/export/review routing.
2. Add `blog_post` to the generated-asset export and review CLIs.
3. Add focused API/CLI routing tests.

Files touched: generated asset router, generated asset CLIs, focused routing
tests, plan, and coordination row.

## Mechanism

The switchboards route `asset=blog_post` to `PostgresBlogPostRepository` and
`export_blog_post_drafts(...)`. `topic_type` is the blog-specific filter,
parallel to `report_type`, `campaign_name`, and `brief_type` on the other asset
types.

## Intentional

- This PR does not update runbooks/status/manifest/check-script. Those are
  coordination/docs wiring and stay in the final stacked PR.
- Unknown-asset tests now use `podcast_episode` because `blog_post` becomes a
  valid generated asset.

## Deferred

- Docs/status/manifest/check-script cleanup stays in the final coordination PR.

## Verification

- `pytest tests/test_extracted_content_asset_api.py tests/test_extracted_content_asset_export_cli.py tests/test_extracted_content_asset_review_cli.py`
  - 27 passed
- `python -m py_compile extracted_content_pipeline/api/generated_assets.py scripts/export_extracted_content_assets.py scripts/review_extracted_content_assets.py`
  - passed
- `bash scripts/check_ascii_python.sh`
  - passed
- `git diff --check`
  - passed

## Estimated diff size

6 files, about +220/-5.
