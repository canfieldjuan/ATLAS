# PR-Content-Ops-Blog-Post-Export

## Why this slice exists

PR #447 added the generated blog-post Postgres repository foundation. Hosts
still need a read-only export helper before the generated-asset API/CLI
switchboards can route `blog_post` alongside reports, landing pages, and sales
briefs.

## Scope (this PR)

1. Add `export_blog_post_drafts(...)` for JSON/CSV review rows.
2. Add focused export-helper tests.
3. Claim this split in coordination.

Files touched: export helper, export tests, plan, and coordination row.

## Mechanism

The helper calls `BlogPostRepository.list_drafts(...)` with tenant scope,
status, topic type, and limit filters. It converts `BlogPostDraft` rows into
flat review/export dictionaries with generation usage, parse-attempt, and
reasoning-context summary fields. CSV rendering mirrors the existing report,
landing-page, and sales-brief export helpers.

## Intentional

- This PR does not wire the helper into any CLI or API. That stays in the next
  stacked PR so the export semantics can be reviewed separately.
- `content`, `charts`, `data_context`, and `metadata` are included in the export
  row because blog drafts are long-form assets and reviewers need the full draft
  body.

## Deferred

- Generated-asset API/CLI switchboards stay in the next stacked PR.
- Docs/status/manifest/check-script cleanup stays in the final coordination PR.

## Verification

- `pytest tests/test_extracted_blog_post_export.py`
  - 5 passed
- `python -m py_compile extracted_content_pipeline/blog_post_export.py`
  - passed
- `bash scripts/check_ascii_python.sh`
  - passed
- `git diff --check`
  - passed

## Estimated diff size

4 files, about +240/-1.
