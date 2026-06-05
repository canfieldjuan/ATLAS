# PR: Content Ops asset review row ids

## Plan

Expose generated asset row `id` and `status` through the existing
list/export path for reports, blog posts, landing pages, and sales briefs.

## Why this slice exists

The generated asset review route accepts `{id, status}`, but the list/export
rows did not include the database id or current status. That makes the hosted
review API hard to use from any UI or operator workflow without asking users
to find ids through SQL.

## Scope (this PR)

- Add optional `id` and `status` fields to the four generated asset draft
  port dataclasses.
- Project `id` and `status` from the Postgres list queries.
- Include those fields in export rows and CSV output.
- Regression-lock the API list response for reports and blog posts.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/blog_ports.py`
- `extracted_content_pipeline/blog_post_export.py`
- `extracted_content_pipeline/blog_post_postgres.py`
- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/landing_page_ports.py`
- `extracted_content_pipeline/landing_page_postgres.py`
- `extracted_content_pipeline/report_export.py`
- `extracted_content_pipeline/report_ports.py`
- `extracted_content_pipeline/report_postgres.py`
- `extracted_content_pipeline/sales_brief_export.py`
- `extracted_content_pipeline/sales_brief_ports.py`
- `extracted_content_pipeline/sales_brief_postgres.py`
- `plans/PR-Content-Assets-Review-Row-Ids.md`
- `tests/test_extracted_content_asset_api.py`

## Mechanism

The Postgres adapters select the row primary key and status, `_row_to_draft`
preserves them on the draft object, and the export helpers pass the draft
dictionary through to API/CSV callers.

## Intentional

- Keep both fields optional/default-empty on port dataclasses so existing
  in-memory tests and host adapters do not need constructor changes.
- Append CSV columns at the end to preserve existing leading-column contracts.

## Deferred

- Frontend review UI that consumes these ids.
- Bulk review actions.

## Verification

- Focused generated-asset API/export tests: 38 passed.
- Python compile check for touched generated-asset port and Postgres adapter
  files: passed.
- Local PR review wrapper: passed.

## Estimated diff size

| Area | Estimate |
|---|---:|
| Generated asset ports/adapters/export helpers | ~42 LOC |
| API regression tests | ~12 LOC |
| Plan + coordination | ~45 LOC |
| Total | ~100 LOC |
