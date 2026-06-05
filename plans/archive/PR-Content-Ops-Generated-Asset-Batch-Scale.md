# PR: Content Ops Generated Asset Batch Scale

## Why This Slice Exists

Generated asset batch review currently loops one scoped status update per draft.
That preserves correctness, but it scales linearly in database round trips when
operators approve or reject larger review queues.

## Scope

Move generated asset batch review to one repository-level bulk update per asset
type while preserving the existing API response shape.

### Files Touched

- `extracted_content_pipeline/api/generated_assets.py`
- `extracted_content_pipeline/blog_post_postgres.py`
- `extracted_content_pipeline/report_postgres.py`
- `extracted_content_pipeline/landing_page_postgres.py`
- `extracted_content_pipeline/sales_brief_postgres.py`
- `extracted_content_pipeline/blog_ports.py`
- `extracted_content_pipeline/report_ports.py`
- `extracted_content_pipeline/landing_page_ports.py`
- `extracted_content_pipeline/sales_brief_ports.py`
- `tests/test_extracted_content_asset_api.py`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Generated-Asset-Batch-Scale.md`

## Mechanism

- Add bulk `update_statuses` repository methods for the four generated asset
  stores.
- Route `POST /content-assets/{asset}/drafts/review-batch` through the bulk
  method.
- Preserve ordered `updated_ids` and `missing_ids` in the API response by
  comparing returned ids against the original request order.
- Treat malformed UUIDs as missing IDs before the bulk SQL call so one bad
  operator-supplied ID cannot poison the whole batch.
- Refresh the stale Content Ops backlog and coordination row.

## Intentional

- No frontend changes.
- No response contract changes.
- No new migration.
- Host implementations of generated asset repository ports must add
  `update_statuses`; the hosted API now depends on the bulk method for batch
  review.

## Deferred

- Reasoning product-depth/source-breadth work.
- Dedicated performance benchmark.

## Verification

- `tests/test_extracted_content_asset_api.py`.
- `scripts/local_pr_review.sh`.
- `git diff --check`.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| API batch dispatch | ~55 |
| Repository bulk methods | ~110 |
| Port protocols | ~40 |
| Tests | ~65 |
| Backlog and coordination docs | ~90 |
| Plan doc | ~65 |
| **Total** | ~425 |
