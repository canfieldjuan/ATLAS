# PR: Content Ops asset bulk review

## Why this slice exists

The current generated asset review surface lets operators approve or reject one
draft at a time. The active AI Content Ops backlog calls out batch review/status
updates before richer preview cards because batch actions improve operator
throughput across all generated asset types.

## Scope (this PR)

1. Add a host-mounted batch review endpoint for generated assets.
2. Add a typed frontend API wrapper for the batch review call.
3. Add checkbox selection and batch approve/reject actions to the generated
   asset review page.
4. Add focused backend route coverage.
5. Replace the merged reasoning-upsert coordination claim with this slice.

### Files touched

- `plans/PR-Content-Ops-Asset-Bulk-Review.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`

## Mechanism

The backend adds `POST /content-assets/{asset}/drafts/review-batch`. It
validates the asset, status, non-empty ids, and configured batch cap, then
reuses the same `_update_asset_status` helper as the single-row endpoint. The
response reports requested ids, updated ids, missing ids, and updated count.

The frontend keeps selection local to the current asset/filter page. Changing
asset, status, limit, or reloading data clears selection so operators do not
accidentally update stale rows.

## Intentional

- No new repository bulk SQL. Reusing the existing scoped `update_status`
  methods keeps tenant filtering identical to the single-row path.
- No cross-asset batch action. Operators review one asset type at a time because
  each asset has a different repository and filter shape.
- No richer preview cards in this slice.

## Deferred

- Native repository-level bulk update SQL if batch sizes become large.
- Richer generated asset previews and component-level frontend tests.

## Verification

```bash
python -m pytest tests/test_extracted_content_asset_api.py
python -m py_compile extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py
npm --prefix atlas-intel-ui run build
bash scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Asset-Bulk-Review.md` | 70 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/api/generated_assets.py` | 62 |
| `tests/test_extracted_content_asset_api.py` | 94 |
| `atlas-intel-ui/src/api/contentOps.ts` | 20 |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | 138 |
| **Total** | **~388** |
