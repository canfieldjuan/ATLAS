# PR: Content Ops Ad Copy Generated Assets

## Why this slice exists

PR #1282 persisted generated `ad_copy` drafts into tenant-scoped
`ad_copy_drafts` rows, but the generated-asset review API and UI still only know
how to list, export, approve, and reject reports, blog posts, landing pages,
sales briefs, social posts, and FAQ Markdown. That leaves the new durable
ad-copy table invisible to operators.

This slice completes the review-queue handoff from #1282: persisted ad-copy
drafts become a first-class generated asset in the existing review API and
`atlas-intel-ui` generated-asset review screen.

The diff is expected to exceed the 400 LOC soft cap for the same reason as the
social-post generated-assets slice: the backend export/review branches,
frontend type/UI branches, CI-enrolled frontend test, and extracted API tests
need to land together or `ad_copy` would be only partially reviewable.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add a package-owned ad-copy export helper that returns the same
   `count`/`limit`/`filters`/`rows` envelope and CSV behavior as sibling
   generated assets.
2. Add `ad_copy` to the generated-assets backend switchboards for list,
   CSV/JSON export, single review, and batch review.
3. Add `ad_copy` to the atlas-intel-ui API type surface and generated-asset
   review asset registry.
4. Render ad-copy rows with channel, format, company/vendor/source facts,
   headline, primary text, CTA, and pain-point metadata.
5. Add `ad_copy` to the completed-run review CTA allowlist so saved ad-copy IDs
   hand off to the review tab without id filtering.
6. Add backend and frontend tests proving the route/UI wiring, and enroll the
   new frontend test in CI.

### Files touched

- `plans/PR-Content-Ops-Ad-Copy-Generated-Assets.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/ad_copy_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/scripts/content-ops-ad-copy-review-assets.test.mjs`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`

## Mechanism

`export_ad_copy_drafts(...)` wraps `PostgresAdCopyRepository.list_drafts` and
maps each `AdCopyDraft` into a flat review/export row. The generated asset
router then treats `ad_copy` like the other non-public generated assets:

```python
if asset == "ad_copy":
    return await export_ad_copy_drafts(PostgresAdCopyRepository(pool), ...)
```

The same `_update_asset_status` and `_update_asset_statuses` helpers call the
repository added in #1282, so review and batch review retain tenant-scoped
`account_id` predicates from the adapter instead of duplicating SQL in the API
layer.

On the frontend, `GeneratedAssetType` and the review screen `ASSETS` registry
include `ad_copy`. Existing fetch/review/export helpers already build URLs from
the asset type, so the UI work is limited to rendering meaningful ad-copy
labels, facts, and preview text. `ContentOpsNewRun.tsx` also includes `ad_copy`
in the generated-output allowlist used by `generatedAssetReviewHref(...)`;
because the backend intentionally does not add id-filter repository support in
this slice, the CTA links to the `ad_copy` draft review tab with no repeated
`id` query keys.

## Intentional

- No public ad-copy URL support. Ad copy is a review/export asset, not a hosted
  public page in this slice.
- No edit/repair flow for ad copy. The existing landing-page edit/repair
  controls remain landing-page-only.
- No ID deep-link filter for `ad_copy`. The #1282 repository lists by status,
  target mode, and channel; adding id-filter repository support can be a
  follow-up if run-result deep links need it.

## Deferred

- Optional ad-copy id deep links can be added after a product path actually
  needs direct review links from generation results.
- Stat/quote-card package defaults remain a future output slice.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_extracted_content_asset_api.py -q (74 passed)
- Passed: python -m py_compile extracted_content_pipeline/ad_copy_export.py extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py
- Passed: cd atlas-intel-ui && npm run test:content-ops-ad-copy-review-assets (4 passed)
- Passed: cd atlas-intel-ui && npm run lint
- Passed: cd atlas-intel-ui && npm run build
- Passed: git diff --check
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/validate_extracted_content_pipeline.sh
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Passed: bash scripts/check_ascii_python.sh
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2992 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-ad-copy-generated-assets-pr-body.md

## Estimated diff size

Actual: 11 files, +617 / -1. This is above the 400 LOC soft cap for the
end-to-end handoff reasons named in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Ad-copy export helper + manifest | ~135 |
| Backend switchboard + tests | ~135 |
| Frontend type/UI/test/workflow enrollment | ~230 |
| Plan doc | ~115 |
| **Total** | **~620** |
