# PR: Content Ops Social Post Generated Assets

## Why this slice exists

PR #1271 persisted generated `social_post` drafts into tenant-scoped
`social_posts` rows, but the generated-asset review API and UI still only know
how to list, export, approve, and reject reports, blog posts, landing pages,
sales briefs, and FAQ Markdown. That leaves the new durable social-post table
invisible to operators.

This slice completes the review-queue handoff from #1271: persisted social
posts become a first-class generated asset in the existing review API and
`atlas-intel-ui` generated-asset review screen.

The diff is over the 400 LOC soft cap because this is the smallest safe
end-to-end handoff: the backend export/review branches, frontend type/UI
branches, CI-enrolled frontend test, and extracted API tests need to land
together or `social_post` would be only partially reviewable.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add a package-owned social-post export helper that returns the same
   `count`/`limit`/`filters`/`rows` envelope and CSV behavior as the sibling
   generated assets.
2. Add `social_post` to the generated-assets backend switchboards for list,
   CSV/JSON export, single review, and batch review.
3. Add `social_post` to the atlas-intel-ui API type surface and generated-asset
   review asset registry.
4. Render social-post rows with channel, company/vendor/source facts and the
   post text as the preview body.
5. Add `social_post` to the completed-run review CTA allowlist so saved social
   post IDs hand off to the review tab without id filtering.
6. Add backend and frontend tests proving the route/UI wiring, and enroll the
   new frontend test in CI.

### Files touched

- `plans/PR-Content-Ops-Social-Post-Generated-Assets.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/social_post_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/scripts/content-ops-social-post-review-assets.test.mjs`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`

## Mechanism

`export_social_post_drafts(...)` wraps `PostgresSocialPostRepository.list_drafts`
and maps each `SocialPostDraft` into a flat review/export row. The generated
asset router then treats `social_post` like the other non-public generated
assets:

```python
if asset == "social_post":
    return await export_social_post_drafts(PostgresSocialPostRepository(pool), ...)
```

The same `_update_asset_status` and `_update_asset_statuses` helpers call the
repository added in #1271, so review and batch review retain tenant-scoped
`account_id` predicates from the adapter instead of duplicating SQL in the API
layer.

On the frontend, `GeneratedAssetType` and the review screen `ASSETS` registry
include `social_post`. Existing fetch/review/export helpers already build URLs
from the asset type, so the UI work is limited to rendering meaningful social
post labels, facts, and preview text. `ContentOpsNewRun.tsx` also includes
`social_post` in the generated-output allowlist used by
`generatedAssetReviewHref(...)`; because the backend intentionally does not add
id-filter repository support in this slice, the CTA links to the
`social_post` draft review tab with no repeated `id` query keys.

## Intentional

- No public social-post URL support. Social posts are review/export assets, not
  hosted public pages in this slice.
- No edit/repair flow for social posts. The existing landing-page edit/repair
  controls remain landing-page-only.
- No ID deep-link filter for `social_post`. The #1271 repository lists by
  status, target mode, and channel; adding id-filter repository support can be a
  follow-up if run-result deep links need it.

## Deferred

- Ad-copy and stat/quote-card package defaults remain future output slices.
- Optional social-post id deep links can be added after a product path actually
  needs direct review links from generation results.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_extracted_content_asset_api.py -q` (70 passed)
- Passed: `python -m py_compile extracted_content_pipeline/social_post_export.py extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py`
- Passed: `cd atlas-intel-ui && npm run test:content-ops-social-post-review-assets` (4 passed)
- Passed: `cd atlas-intel-ui && npm run lint`
- Passed: `cd atlas-intel-ui && npm run build`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (OK: 144 matching tests are enrolled.)
- Passed: `bash scripts/validate_extracted_content_pipeline.sh`
- Passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
- Passed: `python scripts/audit_extracted_standalone.py --fail-on-debt`
- Passed: `bash scripts/check_ascii_python.sh`
- Passed: `bash scripts/run_extracted_pipeline_checks.sh` (2976 passed, 10 skipped, 1 warning)
- Passed: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-social-post-generated-assets-pr-body.md`

## Estimated diff size

Actual: 11 files, +619 / -2. Above the 400 LOC soft cap for the end-to-end
handoff reasons named in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Social-post export helper + manifest | ~135 |
| Backend switchboard + tests | ~135 |
| Frontend type/UI/test/workflow enrollment | ~232 |
| Plan doc | ~119 |
| **Total** | **~621** |
