# PR: Content Ops Upload Source Public Asset Validation

## Why this slice exists

#1236 proved persisted uploaded support-ticket IDs can execute through the
real Content Ops router and save landing-page/blog drafts. The remaining
review-to-public seam is narrower: a saved landing-page draft must only become
visible through the unauthenticated public route after the tenant-scoped review
route approves it.

This PR closes that seam with a deterministic host-route test. It avoids a live
database or deployed browser while proving the same generated-asset review and
public route handlers that the UI calls.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Functional validation

1. Add an in-process generated-assets route test for `landing_page` review
   approval followed by `GET /landing_page/public/{id}`.
2. Assert the public route is fail-closed before approval and after rejection.
3. Enroll the host generated-assets API test in a dedicated Atlas workflow so
   the route validation runs in CI.
4. Keep the storage boundary mocked with an in-memory landing-page repository;
   do not alter production router behavior.

### Files touched

- `tests/test_atlas_content_ops_generated_assets_api.py`
- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `plans/PR-Content-Ops-Upload-Source-Public-Asset-Validation.md`

## Mechanism

The test creates the real generated-asset review router and the real public
landing-page router with the same fake pool and account scope. It monkeypatches
`PostgresLandingPageRepository` inside
`extracted_content_pipeline.api.generated_assets` to an in-memory repository
that records tenant-scoped status updates and returns a public draft only when
its status is approved.

The new workflow runs `tests/test_atlas_content_ops_generated_assets_api.py`
whenever the host API mount, generated-assets route module, workflow file, or
test file changes.

The test then exercises this sequence:

```text
GET public landing page before review -> 404
POST /landing_page/drafts/review approved -> updated true
GET public landing page after approval -> public row
POST /landing_page/drafts/review rejected -> updated true
GET public landing page after rejection -> 404
```

That pins the review/public contract without adding another browser or live
Postgres dependency.

## Intentional

- No production code change is planned. Existing route handlers already expose
  the behavior; this slice adds the missing stitched validation.
- No Playwright/browser smoke in CI. The public route behavior is proved at the
  hosted API seam, matching the current stable CI strategy for Content Ops.
- No blog public-route assertion here. Blog public rendering is covered by the
  blog-specific public tests; this slice targets the landing-page approval gate
  created by the generated-asset review flow.

## Deferred

- Future PR: deployed preview browser validation using a real generated
  uploaded-source landing page URL.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  no matching active entries apply to this route-validation slice.

## Verification

- `pytest tests/test_atlas_content_ops_generated_assets_api.py::test_landing_page_review_approval_controls_public_route -q`
  - 1 passed.
- `pytest tests/test_atlas_content_ops_generated_assets_api.py -q`
  - 14 passed, 1 warning.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  - OK: 140 matching tests are enrolled.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-upload-source-public-asset-validation-pr-body.md`
  - local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Generated-assets route test | ~250 |
| CI workflow enrollment | ~40 |
| **Total** | **~375** |
