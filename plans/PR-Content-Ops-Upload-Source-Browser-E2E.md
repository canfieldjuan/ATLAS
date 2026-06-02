# PR: Content Ops Upload Source Browser E2E

## Why this slice exists

The upload-source handoff arc is now complete in pieces: import returns target
IDs, execute can load those persisted IDs by tenant scope, and New Run applies
those IDs instead of inlining full source rows. What remains is a stitched
validation that proves the browser-visible run contract can produce both
reviewable landing-page and blog-post drafts from persisted imported
support-ticket IDs.

This PR adds that proof at the hosted API seam. It keeps CI deterministic by
using the existing in-process router, fake persisted import rows, and real
landing/blog generation services with a deterministic LLM.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Functional validation

1. Add a route-level execute test where `inputs.source_import_target_ids` load
   support-ticket-shaped imported rows by tenant scope.
2. Generate both `landing_page` and `blog_post` from those persisted rows using
   the real generation services and in-memory draft repositories.
3. Assert saved draft IDs, tenant-scoped repository calls, support-ticket data
   context, and landing/blog draft fields that the review/public UI consumes.

### Files touched

- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-Content-Ops-Upload-Source-Browser-E2E.md`

## Mechanism

The test reuses the existing Content Ops control-surface router and the Atlas
input provider. A fake opportunity repository returns exactly one row for each
imported target ID, matching the fail-closed lookup discipline from #1231. The
execute payload mirrors the browser-applied request shape from #1232:

```json
{
  "outputs": ["landing_page", "blog_post"],
  "inputs": {
    "source_import_target_ids": ["ticket-1", "ticket-2"]
  }
}
```

The route then runs `LandingPageGenerationService` and
`BlogPostGenerationService` with in-memory repositories. Assertions prove the
generated draft IDs are saved, tenant scoped, and grounded in the loaded
support-ticket context.

## Intentional

- No Playwright/browser automation in CI. The repo's current Atlas Intel UI
  E2E tests are source/API contract scripts, and the browser-hosted pass is a
  deployment/manual validation step rather than a stable local CI dependency.
- No live LLM or live database. This is functional validation of the handoff
  contract, not a paid/live generation run.
- No generated-asset approval write in this slice. Existing generated-asset API
  tests already pin approval/public route wiring; this test proves the uploaded
  support-ticket run produces the draft IDs those routes consume.

## Deferred

- Future PR: hosted browser validation on a deployed preview against the real
  UI/API, approving the generated landing/blog assets and opening their public
  URLs.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  existing entries are dependency audit and blog content-quality issues, not
  this upload-source API proof.

## Verification

- `pytest tests/test_atlas_content_ops_input_provider.py -k persisted_source_targets -q`
  - 3 passed, 21 deselected.
- `pytest tests/test_atlas_content_ops_input_provider.py -q`
  - 24 passed, 1 warning.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  - OK: 140 matching tests are enrolled.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-upload-source-browser-e2e-pr-body.md`
  - local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Route-level execute proof | ~150 |
| **Total** | **~245** |
