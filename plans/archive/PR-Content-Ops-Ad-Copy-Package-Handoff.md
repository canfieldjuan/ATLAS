# PR: Content Ops Ad Copy Package Handoff

## Why this slice exists

PR #1275 added the deterministic `ad_copy` output and deliberately deferred
putting it into the review and competitive input-package defaults. Until this
handoff lands, operators can run `ad_copy` explicitly, but the productized
review/competitive source modes still default only to landing pages, blog posts,
sales briefs, and social posts.

This slice completes that handoff by making review and competitive source
packages request ad copy by default, using the same normalized
`source_material` already proven by #1275. It mirrors
`plans/PR-Content-Ops-Social-Post-Package-Handoff.md` and does not overlap
#1268, which is a separate plan-doc-only output-variations lane.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add `ad_copy` to the review source-mode default outputs.
2. Add `ad_copy` to the competitive source-mode default outputs.
3. Align the package offer copy with the expanded output set.
4. Extend the existing host-provider tests to prove the default package outputs
   and execution handoff include `ad_copy`.

### Files touched

- `plans/PR-Content-Ops-Ad-Copy-Package-Handoff.md`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The host input provider already builds normalized `source_material` for review
and competitive source modes. `ad_copy` consumes that same field, so this slice
only expands the two output tuples:

```python
_REVIEW_CAMPAIGN_OUTPUTS = (..., "social_post", "ad_copy")
_COMPETITIVE_CAMPAIGN_OUTPUTS = (..., "social_post", "ad_copy")
```

The execution tests use the existing `_RunnableService` fake in the `ad_copy`
service slot. If the output is present but the dispatcher fails to hand off
`source_material`, the tests fail on the ad-copy step kwargs.

## Intentional

- No new source adapter or generator logic. #1275 already proved `ad_copy` runs
  from normalized source material.
- No UI selector changes. The New Run source-mode path consumes these package
  defaults automatically.
- No persistence or generated-asset review branch. Durable ad-copy review is a
  separate productization slice after this default handoff.
- No changes to #1268 output variations; that open PR is a separate
  workflow/process lane and remains untouched.

## Deferred

- Persist generated ad-copy drafts into the generated-assets review queue after
  the package-default handoff is accepted.
- Stat/quote card output remains the next marketer asset family after ad copy.
- LLM/style/channel variants and #1268-style output variations remain future
  product polish.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_atlas_content_ops_review_input_provider.py -q` (22 passed)
- Passed: `pytest tests/test_atlas_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py -q` (46 passed, 1 warning)
- Passed: `python -m py_compile atlas_brain/_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (OK: 144 matching tests are enrolled.)
- Passed: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-ad-copy-package-handoff-pr-body.md`

## Estimated diff size

Actual: 3 files, +121 / -4. This stays under the 400 LOC soft cap because it
reuses the existing source-mode package and #1275's ad-copy dispatcher.

| Area | Estimated LOC |
|---|---:|
| Host provider defaults and copy | ~13 |
| Host provider tests | ~21 |
| Plan doc | ~91 |
| **Total** | **~125** |
