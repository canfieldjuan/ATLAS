# PR: Content Ops Social Post Package Handoff

## Why this slice exists

PR #1266 added the deterministic `social_post` output and deliberately deferred
putting it into the review and competitive input-package defaults. Until this
handoff lands, operators can run `social_post` explicitly, but the productized
review/competitive source modes still default only to landing pages, blog posts,
and sales briefs.

This slice completes that handoff by making review and competitive source
packages request social posts by default, using the same normalized
`source_material` already proven by #1266. It does not overlap #1268, which is a
plan-doc-only output-variations lane.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add `social_post` to the review source-mode default outputs.
2. Add `social_post` to the competitive source-mode default outputs.
3. Align the package offer copy with the expanded output set.
4. Extend the existing host-provider tests to prove the default package outputs
   and execution handoff include `social_post`.

### Files touched

- `plans/PR-Content-Ops-Social-Post-Package-Handoff.md`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The host input provider already builds normalized `source_material` for review
and competitive source modes. `social_post` consumes that same field, so this
slice only expands the two output tuples:

```python
_REVIEW_CAMPAIGN_OUTPUTS = (..., "social_post")
_COMPETITIVE_CAMPAIGN_OUTPUTS = (..., "social_post")
```

The execution tests use the existing `_RunnableService` fake in the
`social_post` service slot. If the output is present but the dispatcher fails to
handoff `source_material`, the tests fail on the social-post step kwargs.

## Intentional

- No new source adapter or generator logic. #1266 already proved
  `social_post` runs from normalized source material.
- No UI selector changes. The New Run source-mode path consumes these package
  defaults automatically.
- No changes to #1268 output variations; that open PR is a separate
  workflow/process lane and remains untouched.

## Deferred

- Ad copy and stat/quote card package defaults remain future output slices.
- Persisting generated social posts into the generated-asset review queue
  remains future productization.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_atlas_content_ops_review_input_provider.py -q (22 passed)
- Passed: pytest tests/test_atlas_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py -q (46 passed, 1 warning)
- Passed: python -m py_compile atlas_brain/_content_ops_input_provider.py
- Passed: git diff --check
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-social-post-package-handoff-pr-body.md

## Estimated diff size

Actual: 3 files, +125 / -17. This stays under the 400 LOC soft cap because it
reuses the existing source-mode package and #1266's social-post dispatcher.

| Area | Estimated LOC |
|---|---:|
| Host provider defaults and copy | ~17 |
| Host provider tests | ~39 |
| Plan doc | ~86 |
| **Total** | **~142** |
