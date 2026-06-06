# PR: Content Ops Stat Card Package Handoff

## Why this slice exists

PR #1294 added the deterministic `stat_card` output and deliberately deferred
putting it into the review and competitive input-package defaults. Until this
handoff lands, operators can run `stat_card` explicitly, but the productized
review/competitive source modes still default to landing pages, blog posts,
sales briefs, social posts, ad copy, and quote cards.

This slice completes that handoff by making review and competitive source
packages request stat cards by default, using the same normalized
`source_material` already proven by #1294. It mirrors
`plans/PR-Content-Ops-Quote-Card-Package-Handoff.md` and does not overlap
#1268's separate output-variations lane.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input

Slice phase: Vertical slice

1. Add `stat_card` to the review source-mode default outputs.
2. Add `stat_card` to the competitive source-mode default outputs.
3. Align the package offer copy with the expanded output set.
4. Extend the existing host-provider tests to prove the default package outputs
   and execution handoff include `stat_card`.

### Files touched

- `plans/PR-Content-Ops-Stat-Card-Package-Handoff.md`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The host input provider already builds normalized `source_material` for review
and competitive source modes. `stat_card` consumes that same field, so this
slice only expands the two output tuples:

```python
_REVIEW_CAMPAIGN_OUTPUTS = (..., "ad_copy", "quote_card", "stat_card")
_COMPETITIVE_CAMPAIGN_OUTPUTS = (..., "ad_copy", "quote_card", "stat_card")
```

The execution tests use the existing `_RunnableService` fake in the
`stat_card` service slot. If the output is present but the dispatcher fails to
hand off `source_material`, the tests fail on the stat-card step kwargs.

## Intentional

- No new source adapter or generator logic. #1294 already proved `stat_card`
  runs from normalized source material and enforces numeric evidence.
- No UI selector changes. The New Run source-mode path consumes these package
  defaults automatically.
- No persistence or generated-asset review branch. Durable stat-card review
  remains a separate productization slice after this default handoff.
- No visual template/export generation. This slice only makes the executable
  source-mode package request the deterministic output.
- No changes to #1268 output variations; that open PR is a separate lane and
  remains untouched.

## Deferred

- Persist generated stat-card drafts into the generated-assets review queue
  after the package-default handoff is accepted.
- Add visual template/export generation for quote cards and stat cards after
  review/export rows exist.
- LLM/style/channel variants and #1268-style output variations remain future
  product polish.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_atlas_content_ops_review_input_provider.py -q`
  (22 passed)
- Passed: `pytest tests/test_atlas_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py -q`
  (46 passed, 1 warning)
- Passed: `python -m py_compile atlas_brain/_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  (OK: 146 matching tests are enrolled.)
- Passed: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-stat-card-package-handoff-pr-body.md`

## Estimated diff size

Actual: 3 files, +116 / -2. This stays under the 400 LOC soft cap because it reuses
the existing source-mode package and #1294's stat-card dispatcher.

| Area | Estimated LOC |
|---|---:|
| Host provider defaults and copy | ~8 |
| Host provider tests | ~16 |
| Plan doc | ~91 |
| **Total** | **~115** |
