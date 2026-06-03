# PR: Content Ops Quote Card Package Handoff

## Why this slice exists

PR #1289 added the deterministic `quote_card` output and deliberately deferred
putting it into the review and competitive input-package defaults. Until this
handoff lands, operators can run `quote_card` explicitly, but the productized
review/competitive source modes still default only to landing pages, blog posts,
sales briefs, social posts, and ad copy.

This slice completes that handoff by making review and competitive source
packages request quote cards by default, using the same normalized
`source_material` already proven by #1289. It mirrors
`plans/PR-Content-Ops-Social-Post-Package-Handoff.md` and
`plans/PR-Content-Ops-Ad-Copy-Package-Handoff.md`, and does not overlap #1268's
separate output-variations lane.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input

Slice phase: Vertical slice

1. Add `quote_card` to the review source-mode default outputs.
2. Add `quote_card` to the competitive source-mode default outputs.
3. Align the package offer copy with the expanded output set.
4. Extend the existing host-provider tests to prove the default package outputs
   and execution handoff include `quote_card`.

### Files touched

- `plans/PR-Content-Ops-Quote-Card-Package-Handoff.md`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The host input provider already builds normalized `source_material` for review
and competitive source modes. `quote_card` consumes that same field, so this
slice only expands the two output tuples:

```python
_REVIEW_CAMPAIGN_OUTPUTS = (..., "social_post", "ad_copy", "quote_card")
_COMPETITIVE_CAMPAIGN_OUTPUTS = (..., "social_post", "ad_copy", "quote_card")
```

The execution tests use the existing `_RunnableService` fake in the
`quote_card` service slot. If the output is present but the dispatcher fails to
hand off `source_material`, the tests fail on the quote-card step kwargs.

## Intentional

- No new source adapter or generator logic. #1289 already proved `quote_card`
  runs from normalized source material and de-forked bundle handling onto the
  shared source-material adapter.
- No UI selector changes. The New Run source-mode path consumes these package
  defaults automatically.
- No persistence or generated-asset review branch. Durable quote-card review
  remains a separate productization slice after this default handoff.
- No changes to #1268 output variations; that open PR is a separate
  workflow/process lane and remains untouched.
- No `stat_card` output yet. Numeric-claim selection and validation should be
  its own slice.

## Deferred

- Persist generated quote-card drafts into the generated-assets review queue
  after the package-default handoff is accepted.
- Add `stat_card` as a follow-up output with numeric-claim validation.
- LLM/style/channel variants and #1268-style output variations remain future
  product polish.

## Parked hardening

None.

## Verification

- `pytest tests/test_atlas_content_ops_review_input_provider.py -q` (22 passed)
- `pytest tests/test_atlas_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py -q` (46 passed, 1 warning)
- `python -m py_compile atlas_brain/_content_ops_input_provider.py tests/test_atlas_content_ops_review_input_provider.py` (passed)
- `git diff --check` (passed)
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (passed; 144 matching tests enrolled)
- `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>` (passed; no non-diff caller references)

## Estimated diff size

Actual: 3 files, +114 / -2. This stays under the 400 LOC soft cap because it
reuses the existing source-mode package and #1289's quote-card dispatcher.

| Area | Estimated LOC |
|---|---:|
| Host provider defaults and copy | ~8 |
| Host provider tests | ~16 |
| Plan doc | ~90 |
| **Total** | **~114** |
