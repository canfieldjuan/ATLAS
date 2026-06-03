# PR-Deflection-Teaser-Top-Answer

## Why this slice exists

Issue #1286 revises the snapshot teaser selection rule. The shipped backend
keeps the top three eligible drafted answers locked by selecting the first
eligible answer at rank 4 or later, falling back to the last eligible answer
when fewer than four exist. The operator decision in #1286 supersedes that:
show the first eligible drafted answer by rank, intentionally exposing the
most-asked answer when it has scoped resolution evidence.

The reason is product clarity. The free sample should be the most relevant
answer, not the first answer outside a top-N paywall rule. The buyer already
sees the cost/bleed story; scarcity of one drafted answer is less persuasive
than a directly relevant sample.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-teaser
Slice phase: Product polish

1. Change `_select_full_teaser_item` to return the first eligible teaser item
   by rank.
2. Preserve the existing fail-closed eligibility gate: only scoped
   `resolution_evidence` items with an answer body can become the full teaser.
3. Preserve the existing body-withheld preview shape and preview count.
4. Update focused snapshot tests for the new rank-first behavior and the
   fallback when rank 1 has no eligible answer.

### Files touched

- `plans/PR-Deflection-Teaser-Top-Answer.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`

## Mechanism

`_snapshot_teaser(...)` already builds `eligible` in ranked item order:

```python
eligible = tuple(
    (rank, item)
    for rank, item in enumerate(items, start=1)
    if _is_teaser_eligible(item)
)
```

This slice makes `_select_full_teaser_item(eligible)` return `eligible[0]`.
That means:

- rank 1 is exposed when it has scoped resolution evidence and answer text;
- if rank 1 is not eligible, the next eligible ranked answer is exposed;
- if no item is eligible, the existing empty teaser path remains unchanged.

## Intentional

- No "most-resolved" or "confidence" payload is added. The operator explicitly
  chose rank-first selection, and adding a new explanation signal would imply a
  different selection basis.
- The snapshot still exposes at most one full answer; previews remain
  body-withheld.
- This is ATLAS backend selection only. Portfolio article formatting and copy
  can consume the unchanged teaser payload in a separate frontend slice.

## Deferred

- Portfolio copy such as "Sample answer for your #1 most-asked question" is
  deferred to the portfolio renderer because this PR does not touch
  atlas-portfolio.
- Help-center article card styling is deferred to the frontend renderer; the
  backend already emits verbatim question, answer, and structured steps.
- Parked hardening: none. `HARDENING.md` has no current entry touching this
  ownership lane or files.

## Verification

Ran before push:

- `python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_extracted_content_ops_live_execute_harness.py` - passed
- `pytest tests/test_content_ops_deflection_report.py tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n -q` - 36 passed
- `pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_returns_snapshot_for_unpaid_deflection_report -q` - 6 passed
- `bash scripts/validate_extracted_content_pipeline.sh` - passed
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed
- `bash scripts/check_ascii_python.sh` - passed
- `bash scripts/run_extracted_pipeline_checks.sh` - reasoning core 295 passed; extracted content pipeline 3003 passed, 10 skipped, 1 warning
- `bash scripts/local_pr_review.sh --current-pr-body-file "$PR_BODY_FILE"` - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Teaser selector | ~5 |
| Focused tests | ~45 |
| **Total** | **~135** |
