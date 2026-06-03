# PR-Deflection-Teaser-Locked-Dedupe

## Why this slice exists

PR #1288 changed the free snapshot teaser to expose the first eligible answer
by rank. The reviewer called out a pre-existing contract edge that now matters
for the portfolio render: if ranks 1 through `top_n` are not eligible and the
teaser falls through to a later rank, that same rank can also appear in
`locked_questions`.

That produces an internally inconsistent snapshot: one surface exposes the
rank's question and full answer while another marks the same rank as locked.
The product wants the teaser to be free; the locked/FOMO rows should not claim
that same full-answer rank is still withheld.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-teaser
Slice phase: Product polish

1. Build the teaser before `locked_questions` so the selected full-answer rank
   is known.
2. Exclude only the selected full-answer rank from `locked_questions` when it
   falls beyond `top_n`.
3. Preserve the existing fail-closed teaser eligibility gate and body-withheld
   previews.
4. Add a focused regression test for a fall-through teaser rank that would
   otherwise duplicate as locked.

### Files touched

- `plans/PR-Deflection-Teaser-Locked-Dedupe.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

`build_deflection_snapshot(...)` already delegates teaser eligibility to
`_snapshot_teaser(...)`, which returns either one `full_answer` object or
`None`. This slice stores that teaser payload once, extracts the selected
`full_answer.rank`, and filters that rank out of the locked-row projection:

```python
teaser = _snapshot_teaser(items, preview_count=teaser_preview_count)
teaser_full_rank = _teaser_full_answer_rank(teaser)
locked_questions = tuple(
    ...
    for rank, item in enumerate(items[top_n:], start=top_n + 1)
    if rank != teaser_full_rank
)
```

If there is no full teaser answer, the extracted rank is `None` and
`locked_questions` remains unchanged.

## Intentional

- This does not remove teaser previews from `locked_questions`. Previews remain
  body-withheld, so they do not contradict the locked-answer claim the way the
  full teaser answer does.
- This does not add a new payload field. The existing `full_answer.rank` is the
  real selection signal.
- This stays backend-only. Portfolio copy/rendering can use the cleaner
  snapshot payload in a separate frontend slice.
- Cross-layer caller hints were inspected. The control-surface API only
  forwards `build_deflection_snapshot(...)` output, and the docs example still
  covers the unchanged producer shape without a fall-through teaser rank.

## Deferred

- Portfolio copy must read the real `teaser.full_answer.rank` instead of
  hardcoding "#1"; that belongs to the portfolio renderer slice.
- Help-center article card styling remains deferred to the frontend renderer.
- Parked hardening: none. `HARDENING.md` has no current entry touching this
  ownership lane or files.

## Verification

Ran before push:

- `python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py` - passed
- `pytest tests/test_content_ops_deflection_report.py -q` - 36 passed
- `bash scripts/validate_extracted_content_pipeline.sh` - passed
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed
- `bash scripts/check_ascii_python.sh` - passed
- `bash scripts/run_extracted_pipeline_checks.sh` - reasoning core 295 passed; extracted content pipeline 3004 passed, 10 skipped, 1 warning
- `bash scripts/local_pr_review.sh --current-pr-body-file "$PR_BODY_FILE"` - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Snapshot builder | ~20 |
| Focused test | ~35 |
| **Total** | **~140** |
