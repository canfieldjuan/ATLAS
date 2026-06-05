# PR: Blog Repair Failure Diagnostics

## Why this slice exists

PR #847 made the Content Ops blog repair loop debuggable enough to distinguish
exception, unparseable repair, and still-blocked quality outcomes. The live
Haiku smoke still did not save a draft, though, and the final blocker was
`geo_citation_safety_failed`. The current error payload reports blocker codes
but not the final generated candidate, so the next operator cannot tell whether
the model produced a placeholder link, unsupported claim, bad chart reference,
or another citation-safety trigger.

This slice adds bounded failed-candidate diagnostics to the existing error
payload instead of saving a bad draft or guessing with more prompt edits.

## Scope (this PR)

Ownership lane: content-ops/blog-repair-failure-diagnostics

1. Attach a bounded `failed_candidate` snapshot when a parsed blog draft remains
   quality-blocked after repair attempts.
2. Attach the same snapshot when a repair response is unparseable, using the
   candidate that was sent into the failed repair.
3. Keep diagnostics small and structured: metadata, word count, parse/repair
   attempts, and bounded head/tail content excerpts.
4. Add focused tests for both diagnostic paths.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Repair-Failure-Diagnostics.md` | Plan doc for this slice. |
| `extracted_content_pipeline/blog_generation.py` | Add failed-candidate snapshot helper and wire it into repair failure errors. |
| `tests/test_extracted_blog_generation.py` | Assert diagnostics are present and bounded for quality-blocked and repair-unparseable paths. |

## Mechanism

The failed-candidate snapshot helper extracts a small diagnostic object from the
parsed candidate:

- `title`, `slug`, `seo_title`, `target_keyword`, `topic_type`
- `word_count`
- `generation_parse_attempts`, `generation_quality_repair_attempts`
- `content_excerpt_head`, `content_excerpt_tail`, `content_truncated`

The generator attaches that object to:

- `quality_blocked` after all repair attempts parse but still fail the gate.
- `quality_repair_unparseable` when the repair LLM response does not parse, so
  the operator can inspect the candidate that was sent into the failed repair.

The helper does not save the failed draft and does not include the full body.

## Intentional

- No quality-gate change. The gate still decides whether a draft can be saved.
- No live prompt iteration in this slice. The point is to make the next live
  run explain the failure.
- No DB persistence for failed candidates. The diagnostics live in the execution
  result only.

## Deferred

- Running the Haiku live smoke after this lands is the next slice. It should use
  `/tmp/atlas-haiku-override.env` and inspect `failed_candidate` if the draft is
  still blocked.
- Parked hardening: existing `ATLAS-HARDENING.md` items are for older deep-dive
  blog generator/content issues and do not touch this Content Ops blog
  diagnostics path. They remain parked.

## Verification

- pytest `tests/test_extracted_blog_generation.py` -q -> 33 passed.
- bash `scripts/validate_extracted_content_pipeline.sh` -> passed.
- python `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` extracted_content_pipeline -> passed.
- python `scripts/audit_extracted_standalone.py` --fail-on-debt -> passed.
- bash `scripts/check_ascii_python.sh` -> passed.
- bash `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Generator | ~45 |
| Tests | ~35 |
| **Total** | **~160** |
