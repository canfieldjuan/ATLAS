# PR-Content-Ops-FAQ-File-Input-Scale-Smoke

## Why this slice exists

The hosted FAQ path now has 1,000-row coverage for direct
`inputs.source_material` lists and for one-level `source_material.support_tickets`
bundles. The remaining input/output testing gap is the file-backed path that
customers are most likely to use for SMB and mid-market ticket exports. This
slice proves a 1,000-row source file flows through the existing FAQ scale-smoke
script, source-file loader, CLI generator, artifact writer, and compact result
payload.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a 1,000-row JSON bundle file smoke to the existing FAQ scale-smoke tests.
2. Assert the input profile reports 1,000 raw and usable rows with no skipped
   rows or missing source text.
3. Assert the FAQ result and run summary preserve 1,000 rendered ticket sources,
   pass output checks, and write the expected Markdown/result artifacts.

### Files touched

- `plans/PR-Content-Ops-FAQ-File-Input-Scale-Smoke.md`
- `tests/test_smoke_content_ops_faq_scale_run.py`

## Mechanism

The test writes a temporary JSON file in the same bundle shape supported by the
source adapter:

```json
{"support_tickets": [{... 1000 rows ...}]}
```

It then calls `run_scale_smoke(...)`, which shells out to
`scripts/build_extracted_ticket_faq_markdown.py`. That CLI loads the file through
`load_source_campaign_opportunities_from_file(...)`, generates FAQ Markdown, and
writes `faq.md`, `faq_result.json`, and `run_summary.json`.

## Intentional

- This is test-only. The production code already has the source-file loader and
  FAQ CLI path; this slice locks the 1,000-row behavior instead of creating a new
  ingestion path.
- JSON bundle is used because it exercises the file-backed bundle parser and the
  FAQ scale-smoke artifact contract in one cheap test. Existing small tests
  already cover CSV, JSON array, and JSONL format selection.

## Deferred

- Browser upload coverage remains deferred until the UI upload path is active;
  this slice proves the file artifact once it reaches the backend smoke/CLI
  boundary.
- Real database persistence remains in the lifecycle smoke lane rather than this
  artifact-focused unit test lane.

## Verification

- `pytest tests/test_smoke_content_ops_faq_scale_run.py -q` - 19 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 1819 passed, 1 skipped.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| File-backed 1,000-row scale test | ~45 |
| **Total** | **~115** |
