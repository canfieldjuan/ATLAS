# PR-Content-Ops-Campaign-Export-Reasoning-Summary

## Why this slice exists

Recent Content Ops reasoning slices persist reasoning and generation usage
metadata on saved campaign drafts. The host-facing Postgres export already
includes the raw `metadata` JSON blob, but operators reviewing CSV/JSON exports
cannot scan reasoning usage or token totals without opening nested metadata.
This slice adds read-only summary fields to the export surface.

## Scope (this PR)

1. Derive export-only generation usage fields from each draft row's metadata.
2. Derive export-only reasoning summary fields from each draft row's metadata.
3. Cover JSON and CSV export behavior with focused tests.
4. Document the exported fields in the host-facing docs.

### Files touched

- `extracted_content_pipeline/campaign_postgres_export.py`
- `tests/test_extracted_campaign_postgres_export.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`list_campaign_drafts()` continues to read the same database columns. Before
rows are returned, `_serializable_row()` now adds derived keys from
`metadata.generation_usage` and `metadata.reasoning_context`:

- `generation_input_tokens`
- `generation_output_tokens`
- `generation_total_tokens`
- `generation_parse_attempts`
- `reasoning_context_used`
- `reasoning_wedge`
- `reasoning_confidence`

No schema change is needed; CSV output includes the same derived keys as
columns, and JSON output includes them on each row.

## Intentional

- The raw `metadata` blob remains in the export for hosts that need the full
  reasoning payload.
- `reasoning_context_used` is a boolean derived from `metadata.reasoning_context`
  being a non-empty object. It is not a run-level count; the export works at the
  draft-row level.
- Missing metadata yields empty generation fields and
  `reasoning_context_used=false` instead of filtering rows.

## Deferred

- Asset exports for blog posts, reports, landing pages, and sales briefs are
  outside this campaign-draft export helper.
- A richer analytics export over all generated assets can build on these field
  names later.

## Verification

- `python -m pytest tests/test_extracted_campaign_postgres_export.py` (12 passed)
- `python -m py_compile extracted_content_pipeline/campaign_postgres_export.py scripts/export_extracted_campaign_drafts.py tests/test_extracted_campaign_postgres_export.py`
- `bash scripts/run_extracted_pipeline_checks.sh` (1374 passed, 1 existing torch/pynvml warning)
- `git diff --check`
- Non-ASCII byte check for edited Python files (clean)

## Estimated diff size

About 6 files, under 200 changed lines.
