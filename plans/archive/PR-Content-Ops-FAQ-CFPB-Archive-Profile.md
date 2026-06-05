# PR-Content-Ops-FAQ-CFPB-Archive-Profile

## Why this slice exists

The FAQ generator scale smoke now reports generic input profiles, but the CFPB support-ticket demo still hides the source-prep shape for its public complaint archive fetch. That makes it harder to see whether a CFPB run produced too few FAQ-ready rows because the upstream archive returned sparse rows, missing narratives, missing complaint IDs, or because the FAQ generator itself failed.

This slice adds CFPB-specific source profiling so archive extraction failures are acknowledged in the smoke artifact before the data enters the FAQ generator.

## Scope (this PR)

1. Add a profiled CFPB fetch helper that preserves the existing `fetch_cfpb_source_rows(...)` list-returning API.
2. Count scanned rows, usable rows, rows skipped for missing narratives, rows skipped for missing complaint IDs, and the fetch stop reason.
3. Include the CFPB source profile in the CFPB FAQ smoke JSON payload.
4. Add focused tests for the exporter profile and smoke payload.

### Files touched

- `plans/PR-Content-Ops-FAQ-CFPB-Archive-Profile.md`
- `scripts/export_content_ops_cfpb_sources.py`
- `scripts/smoke_content_ops_cfpb_faq_markdown.py`
- `tests/test_export_content_ops_cfpb_sources.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`

## Mechanism

`fetch_cfpb_source_rows_with_profile(...)` will share the same CFPB request parameters as `fetch_cfpb_source_rows(...)`, stream CSV rows once, and return `(rows, profile)`. The existing `fetch_cfpb_source_rows(...)` becomes a wrapper that returns only `rows`, so existing callers do not change.

The profile is derived from the raw CFPB row before conversion:

```python
if not complaint_id:
    skipped_without_id_count += 1
elif not narrative:
    skipped_without_narrative_count += 1
elif source_row:
    rows.append(source_row)
else:
    skipped_other_count += 1
```

The CFPB FAQ smoke uses the profiled helper and writes `source_profile` beside `source_rows`, `source_rows_path`, and the FAQ result.

## Intentional

- The FAQ generator is unchanged; this slice only makes the CFPB archive extraction stage observable.
- The existing `fetch_cfpb_source_rows(...)` API remains list-returning to avoid breaking current callers and tests.
- No live CFPB network dependency is added to tests; fixtures stay as tiny fake CSV payloads.
- The exporter CLI still emits JSONL rows only. The source profile is for programmatic smoke payloads, not a mixed stdout format.

## Deferred

- A future `PR-Content-Ops-FAQ-Archive-Runbook` can document how to interpret CFPB source profiles during large local archive runs.
- A future UI slice can surface source-profile counts on hosted artifacts if the portfolio app needs user-facing diagnostics.

## Verification

- `pytest tests/test_export_content_ops_cfpb_sources.py tests/test_smoke_content_ops_cfpb_faq_markdown.py` passed, 16 tests.
- `bash scripts/run_extracted_pipeline_checks.sh` passed, including 295 reasoning-core tests and 1,597 extracted Content Ops tests.
- `bash scripts/local_pr_review.sh origin/main` passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 69 |
| CFPB exporter | 76 |
| CFPB smoke wrapper | 7 |
| Tests | 109 |
| **Total** | **261** |
