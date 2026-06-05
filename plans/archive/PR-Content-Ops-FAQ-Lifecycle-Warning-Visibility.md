# PR-Content-Ops-FAQ-Lifecycle-Warning-Visibility

## Why this slice exists

The 1,000-row FAQ lifecycle scale test surfaced a useful failure mode: the
lifecycle smoke correctly fail-closes when source normalization emits warnings,
but the payload only says warnings occurred. For large uploads, that hides the
actual reason, such as missing vendor metadata across every row. This slice adds
compact warning visibility at the lifecycle smoke source, without weakening the
fail-closed gate.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add normalization warning diagnostics to the FAQ lifecycle smoke payload.
2. Include warning-code counts in the fail-closed error message.
3. Add tests for both the clean path and a missing-metadata warning failure.
4. Add the lifecycle smoke script and test to the extracted-checks workflow
   path filters so this PR shape self-triggers the suite in CI.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Warning-Visibility.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/smoke_content_ops_faq_lifecycle.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

The lifecycle smoke already receives structured warning dictionaries from
`load_source_campaign_opportunities_from_file(...)`. This slice summarizes those
warnings as:

```python
{
    "warning_count": 1000,
    "warnings_by_code": {"missing_vendor_name": 1000},
    "warning_sample": [...],
    "warnings_truncated": True,
}
```

The existing `--allow-ingestion-warnings` behavior stays unchanged.

## Intentional

- This is not a workaround for bad source rows. The lifecycle smoke still fails
  unless `--allow-ingestion-warnings` is explicitly supplied.
- The payload remains compact by keeping only the first ten warnings.

## Deferred

- Raw-row profiling parity with the scale smoke is deferred; this slice only
  addresses the warning visibility gap surfaced during lifecycle testing.
- Live database execution remains deferred because this environment has neither
  `EXTRACTED_DATABASE_URL` nor `DATABASE_URL` set.

## Verification

- `pytest tests/test_smoke_content_ops_faq_lifecycle.py -q` - 6 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 1827 passed, 1 skipped.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Workflow path filters | ~5 |
| Lifecycle warning summary | ~35 |
| Lifecycle warning tests | ~35 |
| **Total** | **~145** |
