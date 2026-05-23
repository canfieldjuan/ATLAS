# PR-Content-Ops-FAQ-Lifecycle-Result-Summary

## Why this slice exists

The FAQ lifecycle smoke can persist and review generated FAQ Markdown, and the
scale smoke already has compact run-summary fields for large-upload triage. The
lifecycle smoke still writes only the full payload: generation details, draft
export rows, reviewed export rows, and errors. For a real 1,000-row database
run, that makes the artifact harder to scan and pushes operators back into
large nested payloads when they only need the source density, saved/exported row
counts, output-check status, and review status.

This slice adds a compact lifecycle summary beside the existing payload so the
next real-DB 1,000-row run has a stable inspection point.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a compact `lifecycle_summary` block to the FAQ lifecycle smoke result.
2. Include source/input-profile, saved count, export counts, output-checks, and
   error count without duplicating Markdown bodies.
3. Pin the summary for both successful 1,000-row lifecycle runs and fail-closed
   preflight failures.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Result-Summary.md`
- `scripts/smoke_content_ops_faq_lifecycle.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

The lifecycle script will build its existing payload as it does today, then
attach `lifecycle_summary = _lifecycle_summary(payload)`. The helper reads only
aggregate fields already present in the payload:

```python
{
    "status": "ok" if payload["ok"] else "failed",
    "source_rows": payload["source_rows"],
    "input_profile": payload["input_profile"],
    "saved_faq_count": len(payload["saved_ids"]),
    "draft_export_count": len(payload["draft_export"]["rows"]),
    "reviewed_export_count": len(payload["reviewed_export"]["rows"]),
    "output_checks": payload["generation"]["output_checks"],
}
```

Export row counts are derived through a tiny row-count helper so missing exports
stay visible as `None` instead of being confused with zero exported rows.

## Intentional

- This is result visibility only. It does not change database writes, review
  status updates, export behavior, console output, or pass/fail behavior.
- The summary intentionally duplicates small aggregate values already present in
  the full payload so large real-DB artifacts have a stable top-level readout.
- No live database run is added to CI; the existing fake async pool remains the
  test boundary.

## Deferred

- The real local DB 1,000-row lifecycle command remains the next manual test
  once a database URL is available in this checkout.
- A browser upload test remains deferred until the UI upload path is active.
- Root `HARDENING.md` was scanned; no parked FAQ-lane item is required for this
  slice to function.

## Verification

- Passed: `pytest tests/test_smoke_content_ops_faq_lifecycle.py -q` (9 passed)
- Passed: `scripts/run_extracted_pipeline_checks.sh` (1831 passed, 1 skipped)
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Lifecycle summary helper | ~45 |
| Lifecycle tests | ~30 |
| **Total** | **~150** |
