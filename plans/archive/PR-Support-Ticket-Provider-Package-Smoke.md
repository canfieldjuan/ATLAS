# PR-Support-Ticket-Provider-Package-Smoke

## Why this slice exists

The support-ticket input provider now feeds real ticket context into landing-page
and blog generation, and the live smoke proves the LLM/persistence path can save
drafts. The next cheapest validation point is before any DB write or model call:
given an uploaded/exported ticket file, can we package the rows into the exact
Content Ops inputs the generators consume?

This slice adds that pre-LLM proof path. It stays in the support-ticket provider
lane and does not add FAQ-owned generation, file-upload routing, persistence, or
hosted background execution.

This is slightly over the 400 LOC soft budget because the useful slice needs the
CLI, CI enrollment, and representative fixture-style tests together. Shipping
the CLI without tests, or tests without the CI enrollment, would not prove the
pre-LLM package path.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Functional validation

1. Add a small public source-row file loader that exposes the existing CSV/JSON/JSONL loader without changing normalization rules.
2. Add a support-ticket package smoke CLI that loads a file, builds the real support-ticket input package, and emits a JSON readiness summary.
3. Include row counts, skipped/truncated counts, source period, window-filter status, FAQ question count, top clusters, customer wording examples, and warnings in the summary.
4. Add focused tests for undated CSV rows, dated rows, skipped rows, truncation, remaining clusters, uncategorized rows, and CLI failure on empty packaged rows.
5. After PR #919 merged, update its support-ticket landing execute assertion to
   match the current neutral source-period rule for undated ticket exports.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `scripts/smoke_content_ops_support_ticket_package.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_smoke_content_ops_support_ticket_package.py`
- `tests/test_support_ticket_provider_landing_blog_execute.py`
- `plans/PR-Support-Ticket-Provider-Package-Smoke.md`

## Mechanism

The CLI uses the existing source-row file reader, then calls
`build_support_ticket_input_package(...)`. It does not reimplement the package
rules. The emitted JSON summary is a compact view of the package fields that
matter before generation:

```json
{
  "source_row_count": 10,
  "included_ticket_row_count": 9,
  "skipped_ticket_row_count": 1,
  "truncated_ticket_row_count": 0,
  "source_period": "Uploaded support tickets",
  "has_window_filter": false,
  "top_ticket_clusters": []
}
```

`--require-included-rows` exits non-zero if the file loads but no usable customer
wording survives packaging, so hosted or local operators can catch bad exports
before spending model calls.

## Intentional

- No LLM, DB, or generation call in this slice. This is the cheap validation
  layer before those paths.
- No new file-upload route. The hosted ingestion session owns upload and storage
  behavior; this slice validates the provider package after rows are available.
- The summary exposes package outputs instead of raw file contents so it does not
  become a second parser or policy surface.
- The #919 test correction is included because rebasing this PR onto current
  main exposed that the merged test still expected a dated 90-day source period
  for an undated CSV fixture.

## Deferred

- Future PR: hosted upload/intake code can call this same package summary for
  preflight diagnostics once that owning route is ready.
- Future PR: scale validation can run this CLI against larger ticket exports and
  record memory/runtime once the representative customer file is selected.
- Parked hardening: none.

## Verification

- python -m py_compile for `extracted_content_pipeline/campaign_source_adapters.py`, `scripts/smoke_content_ops_support_ticket_package.py`, and `tests/test_smoke_content_ops_support_ticket_package.py` - passed.
- pytest for `tests/test_smoke_content_ops_support_ticket_package.py` and `tests/test_extracted_support_ticket_input_package.py` - 22 passed.
- package smoke against `extracted_content_pipeline/examples/support_ticket_sources.csv` with max rows 5 - passed; 4 included rows, 2 clusters, zero warnings.
- pytest for `tests/test_smoke_content_ops_support_ticket_package.py`, `tests/test_extracted_support_ticket_input_package.py`, and `tests/test_support_ticket_provider_landing_blog_execute.py` - 24 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 1956 passed, 1 skipped.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Public loader export | ~10 |
| Smoke CLI | ~110 |
| Test enrollment | ~1 |
| Tests | ~191 |
| **Total** | **~445** |
