# PR-Deflection-Parser-CSV-Field-Limit

## Why this slice exists

#1463 still lists a parser/admission hardening gap: Python's default
`csv.field_size_limit` (~128 KiB) crashes on one long quoted HTML/body cell.
I reproduced that on current `origin/main` through the real
`inspect_ingestion_file(..., source_rows=True, source_format="csv")` path with
a 200 KiB ticket body: `_csv.Error: field larger than field limit (131072)`.

Root cause: the shared CSV reader boundary in
`extracted_content_pipeline/campaign_customer_data.py` inherits Python's
process-default field limit before delimiter scoring and row loading, so valid
support-ticket exports with large body/comment cells fail before the parser
can return the admission diagnostics #1467 requires.

This PR fixes the root in safe scope by raising the CSV parser field ceiling
at the shared reader boundary before any `csv.reader` is constructed. It is
not a downstream symptom fix in `ingestion_diagnostics`: every CSV consumer
that goes through the shared loader gets the larger ticket-body ceiling.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Robust testing

1. Raise the shared CSV parser field-size ceiling deliberately before delimiter
   scoring and source-row loading.
2. Add focused parser/admission tests proving long quoted body cells no longer
   crash and still produce the expected source-row admission outcome.
3. Keep this slice scoped to long-cell parser survivability; do not add the
   deferred low-coverage reject threshold or streaming/memory redesign.

### Review Contract

Acceptance criteria:
- A source-row CSV with a body/comment cell larger than Python's default
  128 KiB field limit is parsed by the real inspect/admission path instead of
  raising `_csv.Error`.
- The accepted long-cell fixture still yields a normal source-row admission
  decision, not a swallowed exception or generic failure.
- The field-limit helper only raises the process limit upward; it does not
  lower an operator/runtime limit that is already higher.
- Existing CSV diagnostics behavior for normal, zero-usable, and partial
  coverage fixtures remains unchanged.

Affected surfaces:
- `extracted_content_pipeline/campaign_customer_data.py` shared CSV reader
  setup.
- Source-row ingestion diagnostics exercised through
  `extracted_content_pipeline.ingestion_diagnostics.inspect_ingestion_file`.

Risk areas:
- Global `csv.field_size_limit` mutation must be monotonic and local to the
  parser setup.
- Delimiter scoring and row loading both construct `csv.reader`; both must be
  covered by the raised ceiling.
- The fix must not mask malformed CSV dialect/header errors as successful
  parses.

Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-Parser-CSV-Field-Limit.md`
- `tests/test_extracted_campaign_customer_data.py`
- `tests/test_extracted_content_ingestion_diagnostics.py`

## Mechanism

Add a small helper in `campaign_customer_data.py` that reads the current
`csv.field_size_limit()` and raises it to a named support-ticket ceiling when
the current value is lower. Call it before the module's `csv.reader`
constructions used by delimiter scoring and CSV row loading.

Tests exercise the public inspect path rather than the helper alone:

```python
report = inspect_ingestion_file(path, source_rows=True, source_format="csv")
assert report.as_dict()["source_row_admission"]["admission_decision"] == {"status": "ACCEPT"}
```

A helper-level test pins the monotonic behavior by monkeypatching
`csv.field_size_limit`, proving the helper does not lower an already-higher
runtime setting.

## Intentional

- This PR raises the parser ceiling instead of rejecting large cells. Large
  ticket bodies are valid customer support exports; rejecting at 128 KiB would
  treat a Python default as product policy.
- This PR does not introduce streaming/chunked CSV parsing. #1458 remains the
  broader memory/streaming track; this slice only removes the immediate
  parser-limit crash with bounded, focused tests.
- The long-cell test uses synthetic adversarial CSV data because the failure
  is deterministic and tied to Python's parser limit, not to a provider-specific
  threshold policy.

## Deferred

- #1458 streaming upload memory hardening remains separate.
- #1467 low non-zero usable-ratio reject threshold remains blocked on real
  partial-provider evidence.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_content_ingestion_diagnostics.py tests/test_extracted_campaign_customer_data.py -q` -- 32 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 4612 passed, 10 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 14 |
| `plans/PR-Deflection-Parser-CSV-Field-Limit.md` | 121 |
| `tests/test_extracted_campaign_customer_data.py` | 32 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 37 |
| **Total** | **204** |
