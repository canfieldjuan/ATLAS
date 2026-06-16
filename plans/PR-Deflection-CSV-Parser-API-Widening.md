# PR-Deflection-CSV-Parser-API-Widening

## Why this slice exists

Issue #1615 is the deferred third step from #1458. PR #1610 removed avoidable
full-file parser copies while preserving the tuple API. PR #1614 added safe
structured parser error UX. The remaining gap is contract shape: the parser can
stream internally, but CSV callers still receive only `(rows, warnings)`, so the
deflection submit path has to parse all rows before applying its row cap and
cannot represent parser-level source/count/truncation metadata directly.

This slice widens the CSV parser API at the narrowest correct boundary. Existing
tuple callers stay compatible, while deflection submit can request a capped CSV
load result that carries source row count, included row count, and truncation
metadata.

The synced diff is over the 400 LOC soft cap because the slice has to land the
parser result type, the deflection submit wiring, the review-finding fix, and
both adapter-level and route-level regression tests together. Splitting the
tests from the API widening would leave the new contract under-proven.

Review update: Codex thread #1342 found that the first version reused the
post-language row count as the source/truncation basis, hiding parser-cap
truncation when a non-English row appeared inside the capped prefix. This update
fixes the root by separating total parser source count, loaded included count,
post-language eligible count, submitted count, and explicit truncation count.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Production hardening

1. Add a structured CSV load result for source rows with `rows`, `warnings`,
   `source_row_count`, `included_row_count`, and `truncated_row_count`.
2. Preserve existing tuple APIs for successful parser callers.
3. Route deflection CSV upload/blob submits through the widened CSV result so
   the submit `limit` can cap rows during parse while retaining honest source
   row and truncation metadata.
4. Add focused parser/adapter and deflection-submit regressions for capped,
   uncapped, warnings-preserving, and compatibility behavior.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/campaign_customer_data.py`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-CSV-Parser-API-Widening.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_deflection_submit.py`

### Review Contract

- Acceptance criteria:
  - [ ] Existing CSV tuple callers still receive the same `(rows, warnings)`
        shape for uncapped successful parses.
  - [ ] A new CSV result API reports source row count, included row count, and
        truncated row count without including raw row content in metadata.
  - [ ] Deflection CSV submit applies `limit` at parse time and still reports
        source row count, submitted row count, and truncation warnings honestly.
  - [ ] CSV load warnings such as skipped prologue rows survive the widened API.
  - [ ] JSON and Zendesk full-thread submit behavior is unchanged.
- Affected surfaces: extracted package parser, source adapter API, deflection
  submit API, tests, CI enrollment.
- Risk areas: backcompat, parser correctness, user input safety, metadata
  truthfulness, performance.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R7, R10, R12, R14.

## Mechanism

Add a small dataclass for CSV source-row load results. The existing private CSV
loader can build that result while still backing the old tuple-returning helper.
The cap is parser-local: it counts every non-empty data row that passes CSV
shape validation, appends only the first requested rows, and reports truncation
from the difference between source and included row counts.

Expose the result through `campaign_source_adapters` with a CSV-specific helper
instead of widening JSON/JSONL loaders in the same PR. Deflection submit uses
the helper for CSV upload and blob paths. JSON/Zendesk full-thread paths keep
their current tuple shape. Submit diagnostics compute truncation from the parser
cap and post-filter submission separately, so language filtering cannot erase a
real parser cap or create a fake truncation warning by itself.

## Intentional

- This is CSV-only. JSON and Zendesk full-thread sources already have separate
  parser contracts and should not be pulled into this slice unless a later
  shared result type is warranted.
- The old tuple API remains in place. This avoids a broad migration of campaign
  generation, ingestion diagnostics, and other source-row callers.
- The result metadata is numeric only. It does not include row previews, field
  values, or customer content.

## Deferred

- A broader generic source-row result type for JSON/JSONL/full-thread imports is
  deferred until a caller actually needs shared metadata across formats.
- Further upload UX around parser caps is deferred; this slice only preserves
  existing diagnostics and makes their counts parser-backed.

Parked hardening: none.

## Verification

- `.venv/bin/python -m compileall -q extracted_content_pipeline/campaign_customer_data.py extracted_content_pipeline/campaign_source_adapters.py extracted_content_pipeline/api/control_surfaces.py` — passed.
- `.venv/bin/python -m pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_content_deflection_submit.py` — 195 passed after review fix.
- `bash` with `scripts/run_extracted_pipeline_checks.sh` — 4413 passed, 10 skipped.
- `bash` with `scripts/local_pr_review.sh` and the PR body file — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 147 |
| `extracted_content_pipeline/campaign_customer_data.py` | 41 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 51 |
| `plans/PR-Deflection-CSV-Parser-API-Widening.md` | 119 |
| `tests/test_extracted_campaign_source_adapters.py` | 47 |
| `tests/test_extracted_content_deflection_submit.py` | 133 |
| **Total** | **538** |
