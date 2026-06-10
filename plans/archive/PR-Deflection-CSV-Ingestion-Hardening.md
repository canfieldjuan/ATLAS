# PR-Deflection-CSV-Ingestion-Hardening

## Why this slice exists

#1384 calls out that the support-ticket CSV ingestion path is fragile on common
provider exports before launch: UTF-8 BOM can corrupt the first header,
Windows/Excel cp1252 bytes can crash parsing, semicolon-delimited exports can
collapse into one column, and leading metadata rows can silently produce a bad
report. #1386 lists CSV ingestion as a launch-readiness gate, so this slice
closes the parser-hardening P0s before moving to broader preview/product work.

## Scope (this PR)

Ownership lane: deflection/go-live
Slice phase: Production hardening

1. Harden the package-owned CSV dict loaders used by opportunity imports and
   source-row/support-ticket imports.
2. Decode UTF-8 BOM correctly, fall back for cp1252/latin-1 style exports, and
   detect common delimiters (`comma`, `semicolon`, `tab`, `pipe`) before
   `csv.DictReader` runs.
3. Fail loudly with a clear `ValueError` when a CSV has no usable header or rows
   contain more cells than the header, instead of silently returning misleading
   one-column/partial rows.
4. Add focused regression coverage for the internal loader and the public
   deflection submit handoff path.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-CSV-Ingestion-Hardening.md`
- `tests/test_extracted_campaign_customer_data.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_deflection_submit.py`

### Review Contract

Acceptance criteria:
- A UTF-8 BOM export maps the first header normally.
- A cp1252/latin-1 style export with smart punctuation loads without crashing.
- A semicolon-delimited support-ticket CSV reaches the deflection submit route
  as normal source rows.
- A leading metadata/header-mismatch CSV fails with a clear parse error instead
  of silently producing malformed rows.
- Existing JSON/JSONL parsing and quoted comma/multiline CSV behavior remain
  unchanged.

Affected surfaces:
- Extracted Content Pipeline CSV ingestion.
- ATLAS deflection submit route CSV parsing.

Risk areas:
- Delimiter detection must not break normal comma CSVs with quoted commas.
- Fail-loud header mismatch behavior changes formerly silent bad parses into
  `400`/`ValueError` failures.
- Extracted package ownership requires the package gauntlet before push.

Reviewer rules triggered: R1 Requirements match; R2 Test evidence; R3 Security/privacy; R5 Backward compatibility; R6 Error handling; R10 Maintainability.

## Mechanism

Add a shared private CSV dict-row loader in
`extracted_content_pipeline/campaign_customer_data.py`:

```python
text = _read_csv_text(path)          # utf-8-sig, then cp1252 fallback
dialect = csv.Sniffer().sniff(...)   # bounded delimiter set
reader = csv.DictReader(StringIO(text), dialect=dialect)
```

Both `campaign_customer_data._load_csv_rows(...)` and
`campaign_source_adapters._load_source_csv_rows(...)` use that helper so the
opportunity import path and the source-row/support-ticket path get the same
decode, delimiter, and header-mismatch behavior. The deflection submit route
already catches `ValueError` from `load_source_rows_from_file(...)` and returns
the existing bounded parse-error copy.

## Intentional

- No charset-detection dependency is added; cp1252 fallback with replacement is
  enough for the Windows/Excel byte patterns called out in #1384 without
  expanding dependencies.
- Header mismatch now fails closed. That can reject malformed unquoted-comma
  rows, but it prevents paid reports from being built on silently truncated
  ticket text.
- The shared helper remains private to the extracted package. This is parser
  hardening, not a new public API.

## Deferred

- #1384: HTML stripping for ticket bodies before clustering.
- #1384: sanitized real Zendesk/Intercom/Help Scout/Freshdesk export fixtures.
- #1384/#1386: making `inspect` a full pre-payment preview/validation gate.
- #1386: structural checkout authorization before charge.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_campaign_customer_data.py tests/test_extracted_campaign_source_adapters.py tests/test_extracted_content_deflection_submit.py -q`
  - passed; 101 tests cover BOM, cp1252 fallback, semicolon delimiter,
    quoted comma/multiline preservation, metadata-row fail-loud behavior, and
    deflection submit upload parsing.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-bodies/deflection-csv-ingestion-hardening.md`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 73 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 15 |
| `plans/PR-Deflection-CSV-Ingestion-Hardening.md` | 123 |
| `tests/test_extracted_campaign_customer_data.py` | 71 |
| `tests/test_extracted_campaign_source_adapters.py` | 44 |
| `tests/test_extracted_content_deflection_submit.py` | 38 |
| **Total** | **364** |
