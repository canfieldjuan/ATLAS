# PR-Deflection-Parser-JSONL-Line-Diagnostics

## Why this slice exists

#1463 names a parser/admission hardening gap: the guarded JSONL path aborts the
whole upload when one line is malformed. The observed root is in
`campaign_source_adapters._load_source_rows(..., file_format="jsonl")`: it
calls `json.loads(...)` for each non-empty line and lets `JSONDecodeError`
escape, so a single bad source row prevents every other valid row from reaching
the normal source-row admission diagnostics.

This fixes the root at the JSONL loader boundary. The loader should treat
malformed JSONL records like bad rows, not like a bad file: skip that one line,
emit a precise non-fatal warning with the line number, and continue converting
valid rows. This is not a CSV threshold-policy change and does not alter JSON
array/object loading.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Production hardening

1. Add a JSONL malformed-line warning for non-empty lines that fail JSON parse.
2. Preserve valid JSONL rows before/after the malformed line.
3. Surface the warning through source-row diagnostics and the source-row CLI.
4. Preserve physical JSONL line numbers for downstream row-level warnings.
5. Fail closed in the warning-dropping raw-row helper when malformed JSONL
   warnings would otherwise be discarded.
6. Add negative/near-miss tests: one malformed line is skipped, blank lines are
   still ignored, and a fully valid JSONL file stays warning-free.

### Review Contract

Acceptance criteria:
- One malformed JSONL line no longer aborts
  `load_source_rows_with_warnings_from_file`.
- Valid JSONL rows before and after a malformed line are preserved and
  converted.
- The emitted warning includes code, row/line index, and a bounded message that
  does not echo raw source content.
- `inspect_ingestion_file(..., source_rows=True, source_format="jsonl")`
  exposes the warning in `warning_counts` and `warnings`.
- Downstream validation warnings, such as `missing_contact_email`, report the
  physical JSONL line number after skipped malformed lines.
- The legacy raw-row helper does not silently drop malformed-line warnings.
- Valid JSONL and blank-line-only gaps remain warning-free.
- CSV admission and low non-zero coverage policy remain unchanged.

Affected surfaces:
- Extracted source-row JSONL loader, opportunity warning row-index propagation,
  source adapter tests, and ingestion diagnostics tests.

Risk areas:
- Hiding a wholly corrupt JSONL file as a clean zero-row upload.
- Leaking raw customer text in parse-error messages.
- Regressing valid JSONL loading or CSV loader behavior.

Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-Parser-JSONL-Line-Diagnostics.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_ingestion_diagnostics.py`

## Mechanism

- Keep the existing JSONL resolution branch in `_load_source_rows`.
- For each non-empty line, parse with `json.loads`.
- On `JSONDecodeError`, append a `CampaignOpportunityWarning` with a new
  `malformed_jsonl_line` code, `row_index` equal to the physical 1-based line
  number, `field="jsonl"`, and a concise parser message.
- Continue parsing subsequent lines and return `(rows, warnings)`.
- The existing conversion path already concatenates loader warnings with
  row-conversion warnings, so diagnostics/CLI output should pick up the new
  warning without a second reporting path.
- Carry the physical JSONL line number on an internal dict subclass attribute
  so validation warnings emitted after source-row conversion still point to the
  input line, without adding synthetic keys to returned opportunities.
- Make `load_source_rows_from_file(...)` raise on `malformed_jsonl_line` because
  that legacy helper returns only rows and would otherwise discard the warning.

## Intentional

- No hard REJECT envelope for JSONL in this PR. #1467's structured
  `source_row_admission` object is currently CSV-specific, and extending that
  envelope to JSONL is a separate product/API shape decision.
- No raw line echo in warning messages; parse diagnostics can name the line and
  parser reason without committing or logging customer text.
- Empty/whitespace-only JSONL lines remain ignored without warning, matching
  current behavior.

## Deferred

- #1467 low non-zero CSV reject threshold remains blocked on real partial
  provider evidence.
- A broader JSONL admission envelope can be added later if JSONL becomes a
  buyer-facing upload format that needs ACCEPT/REJECT parity with CSV.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_content_ingestion_diagnostics.py -q`
  - 155 passed in 1.34s.
- `./scripts/run_extracted_pipeline_checks.sh`
  - 4590 passed, 10 skipped, 1 warning in 75.01s.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 12 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 56 |
| `plans/PR-Deflection-Parser-JSONL-Line-Diagnostics.md` | 120 |
| `tests/test_extracted_campaign_source_adapters.py` | 109 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 54 |
| **Total** | **351** |
