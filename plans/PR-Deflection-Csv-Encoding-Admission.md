# PR-Deflection-Csv-Encoding-Admission

## Why this slice exists

Issue #1455 is a P0 launch-readiness parser gap: support-ticket CSV ingestion falls back from UTF-8 to CP1252 with `errors="replace"`, so common help-desk exports can silently become mojibake instead of producing trustworthy rows or a visible ingestion warning. The deflection paid funnel now routes CSV uploads through the shared `_load_csv_dict_rows` loader, so hardening that one boundary protects both standalone source loading and `/deflection-reports/submit`.

Diff budget note: post-review boundary coverage pushed the plan estimate over the 400 LOC soft cap. The overage is the focused decoder/test surface for the same byte-admission boundary: BOM-less UTF-16 inference, clean legacy marker text, warned UTF-8 recovery, and the ASCII-plus-one-corrupt-byte legacy fallback warning are coupled tightly enough that splitting them would leave #1455 partially fixed.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Harden the shared CSV text decoder so UTF-16/UTF-32 BOM exports decode explicitly, UTF-8 remains strict, CP1252/Latin-1 smart-text exports decode without replacement, and high Unicode replacement-character ratios surface a load warning instead of passing as clean input.
2. Prove the same decoder is used by the deflection submit upload path with byte-level fixtures, not only direct file-loader tests.

### Review Contract

- Acceptance criteria:
  - UTF-16 BOM CSV input parses into correct support-ticket rows.
  - CP1252/Latin-1 support-ticket CSV input with smart punctuation/accented text parses into correct Unicode text without `errors="replace"` mojibake.
  - A CSV containing a high ratio of Unicode replacement characters returns a visible `CampaignOpportunityWarning`.
  - The deflection submit upload loader accepts UTF-16 BOM bytes through the real temp-file path and returns the decoded rows.
  - Existing delimiter/header behavior is unchanged.
- Affected surfaces:
  - `extracted_content_pipeline/campaign_customer_data.py`
  - `tests/test_extracted_campaign_source_adapters.py`
  - `tests/test_extracted_content_deflection_submit.py`
- Risk areas:
  - Reintroducing silent replacement decoding.
  - Warning on normal CP1252/Latin-1 exports and making clean uploads look suspect.
  - Breaking existing semicolon/BOM and multiline CSV parsing.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-Csv-Encoding-Admission.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

`_read_csv_text` becomes a decoder boundary that returns `(text, warnings)`. It first honors explicit BOMs, then tries strict UTF-8, with a NUL-pattern check for UTF-16 files missing a BOM. If UTF-8 fails, the error path now runs the same raw-byte NUL inference before trying legacy encodings, so BOM-less UTF-16 with non-ASCII text is still admitted. It then compares strict CP1252/Latin-1 fallback with a warning-surfaced UTF-8 recovery; the recovery is selected only when mojibake evidence in the legacy decode outnumbers the replacements the recovery would insert, preserving clean legacy exports that contain legitimate U+00C3/U+00C2 text. Any selected UTF-8 recovery that inserts replacement characters emits a warning, even below the high-ratio threshold used for literal replacement characters already present in valid UTF-8 input. If recovery is not selected but the legacy fallback contains suspicious U+00FF/U+00FE artifacts after a strict UTF-8 failure, the loader returns the legacy-decoded text with a `csv_encoding_ambiguous` warning instead of silently treating that corrupt byte as clean CP1252. The existing `_load_csv_dict_rows` parser appends those decode warnings to the existing leading-row warnings before returning rows.

## Intentional

- This slice does not add full charset-normalizer/chardet dependency behavior. BOM/strict UTF-8/CP1252/Latin-1 covers the launch-critical help-desk export cases without introducing probabilistic detection into the extracted package.
- Clean CP1252/Latin-1 fallback does not warn. Those are common Windows export shapes; a warning is reserved for suspicious decoded content such as high replacement-character ratios.
- `errors="replace"` is used only in the explicit warned UTF-8 recovery branch for mostly-UTF-8 corrupt files. It is no longer the silent CP1252 fallback.
- Delimiter voting is deferred to #1459; this PR keeps the existing dialect detector behavior unless bytes could not be decoded safely.

## Deferred

- #1459 remains the deterministic delimiter-vote/reject branch.
- #1457 remains the header/body-column admission report and is currently claimed by another developer.
- Full charset confidence reporting can be added later if product wants a visible encoding diagnostic for every clean upload.
- Clean UTF-8 bytes that are semantically double-encoded mojibake can still decode without warning; that is a separate content-quality heuristic, not this byte-admission slice.
- Sub-10% NUL contamination that does not satisfy UTF-16 side-pattern inference is still deferred; a future suspicious-NUL warning band can close that without changing this launch-critical admission path.

Parked hardening: none.

## Verification

- `python -m py_compile` on `extracted_content_pipeline/campaign_customer_data.py`, `tests/test_extracted_campaign_source_adapters.py`, and `tests/test_extracted_content_deflection_submit.py` -- passed.
- `pytest tests/test_extracted_campaign_source_adapters.py -q -k "utf16 or legacy or marker or replacement or utf8 or corrupt or semicolon or multiline"` -- 12 passed, 69 deselected.
- `pytest tests/test_extracted_content_deflection_submit.py -q -k "utf16 or bom or embedded_quotes or metadata_header"` -- 4 passed, 50 deselected.
- `pytest tests/test_extracted_campaign_source_adapters.py -q` -- 81 passed.
- `pytest tests/test_extracted_content_deflection_submit.py -q` -- 54 passed.
- `bash` + `scripts/check_ascii_python.sh` -- passed.
- `bash` + `scripts/run_extracted_pipeline_checks.sh` -- 4127 passed, 10 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 204 |
| `plans/PR-Deflection-Csv-Encoding-Admission.md` | 81 |
| `tests/test_extracted_campaign_source_adapters.py` | 197 |
| `tests/test_extracted_content_deflection_submit.py` | 21 |
| **Total** | **503** |
