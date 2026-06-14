# PR-Deflection-Csv-Legacy-Fallback-Warning

## Why this slice exists

Issue #1551 is the next parser/encoding launch-readiness follow-up after #1545.
The #1545 legacy-fallback warning fixed the reviewer's exact `0xFF`/`0xFE`
corrupt-byte repro, but left the defect class open for other strict-UTF-8
failures that decode through CP1252/Latin-1 with implausible fallback
artifacts.

Root cause: the warning predicate is keyed to the output characters from the
cited repro (`U+00FF`/`U+00FE`) rather than the upstream condition that makes
the import suspicious: strict UTF-8 failed, UTF-8 recovery would put a
replacement character where the fallback produced an implausible character for
clean support-ticket text, and the legacy fallback does not show double-encoding
mojibake evidence. The review corrected an important overreach:
common CP1252 bytes such as accented letters, euro signs, and copyright signs
are byte-identical to arbitrary corrupt bytes at this layer, so they cannot be
deterministically separated without a charset detector. The most-upstream safe
fix in this slice is therefore the shared CSV decoder boundary with a narrow
implausible-artifact predicate, not a route-specific warning or a broad
"replacement at boundary" rule.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Replace the two-character legacy-fallback suspicion allowlist with a narrow
   predicate for implausible fallback artifacts.
2. Keep clean CP1252/Latin-1 support-ticket exports warning-free.
3. Add sibling implausible-byte fixtures and common-CP1252 rejection fixtures so
   the fix cannot pass only the original `0xFF`/`0xFE` examples or over-warn
   normal legacy exports.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-Csv-Legacy-Fallback-Warning.md`
- `tests/test_extracted_campaign_source_adapters.py`

### Review Contract

- Acceptance criteria:
  - ASCII UTF-8 CSV content with one implausible byte from the
    CP1252-undefined/C1-control set, at field end or mid-field, returns parsed
    rows plus `csv_encoding_ambiguous`.
  - The existing `0xFF` corrupt-byte warning remains covered.
  - Clean CP1252/Latin-1 exports, including marker-looking legitimate text and
    common field-ending characters such as accented letters, euro signs, and
    copyright signs, keep `warnings == ()`.
  - The warning predicate is expressed as the decoder class condition plus an
    implausible fallback artifact guard, not a broad boundary replacement rule.
- Affected surfaces:
  - Shared CSV decoding for campaign/deflection source loading.
  - Support-ticket upload warning propagation through existing callers.
- Risk areas:
  - Over-warning legitimate legacy encodings.
  - Under-warning silent mojibake for sibling corrupt bytes.
  - Reintroducing a cited-example-only character allowlist.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

## Mechanism

`_decode_utf8_error_csv_bytes` already has both views of the failed decode: the
legacy fallback text and a UTF-8 recovery string with replacement characters.
This slice changes `_legacy_fallback_corruption_warnings` to warn when recovery
contains replacement characters, the matching legacy fallback character is
implausible in clean text (`U+00FF`, `U+00FE`, or a C1 control from the Latin-1
fallback after CP1252 rejects an undefined byte), and the legacy fallback has no
mojibake marker score. Common CP1252 characters are intentionally not warning
signals because the decoder cannot distinguish a legitimate trailing CP1252 byte
from an arbitrary corrupt byte without a charset detector.

## Intentional

- No blanket "legacy fallback happened" warning. Clean CP1252/Latin-1 exports
  are supported input and remain warning-free.
- No broad boundary replacement warning. The review proved that it false-positives
  on normal CP1252 text.
- No blanket "replacement exists in UTF-8 recovery" warning. That would over-warn
  legitimate CP1252/Latin-1 words such as `cafe` with an accented byte or
  smart-quote contractions.
- No charset detector dependency. That remains the full answer for ambiguous
  common bytes, but it is outside this narrow hardening slice.
- No route-specific handling. `/deflection-reports/submit` already consumes this
  shared loader, so the decoder boundary is the correct upstream fix point.

## Deferred

- None.

Parked hardening: none.

## Verification

- Passed: python -m py_compile for the touched Python module and test file.
- Passed: focused legacy/corrupt CSV pytest selection (29 passed, 77 deselected).
- Passed: extracted campaign customer data tests (10 passed).
- Passed: extracted campaign source adapter tests (106 passed).
- Passed: bash scripts/check_ascii_python.sh.
- Passed: bash scripts/run_extracted_pipeline_checks.sh (4168 passed, 10 skipped).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 44 |
| `plans/PR-Deflection-Csv-Legacy-Fallback-Warning.md` | 111 |
| `tests/test_extracted_campaign_source_adapters.py` | 83 |
| **Total** | **238** |
