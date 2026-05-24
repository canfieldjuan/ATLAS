# PR-Support-Ticket-Provider-Package-Scale-Cap

## Why this slice exists

The support-ticket provider can feed landing-page and blog generation, and the
next production risk is row volume before model calls. The provider package has
a `max_rows` cap, but we need a concrete regression test and validation record
showing that larger support-ticket-shaped exports are counted honestly and
truncated before they reach generation inputs.

This slice is independent of the package-smoke CLI PR. It uses the existing
`build_support_ticket_input_package(...)` path directly so it can ship without
stacking on #934.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Robust testing

1. Add regression coverage proving the package default accepts at most 1,000
   rows into `source_material`.
2. Assert oversized input reports original source count, included count,
   skipped count, truncated count, and warning metadata consistently.
3. Run direct package validation against the existing local CFPB-derived 1,000
   and 10,000 row support-ticket-shaped artifacts.
4. Record the validation command, counts, timing, and memory in the extraction
   validation trail.

### Files touched

- `tests/test_extracted_support_ticket_input_package.py`
- `docs/extraction/validation/support_ticket_provider_package_scale_cap_2026-05-24.md`
- `plans/PR-Support-Ticket-Provider-Package-Scale-Cap.md`

## Mechanism

The package builder already accepts loaded rows and enforces `max_rows` before
normalization. The new test builds 1,005 valid ticket rows without passing a
custom limit, then verifies:

- `source_row_count == 1005`
- `included_ticket_row_count == 1000`
- `truncated_ticket_row_count == 5`
- `source_material` contains exactly 1,000 rows
- the truncation warning carries the same count fields

The validation run uses the local CFPB-derived JSONL files already used by the
FAQ scale proofs, but only calls the package builder. No FAQ generation, DB, or
LLM path is exercised here.

## Intentional

- No dependency on the package-smoke CLI PR, because #934 is still under review.
- No checked-in source artifact. The 1,000/10,000 row files are existing ignored
  local validation artifacts, and the committed doc records the command and
  results.
- No change to the default cap in this slice. This proves the current default
  behavior before we decide whether hosted uploads need a lower user-facing
  limit or a background job path.

## Deferred

- Future PR: when #934 lands, rerun the same real-file validation through the
  package-smoke CLI and compare the summary output.
- Future PR: hosted upload/intake can surface the truncation warning to users
  instead of only keeping it in package warnings.
- Parked hardening: none.

## Verification

- Direct package validation against local 1,000-row and 10,000-row CFPB-derived
  support-ticket-shaped JSONL files - passed; 10,000 rows truncated to 1,000
  included rows with 9,000 reported truncations.
- python -m py_compile for `tests/test_extracted_support_ticket_input_package.py` - passed.
- pytest for `tests/test_extracted_support_ticket_input_package.py` - 18 passed.
- validate_extracted_content_pipeline.sh - passed.
- forbid_atlas_reasoning_imports.py for `extracted_content_pipeline` - passed.
- audit_extracted_standalone.py with fail-on-debt - passed.
- check_ascii_python.sh - passed.
- run_extracted_pipeline_checks.sh - 1952 passed, 1 skipped.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Validation doc | ~100 |
| Test | ~35 |
| **Total** | **~215** |
