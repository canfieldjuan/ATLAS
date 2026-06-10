# PR-Deflection-Full-Volume-Submit-Limit

## Why this slice exists

#1440 needs a real full-volume deflection funnel proof using a messy CFPB export,
not the tiny golden upload. The local 50k CFPB JSONL is 106 MB, so the operator
asked to cut the file to the existing 50 MB upload guard instead of treating the
byte cap as a blocker. That still leaves a correctness problem: the hosted
deflection submit route and smoke script clamp the execution to 1000 rows, so a
near-50 MB real upload would silently test only the first 1000 rows.

This slice removes that deflection-submit row clamp while preserving the 50 MB
upload guard, so #1440 can run a deterministic real-export sample through the
hosted intake, snapshot, snapshot email/PDF, payment, and full report
email/PDF path. The CFPB prep also exposed a parser brittleness class: valid
messy CSV with embedded quotes can fail before any report is built. This slice
therefore hardens the submit CSV parser enough to ingest messy high-volume CSV
without turning #1440 into a hand-sanitized-file exercise.

This PR exceeds the 400 LOC soft cap because the row-cap change and CSV
hardening now ship together with failure-first fixtures for embedded quotes,
provider prologue rows, support-export headers, missing optional fields,
mixed-language filtering, and all-non-English fail-closed behavior. Splitting
after #1452 was already open would leave the high-volume proof still dependent
on hand-sanitized CSV.

## Scope (this PR)

Ownership lane: deflection-full-50k-e2e-proof
Slice phase: Functional validation

1. Allow the deflection submit route to use every parsed support-ticket row
   from a CSV/blob that already passed the 50 MB submit guard unless the caller
   explicitly supplies a smaller limit.
2. Keep the general `/execute` FAQ source-material guard unchanged for non-submit
   paths.
3. Update the hosted submit smoke so a near-50 MB real export can omit the
   legacy 1000-row default or explicitly request more than 1000 rows while the
   smoke still validates the reported diagnostics.
4. Add focused regression coverage proving large deflection submit payloads are
   not truncated at 1000 rows while explicit smaller limits still truncate.
5. Harden CSV dialect parsing so valid CSV with embedded quotes/newlines does
   not fail because `csv.Sniffer()` guessed non-standard quote behavior.
6. When a submit CSV includes a language column, keep English rows and skip
   non-English rows before packaging so the internal Hugging Face proof remains
   an English report-quality test, not a multilingual clustering test.
7. Keep provider metadata/prologue rows accepted only when a plausible support
   export header follows, and keep malformed/ragged rows fail-loud.
8. Keep existing provider full-thread fixtures usable by recognizing common
   support export headers such as Intercom conversation fields.

### Review Contract

- Acceptance criteria:
  - [ ] Deflection submit defaults to all rows parsed from the accepted upload,
        not `_MAX_INGESTION_ROWS`.
  - [ ] Explicit smaller `limit` values still reduce submitted rows and report
        truncation diagnostics.
  - [ ] The 50 MB upload/blob guard remains in place.
  - [ ] Non-submit FAQ `/execute` row-limit behavior remains unchanged.
  - [ ] The hosted smoke preflight accepts no explicit limit by default and can
        accept a limit suitable for the near-50 MB #1440 CFPB sample.
  - [ ] Valid messy CSV with embedded quotes/newlines parses without requiring
        hand-normalized quote replacement.
  - [ ] Provider title/prologue rows before a plausible header parse; malformed
        extra-cell rows still fail loudly.
  - [ ] Hugging Face-shaped rows (`subject`, `body`, `answer`, `language`) can
        be submitted with missing IDs/categories, use `answer` as resolution
        evidence, and filter non-English rows.
- Affected surfaces: API, file upload validation, hosted smoke script, tests.
- Risk areas: performance, backward compatibility, file-upload abuse, diagnostic
  truthfulness.
- Reviewer rules triggered: R1, R2, R3, R5, R7, R10, R12.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-Full-Volume-Submit-Limit.md`
- `scripts/build_deflection_messy_csv_fixtures.py`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_build_deflection_messy_csv_fixtures.py`
- `tests/test_extracted_campaign_customer_data.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

Introduce a deflection-submit-specific row ceiling derived from the submit byte
guard for explicit limit validation, and use the parsed file size as the real
per-upload row cap. The submit route chooses `raw_row_count` as the default max
so accepted uploads run at their full parsed size; an explicit lower limit
continues to slice the rows and flow through the existing
`build_support_ticket_input_package` truncation metadata.

CSV dialect detection still uses `csv.Sniffer()` for delimiter support, but the
returned dialect is projected into a hardened dialect that preserves delimiter
choice while forcing standard doubled-quote handling. That keeps semicolon/tab
fixtures working while preventing valid quoted text from becoming false
"more cells than header" failures.

CSV header detection now skips provider title/prologue rows only when a later
row has known support-export header signals. Single-column known headers such
as `ticket_id` remain valid CSV headers so the support-ticket package can report
the real no-usable-wording failure instead of a parser-level header failure.

The submit route applies an English-only filter only when row metadata includes
a language marker. Rows without a language marker continue through the existing
path. Filtered non-English counts are reported in submit diagnostics instead of
being treated as row-limit truncation.

The generic `/execute` validator continues to use `_MAX_INGESTION_ROWS`, so
large inline `source_material` payloads outside the portfolio submit handoff do
not become unbounded request bodies.

## Intentional

- The 50 MB upload/blob guard stays intact. #1440 will cut the real CFPB export
  to fit that guard rather than raising request body size for this slice.
- This does not make the generic `/execute` endpoint accept 50k inline rows.
  The live funnel uses the submit route, and broadening `/execute` would widen a
  public API surface unnecessarily.
- The explicit limit ceiling is tied to the 50 MB byte guard. The real effective
  row cap is the number of rows parsed from the accepted file; the #1440 sample
  currently fits 34,914 rows under that guard.
- The limit is a row-count safety ceiling, not a display cap. The report builder
  remains responsible for deterministic clustering and ranked-question output.
- This does not add translation or multilingual clustering. Mixed-language
  uploads with a language marker are reduced to English rows for this proof
  lane.
- Existing fail-loud fixture labels are updated only where the behavior is now
  intentionally accepted production input. Ragged extra-cell rows stay fail-loud.

## Deferred

- The actual #1440 live run remains after this PR deploys: submit the generated
  near-50 MB CFPB CSV sample to hosted intake, confirm snapshot email/PDF,
  complete payment, and confirm full report email/PDF to
  `canfieldjuan24@gmail.com`.

Parked hardening: none.

## Verification

- Passed: pytest tests/test_extracted_content_deflection_submit.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_support_ticket_input_package.py
  - 235 passed, 1 skipped.
- Passed: pytest tests/test_extracted_campaign_customer_data.py tests/test_build_deflection_messy_csv_fixtures.py tests/test_smoke_content_ops_support_ticket_package.py
  - 39 passed.
- Passed after rebase: pytest tests/test_extracted_content_deflection_submit.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_campaign_customer_data.py tests/test_build_deflection_messy_csv_fixtures.py tests/test_smoke_content_ops_support_ticket_package.py
  - 274 passed, 1 skipped.
- Passed: python scripts/build_content_ops_deflection_report.py tmp/deflection_full_50k_e2e_20260609/cfpb_real_upload_under_50mb.csv --source-format csv --title "CFPB Real Export Deflection Report" --faq-title "CFPB Real Export FAQ Deflection Report" --default-field company_name=CFPB --default-field contact_email=canfieldjuan24@gmail.com --default-field vendor_name=CFPB --output tmp/deflection_full_50k_e2e_20260609/report_build/report.md --summary-output tmp/deflection_full_50k_e2e_20260609/report_build/summary.json --result-output tmp/deflection_full_50k_e2e_20260609/report_build/result.json --require-output-checks
  - Passed on 35,386-row / 52,363,054-byte CFPB sample generated with standard
    CSV quoting and embedded source quotes preserved; generated 37 ranked
    questions, 35,386 ticket sources, 0 proven answers, output checks all true.
- Passed: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Passed: bash scripts/check_ascii_python.sh
  - Passed.
- Passed: python -m py_compile extracted_content_pipeline/api/control_surfaces.py extracted_content_pipeline/campaign_customer_data.py scripts/build_deflection_messy_csv_fixtures.py scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_extracted_content_deflection_submit.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_campaign_customer_data.py tests/test_build_deflection_messy_csv_fixtures.py tests/test_smoke_content_ops_support_ticket_package.py
  - Passed.
- Passed: git diff --check
  - Passed.
- Pending before push:
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Full-Volume-Submit-Limit.md --check
  - bash scripts/local_pr_review.sh

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 118 |
| `extracted_content_pipeline/campaign_customer_data.py` | 107 |
| `plans/PR-Deflection-Full-Volume-Submit-Limit.md` | 184 |
| `scripts/build_deflection_messy_csv_fixtures.py` | 4 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 21 |
| `tests/test_build_deflection_messy_csv_fixtures.py` | 3 |
| `tests/test_extracted_campaign_customer_data.py` | 9 |
| `tests/test_extracted_content_deflection_submit.py` | 187 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 42 |
| **Total** | **675** |
