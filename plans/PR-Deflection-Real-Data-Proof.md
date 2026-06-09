# PR-Deflection-Real-Data-Proof

## Why this slice exists

Issue #1408 asks for the cheapest proof that the deflection funnel works on
real, high-volume customer language rather than only hand-built examples. PR
#1410 closed the first implementation gap by adding deterministic raw CSV
clustering plus a pre-payment cluster-quality preview. The next launch question
is validation: does that path produce a useful report on the CFPB complaint
corpus, and can the messy CSV ingestion edge cases be reproduced without a
one-off manual file?

This slice keeps the validation repeatable without committing the local 102 MB
CFPB export. It adds a generator for small messy CFPB-derived CSV fixtures,
tests the generated cases through the same parser/report path used by the
deflection submit flow, and commits a short validation write-up with the local
CFPB quality/scale observations from the existing `tmp/faq_scale_stress_20260523`
dataset.

This PR exceeds the 400 LOC soft cap because the deliverable is the validation
package itself: generator, negative/positive parser tests, the #1408 write-up,
and the merged-plan archive move. Splitting the report from the generator would
make the write-up non-repeatable.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Archive the merged #1410 plan as housekeeping folded into this branch.
2. Add a repeatable messy CSV fixture generator that derives bounded cases from
   Content Ops source-row JSONL, covering BOM/cp1252, semicolon/tab delimiters,
   HTML bodies, leading metadata rows, ragged rows, and quoted multiline bodies.
3. Add focused tests that run generated cases through the real
   `load_source_campaign_opportunities_from_file(..., file_format="csv")` path
   and distinguish parsed versus fail-loud outcomes.
4. Add a compact #1408 validation report documenting CFPB quality/scale
   findings and messy-case outcomes without committing the large CFPB payloads.

### Review Contract

- Acceptance criteria:
  - [ ] The messy fixture generator is deterministic and does not fetch network
        data; callers provide a JSONL/JSON source-row file.
  - [ ] Generated fixtures include the #1408 edge-case classes and write a
        machine-readable manifest that names expected outcomes.
  - [ ] Tests use the generated CSV bytes/files and the real parser/report load
        path rather than mocking the checker itself.
  - [ ] The validation report states which CFPB data was run locally, the
        observed clustering/report quality, and which ingestion cases parsed,
        failed loud, or need follow-up.
  - [ ] Large local `tmp/` data and generated run artifacts stay out of git.
- Affected surfaces: deflection validation scripts, parser/report fixture tests,
  validation docs, plan archive housekeeping.
- Risk areas: parser false positives, fixture realism, over-claiming real-data
  quality from local-only validation.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/deflection_real_data_proof_2026-06-09.md`
- `plans/INDEX.md`
- `plans/PR-Deflection-Real-Data-Proof.md`
- `plans/archive/PR-Deflection-Raw-CSV-Cluster-Preview.md`
- `scripts/build_deflection_messy_csv_fixtures.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_build_deflection_messy_csv_fixtures.py`

## Mechanism

The new fixture builder reads bounded source rows from JSONL/JSON and normalizes
them into support-ticket-shaped rows (`ticket_id`, `subject`, `message`,
`resolution_text`, `pain_category`). It writes several intentionally messy CSV
files plus a manifest:

- UTF-8 BOM header with cp1252-compatible punctuation in row text.
- Semicolon-delimited and tab-delimited exports.
- HTML-bodied messages with entities.
- Leading metadata/title row before the real header.
- Ragged rows with extra or missing cells.
- Quoted multiline bodies with embedded delimiters.

The tests generate fixtures in a temporary directory, parse the cases with the
same source adapter used by the deflection report CLI and submit flow, and
assert either useful parsed support-ticket rows or a clear failure for cases
that cannot safely map to ticket text.

The committed validation report cites local commands/results from the CFPB 10k
and 50k source-row runs and the generated messy cases. It stays descriptive:
quality observations, runtime/scale notes, and follow-up risks. It does not
claim funnel conversion, SEO traffic, or future ticket reduction.

## Intentional

- This PR does not commit the local CFPB JSONL files or generated large reports;
  they are already local validation inputs under `tmp/` and are too large for
  the repo.
- This PR does not run live portfolio upload or Stripe checkout. It validates
  the parser/report surfaces that ATLAS owns; hosted upload/email validation is
  a separate complete-cycle smoke using real credentials.
- Messy fixtures are small by design. They prove edge-case behavior
  repeatably; CFPB scale is documented from local validation runs.

## Deferred

- Hosted portfolio upload -> ATLAS submit complete-cycle run with real
  credentials remains a follow-up operational validation after the portfolio
  Snapshot email env is confirmed.
- HTML stripping/normalization for HTML-heavy provider exports remains a #1384
  follow-up; this slice proves those inputs do not crash the parser, not that
  raw tags are fully removed from every clustered/evidence surface.
- If the CFPB proof shows weak clustering or thin drafts on a specific topic
  family, the next implementation slice should improve deterministic clustering
  or add a pre-payment inspect gate rather than hiding the quality signal.

Parked hardening: none.

## Verification

- Command: pytest tests/test_build_deflection_messy_csv_fixtures.py
  - 11 passed.
- Command: pytest tests/test_extracted_campaign_customer_data.py::test_load_campaign_opportunities_from_csv_strips_utf8_bom_header tests/test_extracted_campaign_customer_data.py::test_load_campaign_opportunities_from_csv_falls_back_for_cp1252_exports tests/test_extracted_campaign_customer_data.py::test_load_campaign_opportunities_from_semicolon_csv_detects_delimiter tests/test_extracted_campaign_customer_data.py::test_load_campaign_opportunities_from_csv_fails_on_leading_metadata_row tests/test_extracted_campaign_source_adapters.py::test_load_source_rows_from_semicolon_csv_detects_support_ticket_export tests/test_extracted_campaign_source_adapters.py::test_load_source_rows_preserves_quoted_commas_and_multiline_text
  - 6 passed.
- Command: python scripts/build_content_ops_deflection_report.py /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl --source-format jsonl --result-output /tmp/deflection-real-data-proof/cfpb_10000/result.json --summary-output /tmp/deflection-real-data-proof/cfpb_10000/summary.json --output /tmp/deflection-real-data-proof/cfpb_10000/report.md --require-output-checks
  - Passed; 10,000 rows, 33 ranked questions, 14.74s, 144,056 KB max RSS.
- Command: python scripts/build_content_ops_deflection_report.py /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl --source-format jsonl --result-output /tmp/deflection-real-data-proof/cfpb_50000/result.json --summary-output /tmp/deflection-real-data-proof/cfpb_50000/summary.json --output /tmp/deflection-real-data-proof/cfpb_50000/report.md --require-output-checks
  - Passed; 50,000 rows, 39 ranked questions, 74.36s, 619,932 KB max RSS.
- Command: python scripts/build_deflection_messy_csv_fixtures.py /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl --output-dir /tmp/deflection-real-data-proof/messy_cfpb --limit 8 --json
  - Passed; generated eight messy CSV cases and `manifest.json`.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - Passed; `OK: 160 matching tests are enrolled.`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `docs/extraction/validation/deflection_real_data_proof_2026-06-09.md` | 107 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Real-Data-Proof.md` | 146 |
| `plans/archive/PR-Deflection-Raw-CSV-Cluster-Preview.md` | 0 |
| `scripts/build_deflection_messy_csv_fixtures.py` | 306 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_build_deflection_messy_csv_fixtures.py` | 155 |
| **Total** | **720** |
