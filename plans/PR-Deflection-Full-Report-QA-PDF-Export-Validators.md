# PR-Deflection-Full-Report-QA-PDF-Export-Validators

## Why this slice exists

#1612's deterministic tier now proves the shared scorecard, and the repo-neutral
checker layer is still needed before the live runner. Correction from the
#1621 review loop: #1621 added a hosted smoke for `portfolio-ui` inside ATLAS,
but the buyer-facing UI is `atlas-portfolio/web`, so #1621 does **not** count as
buyer hosted-result proof. This slice stays on the ATLAS side only: the live
runner needs one reusable way to decide whether downloaded PDF plus evidence
export artifacts are valid, bounded, internally consistent with
`deflection.v1`, and leak-safe before it stores any sanitized proof summary.

Root cause: the report QA framework can currently score synthetic `pdf` and
`evidence_export` observations, but there is no repo-neutral artifact-level
validator that turns real PDF bytes/text and export JSON into those observations.
A live proof runner would otherwise have to hand-roll artifact checks, which
recreates the same drift #1618/#1620 removed for counts. This slice fixes the
root at the validator seam: PDF/export artifact observations are derived once,
then fed through the same model-anchored scorecard. The buyer hosted-result
surface itself remains an `atlas-portfolio/web` follow-up, not an ATLAS
`portfolio-ui` claim.

The diff may exceed the soft cap because this is another cross-surface bridge:
the checker, negative fixtures, CI enrollment, and plan/archive housekeeping
ship together so the live runner has a tested validator to call.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Add `scripts/check_deflection_full_report_pdf_export_artifacts.py`, a pure
   artifact validator for `report_model`, evidence export JSON, PDF bytes, and
   extracted PDF text.
2. Derive `pdf` and `evidence_export` observations and feed them into
   `build_deflection_full_report_qa_deterministic_harness(...)` with required
   surfaces `("pdf", "evidence_export")`.
3. Add PDF artifact assertions for `%PDF-` bytes, minimum size, required
   customer-facing section text, model-count visibility, complete-export
   pointer copy, and leak-sensitive raw strings.
4. Add evidence-export assertions through the existing scorecard path; no raw
   evidence values are copied into the output scorecard.
5. Add negative tests for bad PDF bytes, missing PDF model count, leak strings,
   and mismatched export totals.
6. Enroll the test in `scripts/run_extracted_pipeline_checks.sh`.
7. Archive the merged #1621 hosted-smoke plan as teardown housekeeping, while
   explicitly not treating it as buyer hosted-result proof.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Deflection-Full-Report-QA-PDF-Export-Validators.md`
- `plans/archive/PR-Deflection-Full-Report-QA-Hosted-Smoke.md`
- `scripts/check_deflection_full_report_pdf_export_artifacts.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_smoke_content_ops_deflection_pdf_export_validators.py`

### Review Contract

Acceptance criteria:

- The checker accepts JSON report/export inputs plus PDF bytes and already
  extracted PDF text; it does not add a PDF parser dependency in this slice.
- The checker returns a sanitized scorecard only: assertion IDs, booleans,
  expected/actual primitive summaries, and model counts. It does not echo raw
  evidence rows, source ID lists, emails, request IDs, result URLs, Stripe IDs,
  private notes, local paths, or PDF text.
- PDF observations are present only when the PDF bytes/text satisfy the
  artifact checks; missing required PDF counts fail the required-metric guard.
- Evidence-export shape/count mismatches fail through the existing
  `deflection_full_report_qa_scorecard.v1` assertions.
- Bad PDF bytes, missing visible count, and leak strings are all proven to fail
  with specific assertion IDs.
- The new Python test is enrolled in the extracted checks suite.

Affected surfaces: full-report QA checker scripts, future live runner,
PDF/export artifacts, and the `deflection.v1` scorecard seam. The buyer-facing
hosted result page is in `atlas-portfolio/web` and is intentionally outside
this ATLAS PR.

Risk areas: printing raw live PDF/export content in the scorecard, weakening
PDF validation to header-only, duplicating count logic that drifts from the
model, and adding a test without CI enrollment.

- Reviewer rules triggered: R1, R2, R9, R10, R12, R13, R14.

## Mechanism

The checker loads `report_model` and evidence export JSON, reads PDF bytes and
extracted PDF text, and computes sanitized artifact assertions. It derives
canonical model counts from the structured sections already consumed by the
scorecard. A count is included in the `pdf` observation only when the PDF text
contains that model value in an acceptable display form; otherwise the
`harness.surface.pdf.count.<metric>.present` assertion fails.

The evidence export is passed directly into the existing scorecard path, while
the checker derives an `evidence_export` observation from the export summary.
The combined scorecard requires `pdf` and `evidence_export`, appends
artifact-specific assertions, and exits non-zero when any assertion fails.

The checker intentionally accepts extracted PDF text instead of parsing PDF
bytes directly. The live runner owns browser/download/extraction mechanics; this
slice owns the model-anchored artifact contract that runner must satisfy.

## Intentional

- No committed live proof bundle. Tests use synthetic fixture text/bytes only.
- No new PDF parser dependency. The validator consumes PDF text supplied by the
  caller so the future live runner can use the most reliable extraction method
  available in that environment.
- The evidence export may contain raw evidence in the input file because it is
  the uncapped audit surface; the checker output still stays sanitized.
- This PR does not claim the ATLAS `portfolio-ui` result page is the buyer
  hosted surface. The actual buyer page is `atlas-portfolio/web`.

## Deferred

- PR-Deflection-Full-Report-QA-Live-Runner: download real hosted PDF/export
  artifacts, extract PDF text, feed this validator, and commit only sanitized
  summaries.
- Buyer hosted-result smoke: move the surface-specific observation hook/smoke to
  `atlas-portfolio/web` and fold it into that repo's existing hosted-results
  smoke rather than duplicating it in ATLAS `portfolio-ui`.
- Browser screenshot/readability scoring remains in the live-runner slice.

Parked hardening: none.

## Verification

- `pytest tests/test_smoke_content_ops_deflection_pdf_export_validators.py -q`
  (5 passed).
- `scripts/run_extracted_pipeline_checks.sh` via bash (4436 passed, 10 skipped).

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Full-Report-QA-PDF-Export-Validators.md` | 145 |
| `plans/archive/PR-Deflection-Full-Report-QA-Hosted-Smoke.md` | 0 |
| `scripts/check_deflection_full_report_pdf_export_artifacts.py` | 336 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_smoke_content_ops_deflection_pdf_export_validators.py` | 215 |
| **Total** | **700** |
