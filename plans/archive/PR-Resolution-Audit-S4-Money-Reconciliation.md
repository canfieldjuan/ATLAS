# PR-Resolution-Audit-S4-Money-Reconciliation

## Why this slice exists

Issue #1993 S4 tracks a current-code money reconciliation gap in the
Resolution Audit report surfaces. The old finding was directionally right, but
this slice reconstructs it from current code rather than accepting the old
wording.

Root cause: support-cost display is not governed by one money contract. The
report producer stores raw float costs from `ticket_count * 13.50`, markdown and
PDF render those values through local half-up integer-dollar helpers, and the
delivery email renders the same value with Python `:.0f` half-even rounding.
The hosted paid result page also had its own whole-dollar `Intl.NumberFormat`
formatter, so the browser page could still disagree with the Python-rendered
deliverables.
That creates two concrete defects:

1. Email or hosted page can disagree with markdown/PDF on `.50` values such as
   `94.50`.
2. Rounded per-row dollars can sum to a different value than the rounded
   headline repeat-cost total, even though both come from the same raw ticket
   counts.

This change fixes the root for displayed support-cost money by using one cents
display helper everywhere the paid Resolution Audit renders support-cost money,
and by keeping annualized support-cost math in Decimal money arithmetic until
the final half-up cents value is computed.
It does not change the report model shape, the report section order, the
snapshot projection, or the assisted-contact basis.

Diff-size note: this slice exceeds the 400 LOC target because the review fixes
must include the hosted paid result page, its report-model support-tax source
selection, and regenerated producer-owned report fixtures in the same PR.
Splitting those would leave the S4 reconciliation contract knowingly
incomplete.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Functional validation

1. Add a shared cents-based support-cost display helper and route report
   markdown, PDF, delivery email, and action-summary support-cost text through
   it; update the hosted paid result page formatter to the same cents display.
2. Preserve existing raw numeric fields (`estimated_support_cost`,
   `assisted_contact_cost`, annualized cost fields) as numeric values; only
   rendered strings change.
3. Close out S3 in the tracker and archive the merged S3 plan in this branch.
4. Add focused tests proving cross-surface `.50` parity and row-total versus
   headline reconciliation.
5. Regenerate the committed deflection report example fixture from the producer
   after the display-string change.
6. Regenerate the committed resolution live-proof markdown fixture from its
   source CSV after the display-string change.

### Review Contract

Acceptance criteria:

- The same support-cost value renders identically in markdown, PDF-derived
  text, delivery email, and the hosted paid result page.
- A report with row costs that would diverge under integer-dollar rounding
  renders cents so the visible row total reconciles to the visible support-tax
  headline.
- Raw report-model cost fields remain numeric and no new hosted report,
  snapshot, email, PDF, or landing-page fields/sections are added.
- The `$13.50 assisted-contact` basis remains the source label.
- The generated frontend report example matches the producer after the display
  string change.
- The hosted paid summary reads the report model's `support_tax` cost/count
  fields before falling back to legacy artifact-summary math.
- The committed resolution live-proof markdown fixture regenerates from its
  CSV source with cents displays.
- Annualized support-cost displays preserve exact half-up cents for fractional
  source windows, including the `5 tickets over 36 days` case.
- S3 tracker state moves from in-review to complete because #2030 merged.

Affected surfaces:

- `extracted_content_pipeline/faq_deflection_report.py`
- `atlas_brain/deflection_pdf_renderer.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- S4-focused tests and the #1993 local tracker.

Risk areas:

- User-visible rendered dollar strings change from whole-dollar estimates to
  cents in Resolution Audit report/email/PDF/hosted-result support-cost
  contexts.
- Existing tests with whole-dollar copy expectations must be updated only where
  they assert support-cost rendering.

Reviewer rules triggered: R1, R2, R7, R9, R10, R12, R14.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `atlas_brain/deflection_pdf_renderer.py`
- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `extracted_content_pipeline/deflection_money.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/manifest.json`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S4-Money-Reconciliation.md`
- `plans/archive/PR-Resolution-Audit-S3-Submit-Row-Cap.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

Introduce one support-cost money renderer that:

- coerces numeric values through `Decimal(str(value))`;
- clamps negative support-cost displays to zero for the existing non-negative
  report contexts;
- quantizes to cents with `ROUND_HALF_UP`;
- renders as `"$N,NNN.NN"`.

Annualized support-cost values use the same money module but multiply the
ticket count, 365-day annualization factor, and `$13.50` benchmark before
dividing by the source-window days. That avoids converting a fractional
annualized ticket count to float before computing cents.

Use that helper in the report producer for markdown/action prose, in the PDF
renderer for report-model-derived PDF text, and in the paid-report delivery
email renderer. The hosted paid result page keeps its local JavaScript
formatter but changes that formatter from whole-dollar output to cents and
sources the paid summary from the generated `support_tax` report-model section
before falling back to legacy artifact-summary math. Signed delta money remains
a separate concern because it is a delta surface, not the S4 paid-report
support-cost total.

Tests build or mutate a small report model with `.50` support-cost values and
assert the rendered strings line up across the producer, PDF model markdown, and
email summary. The hosted result-page test now asserts `.50` paid dashboard and
Jira handoff displays plus the producer-style path where artifact summary omits
repeat totals, and the generated fixture checks prove committed examples were
regenerated from their producers.

## Intentional

- No report-model schema change. Adding `*_cents` fields would be a stronger
  long-term contract, but it would be a hosted payload shape change and is not
  necessary to fix the current display defect.
- No row-rounding allocation. Allocating integer dollars across rows would make
  identical rows display different costs depending on order. Cents rendering is
  the stable reconciliation rule.
- No change to the $13.50 benchmark. This slice makes the existing basis
  truthful and consistent; it does not revisit the benchmark source.
- No snapshot, landing, clustering, date-window, or text-hygiene work.

## Deferred

- Adding stored cents fields to the report model is deferred until the operator
  approves a hosted report-model shape change.
- Signed delta money can adopt the same cents display in a future delta-specific
  slice if product wants cents there too; this PR keeps delta display unchanged.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_report.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_smoke_content_ops_deflection_pdf_export_validators.py
  - 233 passed, 5 skipped.
- python -m compileall extracted_content_pipeline/deflection_money.py extracted_content_pipeline/faq_deflection_report.py atlas_brain/deflection_pdf_renderer.py atlas_brain/content_ops_deflection_delivery.py
- bash scripts/validate_extracted_content_pipeline.sh
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- python scripts/audit_extracted_standalone.py --fail-on-debt
- bash scripts/check_ascii_python.sh
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
- python scripts/sync_pr_plan.py plans/PR-Resolution-Audit-S4-Money-Reconciliation.md --check
- python scripts/generate_deflection_snapshot_example.py --check
- python -m pytest tests/test_content_ops_faq_deflection_snapshot_example_generator.py tests/test_content_ops_faq_report_contract_docs.py
  - 17 passed.
- python scripts/build_content_ops_deflection_report.py docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv --source-format csv --output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md --summary-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json --result-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json --require-output-checks --json
- python -m pytest tests/test_content_ops_deflection_resolution_live_proof.py
  - 3 passed.
- python -m pytest tests/test_content_ops_deflection_report.py
  - 181 passed, 4 skipped.
- npm --prefix portfolio-ui run test:deflection-result
- bash scripts/local_pr_review.sh
  - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 11 |
| `atlas_brain/deflection_pdf_renderer.py` | 8 |
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 18 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 20 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 86 |
| `extracted_content_pipeline/deflection_money.py` | 54 |
| `extracted_content_pipeline/faq_deflection_report.py` | 30 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S4-Money-Reconciliation.md` | 211 |
| `plans/archive/PR-Resolution-Audit-S3-Submit-Row-Cap.md` | 0 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 31 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 68 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 6 |
| `tests/test_content_ops_deflection_report.py` | 169 |
| **Total** | **718** |
