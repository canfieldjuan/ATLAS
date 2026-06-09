# PR-Deflection-Resolution-Evidence-Preview-Signal

## Why this slice exists

Issue #1419 split the remaining publishable-answer gap into small launch gates.
The input package already computes `support_ticket_resolution_evidence_present`
and `support_ticket_resolution_evidence_count`, but the pre-payment deflection
preview only surfaces cluster-quality diagnostics. A question-only export can
therefore look payable even when it can only produce a gap list, not publishable
answers. This slice exposes that existing resolution-evidence signal before
checkout.

Diff-size note: this exceeds the 400 LOC soft cap because the producer schema,
stored snapshot, public proxy, React result page, and exact contract fixtures
must change together or the preview gate fails open on one surface.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Add the already-computed support-ticket resolution-evidence present/count
   values to the deflection submit preview diagnostics.
2. Render that signal in the hosted pre-payment FAQ deflection result page next
   to the free snapshot metrics, with absent evidence framed as "gap list only"
   rather than publishable answers.
3. Add focused backend and frontend tests for both present and absent evidence
   so the preview cannot silently drop the launch gate.

### Review Contract

- The API response carries `resolution_evidence_present` and
  `resolution_evidence_count` from producer metadata; it does not infer the
  values from UI-only strings or paid report fields.
- The pre-payment result page shows a visible resolution-evidence diagnostic
  before checkout and distinguishes present vs absent evidence.
- The copy stays in the clustering/raw-data lane and does not touch PDF,
  email, delivery worker, or paid attachment code.
- Tests cover both branches: present evidence and absent evidence.

- Reviewer rules triggered: R1, R9, R10, R12.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Resolution-Evidence-Preview-Signal.md`
- `portfolio-ui/api/content-ops/deflection/atlas-report.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `tests/test_atlas_content_ops_input_provider.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`

## Mechanism

The deflection submit route already builds input-package metadata from the CSV
rows before returning the snapshot. `_with_deflection_submit_diagnostics(...)`
copies the existing metadata keys into explicit diagnostics fields next to
`cluster_quality`. The deflection artifact gate also copies the same
present/count pair into the unpaid snapshot summary before storing it, so the
hosted result page can project and render the signal without reading paid
artifact fields. The public proxy fails closed if the snapshot omits the two
fields.

## Intentional

- This slice does not add new clustering, synonym handling, HTML normalization,
  or LLM repair. Those are separate in-lane residuals after #1419 part 1.
- This slice reuses producer metadata instead of re-scanning CSV rows in the UI;
  the input package remains the source of truth for whether resolution evidence
  exists.
- This slice does not call or render the paid report artifact. The signal must
  be available pre-payment from the unpaid snapshot/diagnostics envelope.

## Deferred

- #1419 part 2: live proof on a real resolution-bearing export that produces
  publishable answer groups.
- In-lane residual: finish HTML normalization so raw tags cannot reach clustered
  text/output on HTML-heavy exports.
- In-lane residual: improve deterministic synonym grouping for themes with no
  shared tokens.

Parked hardening: none.

## Verification

- Py compile for changed Python modules and tests - passed.
- Node syntax check for changed deflection proxy/result files - passed.
- Focused deflection submit/snapshot contract pytest - 23 passed.
- Full deflection report contract pytest - 41 passed.
- Focused preview metadata and live execute harness pytest - 2 passed.
- Portfolio upload shell test - 20 checks passed.
- Portfolio result page test - 17 checks passed.
- Portfolio ATLAS proxy test - 17 checks passed.
- Extracted content validation, reasoning import audit, standalone audit, and
  ASCII check - passed.
- Extracted CI mirror - 3538 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 2 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 4 |
| `extracted_content_pipeline/api/control_surfaces.py` | 50 |
| `extracted_content_pipeline/faq_deflection_report.py` | 21 |
| `plans/PR-Deflection-Resolution-Evidence-Preview-Signal.md` | 126 |
| `portfolio-ui/api/content-ops/deflection/atlas-report.js` | 12 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 33 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 62 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 39 |
| `portfolio-ui/src/pages/FaqDeflectionResult.tsx` | 55 |
| `tests/test_atlas_content_ops_input_provider.py` | 2 |
| `tests/test_content_ops_deflection_report.py` | 5 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 2 |
| `tests/test_extracted_content_deflection_submit.py` | 24 |
| `tests/test_extracted_content_ops_live_execute_harness.py` | 2 |
| **Total** | **439** |
