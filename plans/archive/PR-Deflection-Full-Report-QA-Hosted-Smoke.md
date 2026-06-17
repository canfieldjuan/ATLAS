# PR-Deflection-Full-Report-QA-Hosted-Smoke

## Why this slice exists

#1612's next tier after the deterministic #1620 harness is a hosted
browser/API smoke: render the buyer-facing result page, fetch the evidence
export API response, and prove their observed counts agree with the persisted
`deflection.v1` report model through the same scorecard seam.

Root cause: the hosted portfolio page can be green on route-specific assertions
while drifting away from the structured report model that the email/PDF/export
surfaces use. The right upstream fix for this slice is not a one-off visual
assertion; it is a reusable hosted-observation path that extracts sanitized
counts from the rendered page/API response and feeds those observations into
the scorecard contract #1620 added.

The diff may exceed the soft cap because this is a cross-runtime slice: the
portfolio route must expose sanitized QA observations, the Node hosted-smoke
test must render/fetch actual hosted surfaces, the Python scorecard CLI must
validate the observations, and both CI lanes need enrollment in the same PR.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Add sanitized result-page QA observation data for the hosted paid dashboard.
2. Add a small Python scorecard CLI that consumes `report_model`,
   `evidence_export`, hosted surface observations, and optional surface caps.
3. Add a portfolio-ui hosted smoke test that renders the paid result page,
   exercises the evidence-export API projection, extracts observations, and
   runs the scorecard CLI.
4. Add negative probes for present-but-incomplete hosted observations and
   model-count mismatches.
5. Enroll the new portfolio-ui test in
   `.github/workflows/portfolio_ui_checks.yml` and the Python CLI test in
   `scripts/run_extracted_pipeline_checks.sh`.
6. Archive the merged #1620 deterministic harness plan as this branch's teardown
   housekeeping.

### Files touched

- `.github/workflows/portfolio_ui_checks.yml`
- `plans/INDEX.md`
- `plans/PR-Deflection-Full-Report-QA-Hosted-Smoke.md`
- `plans/archive/PR-Deflection-Full-Report-QA-Deterministic-Harness.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/package.json`
- `portfolio-ui/scripts/faq-deflection-full-report-qa-hosted-smoke.test.mjs`
- `scripts/check_deflection_full_report_hosted_smoke.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`

### Review Contract

Acceptance criteria:

- Hosted result-page observations contain sanitized counts/displayed-row counts
  only; no request IDs, URLs, source IDs, evidence quotes, customer emails,
  paths, Stripe IDs, or private-note text.
- The portfolio smoke renders the actual `renderResultPage(...)` paid result
  page and calls the actual evidence-export API projection path.
- The smoke feeds result-page and evidence-export observations into the Python
  scorecard CLI with `required_surfaces=["result_page", "evidence_export"]`.
- The scorecard validates against the persisted `deflection.v1` model, not
  against hand-copied expected frontend numbers.
- Missing hosted metrics and wrong observed counts fail with specific
  assertion IDs.
- New Node and Python tests are enrolled in the CI lanes that own them.

Affected surfaces: portfolio hosted result page, portfolio evidence-export API,
full-report QA scorecard harness, and future live proof runner.

Risk areas: silently duplicating model-count logic in frontend tests; leaking
customer evidence through QA attributes or committed artifacts; adding a new
portfolio test without workflow enrollment; making the Python checker pass a
partial observation.

- Reviewer rules triggered: R1, R2, R9, R10, R12, R13, R14.

## Mechanism

`result-page.js` derives a compact `result_page` observation from the same
unlocked report envelope the hosted route renders. It emits the observation as
a JSON data attribute on the paid report section and exports the observation
builder for tests. The observation includes model-backed count values and
displayed-row counts for the sections the hosted dashboard actually renders.

`scripts/check_deflection_full_report_hosted_smoke.py` is the bridge from the
hosted runtime to the extracted scorecard. It loads JSON files, calls
`build_deflection_full_report_qa_deterministic_harness(...)`, writes a sanitized
scorecard if requested, and exits non-zero when the scorecard fails.

The portfolio smoke renders the paid page, extracts the QA observation from the
HTML, calls `evidenceExportFromReport(...)`, builds the `evidence_export`
observation from that real API projection, and invokes the Python checker. The
negative probes remove one metric and alter one count so both detection paths
prove they fire.

## Intentional

- This PR does not hit the deployed portfolio URL or take a browser screenshot.
  It is the hosted API/render smoke tier; the live runner remains separate.
- This PR validates `result_page` and `evidence_export` only. PDF and email are
  separate surfaces with different renderers and will be fed into the same
  scorecard in later slices.
- The QA data attribute intentionally carries counts only. It is not a hidden
  raw report payload.

## Deferred

- PR-Deflection-Full-Report-QA-PDF-Export-Validators: validate PDF bytes/text
  and complete evidence-export bounds/leak rules.
- PR-Deflection-Full-Report-QA-Live-Runner: run the paid Zendesk-shaped proof
  against hosted services and commit only sanitized summaries.

Parked hardening: none.

## Verification

- `pytest tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py -q`
  (3 passed).
- `npm run test:deflection-full-report-qa-hosted-smoke` from `portfolio-ui/`
  (3 hosted-smoke assertions passed).
- `npm run test:deflection-result` from `portfolio-ui/` (27 result-page checks
  passed).
- `npm run test:deflection-upload-shell` from `portfolio-ui/` (40 upload-shell
  checks passed).
- `npm run test:deflection-atlas-proxy` from `portfolio-ui/` (23 proxy/result
  API checks passed).
- `scripts/run_extracted_pipeline_checks.sh` via bash (4431 passed, 10 skipped).

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/portfolio_ui_checks.yml` | 3 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Full-Report-QA-Hosted-Smoke.md` | 147 |
| `plans/archive/PR-Deflection-Full-Report-QA-Deterministic-Harness.md` | 0 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 108 |
| `portfolio-ui/package.json` | 1 |
| `portfolio-ui/scripts/faq-deflection-full-report-qa-hosted-smoke.test.mjs` | 282 |
| `scripts/check_deflection_full_report_hosted_smoke.py` | 93 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py` | 202 |
| **Total** | **840** |
