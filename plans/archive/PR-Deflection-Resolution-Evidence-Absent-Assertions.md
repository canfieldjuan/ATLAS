# PR-Deflection-Resolution-Evidence-Absent-Assertions

## Why this slice exists

PR #1424 exposed the support-ticket resolution-evidence signal in the
pre-payment deflection preview, but the review found the absent branch was not
asserted strongly enough. A mutation that forced the signal to always render as
"Present" could still pass the touched Python and result-page tests. This
follow-up pins the exact launch gate that protects question-only exports from
looking like publishable-answer reports.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Robust testing

1. Add backend absent-direction assertions for question-only deflection
   snapshots and the stored submit snapshot boundary.
2. Add hosted result-page assertions for the absent label and "gap list only"
   warning copy.
3. Keep this as test-only coverage for the already-merged #1424 behavior.

### Review Contract

- A forced always-present regression must fail in Python.
- A forced always-present hosted result page must fail in the `.mjs` test.
- The slice stays in the clustering/raw-data preview lane and does not touch
  PDF, email, delivery-worker, or paid attachment work.

- Reviewer rules triggered: R10, R13.

### Files touched

- `plans/PR-Deflection-Resolution-Evidence-Absent-Assertions.md`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The backend snapshot test builds a report whose FAQ items are all
`draft_needs_review`, then asserts the unpaid snapshot summary reports
`support_ticket_resolution_evidence_present == False` and count `0`. The submit
test now also reads the stored snapshot route and asserts it matches the
question-only snapshot returned at submit time. The hosted result-page test
renders an absent-evidence snapshot and asserts the rendered diagnostic includes
the false marker, "Absent" label, and "gap list only" warning copy.

## Intentional

- This PR does not change production code. The #1424 implementation is already
  merged; this slice closes the reviewer-found test blind spot.
- This PR does not rerun the full deflection harness or extracted CI mirror,
  because the change is test-only and the focused mutation target is narrower.

## Deferred

- #1419 part 2: live proof on a real resolution-bearing export that produces
  publishable answer groups.
- In-lane residual: finish HTML normalization so raw tags cannot reach clustered
  text/output on HTML-heavy exports.
- In-lane residual: improve deterministic synonym grouping for themes with no
  shared tokens.

Parked hardening: none.

## Verification

- Review follow-up absent-signal pytest - 2 passed.
- Review follow-up result page test - 18 checks passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Resolution-Evidence-Absent-Assertions.md` | 80 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 35 |
| `tests/test_content_ops_deflection_report.py` | 38 |
| `tests/test_extracted_content_deflection_submit.py` | 3 |
| **Total** | **156** |
