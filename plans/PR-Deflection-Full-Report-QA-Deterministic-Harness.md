# PR-Deflection-Full-Report-QA-Deterministic-Harness

## Why this slice exists

#1612 asks for a repeatable full-report delivery QA harness that can prove the
customer surfaces agree before we rely on one-off live smokes. #1618 added the
model-anchored scorecard, but the only proof today exercises scorecard inputs
directly. The missing slice is the deterministic harness layer that composes
email, hosted result-page, PDF, and evidence-export observations into one
scorecard-shaped result without committing live artifacts or customer evidence.

This is not a defect-fix PR, but the upstream risk is the same one the scorecard
was built to prevent: surface-specific tests can pass while the delivered
customer bundle drifts away from the persisted `deflection.v1` model. This PR
fixes that at the harness boundary by making the deterministic proof build all
surface observations from one report model and export.

Review fix root cause: the first harness pass required each surface name to be
present, but the real-observation override path did not require each surface to
report its full metric/displayed-row contract. This change fixes that root by
making required-surface completeness part of the harness assertions, not a
deferred hosted-runner concern. The diff now exceeds the soft cap because the
initial harness, review-fix completeness assertions, negative probes, and plan
archive need to ship together for the harness contract to be meaningful.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Add a pure deterministic full-report QA harness that builds sanitized surface
   observations for email, result page, PDF, and evidence export from a
   `deflection.v1` model.
2. Feed those observations into `build_deflection_full_report_qa_scorecard(...)`
   so all customer-facing surfaces are checked through the same contract.
3. Add failure-mode tests proving the harness goes red when a surface reports a
   model-count mismatch, exceeds a capped display count, omits a required
   surface, or omits required metrics for a present surface.
4. Archive the merged #1618 scorecard plan as this branch's teardown
   housekeeping.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Full-Report-QA-Deterministic-Harness.md`
- `plans/archive/PR-Deflection-Full-Report-QA-Scorecard.md`
- `tests/test_content_ops_deflection_report.py`

### Review Contract

Acceptance criteria:

- The deterministic harness returns a sanitized scorecard and does not write or
  require committed email, HTML, PDF, screenshot, or evidence-export artifacts.
- Required surfaces include email, result page, PDF, and evidence export.
- Email/page/PDF count observations are anchored to the report model, not to
  each other.
- Result page and PDF displayed-row observations respect their configured caps:
  displayed rows must equal `min(model total, cap)`.
- Evidence-export observations include complete question/evidence/source totals
  and still rely on the scorecard's export-vs-model checks.
- Missing-surface, count-mismatch, and cap-overrun fixtures fail with specific
  assertion IDs.
- Present-but-incomplete real surface observations fail with specific missing
  metric/displayed-row assertion IDs.
- The scorecard output remains safe to commit: no evidence quotes, source IDs,
  request IDs, result URLs, customer emails, local paths, or Stripe IDs.

Affected surfaces: deterministic full-report QA CI harness, future hosted
result-page smoke, future PDF/export validators, and future live proof runner.

Risk areas: mistaking capped display rows for canonical totals; building a
happy-path-only harness; accidentally making committed proof artifacts a leak
surface; duplicating renderer logic that should stay in the shared scorecard.

- Reviewer rules triggered: R1, R2, R9, R10, R12, R13, R14.

## Mechanism

Add a small package-local helper beside the scorecard:

```python
build_deflection_full_report_qa_deterministic_harness(
    report_model,
    evidence_export=export,
) -> dict
```

The helper derives the canonical counts once, emits observation dictionaries for
the required surfaces, and delegates the actual assertions to the #1618
scorecard. The result includes the normal scorecard plus a small `surfaces`
summary listing which surfaces were observed. The harness also asserts the
required count keys for each required surface and the required displayed-row
keys for capped surfaces, so a hosted/live runner cannot pass by sending a
one-field observation. Tests pass intentionally bad observations to prove the
scorecard fails on the important directions.

## Intentional

- This PR does not render real email, browser HTML, PDF bytes, or screenshots.
  It is the deterministic CI tier from #1612; hosted/browser and live tiers are
  deferred.
- This PR keeps the harness in `faq_deflection_report.py` rather than adding a
  script so extracted-checks can run it without atlas_brain, asyncpg, browser,
  or network imports.
- The harness synthesizes surface observations from the report model because
  the current goal is contract consistency, not visual rendering fidelity.

## Deferred

- PR-Deflection-Full-Report-QA-Hosted-Smoke: fetch the canonical hosted result
  page and downloaded artifacts, then feed observed counts into the same
  scorecard.
- PR-Deflection-Full-Report-QA-Live-Runner: run the paid Zendesk-shaped proof
  and commit only sanitized scorecards/summaries.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_deflection_report.py -q` - 67 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 4423 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 202 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Full-Report-QA-Deterministic-Harness.md` | 138 |
| `plans/archive/PR-Deflection-Full-Report-QA-Scorecard.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 148 |
| **Total** | **489** |
