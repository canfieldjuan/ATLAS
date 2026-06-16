# PR-Deflection-Full-Report-QA-Scorecard

## Why this slice exists

#1612 needs a repeatable full-report delivery QA framework. #1613 closed the
artifact-safety boundary first; this slice adds the shared scorecard contract
the later email/page/PDF/export harnesses will write to.

Root cause: without a shared model-anchored scorecard, each surface proof can
invent its own count checks and either false-fail capped customer surfaces or
false-pass by comparing surfaces to each other instead of to the persisted
`deflection.v1` report model. The scorecard must separate canonical totals
from capped displayed rows and keep the committed artifact sanitized. The diff exceeds the soft cap because the scorecard API, export mismatch checks, surface-cap checks, and leak-redaction regression tests must ship together for the detector to be meaningful.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Add a pure scorecard builder beside the deflection report model.
2. Anchor canonical counts to `report_model` and `evidence_export`, not to a
   rendered surface.
3. Accept future surface observations for exact model-count checks and capped
   row-count checks.
4. Keep the scorecard sanitized: counts, assertion IDs, and surface names only;
   no source IDs, evidence quotes, request IDs, emails, paths, or URLs.
5. Archive the merged #1613 plan as this next branch's teardown housekeeping.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Full-Report-QA-Scorecard.md`
- `plans/archive/PR-Deflection-Full-Report-QA-Redaction-Policy.md`
- `tests/test_check_deflection_full_report_proof_bundle.py`
- `tests/test_content_ops_deflection_report.py`

### Review Contract

Acceptance criteria:

- Scorecard schema is explicit and deterministic.
- Canonical totals include repeat tickets, generated/ranked questions,
  drafted/publishable answers, no-proven-answer gaps, source tickets, support
  tax cost, evidence questions, evidence rows, and source-ID counts.
- Evidence-export totals are checked against the model, including list lengths
  and summary totals.
- Surface observations compare displayed count values to the model and compare
  capped rows to `min(model total, cap)`.
- Mismatches fail with specific assertion IDs.
- Scorecard JSON contains no raw evidence quotes or source IDs.

Affected surfaces: future deterministic CI harness, hosted result-page smoke,
PDF validator, evidence-export validator, and live proof runner.

Risk areas: confusing displayed capped rows with canonical totals; building a
scorecard that leaks the evidence export; silently passing malformed model or
export envelopes.

- Reviewer rules triggered: R1, R2, R9, R10, R12, R13, R14.

## Mechanism

`build_deflection_full_report_qa_scorecard(...)` reads a `deflection.v1`
report model plus optional evidence export and future surface observations. It
derives canonical totals from section data, emits one assertion per invariant,
and returns `{schema_version, ok, counts, assertions}`.

Surface observations are intentionally small dictionaries:
`{"counts": {...}, "displayed_rows": {...}}`. Count observations must equal the
model totals. Displayed rows are allowed to be capped, but only at the configured
cap for that surface/section.

## Intentional

- This PR does not render email, HTML, PDFs, screenshots, or evidence files.
  Those are the next harness slices.
- This PR keeps the scorecard in `faq_deflection_report.py` so it can reuse the
  report section registry and stay package-local for extracted checks.
- This PR archives only this session's just-merged #1613 plan by name.

## Deferred

- PR-Deflection-Full-Report-QA-Deterministic-Harness: fake-transport
  email/page/PDF/export consistency tests using this scorecard.
- PR-Deflection-Full-Report-QA-Live-Runner: live Zendesk-shaped proof runner
  that commits only sanitized scorecards.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_deflection_report.py -q` - 62 passed.
- `pytest tests/test_docs_no_raw_deflection_request_ids.py tests/test_check_deflection_full_report_proof_bundle.py -q` - 36 passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 4417 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 414 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Full-Report-QA-Scorecard.md` | 111 |
| `plans/archive/PR-Deflection-Full-Report-QA-Redaction-Policy.md` | 0 |
| `tests/test_check_deflection_full_report_proof_bundle.py` | 17 |
| `tests/test_content_ops_deflection_report.py` | 243 |
| **Total** | **788** |
