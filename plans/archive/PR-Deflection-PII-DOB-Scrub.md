# PR-Deflection-PII-DOB-Scrub

## Why this slice exists

#1742 defines DOB as high-severity PII for the deflection recall/precision arc,
and `deflection_pii_eval_corpus.py` already accepts and surrogates `dob` labels.
The root cause is that the production report scrub boundary and the committed
tiny corpus never exercised a DOB label, so a high-severity class could remain
unproved by the end-to-end scorer. This slice fixes the root for the currently
supported DOB shape by adding context-cued DOB detection/redaction at the shared
deflection scrub boundary and locking it into the surrogate corpus/scorer proof.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 7

1. Redact context-cued DOB values before deflection report payloads project into
   snapshot, teaser, paid artifact, and PDF surfaces.
2. Add the first DOB surrogate label to the committed tiny eval corpus so the
   scorer proves this high-severity class through the real production path.
3. Add focused positive and near-miss tests for cued DOB shapes, ordinary dates,
   existing compliance/security date-like tokens, corpus surrogation, and scorer
   leak reporting.
4. Review update: include `birthday` and bare `born` as common DOB cues while
   preserving non-date phrases such as "born to lead" and birthday reminders.

### Review Contract

Acceptance criteria:
- `dob` remains a high-severity corpus/scorer class and the tiny fixture includes
  one DOB label.
- Context-cued DOB values such as `DOB 1990-04-17` redact to
  `[redacted-dob]` anywhere the deflection scrubber processes text.
- The scorer reports the tiny DOB label as expected and redacted on the paid
  artifact path, with no DOB leak sample.
- Ordinary date-like report content remains readable unless it is DOB-cued.
- The existing deferred cue-less/open-set person-name leak remains unchanged and
  explicitly out of scope.

Affected surfaces:
- Deflection report payload scrubber.
- Surrogate eval corpus fixture and corpus builder tests.
- PII recall scorer tests and default tiny fixture.

Risk areas:
- Over-redacting non-PII dates, CVEs, ISO/SOC references, years, prices, and
  ticket counts.
- Under-proving DOB by adding only a fixture without exercising the shared
  scrub boundary.

- Reviewer rules triggered: R1 Requirements match, R2 Test evidence, R3
  Security/privacy, R8 Scope, R10 Maintainability, R13 Fix-the-class, R14
  Codebase verification.

### Files touched

- `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json`
- `extracted_content_pipeline/deflection_pii_eval_corpus.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-PII-DOB-Scrub.md`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The shared deflection scrubber gets a DOB regex that requires a DOB/birth-date
cue before redacting a date-shaped value, including common support text cues
such as `birthday` and bare `born`. The replacement preserves the cue and
replaces only the date value with `[redacted-dob]`, so ordinary dates do not get
treated as PII.

The surrogate corpus builder gets the same context-cued DOB detector for its
fail-closed unlabeled-PII pass. The committed tiny fixture adds one DOB label in
agent reply text, which projects into the free and paid surfaces. The scorer then
uses the existing class/severity machinery to prove the label reaches surfaces
and is redacted by the production scrubber.

## Intentional

- This is context-cued DOB redaction only. Broad date redaction is intentionally
  avoided because the report has legitimate dates, years, CVE IDs, ISO/SOC
  references, prices, and counts that must survive for precision.
- Cue-less/open-set person names remain deferred per #1742. This slice should
  not change the existing `person_name-001` leak sample.

## Deferred

- Operator-derived gold corpus, thresholds, and gate flip decisions remain open
  #1742 inputs.
- Model-based NER for cue-less/open-set names remains a separate deferred slice
  under #1734/#1742.
- Additional DOB locales/formats beyond the context-cued shapes covered here can
  be expanded once the real corpus shows they occur.

Parked hardening: none.

## Verification

- pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py tests/test_score_deflection_pii_recall.py -q -- 167 passed after the review update.
- python scripts/score_deflection_pii_recall.py --json -- status ok,
  label_count 10, paid_artifact/paid_pdf `dob` leaks 0 / recall 1.0,
  must-survive violations 0, and only the deferred cue-less `person_name-001`
  leak remains with `free_high_severity_leak_count=1`.
- python -m py_compile extracted_content_pipeline/faq_deflection_report.py extracted_content_pipeline/deflection_pii_eval_corpus.py tests/test_content_ops_deflection_report.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py tests/test_score_deflection_pii_recall.py -- passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas
  runtime import findings: 0.
- bash scripts/check_ascii_python.sh -- ASCII passed for
  extracted_content_pipeline Python files.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188
  matching tests are enrolled.
- Pending before push: bash scripts/push_pr.sh <pr-body-file> -u origin HEAD.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json` | 16 |
| `extracted_content_pipeline/deflection_pii_eval_corpus.py` | 24 |
| `extracted_content_pipeline/faq_deflection_report.py` | 26 |
| `plans/PR-Deflection-PII-DOB-Scrub.md` | 129 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 46 |
| `tests/test_content_ops_deflection_report.py` | 43 |
| `tests/test_score_deflection_pii_recall.py` | 14 |
| **Total** | **298** |
