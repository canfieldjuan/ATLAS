# PR-Deflection-PII-Source-Decision-Preflight

## Why this slice exists

#1742's buildable local bundle chain is now complete through #1801: a labeled
source can be converted into a sanitized review bundle, scored, and promoted to
a validated surrogate-only candidate by one command. The remaining blocker named
by #1742 and the last bundle plans is not another local artifact wrapper; it is
the operator handoff before the real source exists: which transient source is
in scope, what corpus/mix is intended, who owns labeling, and what labeling
quality review happened.

Root cause: those source decisions are currently prose-only open decisions. The
review-bundle pipeline can fail closed once given a labeled source, but there is
no machine-checkable preflight that proves the operator source/mix/labeling
handoff was named before the real-source run starts. This slice fixes that root
at the handoff boundary by adding a small sanitized source-decision preflight
contract. It does not choose the real source, run on real data, commit a
real-source-derived artifact, set thresholds, promote advisory output to a gate,
or close the deferred cue-less/open-set name gap.

This is a vertical slice because it exercises the next real #1742 handoff in
the thinnest buildable form: operator source-decision JSON -> fail-closed
validation -> sanitized ok/blocking envelope that CI can prove without raw PII.

Diff budget note: the synced LOC total is over the 400 LOC soft cap because the
new surface is a raw-source-adjacent preflight gate, so the PR includes
branch-covering negative tests for every detector family plus CI enrollment.
The implementation is intentionally standalone and metadata-only; the larger
share of the diff is the plan and failure-mode tests that prove the gate blocks
unsafe handoff shapes without echoing source details.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 5

1. Add a source-decision preflight CLI for the operator-supplied real-corpus
   handoff, with a small schema version and required source/corpus/labeling
   fields.
2. Emit sanitized JSON envelopes that report blocking error codes without
   echoing raw local paths, raw source IDs, raw ticket text, or raw labels.
3. Fail closed on missing operator decisions, unexpected manifest fields,
   unsafe source-reference shapes, invalid corpus-size/mix targets, unsupported
   corpus metadata values, and missing labeling-quality review.
4. Add CI-enrolled tests for the ok path, each detector branch, and sanitized
   failure output.
5. Keep the real-source run and threshold/advisory-to-gating decisions deferred.

### Review Contract

- Acceptance criteria:
  - [ ] A complete source-decision JSON returns ok with schema version,
        source kind, supply mode, corpus counts, PII class targets, and labeling
        quality-review status.
  - [ ] Missing source, source supply, corpus mix, labeling owner/reviewer, or
        quality-review status fails with stable sanitized error codes.
  - [ ] Unexpected top-level or nested manifest fields fail closed without
        echoing field names or raw field values.
  - [ ] Source references that look like local paths, absolute paths, traversal,
        email addresses, phone numbers, SSNs, payment cards, DOB/date-like
        values, or long raw text fail closed and are never echoed.
  - [ ] Corpus targets must be positive, internally consistent, and include the
        #1742 high-severity classes plus the cue-prefixed/cue-less person-name
        split.
  - [ ] Corpus PII-class and person-name subtype lists accept only the
        metadata allowlist and never echo unsupported raw-looking values.
  - [ ] Labeling owner and reviewer are compared after case/punctuation
        normalization so the same identity cannot pass as an independent
        reviewer.
  - [ ] The command does not read raw ticket data and does not run the
        review-bundle pipeline.
  - [ ] The new test file is enrolled in extracted-pipeline CI.
- Affected surfaces:
  - `scripts/check_deflection_pii_source_decision.py`
  - extracted-pipeline test runner/workflow enrollment
- Risk areas: accidentally treating placeholder decisions as real operator
  approval, raw source/path echo in validation errors, and scope creep into
  thresholds or the real-source run.
- Reviewer rules triggered: R1, R2, R3, R10, R14; boundary-probe required
  because this is a preflight gate for raw-source-adjacent operator handoff.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-PII-Source-Decision-Preflight.md`
- `scripts/check_deflection_pii_source_decision.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_check_deflection_pii_source_decision.py`

## Mechanism

The new CLI reads a small JSON object:

```json
{
  "schema_version": "deflection_pii_source_decision.v1",
  "source": {"kind": "zendesk_export", "supply": "transient_local_file", "reference": "support-q2-sample"},
  "corpus": {
    "target_ticket_count": 50,
    "minimum_ticket_count": 25,
    "pii_class_targets": ["email", "phone", "ssn", "payment_card", "person_name", "dob"],
    "person_name_subtypes": ["cue_prefixed", "cue_less"]
  },
  "labeling": {
    "owner": "support-ops",
    "reviewer": "privacy-review",
    "quality_review": "completed"
  }
}
```

It validates only the decision metadata, not raw tickets. The manifest is
fail-closed at the object boundary: top-level and nested sections use
allowlisted fields, corpus target lists use allowlisted metadata values, and
unknown keys or values report counts without echoing the raw key/value text.
The source reference is intentionally constrained to a short opaque slug rather
than a path, URL, DOB/date-shaped value, or raw example. Labeling owner and
reviewer are normalized before comparison. Failures print
`{"ok": false, "errors": [{"code": "..."}]}` with stable codes; successes print
a compact summary with counts and the reviewed decision fields that are safe to
hand to the next operator step.

## Intentional

- No real-source selection and no real-source run. The command validates the
  shape and presence of decisions; it does not make those decisions.
- No thresholds in this slice. Threshold selection remains an explicit operator
  decision and should not be smuggled into the source handoff.
- No integration into the just-merged pipeline yet. Keeping this standalone
  avoids widening #1801's orchestration surface while the source contract gets
  its own focused review.

## Deferred

- Supplying the real transient source, running the bundle pipeline against it,
  and versioning the surrogate-only candidate remain deferred until an operator
  supplies a passing decision manifest and the source.
- Wiring the decision preflight into the pipeline command can be a follow-up
  once this contract is reviewed.
- Threshold selection, advisory-to-gating promotion, and cue-less/open-set name
  mitigation remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_deflection_pii_source_decision.py -q -- 21 passed.
- python -m pytest tests/test_check_deflection_pii_source_decision.py tests/test_content_ops_deflection_pii_review_bundle_pipeline.py -q -- 29 passed.
- python -m py_compile scripts/check_deflection_pii_source_decision.py tests/test_check_deflection_pii_source_decision.py -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 192 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- git diff --check -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- python scripts/maturity_sweep_file_lane.py scripts/check_deflection_pii_source_decision.py --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-pii-source-decision-preflight.md -- passed during scripts/push_pr.sh.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Deflection-PII-Source-Decision-Preflight.md` | 168 |
| `scripts/check_deflection_pii_source_decision.py` | 344 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_check_deflection_pii_source_decision.py` | 312 |
| **Total** | **829** |
