# PR-Deflection-PII-SSN-Card-Scrub

## Why this slice exists

#1742 now has a CI-visible recall scorer and the current main baseline is
measured: the free headline still fails on a cue-less person name, and the paid
artifact still leaks the tiny corpus private-note SSN/card labels. The cue-less
name gap is explicitly an open-set/NER follow-up in #1742, so this slice takes
the next narrow fixable root.

Root cause: the deflection report payload scrubber has detectors for email,
phone, addresses, role-cued names, and contextual identifiers, but it has no
SSN or payment-card detector. As a result, high-severity SSN/card text carried
in paid artifact fields remains full-span residue after the shared deflection
payload scrub. This change fixes the root for those classes by adding SSN/card
redaction to the shared scrubber, not by hiding them in the scorer.

Review-update root cause: source-link preservation and card candidate matching
were separate admission paths around the shared text scrub. The first pass
taught only the normal text path about SSN/card values, so `source_id`/
`source_ids` could still preserve those values. The first card candidate also
ran before email redaction and used a greedy bounded candidate, so numeric email
local parts, longer hyphenated identifiers, and adjacent expiry/CVV digits could
be handled incorrectly. The update fixes those roots in the shared scrubber
boundary logic rather than adding scorer exceptions.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 4

1. Add SSN and payment-card redaction to the shared deflection payload scrubber.
2. Prove the tiny #1742 scorer no longer reports paid-artifact SSN/card leaks
   while the intentionally deferred cue-less name headline gap remains visible.
3. Keep precision probes for near-miss technical/commercial numbers that should
   survive.
4. Close the Codex review findings for source-link preservation, numeric-email
   ordering, longer hyphenated identifiers, and card-plus-expiry/CVV adjacency.

### Review Contract

Acceptance criteria:
- SSN-shaped text such as `123-45-6789` is redacted from nested report payloads
  and keys as `[redacted-ssn]`.
- Payment-card-shaped text that passes the detector is redacted as
  `[redacted-payment-card]`.
- The #1742 scorer reports `ssn-001` and `payment_card-001` absent from
  `leak_samples`; `person_name-001` remains the free headline leak because
  open-set/NER names are deferred.
- Existing source-id preservation, ISO/CVE/SKU precision, and ordinary numeric
  near-misses still pass.
- SSN/card values in source-link fields are scrubbed while ordinary ticket/source
  IDs remain preserved.
- Numeric email addresses are redacted as complete emails before card matching.
- Longer hyphenated numeric identifiers are not split into partial card
  redactions.
- Cards followed by adjacent expiry/CVV-style digit tokens still redact the card
  prefix.

Affected surfaces:
- Shared deflection paid artifact payload scrub.
- #1742 recall scorer outcome on the committed surrogate-only tiny corpus.

Risk areas:
- Over-redacting ordinary long numeric identifiers as payment cards.
- Accidentally changing source-id preservation or the already-measured
  person-name outcomes.

- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R13, R14.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-PII-SSN-Card-Scrub.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The shared deflection scrubber routes every string through its central text
scrub helper. This slice adds two detector branches there:

- SSN shape detection for bounded `###-##-####` values, emitted as
  `[redacted-ssn]`.
- Payment-card candidate detection for digit values with spaces or hyphens,
  guarded by 13-19 digit normalization and a Luhn check before emitting
  `[redacted-payment-card]`.

The review fix reuses those same detectors in source-link preservation before a
`source_id`/`source_ids` value is allowed through unchanged. It also redacts
emails before card candidates, broadens card candidate collection enough to see
adjacent digit tokens, and redacts only a Luhn-valid prefix when the continuation
is whitespace-separated. Hyphen continuations remain unredacted unless the whole
candidate is a valid card number, which avoids splitting longer hyphenated
identifiers.

The scorer remains unchanged except for its assertions: it should observe the
existing payload scrubber behavior becoming safer. That keeps the measurement
honest and prevents a false-green scorer-only fix.

## Intentional

- No cue-less/open-set name fix; #1742 explicitly treats that as the measured
  NER/open-set gap.
- No private-note source-exclusion policy change; this slice ensures any paid
  artifact text that survives source selection is scrubbed for SSN/card classes.
- No raw PII in scorer leak samples; the existing surrogate-id-only reporting
  stays intact.
- No GitHub self-resolution of bot threads; the code and PR body record the
  fixes, and the next review/live-reconciliation pass can verify thread state.

## Deferred

- Cue-less/open-set person-name detection or NER.
- Larger operator-derived corpus and threshold selection.
- Advisory-to-gating promotion after corpus/threshold decisions.
- Broader private-note/evidence-source policy if product decides private notes
  should be excluded rather than scrubbed.

Parked hardening: none.

## Verification

- pytest tests/test_content_ops_deflection_report.py tests/test_score_deflection_pii_recall.py -q -- 137 passed.
- python scripts/score_deflection_pii_recall.py --json -- reports paid_artifact `ssn` leaks 0 / recall 1.0, paid_artifact `payment_card` leaks 0 / recall 1.0, leak samples limited to `person_name-001` and `person_name-003`, and the deferred free headline cue-less name failure remains `free_high_severity_leak_count=1`.
- python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_score_deflection_pii_recall.py -- passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- bash scripts/check_ascii_python.sh and python scripts/audit_extracted_pipeline_ci_enrollment.py -- ASCII passed; OK: 188 matching tests are enrolled.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-deflection-pii-ssn-card-scrub.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 63 |
| `plans/PR-Deflection-PII-SSN-Card-Scrub.md` | 142 |
| `tests/test_content_ops_deflection_report.py` | 98 |
| `tests/test_score_deflection_pii_recall.py` | 17 |
| **Total** | **320** |
