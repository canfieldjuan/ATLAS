# PR-Deflection-PII-Cued-Name-ISO-Precision

## Why this slice exists

#1742 now has a merged recall/precision scorer and advisory artifact. The
current tiny surrogate corpus exposes two deterministic scrubber gaps that are
small enough to fix before the larger real-corpus and threshold decisions:

- cue-prefixed customer names such as "Customer Jordan Lee" are not redacted
  unless the cue uses "is", ":", "=", or "-";
- "ISO 27001 references" is over-redacted because the generic identifier rule
  treats "27001 references" as a customer/reference identifier.

Root cause: the scrubber grammar is too narrow on customer-name cues and too
broad on numeric identifier phrases near technical standards. This PR fixes the
root at the shared deflection scrubber, not in the scorer, fixture, snapshot,
PDF renderer, or report consumers.

Review boundary probes showed the same root had second-side edges: plain role
cues could reject a real name when followed by a title-cased label or arbitrary
capitalized tier word, broad role cues could over-redact product/team phrases,
and the ISO/HIPAA preserve hook could preserve arbitrary nearby identifiers.
This update keeps the fix at the same shared scrubber root rather than patching
individual report surfaces.

This is deliberately not the open-set name solution. The cue-less name leak in
the tiny corpus remains the deferred NER/open-set-name root called out in #1742.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice

1. Broaden the deflection person-name cue grammar so plain role cues followed
   by a plausible two- or three-token name are redacted, including
   name-plus-label/tier shapes such as "Customer Jane Smith Account" and
   "Customer Jane Smith Premium".
2. Keep the weak plain-role path conservative so product/team phrases such as
   Customer Success Manager, Premium Plan, Annual Subscription, and User
   Experience Research remain readable.
3. Preserve actual ISO-style technical standards before the generic numeric
   identifier rule redacts trailing words such as "references", while still
   scrubbing arbitrary ISO/HIPAA-prefixed identifiers near customer labels.
4. Add behavior tests for the scrubber and scorer outcome:
   - cue-prefixed names are removed from the measured surfaces;
   - ISO 27001 survives as must-survive report content;
   - arbitrary ISO/HIPAA-adjacent customer identifiers are scrubbed;
   - existing near-misses such as Reset Password, product labels, CVE, SKU,
     HIPAA, and tenant source ids still survive.
5. Leave cue-less names, paid-only private-note residue, real corpus derivation,
   and advisory-to-gating thresholds out of this PR.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-PII-Cued-Name-ISO-Precision.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_score_deflection_pii_recall.py`

### Review Contract

- Acceptance criteria:
  - [ ] The scorer reports zero cue-prefixed person-name leaks on the committed
        tiny surrogate corpus.
  - [ ] The scorer reports zero must-survive violations for ISO 27001 on the
        committed tiny surrogate corpus.
  - [ ] The cue-less name leak remains visible and is not blended away or
        hidden by the fix.
  - [ ] ISO/HIPAA-adjacent arbitrary customer identifiers are scrubbed while
        ISO 27001 and ISO-27001 standard references survive.
  - [ ] Plain role names followed by labels or arbitrary capitalized tier words
        are shortened and redacted while common product/team phrases are not
        redacted as names.
  - [ ] Existing safe technical/product/reporting tokens remain unredacted.
  - [ ] No corpus, scorer math, PDF rendering, snapshot structure, or threshold
        gate changes land in this PR.
- Affected surfaces: deterministic deflection report scrubber, paid artifact,
  paid PDF text source, free snapshot/teaser, scorer tests.
- Risk areas: PII leak recall, over-redaction of technical references, false
  positives from broader name cues.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R13, R14.

## Mechanism

The scrubber already has one shared text path used by report artifacts,
snapshots, source-link text, and the scorer. This PR adjusts that shared path:

- the person-name regex admits plain role cues followed by a plausible name,
  then routes those plain-role matches through a stricter title-case and
  reject-token/role-phrase check so phrases like "member was reassigned",
  Reset Password headings, Premium Plan, and Success Manager are not treated as
  people;
- the plain-role decision checks the existing shortest-candidate path before it
  rejects a three-token candidate, so "Jane Smith Account" and
  "Jane Smith Premium" redact Jane Smith while leaving the trailing text;
- the identifier redaction path first scrubs labeled ISO/HIPAA-prefixed opaque
  identifiers, then uses a narrowed callback so only actual standard-number
  matches immediately preceded by a technical-standard prefix are preserved.

The scorer remains a measurement tool. Its expected output changes only because
the shared scrubber stops leaking the cued name and stops over-redacting ISO
content.

## Intentional

- No cue-less-name solution. That requires the deferred open-set NER decision;
  this PR only fixes names with explicit local cues.
- No paid-only private-note fix. SSN/card residue in paid artifact is a
  different evidence-ingestion/private-note root and should not be mixed into
  this scrubber grammar slice.
- No threshold or gating change. #1742 keeps the advisory-to-gating switch as
  an operator decision after the larger eval corpus exists.
- No dependency change. The fix is deterministic regex/callback logic inside
  the existing scrubber.
- Weak bare-role cues are intentionally narrower than explicit "name is" or
  "customer is" cues. That trades off a little weak-cue recall to avoid
  degrading report copy by redacting common product/team labels as names.

## Deferred

- Open-set/cue-less person-name detection and the NER decision.
- Paid artifact private-note exclusion for SSN/card residue.
- Operator-derived larger gold corpus and threshold selection.
- Advisory-to-gating promotion for the headline KPI.

Parked hardening: none.

## Verification

- pytest tests/test_content_ops_deflection_report.py tests/test_score_deflection_pii_recall.py -- 124 passed.
- python scripts/score_deflection_pii_recall.py --json -- headline free high-severity leaks now 1, cue-prefixed person-name leaks 0, must-survive violations 0.
- python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_score_deflection_pii_recall.py -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- bash scripts/local_pr_review.sh -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 133 |
| `plans/PR-Deflection-PII-Cued-Name-ISO-Precision.md` | 148 |
| `tests/test_content_ops_deflection_report.py` | 98 |
| `tests/test_score_deflection_pii_recall.py` | 14 |
| **Total** | **393** |
