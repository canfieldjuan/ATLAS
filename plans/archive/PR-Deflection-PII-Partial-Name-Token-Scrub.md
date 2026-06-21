# PR-Deflection-PII-Partial-Name-Token-Scrub

## Why this slice exists

#1742 now measures the remaining scrubber failures on current `main`. After
#1754, SSN/card leaks are closed and must-survive precision remains clean. The
remaining fixable measured residue is `person_name-003`: a cue-prefixed
three-token name where the scrubber redacts only the first two tokens and leaves
the final token visible on paid artifact/PDF surfaces.

Root cause: the cued-name scrubber tries to preserve trailing labels by
shortening any three-token name candidate when the first two tokens look like a
name. For a real three-token cue-prefixed name such as `Customer Mary Jane
Watson Report`, that shortening runs before the full three-token plain-role
name decision, so it emits `Customer [redacted-name] Watson Report` and leaves a
meaningful name token behind. This change fixes the root in the shared
name-scrub decision order: redact plausible full three-token plain-role names
before falling back to the trailing-label shortening path.

Review-update root cause: the first decision-order fix preserved only
enumerated trailing labels, so an unknown product/status tier such as
`Customer Jane Smith Gold plan` could be redacted as if `Gold` were a name
token. The review update fixes that at the same shared scrubber boundary by
using the word after the candidate (`plan`, `account`, `tier`, and similar
context) to preserve unknown trailing labels without reintroducing the measured
`Mary Jane Watson` residue.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 4

1. Redact full three-token cue-prefixed person names when the whole candidate is
   a plausible plain-role name.
2. Preserve the existing trailing-label behavior for two-token names followed by
   product/status labels, such as `Customer Jane Smith Premium`.
3. Prove the #1742 scorer no longer reports `person_name-003` partial-name
   token leaks while the explicitly deferred cue-less/open-set name gap remains
   visible.

### Review Contract

Acceptance criteria:
- `Customer Mary Jane Watson Report plan was upgraded.` redacts all of `Mary
  Jane Watson` and does not leave `Watson` as partial residue.
- Existing product/role phrase near-misses still survive, including `Customer
  Jane Smith Premium` and `Customer Success Manager`.
- Unknown trailing product/status labels survive when followed by label context,
  such as `Customer Jane Smith Gold plan`.
- The #1742 scorer reports no `partial_name_token` leak samples for
  `person_name-003`.
- The #1742 scorer still reports `person_name-001` cue-less/open-set leaks and
  `free_high_severity_leak_count=1`; this PR must not hide that deferred gap.

Affected surfaces:
- Shared deflection report payload scrubber.
- #1742 recall scorer outcome on the committed surrogate-only tiny corpus.

Risk areas:
- Over-redacting three-token product, role, plan, or team phrases as names.
- Regressing the trailing-label behavior added for two-token names.
- Treating the scorer as the fix instead of changing the shared scrubber.

- Reviewer rules triggered: R1, R2, R3, R8, R10, R12, R13, R14.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-PII-Partial-Name-Token-Scrub.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The shared person-name decision helper already has two useful predicates: one
that recognizes plausible full plain-role names and one that shortens ambiguous
three-token candidates to preserve trailing labels. This slice changes only
their ordering for plain-role cues:

1. If the full candidate is a plausible plain-role name, redact it as a whole.
2. Before full-name redaction, preserve a shortened two-token name plus trailing
   label when the word after the candidate has label context such as `plan`,
   `account`, `subscription`, or `tier`.
3. Otherwise, use the existing shortest-candidate path to preserve trailing
   labels when the first two tokens are the name and the third token is a known
   label.
4. Keep role/product phrase rejection in the plain-role predicate so obvious
   non-name phrases survive.

The scorer remains a measurement surface. Its assertions update only because
the shared scrubber now removes the measured residue.

## Intentional

- No cue-less/open-set name detector or NER; #1742 explicitly names that gap as
  deferred and requires it to stay visible.
- No scorer-side filtering of `person_name-003`; the leak must disappear because
  the scrubbed output no longer contains the token.
- No broad name dictionary or external NLP dependency in this vertical slice.

## Deferred

- Cue-less/open-set person-name detection or NER.
- Larger operator-derived corpus and threshold selection.
- Advisory-to-gating promotion after corpus/threshold decisions.
- Broader name-shape hardening beyond the measured three-token cue-prefixed
  residue if the larger corpus exposes new forms.

Parked hardening: none.

## Verification

- pytest tests/test_content_ops_deflection_report.py tests/test_score_deflection_pii_recall.py -q -- 140 passed.
- python scripts/score_deflection_pii_recall.py --json -- no
  `partial_name_token` samples, `person_name-003` absent from leak samples,
  cue-prefixed recall 1.0, must-survive violations 0, and deferred
  `person_name-001` cue-less leak remains with `free_high_severity_leak_count=1`.
- python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_score_deflection_pii_recall.py -- passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- bash scripts/check_ascii_python.sh and python scripts/audit_extracted_pipeline_ci_enrollment.py -- ASCII passed; OK: 188 matching tests are enrolled.
- Pending before push: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-deflection-pii-partial-name-token-scrub.md.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 29 |
| `plans/PR-Deflection-PII-Partial-Name-Token-Scrub.md` | 134 |
| `tests/test_content_ops_deflection_report.py` | 8 |
| `tests/test_score_deflection_pii_recall.py` | 48 |
| **Total** | **219** |
