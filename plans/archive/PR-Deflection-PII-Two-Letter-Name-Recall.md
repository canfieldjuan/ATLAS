# PR-Deflection-PII-Two-Letter-Name-Recall

## Why this slice exists

#1742 requires the recall harness to make person-name recall visible rather than
blend away known gaps. #1753 added partial person-name token measurement, but it
parked a narrow scorer/corpus limitation: the partial-token detector ignored
two-letter name tokens, so a residue such as `Mary Jane Li -> [redacted-name]
Li` could look fully scrubbed even though the short surname survived.

Root cause: the scorer's person-name token extractor used a global three-character
minimum for all name tokens. That was safe against noisy short words, but it
also excluded real two-letter non-initial surname tokens from the partial-leak
detector.

This change fixes the scorer root for the narrow class by admitting two-letter
non-initial person-name tokens while filtering common two-letter words. It does
not change scrubber behavior, thresholds, advisory-vs-gating policy, or the
explicitly deferred cue-less/open-set name gap.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 5

1. Add a surrogate-only tiny-corpus row for a cue-prefixed person name with a
   two-letter surname so the harness exercises that shape end to end.
2. Teach partial person-name leak scoring to count surviving two-letter
   non-initial name tokens, with near-miss probes for common short words and
   hyphenated/apostrophe compounds.

### Review Contract

- Acceptance criteria:
  - [ ] The tiny corpus includes a short-surname cue-prefixed person-name label
        with no raw source data or real PII.
  - [ ] The scorer still reports the current end-to-end corpus as clean for
        gate-eligible free high-severity leaks and cue-prefixed name leaks.
  - [ ] A direct scorer probe counts `[redacted-name] Li`-style residue as
        `partial_name_token`.
  - [ ] A near-miss common-word probe does not create a false partial-name leak.
  - [ ] A near-miss hyphenated/apostrophe compound such as `Li-ion` or `Li's`
        does not create a false partial-name leak.
  - [ ] Existing cue-less/open-set name accounting stays deferred and visible.
- Affected surfaces: extracted-checks advisory scorer JSON/markdown and the
  committed surrogate-only tiny corpus fixture.
- Risk areas: scorer precision, fixture accounting, CI enrollment, sanitized
  advisory output.
- Reviewer rules triggered: R1, R2, R10, R12, R14; boundary-probe required for
  scorer leak-detection logic.

### Files touched

- `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json`
- `plans/PR-Deflection-PII-Two-Letter-Name-Recall.md`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The fixture adds one more surrogate-only ticket whose agent reply contains a
cue-prefixed synthetic person name with a two-letter surname. The scorer already
runs the full deflection path for each ticket, so this increases the measured
cue-prefixed denominator without changing production scrubber behavior.

The scorer keeps the existing three-character token floor for ordinary
person-name tokens, then allows exactly two-character non-initial tokens unless
they are common short words. That makes a surviving short surname count as
`partial_name_token` while keeping obvious short-word near-misses out of the
leak count. After Codex review, the short-token match also treats hyphen and
apostrophe as token-neighbor characters so unrelated compounds such as `Li-ion`
and `Li's` do not count as a standalone leaked surname.

## Intentional

- No scrubber regex change: the current end-to-end short-surname fixture is
  already fully redacted, and this slice is about making the measurement catch
  the class if it regresses.
- No threshold or gating flip: #1742 still says those are operator decisions
  after the representative corpus is selected.
- No model-based NER/open-set names: cue-less names remain measured and
  deferred, not hidden.
- Codex P2 on short-token matching inside hyphenated/apostrophe compounds is
  fixed in the scorer boundary helper and covered by focused near-miss tests.

## Deferred

- Operator-derived representative corpus, corpus size/archetype mix, threshold
  selection, labeling-quality process, and advisory-to-gating promotion remain
  the #1742 open decisions.
- Cue-less/open-set name recall remains the deferred NER/product decision; this
  PR only improves deterministic scorer coverage for cue-prefixed short
  surnames.

Parked hardening: none.

## Verification

- python -m pytest `tests/test_score_deflection_pii_recall.py` -q -> 24 passed.
- python -m pytest `tests/test_score_deflection_pii_recall.py` `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` -q -> 36 passed.
- python `scripts/score_deflection_pii_recall.py` --json -> status ok; `ticket_count=3`, `label_count=11`, `free_high_severity_gate_eligible_leak_count=0`, `cue_prefixed` expected 7 / leaks 0, deferred open-set name leaks 1.
- python -m py_compile `scripts/score_deflection_pii_recall.py` `tests/test_score_deflection_pii_recall.py` `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` -> passed.
- python `scripts/audit_extracted_pipeline_ci_enrollment.py` -> OK, 188 matching tests enrolled.
- bash `scripts/check_ascii_python.sh` -> passed.
- bash `scripts/run_extracted_pipeline_checks.sh` -> reasoning core 295 passed; extracted content 4831 passed / 15 skipped.
- Pending before push: bash `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json` | 32 |
| `plans/PR-Deflection-PII-Two-Letter-Name-Recall.md` | 119 |
| `scripts/score_deflection_pii_recall.py` | 64 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 22 |
| `tests/test_score_deflection_pii_recall.py` | 58 |
| **Total** | **295** |
