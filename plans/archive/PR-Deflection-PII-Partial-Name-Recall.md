# PR-Deflection-PII-Partial-Name-Recall

## Why this slice exists

#1751 closed the measured cued-name and ISO precision scrubber loop, but its
LGTM review found a residual measurement gap:

- "Customer Mary Jane Watson Report" renders as "Customer [redacted-name] Watson
  Report";
- the surname token is still visible;
- the current scorer only checks whether the full surrogate span survives, so
  it would mark "Mary Jane Watson" as redacted and hide the partial leak.

Root cause: the recall scorer is span-exact for all classes. That is correct
for emails, phone numbers, cards, and identifiers, but it is too coarse for
multi-token person names because a surviving name token can still identify the
customer even when the full name string is gone. This PR fixes the measurement
root by making person-name scoring detect residual name-token leaks and by
adding the residual three-token-name-plus-label shape to the committed tiny
corpus. It does not change the scrubber.

Review found one deeper measurement root: token residue must be attributed only
inside the owning ticket's rendered context, not the aggregate report text,
otherwise shared surnames can create false leaks for unrelated labels. This PR
fixes that attribution root in the scorer.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice

1. Add a committed surrogate-only tiny-corpus row for a cue-prefixed
   three-token name followed by a title-cased label/tier word.
2. Teach the scorer to count a person-name label as leaked when any meaningful
   token from the surrogate name survives on a surface where the full name
   reached the baseline surface, scoped to the label's own ticket context.
3. Mark the leak sample as partial person-name residue without echoing the
   leaked token or full surrogate span.
4. Update tests to prove:
   - the new tiny-corpus row is counted;
   - the scorer reports the partial cue-prefixed leak;
   - leak samples remain redacted and do not contain spans or tokens;
   - the #1751 ISO and must-survive precision wins remain intact.

### Files touched

- `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json`
- `plans/PR-Deflection-PII-Partial-Name-Recall.md`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_score_deflection_pii_recall.py`

### Review Contract

- Acceptance criteria:
  - [ ] The tiny corpus includes a surrogate-only three-token cue-prefixed
        person name followed by a title-cased trailing word.
  - [ ] The scorer reports partial person-name residue as a leak when a
        meaningful name token survives after full-name redaction.
  - [ ] Shared surnames in another ticket do not create false partial leaks for
        a fully redacted label.
  - [ ] Leak samples identify the class/surface/subtype and partial-leak kind
        without echoing the leaked token or full span.
  - [ ] The scorer continues to report zero must-survive violations.
  - [ ] The scrubber behavior is unchanged in this PR.
- Affected surfaces: #1742 scorer, committed tiny eval corpus, scorer tests.
- Risk areas: false-positive partial-token matching, leak-sample safety,
  misleading recall math, accidental scrubber scope creep.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R13, R14.

## Mechanism

The scorer already builds baseline and scrubbed surface text for each label.
This PR keeps that flow and changes only the label leak predicate:

- non-name labels still use exact full-span matching;
- person-name labels first use exact full-span matching, then check meaningful
  name tokens with word boundaries inside that label's own scrubbed ticket
  surface when the full span reached the aggregate baseline surface;
- partial-token hits are recorded as a leak with a leak-kind marker, not with
  the sensitive token.

The fixture adds a second tiny-corpus ticket whose answer contains a
three-token cue-prefixed surrogate name followed by a trailing title-cased word.
That makes the residual detector ceiling visible in the advisory artifact while
leaving the deferred NER/open-set decision untouched.

## Intentional

- No scrubber change. The #1751 reviewer explicitly called out this shape as
  the deterministic-name ceiling; this slice measures it instead of chasing
  another regex round.
- No threshold or gate change. The advisory artifact is allowed to become more
  honestly red while the corpus is still tiny and the operator thresholds are
  deferred.
- No raw PII. The added corpus row uses surrogate-only text and labels.
- Two-character name tokens are intentionally not counted by the partial-token
  detector yet. The current curated surrogate corpus uses longer tokens; short
  surnames need a future scorer/corpus follow-up before they are reliable.

## Deferred

- NER/open-set person-name detection.
- Operator-derived larger gold corpus and threshold selection.
- Paid artifact private-note exclusion for SSN/card residue.
- Advisory-to-gating promotion.

Parked hardening: none.

## Verification

- pytest tests/test_score_deflection_pii_recall.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -q -- 26 passed.
- python scripts/score_deflection_pii_recall.py --json -- reports ticket_count 2, label_count 9, cue-prefixed person-name expected 5/leaks 2/recall 0.6, partial_name_token samples for person_name-003, and must-survive violations 0.
- simulated no-fpdf scorer run -- reports paid_pdf skipped and cue-prefixed expected 3/leaks 1/recall 0.6667.
- python -m py_compile scripts/score_deflection_pii_recall.py tests/test_score_deflection_pii_recall.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- bash scripts/local_pr_review.sh -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json` | 32 |
| `plans/PR-Deflection-PII-Partial-Name-Recall.md` | 132 |
| `scripts/score_deflection_pii_recall.py` | 110 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 20 |
| `tests/test_score_deflection_pii_recall.py` | 79 |
| **Total** | **373** |
