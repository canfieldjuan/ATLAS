# PR-Deflection-PII-Review-Bundle-Score

## Why this slice exists

#1742 has moved from the tiny-corpus scrubber loop into the operator-derived
corpus handoff. #1769 added a single review-bundle directory that can hold the
sanitized source intake JSON summary, sanitized Markdown summary, and
surrogate-only eval corpus artifact. The remaining handoff gap is that scoring
that bundle still requires a separate ad hoc command with two output paths.

Root cause: the review bundle has predictable source-review and corpus filenames,
but `score_deflection_pii_recall.py` only knows about a corpus path plus
individual score-output paths. That makes the "review this derived corpus before
versioning it" handoff less reproducible than the bundle creation step.

This slice fixes the handoff root by adding a bundle scoring mode that reads the
known review-bundle corpus filename and writes predictable recall-score JSON and
Markdown files in the same directory. It does not choose the real operator
source, select thresholds, promote advisory output to a gate, change scrubber
behavior, or close the deferred cue-less/open-set name gap.

Review-fix root cause: the first bundle-scoring pass derived child paths before
validating that `--review-bundle-dir` was actually a directory, and `_load_corpus`
caught file/JSON errors but not decode errors. This change fixes those roots by
rejecting existing non-directory bundle paths during argument parsing and routing
non-UTF-8 corpus files through the same sanitized `corpus_load_failed` path as
missing or malformed JSON corpora.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 3

1. Add a scorer `--review-bundle-dir` mode for the bundle produced by
   `scripts/build_deflection_pii_surrogate_eval_corpus.py`.
2. Write predictable bundle score filenames for recall JSON and recall Markdown.
3. Keep existing `--corpus`, `--output`, `--markdown-output`, and `--json`
   behavior compatible.
4. Add focused tests proving bundle scoring writes the expected artifacts,
   rejects split output flags, fails closed on missing bundle corpus, and does
   not echo raw spans/tokens in score artifacts.
5. Review update: reject existing non-directory bundle paths before deriving
   child paths and treat non-UTF-8 corpus files as sanitized load failures.

### Review Contract

- Acceptance criteria:
  - [ ] Given a review bundle containing the builder artifact filename recorded
        by `REVIEW_BUNDLE_CORPUS_NAME`, the scorer writes the score JSON and
        Markdown filenames recorded by `REVIEW_BUNDLE_SCORE_NAME` and
        `REVIEW_BUNDLE_SCORE_MARKDOWN_NAME` in the same directory.
  - [ ] Bundle score artifacts use the existing recall-score schema and Markdown
        formatter.
  - [ ] `--review-bundle-dir` cannot be combined with individual corpus/output
        flags, so operators do not accidentally score one corpus while writing
        into another bundle.
  - [ ] A missing bundle corpus fails closed with the existing sanitized
        `corpus_load_failed` error path.
  - [ ] Existing non-directory bundle paths are rejected without echoing the
        local path.
  - [ ] Non-UTF-8 bundle corpora fail closed with sanitized score artifacts
        instead of surfacing decoder exceptions.
  - [ ] Score JSON/Markdown do not include raw spans or person-name tokens.
- Affected surfaces:
  - `scripts/score_deflection_pii_recall.py`
  - `tests/test_score_deflection_pii_recall.py`
- Risk areas: bundle/file mismatch, accidental raw PII echo in score artifacts,
  and scope creep into policy or scrubber behavior.
- Reviewer rules triggered: R1, R2, R3, R10, R14; boundary-probe required
  because this writes raw-source-adjacent review artifacts.

### Files touched

- `plans/PR-Deflection-PII-Review-Bundle-Score.md`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The scorer gets a `--review-bundle-dir` option. When present, argument parsing
rejects `--corpus`, `--output`, and `--markdown-output`, then derives the fixed
bundle paths from these constants:

- `REVIEW_BUNDLE_CORPUS_NAME`
- `REVIEW_BUNDLE_SCORE_NAME`
- `REVIEW_BUNDLE_SCORE_MARKDOWN_NAME`

The scoring itself stays unchanged: the command loads the derived corpus,
passes it to `score_corpus`, and writes the existing JSON and Markdown summary
formats. Missing or unreadable corpus input uses the current sanitized
`corpus_load_failed` error path and returns non-zero. Existing non-directory
bundle paths are rejected before derived children are assigned, and decoded-input
failures such as non-UTF-8 corpus files are caught with the same sanitized load
failure envelope.

## Intentional

- No threshold recommendation and no advisory-to-gating flip. This only makes
  the review-bundle measurement handoff reproducible.
- No source selection, no raw-source persistence, and no raw-label persistence.
- No scrubber, detector, scorer math, or NER/open-set-name behavior change.
- The bundle scorer rejects split output flags instead of allowing partial
  bundle writes because #1769 made the bundle path intentionally predictable.

## Deferred

- Operator selection of the real transient source, corpus size/archetype mix,
  labeling ownership, and labeling-quality review remain open #1742 decisions.
- Running the full bundle plus scoring flow against that real source and
  committing/versioning the resulting surrogate-only eval artifact remains a
  follow-up once the source is supplied and reviewed.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_score_deflection_pii_recall.py -q -- 30 passed.
- python -m py_compile scripts/score_deflection_pii_recall.py tests/test_score_deflection_pii_recall.py -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- git diff --check -- passed.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- python scripts/maturity_sweep_file_lane.py <deflection lane files> --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 ... -- ratchet gate passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-PII-Review-Bundle-Score.md` | 135 |
| `scripts/score_deflection_pii_recall.py` | 32 |
| `tests/test_score_deflection_pii_recall.py` | 122 |
| **Total** | **289** |
