# PR-Deflection-PII-Source-Intake-Markdown

## Why this slice exists

#1742 has now closed the fixable tiny-corpus scrubber residue loop: the current
scorer reports zero gate-eligible free high-severity leaks, with only the
explicitly deferred cue-less/open-set name gap still visible. The next true
corpus-first work is the operator-derived labeled source, but the source cannot
be reviewed by passing raw tickets around.

#1765 added a sanitized JSON intake summary for
`deflection_pii_labeled_source.v1`. That is safe, but it is still a machine JSON
artifact. This slice adds the thinnest human-reviewable companion: a Markdown
summary generated from the already-sanitized intake summary, so an operator or
reviewer can inspect corpus mix, class coverage, name subtype split,
must-survive coverage, and sanitized errors without opening raw source files or
raw label spans.

This is a vertical slice because it adds the end-to-end operator-facing artifact
path: labeled local source -> sanitized intake summary -> sanitized Markdown
review artifact. It does not select the real source, label real data, choose
thresholds, turn the advisory into a gate, or close the deferred open-set name
gap.

Diff budget note: the synced LOC total is over 400 because it includes this
plan. The implementation plus tests are 321 net LOC, and the formatter, CLI
path, clobber guard, and raw-echo boundary tests are the indivisible proof for
the review artifact path.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 4

1. Add a Markdown formatter for the existing sanitized labeled-source intake
   summary.
2. Wire the corpus builder CLI to write the Markdown review artifact alongside,
   or independently of, the JSON summary.
3. Add focused tests proving the Markdown includes useful aggregate coverage and
   sanitized error codes, while excluding raw source text, raw label spans, and
   high-risk tokens.

### Review Contract

- Acceptance criteria:
  - [ ] The Markdown is derived only from the sanitized intake summary, not raw
        source records.
  - [ ] Valid-source Markdown reports ticket/label counts, class/severity
        coverage, origin-field coverage, person-name subtype counts, and
        must-survive reason counts.
  - [ ] Invalid-source Markdown reports sanitized error codes/locations without
        echoing raw source text or raw label spans.
  - [ ] The Markdown can be written without writing the surrogate artifact.
  - [ ] JSON summary, Markdown summary, and surrogate artifact output paths
        cannot clobber each other.
  - [ ] Existing JSON summary and artifact behavior remains compatible.
- Affected surfaces:
  - `extracted_content_pipeline/deflection_pii_eval_corpus.py`
  - `scripts/build_deflection_pii_surrogate_eval_corpus.py`
  - `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- Risk areas: accidental raw PII echo in Markdown, CLI path clobbering, and
  broadening the slice into source-selection or threshold policy.
- Reviewer rules triggered: R1, R2, R3, R10, R13, R14; boundary-probe required
  because this renders a raw-source-adjacent validation artifact.

### Files touched

- `extracted_content_pipeline/deflection_pii_eval_corpus.py`
- `plans/PR-Deflection-PII-Source-Intake-Markdown.md`
- `scripts/build_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`

## Mechanism

`deflection_pii_eval_corpus.py` gets a small Markdown formatter that accepts the
already-sanitized summary dictionary returned by `summarize_labeled_source`.
The formatter renders compact tables/lists for counts and errors, using only
safe keys, numeric counts, schema/version labels, and sanitized error metadata.

The CLI gains `--summary-markdown-output`. When present, it builds or reuses the
same sanitized summary used by `--summary-output`, writes the Markdown, and keeps
the existing failure behavior: invalid labeled sources return exit 1 after the
sanitized review artifacts are written. The parser rejects duplicate resolved
output paths across `--output`, `--summary-output`, and
`--summary-markdown-output` so an operator typo cannot overwrite the review
artifact.

## Intentional

- No raw input persistence, no raw source excerpting, and no raw label-span
  samples in Markdown.
- No threshold recommendation or advisory-to-gating flip; the Markdown is a
  review artifact, not policy.
- No new detector/scrubber behavior and no NER/open-set-name work.
- No corpus-size or class-mix gate. The Markdown exposes coverage so the
  operator can make those decisions explicitly.

## Deferred

- Operator selection of the real transient source, corpus size/archetype mix,
  labeling ownership, and labeling-quality review remain open #1742 decisions.
- Building the larger derived surrogate artifact from that real source remains a
  follow-up after the operator supplies/labels the source.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -q -- 22 passed.
- python -m py_compile extracted_content_pipeline/deflection_pii_eval_corpus.py scripts/build_deflection_pii_surrogate_eval_corpus.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -- passed.
- git diff --check -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- python scripts/maturity_sweep_file_lane.py <deflection lane files> --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 ... -- ratchet gate passed.
- bash scripts/run_extracted_pipeline_checks.sh -- reasoning core: 295 passed; extracted content pipeline: 4849 passed, 15 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/deflection_pii_eval_corpus.py` | 167 |
| `plans/PR-Deflection-PII-Source-Intake-Markdown.md` | 132 |
| `scripts/build_deflection_pii_surrogate_eval_corpus.py` | 53 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 102 |
| **Total** | **454** |
