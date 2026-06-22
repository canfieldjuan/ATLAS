# PR-Deflection-PII-Source-Review-Bundle

## Why this slice exists

#1742 has the corpus-first measurement path in place: the surrogate eval schema,
the recall scorer, the advisory artifact, the sanitized labeled-source intake
JSON summary (#1765), and the sanitized Markdown intake summary (#1767). The
remaining true work needs an operator-supplied labeled source, but the current
CLI still asks the operator to coordinate three separate output paths by hand.

This slice adds the thinnest handoff artifact path for that operator source: a
single review-bundle directory that writes the sanitized JSON intake summary,
the sanitized Markdown summary, and the surrogate eval artifact with predictable
filenames. Invalid labeled sources still write only the sanitized review
summaries and do not persist a surrogate artifact.

This is a vertical slice because it exercises the real local handoff flow:
operator-labeled local source -> sanitized intake summaries -> surrogate eval
artifact bundle. It does not choose the real source, set corpus-mix targets,
select thresholds, promote advisory output to a gate, or address the deferred
cue-less/open-set name gap.

Review-fix root cause: bundle mode assigned a stable artifact path before
validation, but the invalid-summary return path exited before overwriting or
removing a prior artifact in that directory. This change fixes the root for
bundle mode by making a blocked source clear the bundle artifact path before it
writes blocked summaries, so a reused bundle cannot contain summaries for one
source and a surrogate corpus from another.

Diff budget note: the synced LOC total is over 400 because it includes this
plan plus the review-fix root-cause/verification record. The code and tests are
the narrow bundle CLI path and the stale-artifact regression required to close
the review finding.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 3

1. Add a `--review-bundle-dir` CLI option for
   `scripts/build_deflection_pii_surrogate_eval_corpus.py`.
2. Write predictable sanitized bundle filenames for summary JSON, summary
   Markdown, and the surrogate eval artifact on valid sources.
3. Preserve the fail-closed invalid-source path: sanitized summaries are written,
   stderr is sanitized, and no surrogate artifact is written.
4. Remove any prior bundle artifact before writing blocked summaries for an
   invalid source rerun.
5. Add focused CLI tests for valid bundles, invalid bundles, stale-artifact
   invalid reruns, and flag-shape rejection.

### Review Contract

- Acceptance criteria:
  - [ ] A valid labeled source can be converted into a review bundle directory
        containing sanitized JSON summary, sanitized Markdown summary, and a
        surrogate-only eval artifact.
  - [ ] An invalid labeled source writes sanitized review summaries, returns a
        non-zero exit, and does not write the eval artifact.
  - [ ] `--review-bundle-dir` cannot be combined with the individual output
        flags, so operators do not accidentally split or clobber the bundle.
  - [ ] Existing individual output flags remain compatible.
  - [ ] No raw source text, raw label spans, or high-risk raw tokens are echoed
        in bundle files, stdout, or stderr.
- Affected surfaces:
  - `scripts/build_deflection_pii_surrogate_eval_corpus.py`
  - `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- Risk areas: raw PII echo in bundle artifacts, stale artifact creation on
  invalid sources, and scope creep into corpus policy or scrubber behavior.
- Reviewer rules triggered: R1, R2, R3, R10, R13, R14; boundary-probe required
  because this writes raw-source-adjacent validation artifacts.

### Files touched

- `plans/PR-Deflection-PII-Source-Review-Bundle.md`
- `scripts/build_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`

## Mechanism

The CLI gains `--review-bundle-dir`. When present, the parser rejects any
individual output flags and the command uses fixed filenames inside the bundle
directory:

- source intake summary JSON
- source intake summary Markdown
- deflection PII surrogate eval corpus JSON

The command uses the same sanitized summary and Markdown formatter already
landed by #1765/#1767. Because the bundle includes an artifact path, the CLI
builds the surrogate corpus once and passes that result into
`summarize_labeled_source`. If validation fails, the sanitized summaries are
written after any existing bundle artifact is removed, the command returns exit
1 with sanitized errors, and the surrogate artifact is not written.

## Intentional

- No raw-source persistence and no raw-label persistence.
- No operator source choice, corpus-size target, threshold recommendation, or
  advisory-to-gating flip.
- No detector/scrubber behavior change and no NER/open-set-name work.
- The bundle option is mutually exclusive with individual output flags to keep
  this handoff predictable and avoid inventing partial-bundle semantics.

## Deferred

- Operator selection of the real transient source, corpus size/archetype mix,
  labeling ownership, and labeling-quality review remain open #1742 decisions.
- Running the bundle against that real source and committing/versioning the
  resulting surrogate-only eval artifact remains a follow-up once the source is
  supplied and reviewed.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -q -- 29 passed.
- python -m py_compile scripts/build_deflection_pii_surrogate_eval_corpus.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -- passed.
- git diff --check -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- python scripts/maturity_sweep_file_lane.py <deflection lane files> --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 ... -- ratchet gate passed.
- bash scripts/run_extracted_pipeline_checks.sh -- reasoning core: 295 passed; extracted content pipeline: 4857 passed, 15 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-PII-Source-Review-Bundle.md` | 138 |
| `scripts/build_deflection_pii_surrogate_eval_corpus.py` | 106 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 182 |
| **Total** | **426** |
