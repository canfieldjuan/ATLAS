# PR-Deflection-PII-Markdown-Advisory-Summary

## Why this slice exists

#1742 section 8 defines the recall/precision harness output as a report
artifact with JSON plus a markdown summary. The merged advisory CI slice
currently writes and uploads only the JSON advisory file, so the measurement is
machine-readable but not reviewer-friendly in the artifact bundle.

Root cause: the scorer CLI only has a JSON `--output` path, and the extracted
checks workflow points the upload at that single JSON file. This slice fixes the
artifact-shape root by adding a redacted markdown summary output and uploading
the artifact directory, while preserving the existing advisory/non-gating
behavior.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 5

1. Add a scorer CLI option that writes a redacted markdown summary beside the
   existing JSON summary.
2. Wire extracted-checks to emit and upload both JSON and markdown advisory
   artifacts.
3. Add focused tests proving the markdown summary contains the headline,
   per-surface/person-name/must-survive facts needed for review without raw PII
   spans or tokens.

### Review Contract

Acceptance criteria:
- `scripts/score_deflection_pii_recall.py --markdown-output <path>` writes a
  markdown file for the existing tiny corpus without changing JSON output.
- The markdown summary includes the all-in headline, gate-eligible headline,
  deferred open-set name count, person-name split, must-survive violation count,
  per-surface recall table, and bounded leak sample metadata.
- The markdown summary does not include raw label spans, must-survive spans, or
  synthetic high-risk tokens from the corpus.
- `.github/workflows/extracted_pipeline_checks.yml` writes both
  the JSON advisory file and the markdown advisory file, then uploads the
  artifact directory.
- This PR does not flip advisory output into a gate and does not change scrubber
  behavior, thresholds, or the cue-less/open-set name deferral.

Affected surfaces:
- `scripts/score_deflection_pii_recall.py`
- `.github/workflows/extracted_pipeline_checks.yml`
- `tests/test_score_deflection_pii_recall.py`
- `tests/test_extracted_pipeline_pii_recall_advisory_artifact.py`

Risk areas:
- Accidentally echoing PII/surrogate spans into the markdown report.
- Weakening the existing JSON artifact or advisory/non-gating workflow contract.
- Making the markdown summary inconsistent with the JSON summary.

- Reviewer rules triggered: R1 Requirements match, R2 Test evidence, R10
  Maintainability, R12 CI test enrollment, R14 Codebase verification.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-PII-Markdown-Advisory-Summary.md`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_extracted_pipeline_pii_recall_advisory_artifact.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The scorer already builds a sanitized summary dictionary: headline counts,
per-surface/class counts, person-name subtype counts, must-survive violation
metadata, and leak samples that carry only surrogate IDs and metadata. This
slice adds a pure renderer that formats that summary into markdown tables. The
renderer consumes the existing sanitized summary rather than raw corpus records,
so it cannot access label spans or must-survive spans.

The CLI gains `--markdown-output`, independent from `--output` and `--json`.
The extracted-checks workflow passes both output paths and uploads the containing
directory so reviewers can inspect either artifact.

## Intentional

- No advisory-to-gating flip. #1742 still leaves threshold and gate timing as
  operator decisions.
- No scrubber or corpus change. The current tiny-corpus result remains:
  deterministic/gate-eligible leaks are clean; the cue-less/open-set
  `person_name-001` gap stays visible.
- The markdown leak table is metadata-only; no spans, raw tokens, or source text.

## Deferred

- Operator-derived gold corpus, thresholds, and advisory-to-gating timing remain
  open #1742 decisions.
- Model-based NER for cue-less/open-set names remains separate.

Parked hardening: none.

## Verification

- `pytest tests/test_score_deflection_pii_recall.py tests/test_extracted_pipeline_pii_recall_advisory_artifact.py -q`
  -- 22 passed.
- `python scripts/score_deflection_pii_recall.py --output tmp/deflection-pii-markdown-advisory/deflection-pii-recall-advisory.json --markdown-output tmp/deflection-pii-markdown-advisory/deflection-pii-recall-advisory.md --json`
  -- status ok; wrote both advisory artifacts.
- Python bytecode compile for the scorer script and focused tests -- passed.
- `git diff --check` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 3 |
| `plans/PR-Deflection-PII-Markdown-Advisory-Summary.md` | 116 |
| `scripts/score_deflection_pii_recall.py` | 187 |
| `tests/test_extracted_pipeline_pii_recall_advisory_artifact.py` | 9 |
| `tests/test_score_deflection_pii_recall.py` | 46 |
| **Total** | **361** |
