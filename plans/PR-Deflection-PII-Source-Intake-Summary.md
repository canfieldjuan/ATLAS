# PR-Deflection-PII-Source-Intake-Summary

## Why this slice exists

#1742's remaining true work is the operator-derived corpus path: pick a real
source, label it transiently, normalize/surrogate it, then use the scorer
against that artifact. The current builder can already emit a surrogate artifact
from labeled local JSON, but there is no safe intake summary that lets the
operator/reviewer inspect corpus mix, class coverage, name subtype split, and
labeling-quality errors without opening raw tickets or raw label spans.

Root cause: the labeled-source handoff has only a build-artifact path. That is
fine once the source is valid, but it leaves the pre-artifact review step
implicit and tempts ad hoc inspection of raw source files. This slice fixes that
handoff root by adding a sanitized summary output for the existing labeled-source
input contract.

This is a narrow corpus-first vertical slice. It does not choose the real source,
label the real data, alter scrubber behavior, pick thresholds, or flip the
advisory into a gate.

Diff-size note: the final diff is over the 400 LOC soft cap after review fixes
because the safety contract needs negative no-echo fixtures for malformed schema
values, unsafe must-survive reasons, and CLI path clobbering in the same PR as
the summary code. Splitting those tests out would leave the blocker fixed
without the guard that proves the class is closed.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 4

1. Add a sanitized labeled-source intake summary over the existing
   `deflection_pii_labeled_source.v1` JSON envelope.
2. Wire the corpus builder CLI to write that summary independently of the
   surrogate artifact so a transient operator source can be reviewed without
   persisting raw PII.
3. Add focused tests proving valid summaries include only aggregate metadata and
   invalid summaries return sanitized errors without raw spans.

### Review Contract

- Acceptance criteria:
  - [ ] The summary requires the explicit labeled-source schema version and
        fails closed on mismatches.
  - [ ] A valid source summary reports ticket count, label count, labels by
        class/severity/origin field, person-name subtype counts, and
        must-survive counts/reasons.
  - [ ] The summary can be written without writing the surrogate artifact.
  - [ ] Invalid input summaries include error codes/locations but do not echo
        raw source text, raw label spans, or high-risk tokens.
  - [ ] Malformed schema-version values and free-form must-survive reasons are
        not persisted verbatim in the summary or CLI stderr.
  - [ ] Summary and artifact output paths cannot clobber each other.
  - [ ] Existing artifact generation behavior remains compatible for the
        current tiny fixture and existing tests.
- Affected surfaces:
  - `extracted_content_pipeline/deflection_pii_eval_corpus.py`
  - `scripts/build_deflection_pii_surrogate_eval_corpus.py`
  - `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- Risk areas: accidental raw PII echo in summary/error output, schema drift,
  CLI compatibility, and over-broad source validation.
- Reviewer rules triggered: R1, R2, R3, R10, R13, R14; boundary-probe required
  because this adds a validation/reporting gate over raw-source-adjacent input.

### Files touched

- `extracted_content_pipeline/deflection_pii_eval_corpus.py`
- `plans/PR-Deflection-PII-Source-Intake-Summary.md`
- `scripts/build_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`

## Mechanism

The corpus module gets a pure summary helper for the labeled-source envelope. It
first checks the source schema version, then reuses the existing
`build_surrogate_eval_corpus` validation/surrogation path so summary readiness
matches artifact readiness. On success, it derives aggregate counts only from
the surrogate artifact labels and must-survive records. On failure, it returns
the existing sanitized error records and does not include raw text or raw spans.

The review-fix pass keeps untrusted metadata from becoming summary metadata:
malformed schema-version values collapse to safe `missing`/`invalid` buckets,
and must-survive reasons count only a fixed safe set
(`security_reference`, `compliance_reference`, `tenant_source_id`, and
`must_survive`), with everything else counted as `other`.

The CLI gains `--summary-output`. Callers may pass only `--summary-output` for a
pre-artifact intake check, or pass both `--summary-output` and `--output` to
write the safe summary plus the surrogate eval artifact. The CLI rejects
identical resolved summary/artifact paths and reuses the already-built surrogate
result when both outputs are requested.

## Intentional

- No raw input persistence, no raw source excerpting, and no raw label-span
  samples in the summary.
- No threshold recommendation or advisory-to-gating flip; the summary is corpus
  readiness evidence, not a policy decision.
- No new detector/scrubber behavior. Existing artifact validation remains the
  source of truth for what is safe to surrogate.
- Review fixes intentionally bucket unknown must-survive reasons instead of
  trying to validate free-form rationale text, because the summary only needs a
  safe aggregate readiness view.

## Deferred

- Operator selection of the real transient source, corpus size/archetype mix,
  labeling ownership, and labeling-quality review remain open #1742 decisions.
- Building the larger derived surrogate artifact from that real source remains a
  follow-up after the operator supplies/labels the source.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- `python -m pytest` on
  `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` -> 17
  passed.
- `python -m py_compile` on
  `extracted_content_pipeline/deflection_pii_eval_corpus.py`,
  `scripts/build_deflection_pii_surrogate_eval_corpus.py`, and
  `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` -> passed.
- `python` `scripts/score_deflection_pii_recall.py` with JSON output -> status
  ok; gate-eligible high-severity free leaks 0, deferred open-set name leaks 1.
- `python` `scripts/audit_extracted_pipeline_ci_enrollment.py` -> OK, 188
  matching tests enrolled.
- `bash` `scripts/check_ascii_python.sh` -> passed.
- `bash` `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `python` `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py`
  `extracted_content_pipeline` -> clean.
- `python` `scripts/audit_extracted_standalone.py` with fail-on-debt -> Atlas
  runtime import findings: 0.
- `bash` `scripts/run_extracted_pipeline_checks.sh` -> reasoning core 295
  passed; extracted content 4841 passed / 15 skipped / 1 warning.
- Pending before push: `bash` `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/deflection_pii_eval_corpus.py` | 114 |
| `plans/PR-Deflection-PII-Source-Intake-Summary.md` | 148 |
| `scripts/build_deflection_pii_surrogate_eval_corpus.py` | 70 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 153 |
| **Total** | **485** |
