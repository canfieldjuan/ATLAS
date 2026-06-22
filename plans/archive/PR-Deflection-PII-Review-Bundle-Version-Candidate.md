# PR-Deflection-PII-Review-Bundle-Version-Candidate

## Why this slice exists

#1742 now has a sanitized local source-review bundle path: build the bundle,
score it, and keep a manifest that records the artifact inventory and score
status. The next handoff gap is still before the real operator source: once a
bundle has been reviewed, there is no narrow command that validates the bundle
is ready and exports only the surrogate eval corpus as the versionable candidate.

Root cause: the build/score bundle commands make the bundle self-describing, but
the versioning step is still convention. An operator could copy the corpus out
of a blocked, stale, unscored, or mismatched bundle by hand. This slice fixes
that handoff root by adding a small fail-closed exporter for reviewed bundles.
It does not choose the real source, commit a real-source-derived artifact,
select thresholds, promote the advisory score to a gate, or change scrubber or
scorer math.

Diff budget note: this is over the 400 LOC target because the new surface is a
validator/promoter, so the PR includes branch-covering tests for every failure
class it detects plus extracted-pipeline CI enrollment. Splitting the tests from
the validator would leave the fail-closed contract unproven.

Review-fix root cause: the first exporter pass validated that the manifest,
corpus, and score each looked individually ready, but it did not prove that the
three files still described the same candidate. That let stale score artifacts,
malformed-but-non-empty corpora, missing headline metrics, and `--force` writes
to other bundle artifacts slip through. The fix cross-checks the score input and
manifest counts against the corpus being copied, reuses the scorer corpus
validator, requires headline metrics, and rejects every reserved bundle output
target.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 5

1. Add a CLI that validates a scored review bundle manifest before exporting
   the surrogate-only eval corpus to an operator-provided candidate path.
2. Fail closed on blocked/stale manifests, fixed filename mismatches, missing
   or invalid corpus files, failed score artifacts, and output clobbering.
3. Add CI-enrolled tests proving the happy path and each detector branch.
4. Review update: prove the manifest, corpus, and score artifact still describe
   the same candidate before export.
5. Keep the real-source run and advisory-to-gating decision deferred.

### Review Contract

- Acceptance criteria:
  - [ ] A bundle with `status: ok`, `score_status: ok`, the fixed corpus
        filename, and an ok score artifact copies the corpus to the requested
        output path.
  - [ ] Blocked, unscored, mismatched, missing-corpus, failed-score, and
        existing-output cases return sanitized error codes and do not write the
        candidate artifact.
  - [ ] Score input counts and manifest corpus counts must match the corpus
        being exported.
  - [ ] Malformed ticket/label/must-survive payloads, missing headline metrics,
        and reserved bundle output targets fail closed.
  - [ ] The exporter copies only the surrogate corpus; it does not persist raw
        source, source summaries, score markdown, or manifest files.
  - [ ] The new test file is enrolled in extracted-pipeline CI.
- Affected surfaces:
  - `scripts/promote_deflection_pii_review_bundle.py`
  - extracted-pipeline test runner/workflow enrollment
- Risk areas: accidentally versioning a stale/blocked bundle, local path or raw
  source echo in errors, and scope creep into threshold policy.
- Reviewer rules triggered: R1, R2, R3, R10, R14; boundary-probe required
  because this is a validator/promoter for raw-source-adjacent artifacts.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-PII-Review-Bundle-Version-Candidate.md`
- `scripts/promote_deflection_pii_review_bundle.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_pii_review_bundle_candidate.py`

## Mechanism

`scripts/promote_deflection_pii_review_bundle.py` accepts a review-bundle
directory and `--output` path. It reads the manifest named by
`REVIEW_BUNDLE_MANIFEST_NAME`, requires the manifest schema/status/score status
to be ready, requires the manifest's fixed corpus and score entries to point to
the known bundle filenames, reads the corpus and score JSON, and then copies
only the corpus named by `REVIEW_BUNDLE_ARTIFACT_NAME` to the requested
candidate path.

The review update reuses the scorer's corpus validator for ticket/label and
must-survive shape, derives counts from the actual corpus file, compares those
counts with both the manifest corpus entry and the score input block, requires
the score headline metrics to be present as integers, and rejects any output path
that resolves to a reserved review-bundle artifact.

The command emits a small JSON envelope with counts/headline metadata on
success. Failures return non-zero and emit sanitized error codes; local paths
and raw source content are not needed for the error contract.

## Intentional

- No real-source run and no committed real-source-derived artifact. Tests use
  the existing surrogate-only tiny corpus.
- No advisory-to-gating flip and no threshold recommendation. A bundle can be a
  valid version candidate even while thresholds remain an operator decision.
- No manifest hash/timestamp semantics in this slice; fixed filenames plus
  manifest/score readiness are enough to prevent hand-copying stale artifacts.

## Deferred

- Operator source selection, corpus size/archetype mix, labeling ownership,
  labeling-quality review, and the real local bundle run remain deferred.
- Committing or otherwise versioning the real-source-derived surrogate artifact
  remains deferred until that source exists and is reviewed.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py tests/test_score_deflection_pii_recall.py tests/test_content_ops_deflection_pii_review_bundle_candidate.py -q -- 84 passed.
- python -m py_compile scripts/promote_deflection_pii_review_bundle.py tests/test_content_ops_deflection_pii_review_bundle_candidate.py -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 191 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- git diff --check -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- python scripts/maturity_sweep_file_lane.py scripts/promote_deflection_pii_review_bundle.py --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-pii-review-bundle-version-candidate.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `plans/PR-Deflection-PII-Review-Bundle-Version-Candidate.md` | 140 |
| `scripts/promote_deflection_pii_review_bundle.py` | 345 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_pii_review_bundle_candidate.py` | 386 |
| **Total** | **874** |
