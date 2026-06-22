# PR-Deflection-PII-Review-Bundle-Pipeline

## Why this slice exists

#1742 now has the local operator handoff in separate validated steps:
`build_deflection_pii_surrogate_eval_corpus.py --review-bundle-dir` creates a
sanitized review bundle, `score_deflection_pii_recall.py --review-bundle-dir`
scores it, and `promote_deflection_pii_review_bundle.py` exports a validated
version candidate. #1797 closed the fail-closed candidate exporter, but the
full local handoff still depends on an operator sequencing three commands by
hand.

Root cause: the review-bundle flow is validated at each command boundary, but
there is no single orchestration command that proves build -> score -> promote
as one local pipeline. That leaves the eventual real-source run more error-prone
than the individual validators: a missed score step, stale bundle directory, or
promotion attempt after a failed build is easy to create manually. This slice
fixes that handoff root by adding a thin one-command pipeline over the existing
validators and proving that it stops before promotion on earlier failures.

Review-fix root cause: the first pipeline pass delegated every path guard to the
child commands, but orchestration introduced cross-command path relationships
the children cannot see: source input vs bundle artifact outputs, candidate
output vs source input, and stale downstream score artifacts when a reused
bundle directory fails during rebuild. This fix moves those orchestration-owned
guards into a pipeline preflight and clears generated bundle artifacts before a
new build starts.

This is a vertical slice because it exercises the real local operator shape
end-to-end on the committed surrogate fixture input contract: labeled source ->
review bundle -> recall score artifacts -> validated candidate export. It does
not choose or run a real operator source, commit a real-source-derived artifact,
select thresholds, promote the advisory score to a gate, change scrubber/scorer
math, or address the deferred cue-less/open-set name gap.

Diff budget note: the synced LOC total is over the 400 LOC soft cap because the
new surface is an orchestration gate, so the PR includes boundary tests for each
stop point (build, score, promote) plus extracted-pipeline CI enrollment. The
implementation is a thin wrapper over existing validators; the larger share of
the diff is the plan and branch-covering tests that prove the wrapper does not
promote after a failed earlier step.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 5

1. Add a local pipeline CLI that accepts a labeled-source JSON path, a review
   bundle directory, and a candidate output path, then runs the existing build,
   score, and promote validators in order.
2. Emit one sanitized pipeline result envelope that reports the failed step or
   the promoted candidate metadata without echoing raw source text or raw label
   spans.
3. Add focused tests for the success path, build-failure stop, score-failure
   stop, and promote-failure/no-clobber behavior.
4. Enroll the new test and script in extracted-pipeline checks.
5. Review update: reject source/candidate/bundle path collisions before build
   and clear stale generated bundle outputs before rebuilding.

### Review Contract

- Acceptance criteria:
  - [ ] A valid labeled source can produce a review bundle, recall score JSON /
        Markdown, manifest score update, and validated candidate artifact from
        one CLI command.
  - [ ] If bundle build fails, the command returns non-zero, records
        `failed_step=build`, writes only sanitized review artifacts/manifest as
        the builder already allows, and does not score or promote.
  - [ ] If scoring fails after a bundle exists, the command returns non-zero,
        records `failed_step=score`, leaves the score failure artifacts, and
        does not promote a candidate.
  - [ ] If promotion fails, the command returns non-zero with
        `failed_step=promote` and does not overwrite an existing candidate
        without `--force`.
  - [ ] The command rejects a candidate output that resolves to the source input,
        even when `--force` is set.
  - [ ] The command rejects a source input that resolves to any reserved
        generated review-bundle artifact path.
  - [ ] Reusing a review-bundle directory clears old generated score artifacts
        before build, so a failed rebuild cannot leave stale score files behind.
  - [ ] The pipeline does not introduce new validation semantics; it delegates
        build, score, and promotion decisions to the existing commands.
  - [ ] Pipeline stdout/stderr and artifacts do not echo raw source text, raw
        label spans, or high-risk raw tokens from the labeled source.
  - [ ] The new test file is enrolled in extracted-pipeline CI.
- Affected surfaces:
  - `scripts/run_deflection_pii_review_bundle_pipeline.py`
  - extracted-pipeline test runner/workflow enrollment
- Risk areas: hiding step-specific failure codes, accidentally promoting after
  a failed earlier step, raw PII echo via captured child output, and scope creep
  into threshold policy or real-source artifact versioning.
- Reviewer rules triggered: R1, R2, R3, R10, R14; boundary-probe required
  because this orchestrates raw-source-adjacent validators and candidate export.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-PII-Review-Bundle-Pipeline.md`
- `scripts/run_deflection_pii_review_bundle_pipeline.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_pii_review_bundle_pipeline.py`

## Mechanism

The new script imports the existing three CLI modules instead of reimplementing
their validators. Before invoking the build step, it resolves the source input,
review-bundle directory, and candidate output, rejects orchestration-level path
collisions, and clears known generated bundle artifacts so a reused directory
cannot carry stale score files into a failed rebuild.

It then captures each child CLI's stdout/stderr in-memory, calls the child
`main()` with explicit argv, and stops at the first non-zero exit:

1. build:
   `build_deflection_pii_surrogate_eval_corpus.py <input> --review-bundle-dir <dir>`
2. score:
   `score_deflection_pii_recall.py --review-bundle-dir <dir>`
3. promote:
   `promote_deflection_pii_review_bundle.py <dir> --output <candidate> [--force]`

On success, the pipeline parses the promoter's JSON payload and prints a compact
pipeline envelope with the schema version, bundle path, candidate path, counts,
and headline metrics. On failure, it prints a compact sanitized error envelope
with `failed_step`, `exit_code`, and parsed child error codes when they are
available. It deliberately avoids forwarding captured child output wholesale, so
a future child-message regression is less likely to echo raw source material
through the orchestration layer.

Tests use a temporary labeled-source JSON with the same safe shape as the
existing corpus-builder tests. Failure tests mutate the source or intermediate
bundle artifacts to prove that the pipeline stops at build/score/promote
boundaries, rejects source/candidate/bundle path collisions, clears stale score
artifacts on reused bundle dirs, and does not produce the candidate on failure.

## Intentional

- No real-source run and no committed real-source-derived artifact; tests use a
  temporary surrogate-only fixture input.
- No threshold recommendation, no advisory-to-gating flip, and no score math
  change. The pipeline only makes the local handoff reproducible.
- No new validator semantics inside the pipeline. Reusing the existing commands
  keeps root validation owned by the builder/scorer/promoter surfaces already
  reviewed in this lane.
- No shelling out to subprocesses. Direct module calls keep the focused tests
  fast and make captured output easier to inspect for no-echo guarantees.

## Deferred

- Operator source selection, corpus size/archetype mix, labeling ownership,
  labeling-quality review, and the real local bundle run remain deferred.
- Committing or otherwise versioning the real-source-derived surrogate artifact
  remains deferred until that source exists and is reviewed.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_pii_review_bundle_pipeline.py -q -- 8 passed.
- python -m pytest tests/test_content_ops_deflection_pii_review_bundle_pipeline.py tests/test_content_ops_deflection_pii_review_bundle_candidate.py tests/test_score_deflection_pii_recall.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py -q -- 92 passed.
- python -m py_compile scripts/run_deflection_pii_review_bundle_pipeline.py tests/test_content_ops_deflection_pii_review_bundle_pipeline.py -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 192 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- git diff --check -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- python scripts/maturity_sweep_file_lane.py scripts/run_deflection_pii_review_bundle_pipeline.py --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-pii-review-bundle-pipeline.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `plans/PR-Deflection-PII-Review-Bundle-Pipeline.md` | 180 |
| `scripts/run_deflection_pii_review_bundle_pipeline.py` | 306 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_pii_review_bundle_pipeline.py` | 346 |
| **Total** | **835** |
