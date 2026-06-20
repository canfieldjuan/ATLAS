# PR-Deflection-PII-Recall-Advisory-Artifact

## Why this slice exists

#1742 Phase 2 calls for the recall/precision harness output to be wired as an
advisory CI artifact before any headline KPI becomes gating. PR #1746 merged the
scorer and CI-enrolled tests, but CI still only proves the scorer can run via
pytest; it does not publish the scorer's JSON result for reviewers/operators to
inspect.

Root cause: the measurement tool is available, but the CI workflow does not
produce a durable measurement artifact. This PR fixes that wiring gap by running
the merged scorer in extracted-checks and uploading its JSON summary as an
artifact. It intentionally leaves the leak KPI advisory, because the larger
operator-derived corpus and threshold decision are still open in #1742.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice

1. Add an extracted-checks step that runs the merged
   `scripts/score_deflection_pii_recall.py` against the committed tiny surrogate
   fixture and writes a JSON summary under a CI artifact directory.
2. Upload that JSON summary with a pinned `actions/upload-artifact` action so
   every extracted-checks run has a reviewer-visible advisory output.
3. Keep the scorer command failure-detecting, but do not fail the workflow on
   the measured headline KPI being nonzero. The current tiny fixture is expected
   to report leaks; that is the point of the advisory artifact.
4. Add a small workflow-contract test so the pinned upload step and scorer
   command cannot disappear while tests stay green.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-PII-Recall-Advisory-Artifact.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_pipeline_pii_recall_advisory_artifact.py`

### Review Contract

- Acceptance criteria:
  - [ ] The extracted-checks workflow runs the merged scorer as a standalone CI
        command and writes deflection-pii-recall-advisory.json.
  - [ ] The JSON is uploaded as an artifact using a pinned upload-artifact SHA,
        not a floating action tag.
  - [ ] The artifact is advisory: the command may fail on script/runtime errors,
        but a nonzero `free_high_severity_leak_count` in the tiny fixture does
        not fail CI.
  - [ ] A CI-facing test verifies the workflow still contains the scorer run,
        artifact path/name, pinned upload action, and no threshold-gating flag.
  - [ ] No scrubber, detector, scorer math, real corpus, or threshold change
        lands in this PR.
- Affected surfaces: one workflow, one workflow-contract test, and this plan.
- Risk areas: accidentally turning the current known leaks into a CI failure,
  using an unpinned action, or creating an artifact step that can silently drift
  away from the scorer.
- Reviewer rules triggered: R1, R2, R6, R10, R12, R14.

## Mechanism

The workflow creates `artifacts/deflection-pii-recall`, runs the existing scorer
with `--output artifacts/deflection-pii-recall/deflection-pii-recall-advisory.json`,
then uploads that directory with `actions/upload-artifact` pinned to the current
v4.6.2 SHA. The scorer still exits nonzero for malformed inputs or runtime
failures, so CI detects broken measurement plumbing. The current measured leak
counts remain data inside the uploaded JSON; the workflow does not parse or gate
on those counts.

The test reads `.github/workflows/extracted_pipeline_checks.yml` as text and
checks the contract markers that matter for this workflow slice: scorer command,
output filename, artifact name, pinned upload action, and absence of a
threshold/gating flag.

## Intentional

- No threshold gate yet. #1742 explicitly keeps threshold values and the
  advisory-to-gating switch as operator decisions.
- No new wrapper script. The merged scorer already owns JSON output and
  structured failure; the workflow should call it directly.
- No real corpus artifact. This uses the committed tiny surrogate fixture until
  the operator-derived corpus is supplied.

## Deferred

- Operator-supplied larger gold corpus and corpus-size/archetype decision.
- Threshold configuration and advisory-to-gating switch.
- Follow-up scrubber/precision fixes for the current tiny-corpus measured gaps.

Parked hardening: none.

## Verification

- git ls-remote https://github.com/actions/upload-artifact.git refs/tags/v4 refs/tags/v4.6.2 -- both resolve to `ea165f8d65b6e75b540449e92b4886f43607fa02`.
- python -m py_compile tests/test_extracted_pipeline_pii_recall_advisory_artifact.py
- pytest tests/test_extracted_pipeline_pii_recall_advisory_artifact.py -- 3 passed.
- python scripts/score_deflection_pii_recall.py --output /tmp/deflection-pii-recall-artifact-test/deflection-pii-recall-advisory.json --json -- status ok and wrote a non-empty JSON artifact.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 16 |
| `plans/PR-Deflection-PII-Recall-Advisory-Artifact.md` | 109 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_pipeline_pii_recall_advisory_artifact.py` | 44 |
| **Total** | **170** |
