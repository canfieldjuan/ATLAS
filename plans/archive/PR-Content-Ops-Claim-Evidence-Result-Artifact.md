# PR-Content-Ops-Claim-Evidence-Result-Artifact

## Why this slice exists

#1475 added the injected-provider runner harness for #1435, so benchmark rows
can now be collected without live provider code in the extracted package. The
next missing deterministic step is the issue's promised output shape: a
machine-readable benchmark artifact that combines completed model runs into
model scores, agreement data, stability data, failure cases, and an explicit
go/no-go decision.

This PR stays below provider and MCP wiring. It makes benchmark results
reviewable once operator-labeled triples and model-run rows exist, but it does
not call providers, read files, write artifacts, or add the structured slot to
the verifier.

Diff-budget note: this is expected to exceed the 400 LOC soft cap because the
artifact builder is a detector/gate surface. The failure-branch tests for
malformed inputs and false-green result rows need to ship with the artifact
contract rather than in a separate test-only follow-up.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic artifact assembler to the existing claim/evidence
   benchmark module.
2. Convert completed `ClaimEvidenceModelRun` values into model scores,
   pairwise agreement rows, stability rows, failure-case rows, threshold
   verdict, and go/no-go status.
3. Fail closed on malformed artifact inputs such as duplicate model ids,
   unknown row triple ids, duplicate row triple ids, non-run values, and
   malformed stability reruns.
4. Add focused tests for passing output, failure-case output, and malformed
   artifact inputs.
5. Keep concrete provider adapters, live model execution, file output, verifier
   inclusion, MCP tools, database behavior, and live-client behavior out of
   scope.

### Review Contract

- Acceptance criteria:
  - [ ] Completed model runs produce deterministic model scores, agreement
        matrix rows, stability rows, threshold verdict, and go/no-go status.
  - [ ] Failure cases identify malformed/missing responses, incorrect support
        judgments, and low-confidence responses without raising.
  - [ ] The artifact exposes a JSON-compatible mapping for downstream writeup
        and operator review slices.
  - [ ] Duplicate model ids, duplicate row triple ids, unknown row triple ids,
        invalid run values, non-mapping stability input, bad stability rerun
        values, unknown stability model ids, mismatched stability run model ids,
        and stability run errors fail closed.
  - [ ] Threshold evaluation remains the single verdict source; the artifact
        does not invent a second scoring policy.
  - [ ] No provider adapter, live prompt execution, file writing, verifier
        rubric inclusion, MCP exposure, database change, or live-client
        behavior is introduced.
- Affected surfaces: extracted benchmark helper, focused benchmark tests, and
  plan.
- Risk areas: benchmark false-green, result-contract drift, incomplete
  stability coverage, malformed model output, future writeup compatibility.
- Reviewer rules triggered: R1, R2, R5, R6, R10, R12.

### Files touched

- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Result-Artifact.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

## Mechanism

The benchmark module gains immutable result-artifact and failure-case values.
The artifact assembler accepts existing benchmark triples, completed primary
model runs, optional stability run groups, and the existing threshold object.
It validates the decoded inputs, builds response maps from successful runner
rows, reuses the existing scoring, agreement, stability, and threshold helpers,
then exposes the result through a JSON-compatible mapping.

Malformed row output remains data: row errors become failure cases and missing
scoreable responses. Malformed artifact structure is different: duplicate model
ids, duplicate row ids, unknown triple ids, bad run values, non-mapping
stability input, and malformed stability reruns produce a failed no-go artifact
so incomplete or ambiguous benchmark data cannot pass silently.

## Intentional

- The artifact assembler stays in the existing benchmark module because it is
  still the package-owned reliability gate, not host-layer provider wiring.
- The go/no-go status is derived only from `BenchmarkVerdict.passed`. The
  artifact does not add policy beyond the already-merged thresholds.
- The artifact returns mappings but does not write JSON files. File output and
  operator-facing reports belong in a later CLI/writeup slice.
- Hardening scan: no parked `HARDENING.md` entries touch this ownership lane or
  these files.

## Deferred

- Concrete provider adapters and live prompt execution.
- Batch CLI for fixture-file model runs.
- File output, markdown writeup, and operator-facing result report.
- Batch fixture directories and operator labeling workflow.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_claim_evidence_benchmark.py - 45 passed.
- python -m py_compile extracted_content_pipeline/claim_evidence_benchmark.py tests/test_extracted_content_claim_evidence_benchmark.py - passed.
- git diff --check - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3734 passed, 10 skipped.
- bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file tmp/content_ops_claim_evidence_result_artifact_pr_body.md - passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_result_artifact_pr_body.md - passed.

## Estimated diff size

Estimated: 3 files, about +635 / -1. This is over the normal 400 LOC target for
the reason named in Why this slice exists.

| Total | 635 |
