# PR-Content-Ops-Claim-Evidence-Artifact-CLI

## Why this slice exists

Issue #1435 now has the deterministic scorer, prompt/schema contract,
injected-provider runner, result artifact, JSON/Markdown bundle, and safe
artifact-directory writer. The remaining operator handoff gap is a small CLI
that turns already-recorded benchmark responses into the artifact directory.

This slice keeps the reliability gate deterministic: it does not call live
models, read the claim registry, expose MCP tools, or decide rubric inclusion.
It only lets an operator take a labeled fixture plus captured structured
responses and produce the same result files that the package-owned writer
already defines.

This slice is over the normal 400 LOC target because the CLI is not useful
without the recorded-response decoder, direct CLI regressions, and CI
enrollment in the same PR. Splitting those pieces would either ship an
untestable command or a decoder with no operator entry point.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic CLI for fixture file + recorded response file + output
   directory.
2. Decode recorded model responses into the existing benchmark run/result
   artifact path.
3. Call the existing artifact-directory writer and print a JSON summary of the
   write result.
4. Fail closed for malformed input files, malformed response envelopes, and
   writer errors.
5. Enroll the new CLI tests in the extracted pipeline wrapper and workflow
   path filters.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-Claim-Evidence-Artifact-CLI.md`
- `scripts/run_content_ops_claim_evidence_artifact.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_claim_evidence_artifact_cli.py`

### Review Contract

Acceptance criteria:
- A valid fixture and valid recorded-response file produce the stable
  claim/evidence JSON and Markdown artifact files in the requested output
  directory.
- The CLI prints a machine-readable JSON envelope with success state, output
  directory, written files, and errors.
- Malformed fixture files, malformed response JSON, unknown response triples,
  and writer failures return non-zero without silent success.
- The CLI does not call providers, read the claim registry, or expose MCP
  behavior.
- New tests are enrolled in the extracted pipeline wrapper and workflow path
  filters.

Affected surfaces: operator CLI script, claim/evidence benchmark decoding, and
focused tests.

Risk areas: false-green benchmark output, accidental filesystem writes, CI
enrollment drift, and scope creep into live provider execution.

Reviewer rules triggered: R1, R2, R6, R10, R12, R14.

## Mechanism

The CLI accepts a fixture path, recorded-response path, and output directory.
It reuses the existing fixture loader for labeled triples, decodes recorded
structured responses into the existing benchmark run dataclasses, builds the
existing result artifact, and delegates filesystem output to the writer from
#1508.

The recorded-response file is deterministic JSON: a top-level object with a
`model_runs` list. Each run names a model and a mapping from benchmark
`triple_id` to the already-captured structured response object. Optional
`stability_runs_by_model_id` can provide repeat runs for the existing stability
metric. Missing or malformed input returns a JSON error envelope and a non-zero
exit code.

## Intentional

- No live provider adapter is introduced here. The CLI consumes captured
  responses only, so it can run in CI and operator dry-runs without model
  credentials or token spend.
- No artifact JSON parser is added. This slice builds a fresh artifact from
  fixture plus response input, then delegates writing to the existing writer.
- No verifier rubric or MCP inclusion is added. That remains gated on actually
  running the benchmark and meeting the issue thresholds.

## Deferred

- Concrete provider adapters and live prompt execution.
- Batch model-run orchestration that calls real models and records response
  files.
- Artifact JSON parsing back into result dataclasses.
- Operator labeling workflow for the final 40-row fixture.
- Verifier rubric inclusion and MCP exposure only if benchmark thresholds pass.

Parked hardening: none.

## Verification

- pytest tests/test_content_ops_claim_evidence_artifact_cli.py: 6 passed.
- pytest tests/test_extracted_content_claim_evidence_benchmark.py tests/test_validate_content_ops_claim_evidence_fixture.py tests/test_content_ops_claim_evidence_artifact_cli.py: 72 passed.
- python -m py_compile scripts/run_content_ops_claim_evidence_artifact.py: passed.
- bash scripts/run_extracted_pipeline_checks.sh: 3909 passed, 10 skipped.
- git diff --check: passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_artifact_cli_pr_body.md: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Content-Ops-Claim-Evidence-Artifact-CLI.md` | 122 |
| `scripts/run_content_ops_claim_evidence_artifact.py` | 257 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_claim_evidence_artifact_cli.py` | 234 |
| **Total** | **618** |
