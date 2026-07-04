# PR-Content-Ops-Claim-Evidence-Artifact-Bundle

## Why this slice exists

Issue #1435 still needs operator-facing benchmark artifacts before the
`verify_claim_evidence` structured-witness slot can be considered for verifier
inclusion. Prior slices built the scorer/evaluator, fixture contract, fixture
loader, validation CLI, prompt/schema contract, injected-provider runner,
machine-readable result artifact, and Markdown writeup. The next deterministic
gap is packaging an already-built result artifact into stable JSON and Markdown
file contents so a later CLI/live-run slice can write exactly those files
without inventing presentation logic at the transport boundary.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic file-bundle value type for claim/evidence benchmark
   result artifacts.
2. Add a pure bundler that returns stable JSON and Markdown entries from an
   existing result artifact.
3. Fail closed on malformed bundler input by returning no-go JSON and Markdown
   that carry the artifact error.
4. Leave disk writing, provider adapters, live prompt execution, and model-run
   CLI orchestration out of this slice.

### Files touched

- `plans/PR-Content-Ops-Claim-Evidence-Artifact-Bundle.md`
- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

### Review Contract

Acceptance criteria:
- The bundler returns exactly two stable entries: JSON result payload and
  Markdown writeup.
- The JSON entry is deterministic, newline-terminated, and parseable back into
  the existing artifact fields including `go_no_go` and `ok`.
- The Markdown entry matches the existing operator-facing renderer.
- Malformed bundler input does not raise; it returns no-go JSON and Markdown
  with the same attributable artifact error.

Affected surfaces: extracted benchmark helper and focused unit tests.

Risk areas: false-green presentation, future CLI/file-output compatibility,
and accidental live/provider scope creep.

Reviewer rules triggered: R1, R2, R10, R14.

## Mechanism

The benchmark module already has a machine-readable result artifact and a
Markdown renderer. This slice adds a tiny file-entry dataclass plus a pure
bundler that normalizes malformed input through the same failed-artifact path as
the renderer, then emits:

- JSON entry named claim_evidence_result.json with sorted, indented JSON from the artifact
  payload.
- Markdown entry named claim_evidence_result.md with the existing writeup.

The function returns content, names, and content types only. A later CLI can
write these entries to an operator artifact directory without duplicating
serialization or Markdown decisions.

## Intentional

- No filesystem writes in this PR. Keeping the extracted benchmark module pure
  preserves the current no-I/O contract and gives the future CLI a narrow helper
  to call.
- No provider adapters or live model execution. The reliability gate still must
  be run and evaluated before verifier/MCP inclusion.
- The JSON payload uses the existing artifact mapping rather than a new schema.
  This avoids a second artifact contract for the same data.

## Deferred

- Disk-writing CLI and operator artifact directory handling.
- Concrete provider adapters and live prompt execution.
- Batch model-run CLI for fixture-file model runs.
- Verifier rubric inclusion and MCP exposure only if benchmark thresholds pass.
- Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_claim_evidence_benchmark.py -- 51 passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3895 passed, 10 skipped.
- git diff --check -- passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_artifact_bundle_pr_body.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-Claim-Evidence-Artifact-Bundle.md` | 105 |
| `extracted_content_pipeline/claim_evidence_benchmark.py` | 52 |
| `tests/test_extracted_content_claim_evidence_benchmark.py` | 63 |
| **Total** | **220** |

Under the 400 LOC target.
