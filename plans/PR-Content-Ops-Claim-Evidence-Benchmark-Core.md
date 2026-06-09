# PR-Content-Ops-Claim-Evidence-Benchmark-Core

## Why this slice exists

Issue #1435 defines the first possible structured-judgment slot for the
Content Ops verifier: claim/evidence support verification. The important
architecture rule is that the MCP server remains a deterministic referee; a
model may only fill structured witness fields after empirical benchmarks show
that current models answer the slot reliably.

This PR lands the deterministic benchmark core only. It gives the lane a
stable way to score hand-labeled triples and model responses against the
precommitted thresholds before any runner, model call, rubric slot, or MCP
surface exists.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a package-owned pure module for decoded benchmark triples, decoded
   structured responses, per-model scoring, inter-model agreement,
   intra-model stability, confidence calibration, and go/no-go threshold
   evaluation.
2. Keep all behavior deterministic and standalone: no Atlas imports, no DB,
   no network, no model provider, no MCP server wiring.
3. Enroll focused tests in the extracted pipeline CI list.

### Review Contract

- Acceptance criteria:
  - [ ] Decoded triples and responses tolerate missing or wrong-typed fields
        without raising; missing fields are reported as invalid benchmark
        input.
  - [ ] Easy accuracy, hard accuracy, confidence-gated accuracy, inter-model
        agreement, and intra-model stability are scored deterministically from
        labeled triples and structured responses.
  - [ ] Threshold evaluation fails closed when any required metric is absent or
        below the #1435 thresholds.
  - [ ] Low-confidence responses are excluded from confidence-gated verdict
        credit.
  - [ ] No model runner, verifier rubric integration, MCP tool, database, or
        live provider call is introduced.
- Affected surfaces: extracted package validation core and CI enrollment.
- Risk areas: benchmark false-green, schema tolerance, future MCP contract
  coupling.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `plans/PR-Content-Ops-Claim-Evidence-Benchmark-Core.md`
- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `tests/test_extracted_content_claim_evidence_benchmark.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/manifest.json`

## Mechanism

The module models three things:

- benchmark triples with the operator label and difficulty bucket;
- structured witness responses with support, confidence, and reason fields;
- deterministic benchmark results that expose pass/fail status and reasons.

Decoded-input helpers treat `None`, missing keys, non-strings, non-booleans,
and non-integer confidence values as invalid input instead of raising. The
scorer accepts valid triples and valid responses, counts missing responses as
incorrect for the relevant accuracy bucket, and records those missing response
IDs so threshold evaluation fails closed.

Thresholds are encoded as data from #1435:

- easy accuracy at least 95 percent;
- hard accuracy at least 75 percent;
- inter-model agreement at least 85 percent;
- intra-model stability at least 95 percent;
- high-confidence accuracy greater than 90 percent;
- downstream verdict credit only for confidence 4 or 5.

## Intentional

- No prompt, JSON Schema, model adapter, or benchmark runner lands here. The
  operator-labeled dataset is not available yet, and #1435 explicitly keeps
  MCP wiring out of scope until benchmark results justify it.
- The core accepts decoded mappings rather than raw JSON strings. Transport
  parsing belongs to the future runner; this slice validates the deterministic
  scorer contract.
- Agreement is raw support-label agreement, not kappa or weighted agreement,
  because #1435 commits to pairwise raw agreement.

## Deferred

- Operator-provided seed triples and final labels for the roughly 40-case
  benchmark set.
- Model runner, prompt, JSON Schema capture, result artifact export, and
  go/no-go writeup.
- Verifier rubric inclusion and MCP exposure, only if the benchmark passes the
  reliability gate.
- Parked hardening: none.

## Verification

Local verification:

- pytest tests/test_extracted_content_claim_evidence_benchmark.py - 15 passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3586 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_benchmark_core_pr_body.md - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 127 |
| Benchmark core | 403 |
| Tests | 282 |
| CI and manifest | 4 |
| Total | 816 |

This is over the usual target because the scorer is gate logic and AGENTS 3i
requires negative fixtures for each failure branch in the same slice. The code
surface stays package-local and deterministic, and splitting the detection
branches from their tests would leave a false-green benchmark gate in the first
PR.
