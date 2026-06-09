# PR-Content-Ops-Claim-Evidence-Fixture-Contract

## Why this slice exists

#1443 landed the deterministic scorer for issue #1435, but the next blocker is
still labeled benchmark data. The operator needs a concrete input shape for the
roughly 40 claim/evidence triples before the model runner and results artifact
can be useful.

This PR makes that data contract explicit and testable. It validates decoded
fixture rows, reports malformed rows without raising, catches duplicate triple
ids, and can enforce the final #1435 composition target. It still does not run
models, call providers, write results, or change the verifier/MCP surface.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic fixture validator to the existing benchmark core.
2. Add an operator-facing fixture contract document with example JSON rows and
   the final benchmark composition target.
3. Add tests for decoded-input tolerance, duplicate detection, seed-set
   tolerance, and final-shape failure branches.

### Review Contract

- Acceptance criteria:
  - [ ] Decoded fixture input that is `None`, a mapping, a string, or another
        non-list shape is reported invalid without raising.
  - [ ] Malformed rows reuse the existing triple decoder errors with row indexes.
  - [ ] Duplicate `triple_id` values are reported as invalid fixture input.
  - [ ] Seed sets can validate without requiring the final 40-row composition.
  - [ ] Final benchmark validation enforces 15 easy support, 15 easy
        non-support, and 10 hard cases.
  - [ ] No model runner, prompt, provider call, result artifact, verifier
        rubric integration, MCP tool, database, or live-client behavior is
        introduced.
- Affected surfaces: extracted benchmark validation core, tests, and docs.
- Risk areas: benchmark false-green, operator labeling ambiguity, schema
  tolerance.
- Reviewer rules triggered: R1, R2, R5, R10.

### Files touched

- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Fixture-Contract.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

## Mechanism

The fixture validator accepts decoded rows, not raw JSON transport. It returns a
frozen result with valid triples, explicit errors, and counts for easy support,
easy non-support, and hard cases.

The validator has two modes:

- seed mode validates shape and duplicates only, so the operator can start with
  5 to 10 real registry triples;
- final mode also enforces the #1435 target composition of 40 rows.

The existing `ClaimEvidenceTriple` decoder remains the row-level schema. This
slice only adds collection-level checks around that decoder.

## Intentional

- No JSON file loader lands here. The future runner owns file I/O and provider
  execution; this slice stays at the decoded-data contract boundary.
- No model response fixture validation is added. #1443 already validates
  decoded structured responses; this PR focuses on the missing operator-label
  input contract.
- Final composition is issue-specific and deliberately explicit rather than
  configurable for now.

## Deferred

- JSON/JSONL loader and benchmark runner.
- Prompt and JSON Schema capture for provider calls.
- Results table, agreement matrix, failure-case list, and go/no-go writeup.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.
- Parked hardening: none.

## Verification

Local verification:

- pytest tests/test_extracted_content_claim_evidence_benchmark.py - 21 passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3593 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_fixture_contract_pr_body.md - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/content_ops_claim_evidence_benchmark_fixtures.md` | 61 |
| `extracted_content_pipeline/claim_evidence_benchmark.py` | 97 |
| `plans/PR-Content-Ops-Claim-Evidence-Fixture-Contract.md` | 104 |
| `tests/test_extracted_content_claim_evidence_benchmark.py` | 85 |
| **Total** | **347** |
