# PR-Content-Ops-Claim-Evidence-Fixture-Loader

## Why this slice exists

#1448 made the operator-labeled benchmark fixture contract explicit after
#1443 landed the deterministic scorer. The remaining practical gap before a
runner can be useful is getting raw fixture text into that decoded contract
without each caller inventing its own parser behavior.

This PR lands only the deterministic loading boundary for issue #1435. It
parses JSON array text and JSONL object streams into decoded rows, delegates to
the existing fixture validator, and reports malformed input without raising.
It still does not read files, call models, prompt providers, write result
artifacts, alter verifier behavior, or expose MCP transport.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a pure fixture-text loader to the benchmark core for JSON and JSONL
   payloads.
2. Thread loader results through the existing fixture validator so seed and
   final-shape validation behave exactly like decoded input validation.
3. Document the accepted JSON and JSONL fixture formats for the operator.
4. Add focused tests for malformed text, unsupported formats, JSONL row
   indexing, and final-shape delegation.

### Review Contract

- Acceptance criteria:
  - [ ] Non-string fixture text is reported invalid without raising.
  - [ ] Unsupported fixture formats are reported invalid without raising.
  - [ ] JSON array text validates through the existing decoded fixture
        contract.
  - [ ] JSON text that is not an array reports a loader-level error before
        decoded fixture validation.
  - [ ] JSONL text parses one object per non-empty line and reports malformed
        line numbers.
  - [ ] JSONL arrays are rejected so line/object semantics stay unambiguous.
  - [ ] Final-shape validation remains delegated to the existing fixture
        validator.
  - [ ] No file I/O, model/provider runner, prompt/schema capture, result
        artifact, verifier rubric wiring, MCP tool, database, or live-client
        behavior is introduced.
- Affected surfaces: extracted benchmark validation core, tests, and docs.
- Risk areas: benchmark false-green, malformed operator input, schema
  tolerance.
- Reviewer rules triggered: R1, R2, R5, R10.

### Files touched

- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Fixture-Loader.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

## Mechanism

The loader accepts raw text plus a format name. For JSON, it decodes one JSON
array and passes the resulting list into the existing fixture validator. For
JSONL, it decodes each non-empty line as one object, preserves line numbers in
parse errors, rejects per-line arrays, and then passes the list of decoded
objects into the same fixture validator.

The result type mirrors the existing fixture result shape: valid triples plus
explicit errors. When parsing fails, the result has no triples and only
loader-level errors. When parsing succeeds, validator errors are returned
unchanged so callers see the same row-contract messages regardless of whether
they started from decoded rows or text.

## Intentional

- No filesystem loader lands here. The future runner owns source paths,
  storage, and operator workflow; this slice stays at the raw-text boundary.
- Format names are explicit rather than inferred from file extensions because
  this package should not know about filenames.
- JSONL rejects arrays on a line even though JSON can parse them. The benchmark
  fixture contract is one object per JSONL line, and accepting arrays there
  would make row numbering ambiguous.

## Deferred

- File-path loading and CLI/operator handoff.
- Provider/model runner and prompt/schema capture.
- Results table, inter-model agreement matrix, failure-case list, and go/no-go
  writeup.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.
- Parked hardening: none.

## Verification

Local verification:

- pytest tests/test_extracted_content_claim_evidence_benchmark.py - 30 passed.
- python -m py_compile extracted_content_pipeline/claim_evidence_benchmark.py tests/test_extracted_content_claim_evidence_benchmark.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3690 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_fixture_loader_pr_body.md - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/content_ops_claim_evidence_benchmark_fixtures.md` | 34 |
| `extracted_content_pipeline/claim_evidence_benchmark.py` | 66 |
| `plans/PR-Content-Ops-Claim-Evidence-Fixture-Loader.md` | 112 |
| `tests/test_extracted_content_claim_evidence_benchmark.py` | 99 |
| **Total** | **311** |
