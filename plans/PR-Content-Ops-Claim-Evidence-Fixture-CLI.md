# PR-Content-Ops-Claim-Evidence-Fixture-CLI

## Why this slice exists

#1451 added the deterministic raw-text loader for issue #1435, but operators still need a repeatable handoff command for local fixture files before a model runner exists. Without a CLI, fixture validation remains a Python-callable contract instead of a usable operator workflow.

This PR lands only the deterministic file-validation handoff. It reads a local JSON or JSONL fixture file, runs the merged loader and validator, and emits a machine-readable validation envelope. It still does not call models, prompt providers, write benchmark result artifacts, alter verifier behavior, or expose MCP transport.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a repo script that validates an operator fixture file through the merged loader and fixture validator.
2. Support explicit `json` / `jsonl` format selection and `auto` inference from file suffix.
3. Emit a JSON result envelope with validity, errors, row counts, and bucket counts.
4. Add focused CLI tests and enroll the new test file in the extracted pipeline wrapper.

### Review Contract

- Acceptance criteria:
  - [ ] Missing fixture files return non-zero with a JSON error envelope.
  - [ ] Unsupported file suffixes in auto mode return non-zero without reading as a guessed format.
  - [ ] Explicit `json` and `jsonl` modes validate files through the merged loader.
  - [ ] Final-shape mode delegates to the merged final composition validator.
  - [ ] Successful validation returns zero with counts and no errors.
  - [ ] Failed validation returns non-zero with the loader or validator errors.
  - [ ] The new CLI test file is enrolled in the extracted pipeline check wrapper.
  - [ ] No model/provider runner, prompt/schema capture, benchmark result artifact, verifier rubric wiring, MCP tool, database, or live-client behavior is introduced.
- Affected surfaces: operator validation script, tests, workflow filters, extracted check enrollment, and docs.
- Risk areas: benchmark false-green, malformed operator input, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `plans/PR-Content-Ops-Claim-Evidence-Fixture-CLI.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/validate_content_ops_claim_evidence_fixture.py`
- `tests/test_validate_content_ops_claim_evidence_fixture.py`

## Mechanism

The script parses a fixture path and format option. In `auto` mode it accepts only .json and .jsonl suffixes, so unknown filenames fail closed instead of being guessed. It reads UTF-8 text, passes that text into the merged fixture loader, and prints one JSON object containing `ok`, `errors`, `triple_count`, `easy_supports_count`, `easy_not_supports_count`, and `hard_count`.

The script returns zero only when the fixture result is valid. Any read error, format error, loader error, or validator error returns a non-zero exit code and the same result-envelope shape.

## Intentional

- The CLI validates one local file at a time. Batch runs and corpus management belong with the future benchmark runner.
- Auto format inference is suffix-only and fail-closed. It does not inspect file contents because content sniffing would hide operator mistakes.
- The output envelope includes aggregate counts, not the full fixture rows, so logs avoid dumping labeled evidence text by default.

## Deferred

- Batch fixture directories and operator labeling workflow.
- Provider/model runner and prompt/schema capture.
- Results table, inter-model agreement matrix, failure-case list, and go/no-go writeup.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.
- Parked hardening: none.

## Verification

Local verification:

- pytest tests/test_validate_content_ops_claim_evidence_fixture.py tests/test_extracted_content_claim_evidence_benchmark.py - 37 passed.
- python -m py_compile scripts/validate_content_ops_claim_evidence_fixture.py tests/test_validate_content_ops_claim_evidence_fixture.py - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3697 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_fixture_cli_pr_body.md - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `docs/content_ops_claim_evidence_benchmark_fixtures.md` | 15 |
| `plans/PR-Content-Ops-Claim-Evidence-Fixture-CLI.md` | 87 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `scripts/validate_content_ops_claim_evidence_fixture.py` | 125 |
| `tests/test_validate_content_ops_claim_evidence_fixture.py` | 155 |
| **Total** | **387** |
