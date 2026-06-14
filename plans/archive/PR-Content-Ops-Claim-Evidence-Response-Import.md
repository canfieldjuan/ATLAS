# PR-Content-Ops-Claim-Evidence-Response-Import

## Why this slice exists

Issue #1435 now has deterministic fixture validation, prompt/schema capture,
provider-injected harnesses, artifact building, saved-artifact validation, and
prompt-packet export. The remaining operator gap before any live provider work
is the return path: once model witnesses fill the exported packets, there is no
deterministic command that validates those returned packet responses and writes
the recorded-response JSON shape consumed by the existing artifact CLI.

This slice closes that operator handoff without adding provider adapters,
credentials, live prompt execution, verifier rubric inclusion, MCP transport, or
registry mutation. It lets an operator run the current zero-provider benchmark
loop as files:

1. validate labeled fixture;
2. export prompt packets;
3. collect model-filled response rows out of band;
4. import those returned rows into recorded-response JSON;
5. feed that recorded-response JSON into the existing artifact CLI.

This slice may exceed the normal 400 LOC target because the importer, docs,
failure-branch tests, and same-PR CI enrollment need to ship together. Splitting
tests or enrollment away would create an unprotected operator command in the
benchmark path.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add a deterministic prompt-response import CLI that reads exported
   prompt-packet JSON or JSONL plus returned response-row JSON or JSONL and
   writes the existing recorded-response JSON shape for the artifact CLI.
2. Validate every returned row against exported packet identity:
   `model_id`, `triple_id`, and contract version must match an exported packet.
3. Decode each returned `response` through the existing strict
   `verify_claim_evidence.v1` response contract before writing output.
4. Fail closed for malformed packet/response files, unsupported formats,
   duplicate main rows, duplicate stability rows, missing main coverage,
   unmatched rows, invalid structured responses, directory output, symlinked
   output, input/output path collisions, and non-UTF-8 input.
5. Document the operator file handoff and enroll the new tests in extracted
   checks.

### Review Contract
- Acceptance criteria:
  - [ ] Returned main response rows produce `model_runs` grouped by model id in
        the recorded-response JSON shape consumed by the artifact CLI.
  - [ ] Returned stability response rows produce
        `stability_runs_by_model_id` grouped by model id and run id.
  - [ ] The importer rejects rows that do not match an exported
        model/triple/contract packet.
  - [ ] The importer rejects duplicate model/triple main rows and duplicate
        model/run/triple stability rows.
  - [ ] The importer rejects invalid structured responses before writing output.
  - [ ] The importer writes no output on validation, read, format, directory, or
        symlink failures, and refuses output paths that match either input path.
  - [ ] No provider calls, credentials, verifier rubric inclusion, MCP
        transport, DB writes, or registry mutation are added.
- Affected surfaces: operator CLI, docs, extracted-checks CI enrollment.
- Risk areas: malformed input handling, label leakage, artifact-input
  backcompat, CI enrollment.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `plans/PR-Content-Ops-Claim-Evidence-Response-Import.md`
- `scripts/import_content_ops_claim_evidence_prompt_responses.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_claim_evidence_response_import_cli.py`

## Mechanism

The CLI reads two deterministic file sets:

- prompt packets from the previous export command, in JSON array or JSONL
  object stream form;
- returned response rows in JSON array or JSONL object stream form.

Returned response rows use the model-facing packet identity plus a structured
response:

```json
{
  "model_id": "claude-sonnet",
  "triple_id": "real-001",
  "contract_version": "verify_claim_evidence.v1",
  "response": {
    "supports": true,
    "confidence": 5,
    "reason": "The quote directly states the claim."
  }
}
```

Rows may optionally set `run_type` to `stability` and provide a non-empty
`run_id`; those rows are grouped into `stability_runs_by_model_id`. Rows without
`run_type`, or with `run_type` set to `main`, become the primary `model_runs`.

The importer builds an allowed packet key set from the exported packets, then
validates and canonicalizes every returned row. It writes a JSON object with
`model_runs` and `stability_runs_by_model_id`, matching the input contract of
the existing artifact CLI. It writes only after all rows pass validation and
main rows cover every exported packet exactly once.

## Intentional

- No live provider adapter in this slice. Returned rows are collected out of
  band so this PR stays deterministic and credential-free.
- No verifier/MCP inclusion. The benchmark still needs actual model results and
  threshold evaluation before a structured-witness slot can enter the rubric.
- No `expected_supports` or other operator labels in packet-response rows or
  importer output. Labels stay in the fixture path and artifact scoring path,
  not the witness-return path.
- Stability rows are accepted but not required by the importer. The existing
  artifact builder remains responsible for benchmark threshold and stability
  coverage decisions.

## Deferred

- Concrete provider adapters and live prompt execution remain future work.
- Batch orchestration that exports packets, runs providers, imports responses,
  and writes artifacts in one command remains future work.
- Go/no-go writeup from real benchmark results remains future work.
- Verifier rubric inclusion and MCP exposure remain blocked until benchmark
  thresholds pass.

Parked hardening: none.

## Verification

- `python -m py_compile scripts/import_content_ops_claim_evidence_prompt_responses.py tests/test_content_ops_claim_evidence_response_import_cli.py` passed.
- `pytest tests/test_content_ops_claim_evidence_response_import_cli.py -q` passed:
  14 passed.
- Focused claim/evidence benchmark sweep passed: 109 passed.
- Extracted content package guardrails passed.
- `bash scripts/run_extracted_pipeline_checks.sh` passed: 4080 passed, 10
  skipped.
- Body-aware local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `docs/content_ops_claim_evidence_benchmark_fixtures.md` | 45 |
| `plans/PR-Content-Ops-Claim-Evidence-Response-Import.md` | 155 |
| `scripts/import_content_ops_claim_evidence_prompt_responses.py` | 436 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_claim_evidence_response_import_cli.py` | 420 |
| **Total** | **1061** |
