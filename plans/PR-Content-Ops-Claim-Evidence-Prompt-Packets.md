# PR-Content-Ops-Claim-Evidence-Prompt-Packets

## Why this slice exists

Issue #1435 has the deterministic benchmark path through fixture validation,
prompt/schema contract, injected-provider runner, recorded-response artifact
CLI, artifact writer/parser, and saved-result validator. The remaining gap
before live provider adapters is an operator handoff for the exact prompts that
must be sent to model witnesses.

Without a file-level prompt packet, the operator either has to reverse-engineer
the in-memory `build_claim_evidence_prompt_contract` output or wait for live
provider code before reviewing the benchmark prompts. This slice makes the
first real handoff explicit: validated fixture rows in, stable prompt/schema
packets out. It keeps provider credentials, network calls, result scoring, and
verifier/MCP inclusion deferred.

This slice is over the normal 400 LOC target because the operator CLI, docs,
failure-branch tests, and same-PR CI enrollment need to ship together. Splitting
the tests or enrollment away would create an unprotected operator command.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add a deterministic CLI that reads a claim/evidence fixture JSON or JSONL
   file and writes prompt packets for selected model ids.
2. Validate the fixture through the existing #1435 loader before any packet is
   written.
3. Emit each packet with model id, triple id, contract version, prompt, and the
   strict response schema from the package contract.
4. Support JSON array and JSONL output so operators can inspect the request set
   or hand it to later provider/batch tooling.
5. Keep provider calls, API credentials, recorded responses, result artifacts,
   verifier rubric inclusion, MCP transport, DB writes, and registry mutation
   out of scope.

### Review Contract

Acceptance criteria:
- A valid fixture produces one prompt packet per model id and triple, using the
  existing prompt/schema contract.
- Invalid fixture files, malformed packet-output format, missing model ids, and
  unwritable packet destinations fail closed with a JSON error envelope and
  non-zero exit without model calls.
- JSON and JSONL packet outputs are deterministic and parseable.
- The command does not call providers, read API credentials, write benchmark
  results, expose verifier/MCP behavior, or mutate registry/DB state.

Affected surfaces: one operator script, focused CLI tests, extracted pipeline
test enrollment, and #1435 operator documentation.

Risk areas: prompt/schema drift before live runs, false-green packet writes,
unparseable operator artifacts, and scope creep into live model execution.

Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `plans/PR-Content-Ops-Claim-Evidence-Prompt-Packets.md`
- `scripts/export_content_ops_claim_evidence_prompt_packets.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_claim_evidence_prompt_packets_cli.py`

## Mechanism

The script loads fixture text from disk, infers or accepts the fixture format,
and delegates row validation to `load_claim_evidence_fixture_text`. For every
validated triple and every normalized model id, it calls
`build_claim_evidence_prompt_contract` and serializes a packet object containing
the model id, triple id, claim/source metadata, contract version, prompt, and
response schema.

The script writes either a JSON array or JSONL object stream to an
operator-selected output path. It creates parent directories, rejects directory
or symlink output targets, and prints a sorted JSON status envelope with counts,
output path, and errors. Exit codes distinguish successful packet writes,
validated-input/contract errors, and file/write failures.

## Intentional

- This slice exports prompt packets only. It does not add Anthropic/OpenAI
  adapters, choose production model names, read credentials, or execute prompts.
- The prompt text and schema remain package-owned through the existing builder;
  the CLI only serializes that contract for operator/provider handoff.
- The output shape is intentionally simple JSON/JSONL so later live-provider or
  batch-runner slices can consume it without parsing Markdown.

## Deferred

- Concrete provider adapters and live prompt execution.
- Batch model-run orchestration that consumes prompt packets and records
  structured responses.
- Operator labeling workflow for the final 40-row fixture.
- Benchmark result scoring from live response files.
- Verifier rubric inclusion and MCP exposure only if benchmark thresholds pass.

Parked hardening: none.

## Verification

- py_compile for the prompt packet script and focused test: passed.
- focused prompt packet CLI pytest: 8 passed.
- claim/evidence focused pytest sweep: 95 passed.
- extracted content package guardrails: passed.
- full extracted pipeline wrapper: 4049 passed, 10 skipped.
- git diff whitespace check: passed.
- body-aware local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `docs/content_ops_claim_evidence_benchmark_fixtures.md` | 29 |
| `plans/PR-Content-Ops-Claim-Evidence-Prompt-Packets.md` | 123 |
| `scripts/export_content_ops_claim_evidence_prompt_packets.py` | 267 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_claim_evidence_prompt_packets_cli.py` | 264 |
| **Total** | **688** |
