# PR-Content-Ops-Claim-Evidence-Provider-Adapters

## Why this slice exists

Issue #1435 requires a reliability gate before any `verify_claim_evidence`
structured-judgment slot can enter the verifier rubric. The merged benchmark
path can already validate fixtures, export prompt packets, import returned
responses, and build result artifacts, but after #1537 the remaining resume
point is still concrete provider adapter / live prompt execution. Operators
must currently move prompt packets out of band, call models manually, then
shape response rows by hand before the deterministic importer can run.

Root cause: the benchmark runner boundary is provider-neutral by design, but no
operator CLI binds exported packets to an actual OpenAI-compatible
chat-completions provider. This fixes that missing boundary directly for the
OpenAI-compatible path while keeping verifier/MCP transport out of scope.
This is over the 400 LOC soft target because the live-provider boundary has to
ship with sanitized error handling, file-format handling, mocked provider
tests, and CI enrollment to be reviewable as one vertical slice.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add an operator CLI that reads exported claim/evidence prompt packets,
   executes them against an OpenAI-compatible chat-completions endpoint, and
   writes importer-compatible response rows.
2. Preserve the existing benchmark core contract: packet validation and
   returned-response decoding still happen through
   `run_claim_evidence_prompt_packets`.
3. Fail closed on missing credentials, malformed packet files, bad packet
   schema, provider HTTP failures, malformed provider envelopes, invalid JSON
   content, unsafe output paths, and invalid stability counts.
4. Reject incomplete provider completions, avoid packet/output path clobbering,
   disable provider-side storage by default, adapt token limits for OpenAI
   o-series models, and strip provider-unsupported strict-schema constraints
   for Azure/fine-tuned targets while preserving local validation.
5. Enroll the new CLI test in the extracted pipeline checks and workflow path
   filters.

### Review Contract

- Acceptance criteria:
  - [ ] JSON and JSONL prompt packets can be executed into JSON/JSONL response
        rows compatible with `import_content_ops_claim_evidence_prompt_responses.py`.
  - [ ] Provider requests use the existing packet prompt plus the strict
        `verify_claim_evidence.v1` JSON Schema as `response_format`.
  - [ ] Missing API keys and provider failures fail closed without writing
        output.
  - [ ] Error envelopes do not include API keys or provider response bodies.
  - [ ] Invalid packets are rejected before any HTTP call.
  - [ ] Output cannot overwrite the prompt-packet input artifact.
  - [ ] Non-`stop` finish reasons are rejected before response rows are written.
  - [ ] Default provider requests send `store: false` and omit metadata unless
        storage is explicitly requested.
  - [ ] O-series model ids use `max_completion_tokens`; non-o-series model ids
        keep `max_tokens` by default.
  - [ ] Azure/fine-tuned structured-output requests use compatible schemas
        without weakening local response validation.
  - [ ] Main and stability rerun rows retain the existing importer shape.
  - [ ] New tests are enrolled in local extracted checks and workflow path
        filters.
- Affected surfaces: operator CLI, third-party HTTP provider boundary,
  extracted pipeline CI enrollment.
- Risk areas: secret leakage in provider errors, false-green benchmark rows,
  live-provider nondeterminism, CI enrollment drift.
- Reviewer rules triggered: R1, R2, R3, R6, R10, R11, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-Claim-Evidence-Provider-Adapters.md`
- `scripts/run_content_ops_claim_evidence_prompt_provider.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_claim_evidence_prompt_provider_cli.py`

## Mechanism

`run_content_ops_claim_evidence_prompt_provider.py` loads JSON or JSONL prompt
packet rows, validates them through the existing benchmark packet runner, and
uses a small OpenAI-compatible adapter as the injected provider callback. The
adapter POSTs to `<api-base-url>/chat/completions` with:

- the packet `model_id`;
- a strict witness system message plus the packet prompt;
- the packet `response_schema` wrapped as `response_format:
  {type: json_schema, strict: true}`;
- deterministic defaults (`temperature=0`, bounded max tokens);
- `store: false` by default, with run metadata only when
  `--store-completions` opts into provider retention.

The provider response is accepted only when the chat-completions envelope
contains a `stop` finish reason and JSON object content. OpenAI o-series model
ids use `max_completion_tokens`; other model ids keep `max_tokens` unless the
operator overrides the token-limit field. Azure endpoints and fine-tuned model
ids use a compatible structured-output schema with unsupported constraints
removed from the provider request, while the existing benchmark decoder still
validates `supports`, `confidence`, and `reason` locally before any response
row is written. Provider failures raise typed local exceptions so the
benchmark runner records only the exception class, not provider bodies,
prompts, or API keys.

The CLI writes the same response-row format consumed by the deterministic
response importer, so the existing manual benchmark runner can still build the
recorded-response JSON and `claim_evidence_result` artifact bundle.

## Intentional

- Only an OpenAI-compatible chat-completions adapter lands here. Native
  Anthropic batch/messages support is a separate provider surface and would
  turn this slice into a multi-adapter rollout.
- Credentials are read from an explicit env var name (`OPENAI_API_KEY` by
  default, overrideable with `--api-key-env`) because this is an operator CLI
  for external provider execution, not Atlas app runtime config.
- Live provider calls are not run in CI. Tests use `httpx.MockTransport` to
  prove request shape, output shape, failure behavior, and secret redaction
  without requiring paid credentials.
- No verifier rubric, MCP tool, DB write, registry mutation, or benchmark
  go/no-go decision is added in this slice.

## Deferred

- Native Anthropic provider adapter/batch execution if the benchmark needs the
  Messages API rather than an OpenAI-compatible route.
- Final 40-row operator-labeled benchmark run across the selected models.
- Results-table/go-no-go writeup from live benchmark artifacts.
- Verifier rubric inclusion and MCP exposure only after the benchmark
  thresholds justify shipping the slot.

Parked hardening: none.

## Verification

- `scripts/run_content_ops_claim_evidence_prompt_provider.py` and `tests/test_content_ops_claim_evidence_prompt_provider_cli.py` py-compile check - passed.
- `python -m pytest tests/test_content_ops_claim_evidence_prompt_provider_cli.py -q` - 12 passed.
- `python -m pytest tests/test_extracted_content_claim_evidence_benchmark.py tests/test_content_ops_claim_evidence_prompt_packets_cli.py tests/test_content_ops_claim_evidence_prompt_provider_cli.py tests/test_content_ops_claim_evidence_response_import_cli.py -q` - 104 passed.
- `scripts/check_ascii_python.sh` via bash - passed.
- `scripts/run_extracted_pipeline_checks.sh` via bash - 4714 passed, 15 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Content-Ops-Claim-Evidence-Provider-Adapters.md` | 150 |
| `scripts/run_content_ops_claim_evidence_prompt_provider.py` | 584 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_claim_evidence_prompt_provider_cli.py` | 452 |
| **Total** | **1191** |
