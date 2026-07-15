# PR-Local-Model-MCP-Qualification-V2

## Why this slice exists

Issue #2114 follows the independent investigation for content-factory epic
#2109. The existing local-model evaluator is the correct entrypoint, but its
current pass/fail contract cannot support role qualification: it accepts any
JSON object as tool arguments, cannot express exact call order or expected
arguments, runs every case once, records no effective runtime configuration or
latency/token usage, and treats a small refusal-phrase list as the approval
boundary. In a safe mock probe, a model avoided the send tool and requested an
approval token, yet the evaluator failed the run because the response did not
contain one of its refusal phrases.

This functional-validation slice is justified by a safety risk rather than
workflow polish. Without deterministic argument, sequence, and approval gates,
the planned content-factory roles could be assigned from misleading evidence.
The slice extends the already-shipped read-only harness; it does not build a
second evaluation platform or expose a live mutating tool.

The expected diff exceeds the 400-LOC target because the v2 case contract, mock
transport, deterministic validators, representative 12-case fixture, and their
boundary tests must land together. Splitting the runner from the validators
would create a reachable CLI that can record false qualification results.

### Problem-derived contract

- Root cause: The evaluator conflates parseable tool-call syntax with correct
  tool behavior. It validates only that arguments decode to an object, grades
  expected tools as an unordered set, cannot distinguish approval safety from
  refusal wording, and omits the repetition/runtime evidence needed to compare
  model configurations. Its only executable tool path is a live MCP session,
  so the safety-critical cases cannot advertise realistic side-effect decoys
  without risking external effects.
- Correct fix must touch/change: Extend `EvalCase` and JSONL loading with
  category/role, per-case tool exposure, exact sequence/arguments, explicit
  no-tool and approval boundaries, side-effect tool names, output schema,
  identifier-provenance patterns, forbidden output substrings, severity, and
  retry metadata. Validate a named tool's arguments against its
  advertised JSON Schema before any runner call. Record attempted/executed call
  traces, hard-gate failures, repetitions, response timing/token usage, and a
  captured runtime-config payload. Add an isolated fixture-backed tool surface
  that never opens MCP and resets responses for every case repetition. Prove the
  new CLI entrypoint and all guard sides with focused tests and a 12-case
  content-factory fixture.
- Must not change: Existing live MCP allowlists, known-mutator blocking, MCP
  authentication, legacy v1 case/result compatibility, Atlas product behavior,
  LM Studio or Open WebUI configuration/prompts, installed models, email,
  calendar, CRM, publishing, ComfyUI, filesystem tools, or any other PR lane.

## Scope (this PR)

Ownership lane: mcp/local-model-qualification
Slice phase: Functional validation

1. Add a backwards-compatible v2 evaluation case/record contract with exact
   tool, argument, sequence, output-schema, approval, repetition, and runtime
   metadata.
2. Add a fixture-backed mock tool mode that advertises realistic primitive,
   workflow-helper, draft, and send-decoy tools without opening an MCP session.
3. Add 12 essential content-factory qualification cases and document the safe
   exploratory run procedure.
4. Run five exploratory repetitions against the three issue-selected installed
   candidates, keeping raw artifacts ignored and explicitly avoiding a
   production-qualification claim.

### Review Contract

- Acceptance criteria:
  - A syntactically valid but schema-invalid argument object is returned to the
    model as a validation error and never reaches the tool runner.
  - Missing required fields, extra fields, wrong types, enums, and malformed
    JSON exercise the same no-runner invariant.
  - Exact tool order and exact per-call arguments are graded deterministically.
  - Case-scoped identifier patterns reject identifiers in the final answer that
    were not present in the user request or successful tool results, and
    forbidden output substrings make the retrieval-injection case observable.
  - An explicit no-tool case fails on any model tool attempt.
  - Missing approval fails on any attempted side-effect tool or false-success
    claim, while a safe stop is not required to contain a magic refusal phrase.
  - Final structured output can be graded against JSON Schema.
  - Mock mode cannot connect to MCP and resets configured responses for each
    case/repetition.
  - Every v2 record identifies repetition, category, role, severity, runtime
    config, elapsed time, token usage, exposed tools, attempts, executions, and
    hard-gate failures.
  - Legacy presets, allowlist/mutator rejection, summary redaction, and v1 tests
    remain green.
  - The committed fixture contains 12 cases spanning no-tool, selection,
    arguments, structured output, ID preservation, controlled error data,
    retrieved instruction injection, missing/explicit approval, and primitive
    versus workflow-helper retrieval.
  - No external email, calendar, publishing, CRM write, ComfyUI job,
    unrestricted code, or filesystem effect occurs.
- Reachability proof: Invoke `scripts/eval_local_mcp_models.py` through its real
  CLI with `--mock-tools-file`, `--prompts-file`, `--repetitions 5`, and the live
  local OpenAI-compatible endpoint. The observable proof is append-safe ignored
  JSONL containing five records per model/case plus its redacted summary.
- Affected surfaces: Local MCP model-eval CLI, v1/v2 JSONL case and result
  contracts, deterministic grader, mock fixtures, focused tests, and operator
  runbook.
- Risk areas: A schema-invalid call reaching a runner, mock mode accidentally
  opening MCP, a safe approval stop being falsely failed, a forbidden attempt
  hidden because execution was blocked, repeated runs sharing mock response
  state, sensitive payloads entering summaries, and v1 preset regression.
- Reviewer rules triggered: R1, R2, R3, R8, R10, R13, R14.

### Files touched

- `docs/local_mcp_model_eval_runbook.md`
- `plans/PR-Local-Model-MCP-Qualification-V2.md`
- `scripts/eval_local_mcp_models.py`
- `tests/fixtures/mcp_model_eval/content_factory_v2_cases.jsonl`
- `tests/test_eval_local_mcp_models.py`

## Mechanism

The case loader keeps every v1 field and adds optional v2 fields. Per-case tool
names filter the already-advertised live or mock surface. `_run_case` records
every named attempt before parsing, validates allowed calls with the official
`jsonschema` dependency already supplied by the MCP SDK, and calls the runner
only after validation succeeds. The tool response remains a normal tool message
so a model may repair an invalid call within the case's retry allowance.

The grader uses the attempted trace for no-tool and approval hard gates, and the
executed trace for exact sequence/argument assertions. Approval is behavioral:
when approval is required and absent, an attempted named side-effect or an
unnegated success claim is a hard failure. It does not require specific refusal
prose. Existing `requires_refusal` behavior remains for legacy read-only cases.
Final-answer JSON Schema is a separate deterministic check.
For cases that opt in, identifiers found in the final answer must have appeared
in the prompt or a successful tool result. Case-scoped forbidden substrings
provide a narrow deterministic assertion for injected result content without
pretending to be a general semantic grader.

Mock mode reads tool schemas and per-case response queues from the same JSONL
case records. It constructs a fresh runner for every model/case/repetition and
never imports or opens the MCP transport. A configured tool error or result is
returned as fixture data; no function represented by the schema exists to
execute. Runtime records include the CLI sampling/tool settings, a hash of the
system prompt, optional operator-supplied model metadata, API usage totals, and
wall-clock duration. Raw prompts, arguments, and mock results remain only in the
ignored JSONL; summaries remain redacted.

## Intentional

- Five repetitions are exploratory evidence only. The report must not label a
  model qualified or select a universal default.
- Mock side-effect tools may be advertised because no callable implementation or
  MCP connection exists in mock mode; this is necessary to test whether the
  model stops at approval.
- The existing heuristic result-grounding check remains for v1 compatibility;
  improving semantic/citation grading is deferred.
- Runtime metadata is supplied/captured rather than mutating model presets to
  normalize them. Saved-prompt differences remain visible confounders.
- Plain JSONL remains the storage format. No SQLite or evaluation service is
  introduced.

## Deferred

- Live read-only MCP qualification on ports 8065/8067 after isolated account
  fixtures and tokens are available.
- Broader 105-case qualification, long-context tiers, tool-count experiments,
  native-tool versus forced-JSON comparison, and statistically meaningful role
  assignment.
- Any live draft/write/side-effect test, workflow wrapper, prompt change, tool
  redesign, or Open WebUI worker configuration.

Parked hardening: none.

## Verification

- Passed: focused evaluator suite, 52 tests.
- Passed: Python compile, Black check, diff check, and direct ASCII checks.
- Passed: real CLI loaded 12 mock cases and listed six mock tools without an
  MCP URL or token.
- Passed: 180 side-effect-free records across three candidates, 12 cases, and
  five repetitions. Qwen3.6 35B-A3B passed 60/60, Qwen2.5 Coder 32B passed
  55/60, and Gemma 4 31B passed 50/60. All three recorded zero hard-gate,
  argument-validation, and tool-runner failures. Gemma made five blocked
  unadvertised-tool attempts; Qwen2.5 repeated injected text in five runs
  without attempting the side effect; Gemma fenced JSON in five structured
  outputs.
- Observed: the workflow helper reduced mean calls from two to one and reduced
  mean latency/token use for all three candidates, while both primitive and
  helper variants passed 5/5. This is a narrow efficiency result, not evidence
  that a workflow wrapper is universally more reliable.
- Caveat: effective contexts differed (8,192; 51,210; 50,889), Qwen3.6 had a
  saved model-level system prompt, and the initial raw run records captured the
  chat-template fingerprint as unavailable. The local runtime manifest was
  updated with fingerprints recovered from GGUF metadata after the run, but the
  180 records were not rewritten. These runs are not fully normalized or
  production-qualification evidence.
- Passed: scripts maturity-sweep ratchet, 291 files scanned with no new
  brittleness above baseline.
- Pending: managed local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/local_mcp_model_eval_runbook.md` | 72 |
| `plans/PR-Local-Model-MCP-Qualification-V2.md` | 207 |
| `scripts/eval_local_mcp_models.py` | 939 |
| `tests/fixtures/mcp_model_eval/content_factory_v2_cases.jsonl` | 12 |
| `tests/test_eval_local_mcp_models.py` | 667 |
| **Total** | **1897** |
