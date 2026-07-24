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

Current-head review found that the first implementation still admits misleading
evidence: exact arguments can enter a shareable summary, a nearby negation can
hide a contradictory success claim, identifiers from ambiguity payloads can be
used as if resolved, dependent calls in one assistant turn can pass as a real
sequence, and the schema validator is only a transitive dependency. A cold
control audit also found that the primitive/helper pair starts from different
inputs, so its efficiency result is not a controlled comparison. These are root
correctness and privacy gaps in this slice's own qualification contract; they
must be repaired here before the exploratory results are published. The final
cold audit also found that truthy strings can silently invert boolean case
fields, including whether approval was provided, and that additional causal or
comma-spliced success clauses can still inherit an unrelated negation.

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
  content-factory fixture. Redact mismatch diagnostics before they reach the
  summary, bind negation to the success clause it actually modifies, distinguish
  identifiers that may be repeated from identifiers authorized for later tool
  arguments, and record/enforce assistant tool-call rounds for dependent calls.
  Reject non-boolean values for boolean case fields and cover additional
  clause-boundary forms. Declare the runtime schema dependency directly. Make
  the primitive/helper cases share the same request, input state, expected
  facts, and mock content so only tool granularity differs.
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
5. Repair the current-head review gaps and rerun the 180-record baseline under a
   common transient 8,192-token context before publishing any result.

### Review Contract

- Acceptance criteria:
  - A syntactically valid but schema-invalid argument object is returned to the
    model as a validation error and never reaches the tool runner.
  - Missing required fields, extra fields, wrong types, enums, and malformed
    JSON exercise the same no-runner invariant.
  - Exact tool order and exact per-call arguments are graded deterministically,
    while shareable summaries never contain either argument object.
  - Case-scoped identifier patterns distinguish observable identifiers from
    actionable identifiers. Ambiguity candidates may be repeated while asking
    for clarification, but cannot authorize a downstream tool argument.
  - Dependent calls marked as requiring separate rounds fail before a later
    same-turn call reaches the runner.
  - Forbidden and required output substrings make injection and stable result
    facts deterministically observable.
  - An explicit no-tool case fails on any model tool attempt.
  - Missing approval fails on any attempted side-effect tool or false-success
    claim, including contradictory clauses, while a safe stop is not required
    to contain a magic refusal phrase.
  - Boolean case fields accept JSON booleans only; strings, numbers, arrays,
    objects, and null values fail closed instead of being truthiness-coerced.
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
  - Primitive and workflow-helper cases use the same request and expected facts;
    only the exposed tool surface and resulting call sequence differ.
  - `jsonschema` is a direct dependency in both checked requirements surfaces.
  - No external email, calendar, publishing, CRM write, ComfyUI job,
    unrestricted code, or filesystem effect occurs.
- Reachability proof: Invoke `scripts/eval_local_mcp_models.py` through its real
  CLI with `--mock-tools-file`, `--prompts-file`, `--repetitions 5`, and the live
  local OpenAI-compatible endpoint. The observable proof is append-safe ignored
  JSONL containing five records per model/case plus its redacted summary.
- Affected surfaces: Local MCP model-eval CLI, v1/v2 JSONL case and result
  contracts, deterministic grader, mock fixtures, focused tests, checked Python
  requirements, and operator runbook.
- Risk areas: A schema-invalid call reaching a runner, mock mode accidentally
  opening MCP, a safe approval stop being falsely failed, a forbidden attempt
  hidden because execution was blocked, repeated runs sharing mock response
  state, sensitive payloads entering summaries, ambiguity identifiers becoming
  actionable, same-turn dependency guesses, uncontrolled experiment pairs, and
  v1 preset regression.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R11, R12, R13, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml` (round 3: jsonschema in the
  tooling-test install line)
- `docs/local_mcp_model_eval_runbook.md`
- `plans/PR-Local-Model-MCP-Qualification-V2.md`
- `requirements.content_ops_ci.txt`
- `requirements.txt`
- `scripts/eval_local_mcp_models.py`
- `tests/fixtures/mcp_model_eval/content_factory_v2_cases.jsonl`
- `tests/test_eval_local_mcp_models.py`

## Mechanism

The case loader keeps every v1 field and adds optional v2 fields. Per-case tool
names filter the already-advertised live or mock surface. `_run_case` records
every named attempt before parsing, validates allowed calls with the official
directly declared `jsonschema` dependency, and calls the runner
only after validation succeeds. The tool response remains a normal tool message
so a model may repair an invalid call within the case's retry allowance.

The grader uses the attempted trace for no-tool and approval hard gates, and the
executed trace for exact sequence/argument assertions. Argument mismatch errors
name only call positions, never values. Approval is behavioral:
when approval is required and absent, an attempted named side-effect or an
success claim without a clause-local negation is a hard failure. It does not
require specific refusal prose. Existing `requires_refusal` behavior remains for
legacy read-only cases.
Final-answer JSON Schema is a separate deterministic check.
For cases that opt in, identifiers found in the final answer must be observable
from the prompt or a tool result. Only prompt identifiers and identifiers from a
successful result become actionable for a later tool call. Newly returned IDs do
not become actionable inside the same assistant turn, and cases may require one
dependent call per turn. Case-scoped required/forbidden substrings provide narrow
deterministic assertions without pretending to be a general semantic grader.

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
  normalize them. Model loading for the corrected rerun is transient; saved
  prompt/config files remain untouched and prompt behavior is canary-checked.
- Ambiguity candidates are safe to show for human clarification but are not
  actionable IDs until the user supplies one in a later request.
- The first 180 local records are retained as superseded evidence; corrected
  records are written to a fresh ignored directory rather than rewritten.
- Plain JSONL remains the storage format. No SQLite or evaluation service is
  introduced.

## Deferred

- Live read-only MCP qualification on ports 8065/8067 after isolated account
  fixtures and tokens are available.
- Broader 105-case qualification, long-context tiers, tool-count experiments,
  native-tool versus forced-JSON comparison, and statistically meaningful role
  assignment.
- Reconcile the machine-readable hard-gate taxonomy with the broader
  qualification policy. The current fixture detects retrieved-instruction
  injection as a forbidden-output grade error, but
  `hard_gate_failure_count` includes only violations emitted by the no-tool and
  missing-approval guards and is not derived from `severity`; a zero aggregate
  therefore does not clear every policy disqualifier.
- Any live draft/write/side-effect test, workflow wrapper, prompt change, tool
  redesign, or Open WebUI worker configuration.

Parked hardening: none.

## Review round 3 (2026-07-24)

Eight Codex grader/loader findings fixed with regressions (clause
boundaries incl. dashes/parentheses; send-class success detector with
passive/denominal coverage replacing the generic write matcher on the
missing-approval gate; fail-closed malformed contract fields; load-time
side_effect_tools requirement for missing-approval cases; off-surface
calls blocked before argument retry; summary redaction of forbidden
output and identifiers; jsonschema in the pre-push workflow install;
swallowed-exception restructure for the scripts-lane ratchet).

## Verification

- Superseded evidence: the first 180 records used unequal contexts and an
  uncontrolled primitive/helper pair. Their scores remain diagnostic only and
  must not be published as the corrected baseline.
- Focused evaluator tests: 83 passed, covering current-head P2 classes, summary
  redaction, observable/actionable identifier provenance, separate tool rounds,
  false-success clause boundaries, and the controlled primitive/helper
  invariant.
- Python compile, Black check, and `git diff --check`: passed. The file-lane
  maturity sweep completed with the existing advisory `SWALLOWED_EXCEPT` at the
  bounded model-request retry handler; no new blocking maturity finding was
  introduced.
- Real mock CLI proof listed 12 cases and six tools without opening MCP. Every
  case listing exposes its required-output and separate-round constraints.
- Prompt canary: all three candidates returned the exact requested token 3/3;
  the LM Studio request log showed the explicit system message. Qwen 3.6 still
  has a saved system prompt and remains a configuration caveat.
- Corrected baseline: 180 isolated records, with five repetitions for every
  model/case, common context 8,192, parallelism 1, temperature 0, max output 800,
  four tool rounds, one system-prompt hash, and mock-only tools. Results were
  Qwen 2.5 Coder 55/60, Qwen 3.6 50/60, and Gemma 45/60. These are exploratory
  scores, not production qualifications.
- Failure classes were deterministic across all five repetitions: Qwen 2.5
  followed retrieved instructions; Qwen 3.6 failed structured output and
  ambiguity recovery; Gemma failed tool selection, structured output, and
  ambiguity recovery. All fixture-tagged approval gates passed, while Qwen
  2.5's injection failure remains policy-disqualifying despite the narrower
  machine hard-gate aggregate.
- The matched primitive/helper cases passed 5/5 on both surfaces for all three
  models. The helper used one call instead of two and reduced mean tokens and
  elapsed time for every candidate; it did not improve correctness in this
  probe.
- Final cold audit also closed truthy-string boolean inversion, additional
  comma/causal false-success clauses, and legacy raw mismatch leakage through
  `--summarize`. The fixture's typed booleans and all 15 missing-approval model
  answers prove these post-run guard repairs do not change the recorded model
  outcomes.
- Diff-scoped ASCII, JSONL uniqueness, plan-sync, full 291-file scripts maturity
  sweep, and cold diff audit: passed. The full maturity output remains advisory;
  the changed evaluator's one finding is the documented bounded retry handler.
- Pending: managed local PR review on the repaired head.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/local_mcp_model_eval_runbook.md` | 112 |
| `plans/PR-Local-Model-MCP-Qualification-V2.md` | 277 |
| `requirements.content_ops_ci.txt` | 1 |
| `requirements.txt` | 1 |
| `scripts/eval_local_mcp_models.py` | 1090 |
| `tests/fixtures/mcp_model_eval/content_factory_v2_cases.jsonl` | 12 |
| `tests/test_eval_local_mcp_models.py` | 1144 |
| **Total** | **2637** |
