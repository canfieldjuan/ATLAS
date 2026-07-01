# PR-Local-MCP-Model-Eval-Harness

## Why this slice exists

The operator wants to compare local LM Studio models against Atlas MCP tools
before allowing any model near write-capable servers. The root cause is that
local model testing is currently an ad hoc UI exercise: a model can appear
helpful without proving it selected the right MCP read tool, passed valid
arguments, respected read-only boundaries, and summarized the tool result
faithfully. This fixes the root for the first evaluation step by adding a
repeatable read-only harness; it does not attempt to certify write actions.
This intentionally exceeds the 400 LOC target because the usable vertical slice
needs the CLI, MCP/OpenAI-compatible tool loop, built-in read-only presets,
JSONL result recording, and the safety tests that prove the write boundary
does not call MCP.

## Scope (this PR)

Ownership lane: mcp/local-model-evals
Slice phase: Vertical slice

1. Add a local script that connects to a Streamable HTTP MCP endpoint, filters
   the advertised tool surface through a read-only allowlist, and runs one or
   more OpenAI-compatible local models through a bounded tool-use loop.
2. Persist per-model/per-case JSONL records that include called tools, blocked
   tool attempts, final answers, and pass/fail grading against expected read
   tools.
3. Include the first safe presets for read-only invoicing and read-only Content
   Ops deflection. No write-capable tool is advertised to the model.

### Review Contract

- Acceptance criteria:
  - The harness never exposes tools outside the selected read-only allowlist.
  - A model-emitted tool call whose name is not advertised is blocked locally
    and recorded; the MCP server is not called for that tool.
  - Built-in evaluation cases can require expected read tools and fail when the
    model does not call them.
  - The script can list the allowed tool surface without calling a model.
  - JSONL output is append-safe and includes enough context to compare models.
- Affected surfaces:
  - Operator-only script under `scripts/`.
  - Unit tests for harness filtering/grading/tool-loop helpers.
- Risk areas:
  - Accidentally advertising mutating MCP tools.
  - Treating a blocked write attempt as a successful run.
  - Requiring live LM Studio or live MCP services during unit tests.
- Reviewer rules:
  - R1 requirements match, R2 test evidence, R3 security/auth, R8 contracts,
    R13 class-fix discipline for boundary logic, R14 codebase verification.

### Files touched

- `plans/PR-Local-MCP-Model-Eval-Harness.md`
- `scripts/eval_local_mcp_models.py`
- `tests/test_eval_local_mcp_models.py`

## Mechanism

`scripts/eval_local_mcp_models.py` accepts one or more model ids, an
OpenAI-compatible base URL (LM Studio defaults to `http://127.0.0.1:1234/v1`),
and a Streamable HTTP MCP URL. It lists MCP tools through the official MCP
client, converts only allowlisted tools into OpenAI tool schemas, and runs a
bounded chat-completions loop. Tool calls are executed only when the tool name
is in the advertised allowlist; unknown or mutating names become blocked
observations and are written to the JSONL result.

Built-in presets keep the first path safe:

- `invoicing-readonly`: the eight read-only invoicing tools.
- `content-ops-deflection-readonly`: `search`, `fetch`, and `fetch_delta`.

The script also supports custom read-only allowlists for local experiments, but
it rejects known mutating tool names in `--allow-tool` so a quick CLI typo cannot
turn the harness into a write proxy.

## Intentional

- Streamable HTTP only for this first slice. Stdio spawning would add process
  orchestration and server lifecycle decisions; local connector testing already
  uses Streamable HTTP endpoints.
- Read-only allowlists live in the harness rather than inferring safety from
  tool names. Name-based heuristics are too easy to get wrong.
- No live model or live MCP service in unit tests. Tests mock the model/MCP
  boundaries and prove the filtering, blocking, and grading branches.

## Deferred

- Stdio MCP server spawning for fully local one-command evals.
- A broader fixture suite with prompt-injection-in-tool-output cases and
  model leaderboard summaries.
- Draft-only write evaluation against a sandbox/test tenant after read-only
  models prove reliable.

Parked hardening: none.

## Verification

- Python compile for the harness and tests -- pass.
- Focused pytest for the harness tests -- 7 passed.
- CLI help command -- pass.
- Invoicing read-only case listing command -- pass.
- Plan sync check command -- pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Local-MCP-Model-Eval-Harness.md` | 112 |
| `scripts/eval_local_mcp_models.py` | 613 |
| `tests/test_eval_local_mcp_models.py` | 241 |
| **Total** | **966** |
