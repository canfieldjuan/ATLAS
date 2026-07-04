# PR-Local-MCP-Eval-Live-Runbook

## Why this slice exists

The first local MCP model-eval slice added the harness, but operators still have
to infer the safe live-run procedure, output expectations, and pass/fail
interpretation from the CLI. The root cause is not missing model logic; it is
that a live read-only evaluation has no committed operator contract or sanitized
result shape. This slice fixes that root by adding the smallest runbook and
machine-readable summary path needed to run local/OpenAI-compatible models
against real read-only MCP servers, capture ignored JSONL output, and review a
safe summary before any write-capable MCP tools are considered.

This is functional validation, not write-tool enablement. The slice proves the
read-only harness can produce reviewable evidence from real runs while keeping
live data out of git.

The diff exceeds the 400 LOC soft cap because the slice has to ship the
operator runbook, summary implementation, synthetic fixture, and detector tests
together; splitting the docs or tests away would leave the live-run contract
unguarded.

## Scope (this PR)

Ownership lane: mcp/local-model-evals
Slice phase: Functional validation

1. Add a harness summary artifact so completed eval JSONL can be converted into
   a small, operator-readable JSON summary: models, cases, pass counts, failed
   cases, advertised tools, blocked tool attempts, and tool errors.
2. Add a runbook for live read-only MCP evals covering prerequisites, safe
   server startup commands, model-run commands, summary generation, and what
   must remain out of git.
3. Add a sanitized fixture that documents the JSONL result shape without using
   live customer data.

### Review Contract

- Acceptance criteria:
  - Summary generation consumes existing harness JSONL records and never needs
    live LM Studio or live MCP services in unit tests.
  - The summary flags failed cases, blocked tool attempts, and tool errors so a
    human can decide whether a model is safe to keep testing.
  - The runbook starts from read-only MCP servers only and does not document any
    write-capable server run as part of this lane.
  - The committed fixture is synthetic and contains no real customer, invoice,
    report, token, or URL data.
  - Live eval outputs remain under ignored `artifacts/` paths.
- Affected surfaces:
  - Operator-only harness script under `scripts/`.
  - Operator runbook under `docs/`.
  - Unit tests and sanitized fixture for the summary path.
- Risk areas:
  - Accidentally treating a model with blocked write attempts as safe.
  - Committing live eval output with customer data.
  - Documenting a command that exposes write-capable MCP tools.
  - Letting summary code drift from the JSONL record shape emitted by the
    harness.
- Reviewer rules:
  - R1 requirements match, R2 test evidence, R3 security/auth, R8 contracts,
    R10 evaluator/gate behavior, R14 codebase verification.

### Files touched

- `docs/local_mcp_model_eval_runbook.md`
- `plans/PR-Local-MCP-Eval-Live-Runbook.md`
- `scripts/eval_local_mcp_models.py`
- `tests/fixtures/mcp_model_eval/synthetic_results.jsonl`
- `tests/test_eval_local_mcp_models.py`

## Mechanism

The existing `scripts/eval_local_mcp_models.py` remains the live runner. This
slice adds summary generation as an offline operation over the runner's JSONL
output so the operator can do:

```bash
python scripts/eval_local_mcp_models.py \
  --preset invoicing-readonly \
  --model <local-model-id> \
  --output artifacts/mcp_model_eval/live/invoicing-readonly.jsonl

python scripts/eval_local_mcp_models.py \
  --summarize artifacts/mcp_model_eval/live/invoicing-readonly.jsonl \
  --summary-output artifacts/mcp_model_eval/live/invoicing-readonly.summary.json
```

The summary path parses each JSONL object, groups by model, counts cases,
records pass/fail totals, and preserves only bounded metadata needed for review:
case ids, grade errors, called tools, blocked tool names, tool-error previews,
and advertised tools. It does not copy prompts, final answers, tool arguments,
or tool-result previews into the summary.

The runbook documents how to start the two read-only MCP servers, how to list
tools before running a model, how to execute the invoicing and deflection
presets, and how to interpret the resulting summary. The checked-in fixture is
synthetic and exists only to lock the parser/summary contract.

## Intentional

- No committed live eval output. Real runs may include customer-support or
  invoice details, so live JSONL and summaries stay under ignored `artifacts/`.
- No write-capable MCP server commands. This lane is still proving read-only
  behavior before any write tools are exposed to models.
- No automatic MCP server process launcher in this slice. Operators can already
  start the servers directly; process orchestration is deferred until the
  result contract is proven useful.
- The summary omits prompts, final answers, tool arguments, and tool-result
  previews by design. The raw JSONL remains available locally when the operator
  needs detailed inspection.

## Deferred

- One-command local MCP server spawning and health checks.
- Prompt-injection-in-tool-output eval cases and model leaderboard reports.
- Draft-only write evaluation against a sandbox/test tenant after read-only
  model behavior is proven.

Parked hardening: none.

## Verification

- `python -m py_compile scripts/eval_local_mcp_models.py tests/test_eval_local_mcp_models.py`
- `pytest tests/test_eval_local_mcp_models.py -q` -- 28 passed.
- `python scripts/eval_local_mcp_models.py --preset invoicing-readonly --list-cases`
- `python scripts/eval_local_mcp_models.py --summarize tests/fixtures/mcp_model_eval/synthetic_results.jsonl --summary-output /tmp/synthetic_mcp_eval_summary.json`
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/eval_local_mcp_models.py' --top 25` -- ratchet gate passed.
- `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/local_mcp_model_eval_runbook.md` | 161 |
| `plans/PR-Local-MCP-Eval-Live-Runbook.md` | 139 |
| `scripts/eval_local_mcp_models.py` | 131 |
| `tests/fixtures/mcp_model_eval/synthetic_results.jsonl` | 2 |
| `tests/test_eval_local_mcp_models.py` | 61 |
| **Total** | **494** |
