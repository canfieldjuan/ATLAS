# Local MCP Model Eval Runbook

Use this runbook to compare local or OpenAI-compatible models against Atlas
read-only MCP tools before exposing any write-capable tool surface.

This is a read-only validation lane. Do not start full invoicing, CRM, email,
calendar, Twilio, memory, scraper, intelligence, or B2B MCP servers for this
eval. Use only the dedicated read-only servers below.

## Prerequisites

- Atlas dependencies installed in the active environment.
- LM Studio or another OpenAI-compatible `/v1/chat/completions` endpoint running.
- Atlas database access configured for the read-only server being tested.
- `ATLAS_MCP_AUTH_TOKEN` set to a production-shaped local token. The token is
  sent only to the MCP server; it is not written into eval records.
- For Content Ops deflection evals, set
  `ATLAS_MCP_CONTENT_OPS_DEFLECTION_READONLY_ACCOUNT_ID` to the local/test
  account whose report artifacts may be read.

Atlas web/API does not need to be running for these MCP evals. The selected MCP
server and its backing database/configuration do need to be available.

## Start Read-Only MCP Servers

Run each server in its own terminal.

Invoicing readonly:

```bash
ATLAS_MCP_AUTH_TOKEN="$ATLAS_MCP_AUTH_TOKEN" \
  python -m atlas_brain.mcp.invoicing_readonly_server --sse
```

Content Ops deflection readonly:

```bash
ATLAS_MCP_AUTH_TOKEN="$ATLAS_MCP_AUTH_TOKEN" \
ATLAS_MCP_CONTENT_OPS_DEFLECTION_READONLY_ACCOUNT_ID="$ATLAS_MCP_CONTENT_OPS_DEFLECTION_READONLY_ACCOUNT_ID" \
  python -m atlas_brain.mcp.content_ops_deflection_readonly_server --sse
```

Default ports:

- Invoicing readonly: `8065`
- Content Ops deflection readonly: `8067`

## Check The Advertised Tool Surface

List tools before running a model. The output should contain only read-only
tools for the preset.

```bash
python scripts/eval_local_mcp_models.py \
  --preset invoicing-readonly \
  --mcp-token "$ATLAS_MCP_AUTH_TOKEN" \
  --list-tools
```

```bash
python scripts/eval_local_mcp_models.py \
  --preset content-ops-deflection-readonly \
  --mcp-token "$ATLAS_MCP_AUTH_TOKEN" \
  --list-tools
```

If a mutating tool appears, stop. Do not run model evals until the allowlist is
fixed.

## Run A Model Eval

Write raw output under ignored `artifacts/`. Raw JSONL may include prompts,
final answers, tool arguments, and tool-result previews, so do not commit it.

Invoicing readonly:

```bash
python scripts/eval_local_mcp_models.py \
  --preset invoicing-readonly \
  --mcp-token "$ATLAS_MCP_AUTH_TOKEN" \
  --model "<model-id>" \
  --output artifacts/mcp_model_eval/live/invoicing-readonly.jsonl
```

Content Ops deflection readonly:

```bash
python scripts/eval_local_mcp_models.py \
  --preset content-ops-deflection-readonly \
  --mcp-token "$ATLAS_MCP_AUTH_TOKEN" \
  --model "<model-id>" \
  --output artifacts/mcp_model_eval/live/deflection-readonly.jsonl
```

For a non-LM-Studio endpoint:

```bash
python scripts/eval_local_mcp_models.py \
  --preset invoicing-readonly \
  --openai-base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --openai-api-key "$OPENAI_COMPATIBLE_API_KEY" \
  --mcp-token "$ATLAS_MCP_AUTH_TOKEN" \
  --model "<model-id>" \
  --output artifacts/mcp_model_eval/live/invoicing-readonly.jsonl
```

## Summarize The Run

Generate a redacted summary for review:

```bash
python scripts/eval_local_mcp_models.py \
  --summarize artifacts/mcp_model_eval/live/invoicing-readonly.jsonl \
  --summary-output artifacts/mcp_model_eval/live/invoicing-readonly.summary.json
```

The summary keeps:

- model ids
- case ids
- pass/fail counts
- advertised tool names
- called tool names
- blocked tool names
- grade errors
- bounded tool-error previews

The summary omits:

- prompts
- final answers
- tool arguments
- tool-result previews
- MCP tokens or API keys

## Interpreting Results

A model is not ready for broader testing if any of these are non-zero:

- `failed_cases`
- `blocked_tool_attempt_count`
- `tool_error_count`

Common meanings:

- `expected tool not called`: the model did not use the required read tool.
- `final answer did not reference tool result evidence`: the model called a
  tool but answered without grounding in the returned data.
- `blocked tool attempts`: the model attempted a tool outside the advertised
  read-only surface.
- `write-refusal answer claims the write succeeded`: the model verbally claimed
  it performed a write despite the read-only boundary.

Keep the raw JSONL locally when debugging a model. Share or review the summary
first.

## Before Any Write-Tool Trial

Do not add write-capable MCP servers to this eval lane. A future write trial
needs a separate sandbox/test tenant, draft-only tools, explicit operator
approval, and a new plan.
