# Content Ops Marketer Verify Live Run Artifact

This runbook captures the evidence expected when the verify-only Content Ops
marketer MCP is tested against both public connector surfaces:

- Claude rich profile: `verify_draft`
- ChatGPT search/fetch adapter profile: `search`, `fetch`

Use this only after public Claude and ChatGPT connector registrations are
available. Until then, the deterministic local evidence is the launcher,
discovery, OAuth e2e, and dual-client wrapper tests.

## Scope

This artifact proves the public OAuth surfaces can be reached and list the
expected tools. It does not prove generation, publishing, registry mutation, or
durable verdict persistence.

## Secret Hygiene

Never paste these values into the artifact, PR, issue, terminal transcript, or
docs:

- approval token
- client secret
- authorization code
- access token
- refresh token
- full OAuth state-file content
- customer draft content that was not created for the smoke

Allowed evidence is limited to public URLs, command names, exit statuses,
sanitized success lines, tool names, timestamps, commit SHAs, and operator
notes.

## Preflight

Confirm the target commit is on `main`:

```bash
git log --oneline -5 origin/main
```

Create or reuse a local approval token file. Do not print the value:

```bash
mkdir -p .secrets
chmod 700 .secrets
python - <<'PY' > .secrets/content-ops-marketer-verify-approval-token
import secrets

print(secrets.token_urlsafe(32))
PY
chmod 600 .secrets/content-ops-marketer-verify-approval-token
```

Dry-run the rich verifier launcher:

```bash
.venv/bin/python scripts/start_content_ops_marketer_verify_oauth_server.py \
  --approval-token-file .secrets/content-ops-marketer-verify-approval-token \
  --dry-run
```

Dry-run the ChatGPT adapter launcher:

```bash
.venv/bin/python scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py \
  --approval-token-file .secrets/content-ops-marketer-verify-approval-token \
  --dry-run
```

Dry-run validates env and prints the Funnel plus smoke commands. It does not
start either MCP server. Start both servers without `--dry-run` and keep each
one running in a separate terminal or tmux pane while you capture the live
artifact:

```bash
# Terminal A: rich verifier
.venv/bin/python scripts/start_content_ops_marketer_verify_oauth_server.py \
  --approval-token-file .secrets/content-ops-marketer-verify-approval-token

# Terminal B: ChatGPT adapter
.venv/bin/python scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py \
  --approval-token-file .secrets/content-ops-marketer-verify-approval-token
```

Follow the Funnel route commands printed by each launcher while the servers are
running, then run discovery for each public surface:

```bash
.venv/bin/python scripts/check_content_ops_marketer_verify_oauth_discovery.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer/mcp

.venv/bin/python scripts/check_content_ops_marketer_verify_oauth_discovery.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt/mcp
```

## Dual-Client Smoke

Run the combined smoke after both public surfaces pass discovery:

```bash
.venv/bin/python scripts/check_content_ops_marketer_verify_dual_client_rollout.py \
  --rich-issuer-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer \
  --rich-resource-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer/mcp \
  --chatgpt-adapter-issuer-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt \
  --chatgpt-adapter-resource-url https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt/mcp \
  --approval-token-file .secrets/content-ops-marketer-verify-approval-token
```

Expected tool evidence:

- `claude-rich`: `verify_draft`
- `chatgpt-search-fetch`: `fetch`, `search`

## Artifact Template

Copy this template into the tracker issue or a follow-up PR after replacing the
placeholders. Keep it sanitized.

```md
## Content Ops marketer verify live run artifact

Date:
Operator:
Repo commit:
Environment:

### Public surfaces

- Claude rich issuer:
- Claude rich resource:
- ChatGPT adapter issuer:
- ChatGPT adapter resource:

### Commands run

- Rich launcher dry-run: PASS | FAIL
- Adapter launcher dry-run: PASS | FAIL
- Rich verifier startup: PASS | FAIL
- ChatGPT adapter startup: PASS | FAIL
- Rich discovery smoke: PASS | FAIL
- Adapter discovery smoke: PASS | FAIL
- Dual-client rollout smoke: PASS | FAIL

### Sanitized dual-client output

- Claude rich profile: PASS | FAIL
  - Expected tools: verify_draft
  - Observed tools:
- ChatGPT search/fetch profile: PASS | FAIL
  - Expected tools: fetch, search
  - Observed tools:

### Connector registration notes

- Claude connector registration:
- ChatGPT connector registration:
- Reconnect/retry needed:

### Follow-up decisions

- Durable verdict/session persistence needed: yes | no
- Approval-page account picker needed: yes | no
- Reason:
```

## Stop Conditions

Stop and do not attach the artifact if any command output includes a token,
secret, authorization code, refresh token, or customer draft content. Redact the
local file, rerun the smoke, and only then post sanitized evidence.

If the ChatGPT `fetch` leg fails after `search` succeeds, do not assume durable
persistence is required yet. First confirm the adapter process stayed running,
the same tenant account was bound, and the returned verdict id was copied
exactly. Add durable persistence only after live evidence shows process-local
state is the real failure.
