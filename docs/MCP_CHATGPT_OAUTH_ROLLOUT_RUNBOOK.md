# ChatGPT OAuth rollout runbook for Atlas MCP servers

This runbook captures the pattern that successfully connected the
read-only invoicing MCP server to ChatGPT online. Use it for the next Atlas
MCP server instead of rediscovering the same OAuth, Tailscale, and smoke-test
issues one server at a time.

## Proven outcome

The first completed server is:

- Local module: `atlas_brain.mcp.invoicing_readonly_server`
- Public MCP URL: `https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp`
- Auth mode: OAuth
- Tool surface: exactly eight read-only invoicing tools
- Verification: discovery smoke, OAuth e2e smoke, and actual ChatGPT connector
  approval all succeeded.

The working implementation landed through:

- `plans/PR-Invoicing-Readonly-ChatGPT-OAuth.md`
- `plans/PR-Invoicing-Readonly-OAuth-Discovery-Smoke.md`
- `plans/PR-Invoicing-Readonly-OAuth-E2E-Smoke.md`
- `plans/PR-Invoicing-Readonly-OAuth-Operator-Launcher.md`

Use those PRs as concrete references.

## Current handoff: 2026-05-19

The invoicing connector rollout now has two proven ChatGPT-facing surfaces:

| Surface | Public MCP URL | Port | Status |
|---|---|---:|---|
| Read-only invoicing | `https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp` | 8065 | Proven earlier; read-only pattern source |
| Draft-writer invoicing | `https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp` | 8066 | Proven live with blocked-draft write smoke |

Recent MCP connector PRs that matter for the next session:

- `#623` added the read-only Tailscale Funnel route checker for the
  draft-writer path and connector-specific protected-resource metadata route.
- `#626` allowed the configured public Tailscale host in draft-writer OAuth
  transport security while keeping DNS-rebinding protection enabled.
- `#629` fixed invoice-number readback by making repository invoice-number
  lookup case-insensitive. This closed the live bug where
  `INV-2026-May-0185` could be created but not fetched by number.
- `#632` added `scripts/check_invoicing_draft_writer_live_write.py`, the
  explicit blocked-draft write smoke.
- `#633` added optional OAuth state-file persistence for invoicing MCP clients
  and refresh tokens.

Current local draft-writer runtime state after restart:

- Server command: `.venv/bin/python scripts/start_invoicing_draft_writer_oauth_server.py --approval-token-file .secrets/invoicing-draft-writer-approval-token`
- Required env for restart durability:
  `ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_STATE_FILE=.secrets/invoicing-draft-writer-oauth-state.json`
- Public URL: `https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp`
- Listening port: `8066`
- State file: `.secrets/invoicing-draft-writer-oauth-state.json`
- Approval token file: `.secrets/invoicing-draft-writer-approval-token`
- The state file stores registered OAuth clients and refresh tokens only. It
  does not store pending approvals, authorization codes, or access tokens.

Post-restart verification that passed on 2026-05-19:

```bash
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_discovery.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp

.venv/bin/python scripts/check_invoicing_draft_writer_oauth_e2e.py \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp

.venv/bin/python scripts/check_invoicing_draft_writer_live_write.py \
  --create-blocked-draft \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

Expected live-write result: action `reused`, invoice
`INV-2026-May-0185`, blocker `no_email`, warnings `subtotal_zero` and
`no_contact_id`. That draft is intentionally blocked and should not be sent.

Known limits to carry forward:

- The server is not daemonized. It runs in a foreground/persistent session; a
  systemd user-service or other process manager is a future operational slice.
- ChatGPT may need one reconnect after enabling a new state file so the file can
  be seeded with its client and refresh token.
- Dynamic OAuth registration is public by design, but persisted client state is
  capped. Do not remove that cap without replacing it with a better abuse
  boundary.
- Copilot errored on recent PR reviews and Codex review hit usage limits, so
  local `scripts/local_pr_review.sh` plus focused tests were the real review
  gate for `#633`.

Recommended next MCP connector slices:

1. Add a process-manager/operator service for draft-writer OAuth if the server
   needs to survive terminal/session closure.
2. Repeat the same read-only-first OAuth pattern for the next MCP domain
   instead of exposing a full write server.
3. Only add more invoicing write tools after a separate guardrail document and
   allowlist smoke exist for that exact tool surface.

## Fast path for the next server

Do not start by debugging ChatGPT. Build and verify the server in this order:

1. Define the safe tool surface.
2. Add OAuth mode.
3. Add public discovery smoke.
4. Add OAuth e2e smoke.
5. Add an operator launcher.
6. Run discovery and e2e smokes against the public Tailscale URL.
7. Only then connect ChatGPT.

If any step fails, fix that layer before moving on. This prevents wasting time
inside the ChatGPT UI when the real issue is local env, OAuth metadata, or
Tailscale routing.

## Step 1: choose a safe surface

Before exposing a server publicly, classify every tool:

- `read`: safe to expose once authenticated.
- `write`: creates, updates, sends, deletes, approves, records payments, or
  triggers external side effects.
- `sensitive-read`: reads customer financial, calendar, CRM, or personal data.
  This still needs auth and usually deserves a read-only surface first.

Recommended first rollout for each domain is a read-only connector. Create a
separate `<domain>_readonly_server.py` when the existing server mixes reads and
writes.

For read-only surfaces, add a boundary smoke equivalent to
`scripts/check_invoicing_readonly_mcp_connector.py`:

- Assert unauthenticated MCP requests are rejected.
- Authenticate and call only `list_tools`.
- Require the exact expected read-only tool list.
- Reject known mutating tools explicitly.

Do not call real data tools in connector boundary smoke.

For invoicing write access, do not expose the full `atlas-invoicing` server.
Use the draft-only contract in
`docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md` before adding any mutating
ChatGPT connector. The first write surface is draft creation/update only;
send, payment, void, PDF export, and service mutation stay out of scope.

The first write connector is `atlas-invoicing-draft-writer` at:

```text
https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

Use the draft-writer launcher and smokes:

```bash
mkdir -p .secrets
chmod 700 .secrets
python - <<'PY' > .secrets/invoicing-draft-writer-approval-token
import secrets

print(secrets.token_urlsafe(32))
PY
chmod 600 .secrets/invoicing-draft-writer-approval-token
.venv/bin/python scripts/start_invoicing_draft_writer_oauth_server.py \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --dry-run
tailscale funnel --bg --yes \
  --set-path /invoicing-draft-writer \
  http://127.0.0.1:8066
tailscale funnel --bg --yes \
  --set-path /.well-known/oauth-protected-resource/invoicing-draft-writer \
  http://127.0.0.1:8066/.well-known/oauth-protected-resource/invoicing-draft-writer
.venv/bin/python scripts/check_invoicing_draft_writer_funnel_routes.py
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_discovery.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_e2e.py \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
.venv/bin/python scripts/check_invoicing_draft_writer_live_write.py \
  --create-blocked-draft \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

Run the live write smoke only after the no-mutation e2e smoke passes. It creates
or reuses one idempotent blocked test draft and verifies invoice-number readback
plus pending-draft blockers.

If the e2e smoke reaches token exchange but the MCP session fails with
`421 Misdirected Request` / `Invalid Host header`, the OAuth server's transport
host allowlist does not include the public Tailscale host. Keep DNS-rebinding
protection enabled and add the configured issuer/resource host to the allowlist.

## Step 2: add OAuth mode

Use `atlas_brain/mcp/invoicing_readonly_oauth.py` as the template. The minimum
OAuth provider needs:

- Dynamic client registration.
- Authorization-code flow with PKCE.
- Operator approval page.
- Access-token verification.
- Refresh-token support if the MCP library requests it.
- Client binding for authorization codes and refresh tokens.

Server wiring should follow
`atlas_brain/mcp/invoicing_readonly_server.py`:

- Keep bearer mode as default for direct clients.
- Add `<SERVER>_AUTH_MODE=oauth`.
- Configure `AuthSettings`.
- Configure `ClientRegistrationOptions`.
- Configure `ProviderTokenVerifier`.
- Use MCP library routes for:
  - `/.well-known/oauth-authorization-server`
  - `/authorize`
  - `/token`
  - `/register`
  - protected-resource metadata

Auth mode variables should be server-specific. For example:

```bash
ATLAS_MCP_<SERVER>_AUTH_MODE=oauth
ATLAS_MCP_<SERVER>_OAUTH_ISSUER_URL=https://atlas-brain.tailc7bd29.ts.net/<path>
ATLAS_MCP_<SERVER>_OAUTH_RESOURCE_URL=https://atlas-brain.tailc7bd29.ts.net/<path>/mcp
ATLAS_MCP_<SERVER>_OAUTH_APPROVAL_TOKEN=<long-random-token>
```

Use a long random approval token. Never commit it. Never print it in scripts.

## Step 3: make the approval form prefix-safe

The approval form must not post to an absolute root path such as
`/oauth/approve` when the public URL is path-prefixed. The safe shape is a form
with no `action` attribute:

```html
<form method="post">
```

Browsers submit that form to the current external URL, preserving the public
prefix. Do not rely on ASGI `root_path`; Tailscale and other proxies may strip
or omit it.

## Step 4: understand the Tailscale metadata route

This was the main routing trap.

For a path-prefixed MCP resource:

```text
resource_url = https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp
```

the MCP library advertises protected-resource metadata at:

```text
https://atlas-brain.tailc7bd29.ts.net/.well-known/oauth-protected-resource/invoicing-readonly/mcp
```

That route is at host root, not under `/invoicing-readonly`.

The working Tailscale Funnel route preserved the backend path:

```bash
tailscale funnel --bg --yes \
  --set-path /.well-known/oauth-protected-resource \
  http://127.0.0.1:<PORT>/.well-known/oauth-protected-resource
```

Do not point that route at only `http://127.0.0.1:<PORT>`. Tailscale strips the
matched public prefix before forwarding, and the backend will receive the wrong
path.

Expected status after configuration:

```bash
tailscale funnel status
```

should include:

```text
|-- /<server-path>                         proxy http://127.0.0.1:<PORT>
|-- /.well-known/oauth-protected-resource  proxy http://127.0.0.1:<PORT>/.well-known/oauth-protected-resource
```

## Step 5: add discovery smoke

Add a script equivalent to
`scripts/check_invoicing_readonly_oauth_discovery.py`.

It should:

- Read issuer/resource URLs from env or CLI flags.
- Fetch `<issuer>/.well-known/oauth-authorization-server`.
- Fetch the RFC 9728 protected-resource metadata URL derived from the resource
  URL.
- Probe the MCP resource without auth.
- Require HTTP `401`.
- Require `WWW-Authenticate` to include the protected-resource metadata URL.
- Validate issuer/resource/scopes in JSON.

It should not:

- Use the approval token.
- Exchange OAuth tokens.
- Call data tools.

This script tells you whether public routing and metadata discovery are ready
before involving ChatGPT.

## Step 6: add OAuth e2e smoke

Add a script equivalent to `scripts/check_invoicing_readonly_oauth_e2e.py`.

It should:

- Dynamically register a client.
- Start authorization with PKCE.
- Read the approval `request_id` from the redirect.
- POST the operator approval token to the approval endpoint.
- Extract the authorization code without following the redirect to ChatGPT.
- Exchange the code for a bearer token.
- Open an MCP session with that token.
- Call only `list_tools`.
- Assert the exact expected tool list.

It should not print:

- Approval token.
- Client secret.
- Authorization code.
- Access token.
- Refresh token.

Run e2e with the repo virtualenv:

```bash
.venv/bin/python scripts/check_<server>_oauth_e2e.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/<path> \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/<path>/mcp
```

Do not rely on the script shebang unless the system Python has the MCP package.

## Step 7: add an operator launcher

Add a launcher equivalent to
`scripts/start_invoicing_readonly_oauth_server.py`.

It should:

- Load `.env` and `.env.local`.
- Let process env override dotenv values.
- Force OAuth mode for that server.
- Validate issuer URL, resource URL, approval token length, and port.
- Start `<python> -m atlas_brain.mcp.<server> --sse`.
- Run in foreground by default.
- Support `--dry-run`.
- Print the Tailscale route, discovery smoke, and e2e smoke commands.
- Mask all secrets.

It should not:

- Mutate Tailscale config automatically.
- Daemonize silently.
- Write tokens to stdout.
- Start from shell-sourced env assumptions.

## Step 8: ChatGPT connector setup

After discovery and e2e smokes pass:

1. Add the connector in ChatGPT using the public MCP URL:

   ```text
   https://atlas-brain.tailc7bd29.ts.net/<path>/mcp
   ```

2. Choose OAuth authentication.
3. ChatGPT should redirect to the server approval page.
4. Paste the local approval token from `.env`.
5. Approve the connector.
6. Confirm ChatGPT can list/use only the intended tools.

If ChatGPT fails before the approval page, rerun discovery smoke.
If it fails after approval, rerun e2e smoke and tail the server log.

## Reusable test checklist

Each server PR should include tests for:

- OAuth setting validation fails closed.
- Approval page does not use an absolute root action.
- Authorization codes are bound to the issuing client.
- Refresh tokens are bound to the issuing client.
- Discovery smoke URL derivation for path-prefixed resources.
- E2E smoke does not print secrets.
- E2E smoke fails on extra/mutating tool exposure.
- Launcher dotenv parsing and precedence.
- Launcher prints configured port in Funnel guidance.
- Launcher dry-run does not start subprocess.

## Common failure modes

### Discovery endpoint returns 401

The server is probably running in bearer mode, not OAuth mode. Check:

```bash
ATLAS_MCP_<SERVER>_AUTH_MODE=oauth
```

Use the operator launcher instead of shell-sourcing `.env` manually.

### Protected-resource metadata returns 502 or 404

The Tailscale root well-known route is missing or forwarding to the wrong
backend path. Add:

```bash
tailscale funnel --bg --yes \
  --set-path /.well-known/oauth-protected-resource \
  http://127.0.0.1:<PORT>/.well-known/oauth-protected-resource
```

### Approval form posts to the wrong path

The form probably uses an absolute `action="/oauth/approve"`. Remove the action
attribute.

### E2E smoke fails with `ModuleNotFoundError: No module named 'mcp'`

Run with the repo virtualenv:

```bash
.venv/bin/python scripts/check_<server>_oauth_e2e.py ...
```

### ChatGPT works until restart, then fails

Set the server-specific OAuth state file env var before startup:

```bash
ATLAS_MCP_INVOICING_READONLY_OAUTH_STATE_FILE=.secrets/invoicing-readonly-oauth-state.json
ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_STATE_FILE=.secrets/invoicing-draft-writer-oauth-state.json
```

The state file persists registered OAuth clients and refresh tokens only. It is
local operator secret material and should stay under `.secrets/`. Pending
approval requests, one-time authorization codes, and access tokens remain
process-local.

## What to defer until after read-only works

Do not add write access as part of the initial connector rollout. First prove:

- OAuth works.
- The safe read-only tool list is exact.
- ChatGPT uses the connector predictably.
- Logs are understandable when the flow fails.

Only then design a write-capable surface with separate approval rules,
idempotency, audit logs, and explicit confirmation semantics.
