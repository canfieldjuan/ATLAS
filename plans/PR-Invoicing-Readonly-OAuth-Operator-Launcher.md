## Why this slice exists

The read-only invoicing OAuth connector now has discovery and e2e smokes, but
starting the server is still too fragile. In the last live test, shell-sourcing
`.env` failed in the interactive environment, so we had to use an ad hoc Python
launcher to pass OAuth env vars into the server process.

This slice turns that ad hoc launch path into a reusable operator script.

## Scope (this PR)

1. Add an operator launcher for the read-only invoicing OAuth MCP server.
2. Load `.env` and `.env.local` without printing secrets.
3. Validate OAuth mode, issuer URL, resource URL, approval token length, and
   port before starting.
4. Start `atlas_brain.mcp.invoicing_readonly_server --sse` with the current
   Python interpreter.
5. Print the follow-up discovery/e2e smoke commands and the required Tailscale
   Funnel route.
6. Add unit tests for env loading, validation, command construction, dry-run,
   and secret-safe output.

### Files touched

- `scripts/start_invoicing_readonly_oauth_server.py`
- `tests/test_start_invoicing_readonly_oauth_server.py`
- `CLAUDE.md`
- `plans/PR-Invoicing-Readonly-OAuth-Operator-Launcher.md`

## Mechanism

The launcher reads simple `KEY=value` dotenv lines from `.env` and `.env.local`
with `.env.local` taking precedence. It overlays those values on the current
environment, forces `ATLAS_MCP_INVOICING_READONLY_AUTH_MODE=oauth`, validates
the required OAuth keys, then calls:

```bash
<python> -m atlas_brain.mcp.invoicing_readonly_server --sse
```

By default it runs in the foreground so the operator can see uvicorn logs. A
`--dry-run` mode validates configuration and prints the command/smoke guidance
without starting the process.

## Intentional

- The script does not manage Tailscale Funnel state. It prints the exact route
  command, but the operator still controls public exposure.
- The script does not daemonize or write a PID file. Foreground execution is
  simpler and safer for this local operator workflow.
- The script never prints `ATLAS_MCP_AUTH_TOKEN` or
  `ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN`.

## Deferred

- Process supervision/systemd integration is deferred until the connector needs
  unattended uptime.
- Durable OAuth client/token persistence remains deferred until restart
  resilience is actually needed.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile scripts/start_invoicing_readonly_oauth_server.py tests/test_start_invoicing_readonly_oauth_server.py
.venv/bin/python -m pytest tests/test_start_invoicing_readonly_oauth_server.py
bash scripts/local_pr_review.sh --allow-dirty
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Launcher script | ~210 |
| Tests | ~190 |
| Docs/plan | ~110 |
| **Total** | **~510** |

This exceeds the soft cap because the operator helper ships with parser and
secret-safety tests.
