## Why this slice exists

The draft-writer OAuth connector now has correct launch guidance, but the
operator still has to visually inspect `tailscale funnel status` before running
public discovery/e2e. That is fragile: the current machine has read-only
invoicing routes configured, but not the draft-writer routes.

This slice adds a read-only route checker so the operator can fail fast before
connecting ChatGPT or running public OAuth smokes.

## Scope (this PR)

1. Add a draft-writer Funnel route checker.
2. Validate the primary `/invoicing-draft-writer` route proxies to the
   configured local port.
3. Validate the per-connector protected-resource metadata route proxies to the
   same local port and preserves the backend metadata path.
4. Print the exact `tailscale funnel` commands when routes are missing or
   wrong.
5. Add unit tests for route parsing, happy path, route drift, and CLI failure.

### Files touched

- `scripts/check_invoicing_draft_writer_funnel_routes.py`
- `tests/test_check_invoicing_draft_writer_funnel_routes.py`
- `scripts/start_invoicing_draft_writer_oauth_server.py`
- `tests/test_start_invoicing_draft_writer_oauth_server.py`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `plans/PR-Invoicing-Draft-Writer-Funnel-Route-Check.md`

## Mechanism

The checker runs:

```bash
tailscale funnel status --json
```

It derives the public host and app path from the resource URL. For the default
resource URL, it checks:

```text
Web["atlas-brain.tailc7bd29.ts.net:443"].Handlers["/invoicing-draft-writer"].Proxy
Web["atlas-brain.tailc7bd29.ts.net:443"].Handlers["/.well-known/oauth-protected-resource/invoicing-draft-writer"].Proxy
```

The script does not mutate Funnel state. It only reports OK or prints the
commands the operator should run.

## Intentional

- This is read-only. The operator still chooses when to expose the write
  connector publicly.
- The metadata route is per-connector, not the host-root metadata prefix. That
  avoids clobbering the existing read-only invoicing connector on the same
  Tailscale hostname.
- The checker validates proxies exactly enough to catch wrong-port and
  wrong-path mistakes, but it does not attempt to verify HTTPS or OAuth
  metadata. Discovery/e2e smokes remain responsible for that.

## Deferred

- A write-capable route installer is deferred. Explicit operator action is
  safer for first public exposure of write access.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile scripts/check_invoicing_draft_writer_funnel_routes.py tests/test_check_invoicing_draft_writer_funnel_routes.py
.venv/bin/pytest tests/test_check_invoicing_draft_writer_funnel_routes.py -q
bash scripts/local_pr_review.sh --allow-dirty
git diff --check
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Checker script | ~190 |
| Tests | ~160 |
| Launcher guidance update | ~45 |
| Docs/plan | ~165 |
| **Total** | **~560** |

This exceeds the soft cap because the route checker ships with parser and CLI
boundary tests in the same slice, and because the launcher guidance had to be
tightened to avoid clobbering the read-only connector's metadata route.
