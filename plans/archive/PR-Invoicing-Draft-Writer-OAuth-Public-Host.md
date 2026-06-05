## Why this slice exists

The public draft-writer route and OAuth discovery now work, but the e2e smoke
fails when it opens an authenticated MCP session through Tailscale. Server logs
show:

```text
421 Misdirected Request
Invalid Host header: atlas-brain.tailc7bd29.ts.net
```

FastMCP auto-enables DNS-rebinding protection for localhost-backed servers. The
draft-writer process is bound to `127.0.0.1`, but public clients reach it with
the Tailscale hostname. The MCP transport path therefore needs an explicit
allowlist for the configured public issuer/resource host.

## Scope (this PR)

1. Add a draft-writer transport-security helper for OAuth HTTP mode.
2. Allow localhost plus the configured public issuer/resource hosts.
3. Keep DNS-rebinding protection enabled instead of disabling it broadly.
4. Add tests proving public hosts are allowlisted and unrelated hosts are not.
5. Update operator docs to identify a `421 Invalid Host header` as this exact
   guardrail.

### Files touched

- `atlas_brain/mcp/invoicing_draft_writer_server.py`
- `tests/test_invoicing_draft_writer_oauth.py`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `plans/PR-Invoicing-Draft-Writer-OAuth-Public-Host.md`

## Mechanism

The server derives allowed `Host` values from:

- `ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL`
- `ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL`
- localhost defaults used by local smoke tests

Then it assigns `mcp.settings.transport_security` before creating the
Streamable HTTP app:

```python
TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[...],
)
```

## Intentional

- This does not disable DNS-rebinding protection. The public hostname is
  allowed explicitly.
- This is scoped to the draft-writer server because this is the connector we
  are actively rolling out. Read-only can get the same tightening in a separate
  compatibility slice if needed.

## Deferred

- Dedicated hostname support remains deferred. The allowlist is derived from
  the configured public URLs, so it will work for either path-prefixed or
  dedicated-host deployments.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile atlas_brain/mcp/invoicing_draft_writer_server.py tests/test_invoicing_draft_writer_oauth.py
.venv/bin/pytest tests/test_invoicing_draft_writer_oauth.py tests/test_check_invoicing_draft_writer_oauth_e2e.py -q
bash scripts/local_pr_review.sh --allow-dirty
git diff --check
```

Manual rollout verification after merge:

```bash
.venv/bin/python scripts/check_invoicing_draft_writer_funnel_routes.py
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_discovery.py --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_e2e.py --approval-token-file .secrets/invoicing-draft-writer-approval-token --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Server | ~45 |
| Tests | ~55 |
| Docs/plan | ~105 |
| **Total** | **~205** |
