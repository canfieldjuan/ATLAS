## Why this slice exists

The read-only invoicing MCP server is now connected to ChatGPT online through
OAuth, but getting there took multiple slices and several operational traps:
path-prefixed OAuth metadata, Tailscale route rewriting, shell `.env` loading,
and safe no-invoice smoke coverage. The next Atlas MCP servers should not take
a day each.

This slice documents the reusable rollout pattern for future MCP connectors.
It exceeds the 400 LOC soft cap because the runbook is intentionally
self-contained: it captures the implementation sequence, Tailscale route shape,
smoke-test contracts, launcher pattern, failure modes, and write-access
deferral in one operational artifact. Splitting it would make the next session
reconstruct context across multiple docs.

## Scope (this PR)

1. Add a general ChatGPT OAuth rollout runbook for Atlas MCP servers.
2. Capture the proven read-only invoicing implementation as the reference.
3. Document the fast path, required smokes, Tailscale route shape, and common
   failure modes.
4. Link the runbook from the MCP section in `CLAUDE.md`.

### Files touched

- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `CLAUDE.md`
- `plans/PR-MCP-ChatGPT-OAuth-Rollout-Runbook.md`

## Mechanism

The runbook explains the sequence future sessions should follow:

```text
safe tool surface -> OAuth mode -> discovery smoke -> e2e smoke -> launcher -> ChatGPT
```

It names the reusable implementation files from the invoicing rollout and
captures the exact Tailscale protected-resource metadata route that was needed
for path-prefixed MCP URLs.

## Intentional

- This is docs-only. It does not add OAuth to another server yet.
- The runbook explicitly says write access remains deferred until read-only
  connectors are proven.
- No real approval token, bearer token, client secret, authorization code, or
  access token is documented.

## Deferred

- A shared OAuth helper library is deferred until a second server proves the
  duplication is real.
- Write-capable MCP connector design is deferred until read-only connector
  behavior is stable.

## Verification

Commands run:

```bash
git diff --check
rg -n "ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN=.*[A-Za-z0-9_-]{20,}|approval-token-with-enough-entropy|bearer-token-value" docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md CLAUDE.md
```

The `rg` command returned no matches.

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Runbook | ~331 |
| CLAUDE pointer | ~6 |
| Plan | ~70 |
| **Total** | **~407** |
