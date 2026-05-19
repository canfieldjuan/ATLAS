## Why this slice exists

The localhost draft-writer OAuth smoke passes when served at the app root, but
the public ChatGPT URL is path-prefixed:

```text
https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

Tailscale Funnel needs two routes for that shape: the primary
`/invoicing-draft-writer` route and the host-root protected-resource metadata
route. The launcher currently prints only the metadata route, leaving the
operator to infer the primary connector route.

## Scope (this PR)

1. Teach the draft-writer OAuth launcher to derive the public app path from
   `resource_url`.
2. Print the required primary Funnel route before the metadata route.
3. Update rollout docs with the two-route setup.
4. Add tests for path-prefixed and root-resource guidance.

### Files touched

- `scripts/start_invoicing_draft_writer_oauth_server.py`
- `tests/test_start_invoicing_draft_writer_oauth_server.py`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `plans/PR-Invoicing-Draft-Writer-Funnel-Route-Guidance.md`

## Mechanism

The launcher parses the configured resource URL path. If it ends with `/mcp`,
the launcher treats the parent path as the public app path. For the default
resource URL, that produces `/invoicing-draft-writer` and prints:

```bash
tailscale funnel --bg --yes \
  --set-path /invoicing-draft-writer \
  http://127.0.0.1:<PORT>
```

It still prints the protected-resource metadata route separately because the
MCP library advertises that metadata at host root.

## Intentional

- This does not change server routing. FastMCP still serves endpoints at app
  root; Tailscale owns the public prefix.
- Root-resource local smoke guidance remains valid. When the resource path is
  just `/mcp`, the launcher prints a root Funnel command instead of a path
  prefix.

## Deferred

- Dedicated hostname support remains deferred. Path-prefix routing is the
  current operator model because it matches the read-only connector setup.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile scripts/start_invoicing_draft_writer_oauth_server.py tests/test_start_invoicing_draft_writer_oauth_server.py
.venv/bin/pytest tests/test_start_invoicing_draft_writer_oauth_server.py -q
bash scripts/local_pr_review.sh --allow-dirty
git diff --check
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Launcher | ~35 |
| Tests | ~30 |
| Docs/plan | ~100 |
| **Total** | **~165** |
