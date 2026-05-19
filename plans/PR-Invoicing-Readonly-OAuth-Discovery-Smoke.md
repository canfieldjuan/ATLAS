## Why this slice exists

PR #600 added OAuth mode for the read-only invoicing MCP server, but the
remaining operational risk is discovery: ChatGPT has to reach OAuth metadata,
protected-resource metadata, and the operator approval page through the public
URL shape. The current hosted URL is path-prefixed under `/invoicing-readonly`,
so a local green provider test does not prove the public connector URL works.

This slice adds a small public-discovery smoke and removes the approval form's
dependency on ASGI `root_path` so proxied path prefixes do not get dropped on
form submit.

## Scope (this PR)

1. Make the OAuth approval form submit back to the current external URL by
   omitting the absolute `action` attribute.
2. Add a read-only smoke script that validates OAuth authorization-server
   metadata, protected-resource metadata, and the unauthenticated MCP 401
   challenge.
3. Add focused tests for URL construction, smoke failure behavior, and the
   prefix-safe approval form.
4. Document the smoke command alongside the ChatGPT OAuth setup notes.

### Files touched

- `atlas_brain/mcp/invoicing_readonly_oauth.py`
- `tests/test_invoicing_readonly_oauth.py`
- `scripts/check_invoicing_readonly_oauth_discovery.py`
- `tests/test_check_invoicing_readonly_oauth_discovery.py`
- `CLAUDE.md`
- `plans/PR-Invoicing-Readonly-OAuth-Discovery-Smoke.md`

## Mechanism

The approval page renders `<form method="post">` with a hidden `request_id`.
Browsers submit a form with no `action` to the current document URL, preserving
whatever public prefix the proxy exposed. This is safer than trying to rebuild
the public prefix from `request.scope["root_path"]`, which may be absent behind
Tailscale or another path-stripping proxy.

The new smoke script derives:

```text
<issuer>/.well-known/oauth-authorization-server
<resource-origin>/.well-known/oauth-protected-resource<resource-path>
```

It fetches both JSON documents, validates the expected issuer/resource/scopes,
then probes the MCP URL without credentials and requires a 401 with a
`resource_metadata` challenge pointing at the protected-resource metadata URL.

## Intentional

- The smoke does not complete a full OAuth code exchange. It is a public
  routing/discovery check that requires no approval token and reads no invoice
  data.
- The approval form keeps the hidden `request_id`; no state is moved into query
  parsing during POST.

## Deferred

- A full browser/ChatGPT OAuth connection test remains operational because it
  needs the live public host and manual operator approval token.
- Durable OAuth token/client persistence remains deferred from PR #600.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile atlas_brain/mcp/invoicing_readonly_oauth.py scripts/check_invoicing_readonly_oauth_discovery.py
.venv/bin/python -m pytest tests/test_invoicing_readonly_oauth.py tests/test_check_invoicing_readonly_oauth_discovery.py
bash scripts/local_pr_review.sh --allow-dirty
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Approval form adjustment | ~10 |
| Discovery smoke script | ~170 |
| Tests | ~160 |
| Docs/plan | ~110 |
| **Total** | **~450** |

This is slightly over the soft cap because the smoke script ships with fixture
tests instead of being an untested operational helper.
