# PR-Content-Ops-MCP-Claude-Public-Client-Metadata

## Why this slice exists
Live Claude.ai setup still required manual OAuth Client ID settings because the
authorization-server metadata advertised only confidential-client token auth
methods. The Content Ops OAuth provider can already register and exchange public
clients using `token_endpoint_auth_method=none`, so the metadata should stop
hiding that supported path.

## Scope (this PR)
Ownership lane: content-ops/review-contract
Slice phase: Production hardening

Advertise public-client OAuth metadata for the Content Ops marketer verifier and
ChatGPT adapter without changing token exchange, registration, approval, or MCP
tool behavior.

### Files touched
- `plans/PR-Content-Ops-MCP-Claude-Public-Client-Metadata.md`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract
- Acceptance criteria:
  - [ ] Content Ops OAuth metadata includes `none` in token endpoint auth methods.
  - [ ] Confidential-client methods remain advertised for existing ChatGPT/invoicing-style clients.
  - [ ] Registration, token, approval, and protected-resource routes remain present.
  - [ ] Bearer-mode behavior is unchanged.
- Affected surfaces: Content Ops marketer OAuth metadata only.
- Risk areas: OAuth discovery, connector compatibility, backcompat.
- Reviewer rules triggered: R1, R2, R5, R10, R13

## Mechanism
After FastMCP builds the OAuth Starlette app, replace only the built-in
authorization-server metadata route for the Content Ops app with an equivalent
metadata response that also lists `none`. The underlying MCP OAuth provider and
token authenticator already support public clients, so this is an advertisement
fix rather than a new auth path.

## Intentional
This does not fork the upstream MCP OAuth handlers or alter dynamic client
registration defaults. Clients that omit `token_endpoint_auth_method` still get
the upstream confidential-client default; clients that request `none` can
discover that the server supports it.

## Deferred
Full automatic Claude DCR live verification remains a live-connector artifact
step after this metadata is deployed. Parked hardening: none.

## Verification
- Passed: focused Content Ops MCP metadata tests, 32 passed.
- Passed: py_compile for the server modules.
- Passed: git diff whitespace check.
- Passed: local PR review with body file.

## Estimated diff size
| Area | Estimated LOC |
| --- | ---: |
| Total | ~186 |

4 files, +184 / -2.
