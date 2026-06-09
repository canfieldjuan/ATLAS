# PR-Content-Ops-MCP-Claude-Hosted-PKCE-Smoke

## Why this slice exists
PR #1422 merged with a non-blocking review NIT: the Claude-hosted OAuth checker
uses a hardcoded placeholder challenge while declaring `code_challenge_method=S256`.
Strict OAuth providers can reject that before the checker reaches the root-route
behavior it is supposed to validate.

## Scope (this PR)
Ownership lane: content-ops/review-contract
Slice phase: Production hardening

Replace the checker placeholder with a valid generated S256 PKCE challenge and
prove the generated authorize URL carries the expected challenge shape.

### Files touched
- `plans/PR-Content-Ops-MCP-Claude-Hosted-PKCE-Smoke.md`
- `scripts/check_content_ops_marketer_verify_claude_hosted_oauth.py`
- `tests/test_check_content_ops_marketer_verify_claude_hosted_oauth.py`

### Review Contract
- Acceptance criteria:
  - [ ] Checker generates a real verifier-derived S256 PKCE challenge.
  - [ ] Authorize URL still targets root `/authorize` with the same resource and scope fields.
  - [ ] Tests reject the previous placeholder shape by checking decoded query values.
  - [ ] Existing redirect failure-branch coverage remains intact.
- Affected surfaces: Claude-hosted OAuth compatibility checker only.
- Risk areas: OAuth correctness, checker precision, maintainability.
- Reviewer rules triggered: R2, R10, R13

## Mechanism
Add a small local PKCE helper using SHA-256 and base64url without padding, then
call it from the checker before building the authorization URL. The helper stays
inside the checker script because this is a smoke-tool concern, not shared MCP
server behavior.

## Intentional
No live OAuth token exchange is added. This checker still validates the
root-authorize redirect only; the full e2e checker remains the surface for
registration, approval, token exchange, and tool listing.

## Deferred
Comprehensive probing of all five Claude root OAuth aliases remains deferred to
a later root-route health-check slice if live operations show a need. Parked
hardening: none.

## Verification
- Passed: focused checker pytest, 5 passed.
- Passed: py_compile for the checker.
- Passed: git diff whitespace check.
- Passed: local PR review with body file.

## Estimated diff size
| Area | Estimated LOC |
| --- | ---: |
| Total | ~90 |

3 files, +88 / -2.
