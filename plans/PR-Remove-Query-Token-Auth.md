# PR-Remove-Query-Token-Auth

## Why this slice exists

Issue #1656 M4 verifies that `atlas_brain/auth/dependencies.py` accepts JWTs
from `?token=` via the shared `_extract_token()` helper. URL-carried
credentials can leak through access logs, browser history, analytics, and
referrers, so auth must stay in the Authorization header or the existing
`atlas_token` cookie.

Root cause: the shared JWT extraction helper treated the query string as an
equivalent credential source after checking the header and cookie. This PR
fixes the root for every `require_auth()` and `optional_auth()` caller by
removing the query-parameter source from the shared helper instead of patching
one route.

## Scope (this PR)

Ownership lane: security/hardening-1656
Slice phase: Production hardening

1. Remove `request.query_params.get("token")` from the JWT extraction path.
2. Add focused tests proving query-only auth is rejected while Bearer header
   and `atlas_token` cookie auth are still admitted.

### Review Contract

- Acceptance criteria:
  - [ ] A request with only `?token=<jwt>` is treated as unauthenticated by
        `require_auth()` and `optional_auth()`.
  - [ ] `Authorization: Bearer <jwt>` still extracts the JWT.
  - [ ] The `atlas_token` cookie still extracts the JWT.
  - [ ] API-key auth remains header-only and is not broadened.
- Affected surfaces: auth dependencies, SaaS JWT auth callers.
- Risk areas: security, backward compatibility for any client still using URL
  tokens.
- Reviewer rules triggered: R1, R2, R3, R5, R14; boundary-probe required
  before LGTM because this changes a security guard.

### Files touched

- `atlas_brain/auth/dependencies.py`
- `plans/PR-Remove-Query-Token-Auth.md`
- `tests/test_auth_dependencies.py`

## Mechanism

`_extract_token()` remains the single JWT source selector for `require_auth()`
and `optional_auth()`, but it now returns only a Bearer token from the
Authorization header or the `atlas_token` cookie. Query strings are ignored,
so query-only requests fail at the normal missing-auth branch.

## Intentional

- This is a deliberate breaking change for URL-token clients. That
  compatibility loss is the point of #1656 M4 because URL credentials are not a
  safe auth transport.
- No config flag or transition shim is added; keeping the query fallback alive
  behind a flag would preserve the leak-prone path.

## Deferred

- #1656 M1 RLS, M2 JSONB encryption, M3 alert delivery, M6 incident response
  docs, M8 disclosure policy/security.txt, M9 CVE SLA, and H3 provider-side
  credential rotation remain separate slices.

Parked hardening: none.

## Verification

- `tests/test_auth_dependencies.py` via `pytest` -- 30 passed.
- `scripts/local_pr_review.sh` with the PR body file -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/auth/dependencies.py` | 4 |
| `plans/PR-Remove-Query-Token-Auth.md` | 81 |
| `tests/test_auth_dependencies.py` | 95 |
| **Total** | **180** |
