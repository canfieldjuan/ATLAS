# PR-EOM-Settings-Admin-Auth

## Why this slice exists

Follow-up to #2332 (Codex review): `PATCH /api/v1/settings/notifications` — and every
other `/api/v1/settings/*` route — is **publicly reachable with no authentication**.
A Tailscale Funnel (`atlas-brain.tailc7bd29.ts.net`, the same host the marketing site's
JS posts leads to) proxies all of `/api` to `127.0.0.1:8012`. **Empirically confirmed:**
`GET https://atlas-brain.tailc7bd29.ts.net/api/v1/settings/notifications` → HTTP 200, no
auth. Tracked as **#2335**.

### Problem-derived contract

- **Root cause (not the symptom):** the reported symptom is "`ntfy_url` is
  attacker-mutable," but the root cause is that the **entire `/settings` router is an
  unauthenticated mutation surface exposed on the public funnel** (7 PATCH handlers
  persisting `ntfy_url`, `imap_host`, `ollama_url`, `asr_url`, `ha_url`, `mqtt_host`,
  `default_from` + DoS toggles to `.env.local`). Fixing one field is a symptom patch; the
  class is "the router has no auth."
- **Secondary root cause (browser cannot hold a static secret):** `atlas_brain/main.py`
  serves the built `atlas-ui/dist` as static files on the SAME public funnel origin, so any
  build-time secret (a Vite `VITE_*` value) is inlined into publicly-fetchable JS. A bearer
  the browser ships is therefore not a secret. The browser must instead **prove possession
  of the token once and carry an opaque, JS-unreadable credential** (an HttpOnly cookie).
- **Secondary root cause (placeholder digests):** a format-only digest check accepts a
  placeholder-derived digest (`sha256("change-me")`), letting a lazy deploy silently
  re-open the surface. The established `validate_generated_service_token_digest`
  (`atlas_brain/eom_api/auth.py`) already rejects placeholders; the fix must reuse it.
- **Correct fix must touch/change:** add a fail-closed digest-validated dependency applied
  **router-level** on `atlas_brain/api/settings.py`, accepting **either** a bearer token
  (CLI/API) **or** an HttpOnly signed **session cookie** (browser); add a login/logout
  endpoint (`atlas_brain/api/settings_session.py`) that exchanges the token for that cookie
  and is NOT itself behind the gate; wire both routers in `atlas_brain/api/__init__.py`; add
  the deploy-only digest config (`ATLAS_SETTINGS_ADMIN_TOKEN_SHA256`). Fail-closed: no/blank/
  malformed/placeholder digest ⇒ 503 (unavailable), never open.
- **Must not change:** no behavior of any settings handler's logic (only the auth gate); no
  other router; the funnel config; `atlas_brain/main.py`'s dist-serving mount; the
  receivables/invoicing auth module (reuse its validator, don't modify it); the lead-ntfy
  path (#2332). The browser **login form UI** ships as the immediate follow-up slice (split
  at the Python/`.tsx` line); the broader "authenticated ntfy tokens" work stays deferred in
  #2335.

## Scope (this PR)

Ownership lane: eom-security/settings-auth
Slice phase: Vertical slice
Max files: 7

1. Add `SettingsAdminConfig` (`ATLAS_SETTINGS_ADMIN_TOKEN_SHA256`, digest-only) + wire it
   into `Settings`.
2. Add `require_settings_admin` (fail-closed; bearer OR signed session cookie) validating the
   digest via `validate_generated_service_token_digest`, and apply it router-level to the
   `/settings` router.
3. Add `atlas_brain/api/settings_session.py`: `POST/DELETE /settings/session` login/logout
   (not behind the gate) that mints/clears the HttpOnly session cookie.
4. Add the auth-gate + session test matrix.

### Review Contract

- Acceptance criteria:
  - Unconfigured/blank/**placeholder**/malformed server digest ⇒ every `/settings` route AND
    the login endpoint return **503**, even with a valid-looking bearer (fail-closed) —
    `tests/test_settings_auth.py::test_unconfigured_or_placeholder_or_malformed_digest_returns_503`.
  - Configured but no/malformed credential ⇒ **401** —
    `::test_configured_missing_credential_returns_401` +
    `::test_configured_bad_authorization_returns_401`.
  - Correct bearer ⇒ reaches the handler (**200**) — `::test_correct_bearer_passes_the_gate`.
  - Login with the correct token ⇒ **200** + a **HttpOnly, Secure, SameSite=Strict** cookie
    scoped to `/api/v1/settings` — `::test_login_with_bearer_sets_hardened_cookie` +
    `::test_login_with_json_body_token_succeeds`.
  - A valid session cookie authenticates the gate — `::test_valid_session_cookie_passes_the_gate`;
    a tampered / expired / foreign-signed cookie ⇒ **401** —
    `::test_tampered_session_cookie_is_invalid_returns_401`,
    `::test_expired_session_cookie_is_invalid_returns_401`,
    `::test_session_cookie_with_invalid_signature_is_rejected`.
  - **Digest disclosure cannot forge a cookie** (signing key is independent, not
    digest-derived): a cookie signed with the token digest ⇒ **401** —
    `::test_cookie_forged_from_the_digest_alone_is_rejected`. Missing signing secret ⇒ login
    **503** but bearer still 200 — `::test_login_returns_503_when_session_signing_secret_missing`,
    `::test_bearer_still_works_without_session_signing_secret`.
  - A mutation is blocked without a credential — **401** —
    `::test_patch_mutation_blocked_when_credential_missing`.
  - The gate is **router-level**, covering ALL `/settings/*` routes (≥7) —
    `::test_every_settings_route_is_gated`.
  - Non-ascii bearer does not 500 — `::test_non_ascii_bearer_is_invalid_returns_401`.
  - Logout clears the cookie — `::test_logout_clears_the_cookie`.
  - A weak/non-generated token is rejected even if its digest is provisioned (strength, not
    just a denylist) — `::test_nongenerated_token_is_invalid_even_if_its_digest_is_configured`.
  - A correctly-shaped but low-entropy token (repeated-char payload) is rejected by the
    entropy floor — `::test_low_entropy_but_correctly_shaped_token_is_invalid`.
  - A malformed cookie expiry (oversized, non-ASCII digit like `²`, non-decimal) does not 500
    — `::test_verify_settings_session_rejects_malformed_expiry` +
    `::test_session_cookie_with_ascii_malformed_expiry_returns_401`.
  - The PRODUCTION `atlas_brain.api` aggregate wires the session route AND gates the settings
    routes — `::test_production_aggregate_router_serves_the_session_route`.
- Reachability proof: real entrypoint `PATCH/GET https://atlas-brain.tailc7bd29.ts.net/api/v1/settings/*`
  (public funnel → 127.0.0.1:8012). Observable effect: unauthenticated request → 401/503
  (was 200). Verified pre-fix with a live `GET → 200`.
- Affected surfaces: `atlas_brain/api/settings.py` (router dep), `atlas_brain/api/settings_auth.py`
  (guard + cookie crypto), `atlas_brain/api/settings_session.py` (login/logout),
  `atlas_brain/api/__init__.py` (wiring), `atlas_brain/config.py` (`SettingsAdminConfig`).
- Risk areas: (1) fail-open on misconfiguration (guarded — fail-closed 503 via the shared
  validator, which also rejects placeholders); (2) session-cookie forgery (guarded — HMAC
  signature keyed by an INDEPENDENT server secret, not the token digest, so digest disclosure
  cannot forge; `hmac.compare_digest`, absolute expiry);
  (3) CSRF on the cookie-authed mutations (guarded — `SameSite=Strict`; mutations stay
  PATCH+JSON; no CORS widening); (4) timing (`hmac.compare_digest`).
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R3 (security/auth —
  THE point), R5 (backward compatibility — the `/settings` routes now REQUIRE auth where they
  were open; unauthenticated callers get 401/503, which is the intended break; CLI/API callers
  keep working via the bearer path), R11 (dependencies/config — new token env), R12 (deployment
  safety — token provisioning + restart), R13 (defect class — the auth-gate closure below).

### Boundary-change enumeration

**Closure declaration (auth-gate inventory).** The set of routes that require the settings
admin credential is **CLOSED** and **code-owned (DERIVED** from a single router-level
`dependencies=[Depends(require_settings_admin)]` on the `/settings` router, not a
per-endpoint list): membership is *every* route registered on that router (currently 14
GET/PATCH across 7 sections), and any route added to it inherits the gate automatically. The
login/logout router (`/settings/session`) is deliberately OUTSIDE the gate — you must be able
to log in before you hold a session — but performs the identical fail-closed token check, so
it is not an unauthenticated surface. **Outside-the-set default: deny.** A request without a
valid bearer or session cookie, OR a server with no valid digest, is refused (401 / 503).
The safe/expensive side is chosen deliberately: a missing/placeholder token makes the admin
API *unavailable* (503) rather than open, because the failure mode being prevented is exactly
"config drift silently re-opens a public mutation surface."
- Guard-relevant fields: `SettingsAdminConfig.token_sha256` (deploy-only digest; empty /
  non-`[0-9a-f]{64}` / placeholder ⇒ 503), the `Authorization` bearer header, the
  `atlas_settings_admin_session` cookie.
- Caller × input shape: valid digest + (correct bearer OR valid unexpired cookie) → allow;
  valid digest + missing/wrong/non-bearer/non-ascii/tampered/expired/foreign-signed → 401;
  blank/malformed/placeholder digest → 503 (any credential).

### Deployed-config probing

- Deployed/default config values: `token_sha256` defaults to `""` ⇒ **fail-closed 503**.
  The deploy provisions `ATLAS_SETTINGS_ADMIN_TOKEN_SHA256` (SHA-256 of a raw token kept
  only on the caller / operator).
- Explicit value probe: configured digest + correct bearer → 200 (`::test_correct_bearer_passes_the_gate`).
- Absent/placeholder/malformed value probe: → 503
  (`::test_unconfigured_or_placeholder_or_malformed_digest_returns_503`).
- Side-effect ordering: auth is a router dependency, so it runs BEFORE any handler body —
  no settings mutation or `.env.local` write can occur before the auth decision.

### Files touched

- `atlas_brain/api/__init__.py`
- `atlas_brain/api/settings.py`
- `atlas_brain/api/settings_auth.py`
- `atlas_brain/api/settings_session.py`
- `atlas_brain/config.py`
- `plans/PR-EOM-Settings-Admin-Auth.md`
- `tests/test_settings_auth.py`

## Mechanism

`require_settings_admin` (`atlas_brain/api/settings_auth.py`) resolves the deploy-only digest
from `settings.settings_admin.token_sha256` via an overrideable dependency and validates it
with `validate_generated_service_token_digest` — empty/malformed/placeholder ⇒ 503
(fail-closed). It then allows the request if a valid `atlas_settings_admin_session` cookie is
present (HMAC signature keyed by an INDEPENDENT server secret `ATLAS_SETTINGS_ADMIN_SESSION_SECRET`
— NOT derived from the token digest, so a digest-only disclosure cannot forge a cookie —
absolute expiry, `hmac.compare_digest`)
OR a valid `Authorization: Bearer <token>` is presented; otherwise 401. It is attached with
`APIRouter(prefix="/settings", dependencies=[Depends(require_settings_admin)])`, gating every
route before any handler runs. `settings_session.py` exposes `POST /settings/session` (verify
the token, set the HttpOnly/Secure/SameSite=Strict cookie) and `DELETE /settings/session`
(clear it), NOT behind the gate.

**Token-strength enforcement (not just a denylist).** `token_matches` first runs the shared
`validate_generated_service_token` (prefix `eomset_v1_` + fixed-length url-safe random payload
+ a `minimum_unique_characters=16` entropy floor, matching the sibling funnel gate)
on the PRESENTED token, mirroring the receivables gate — so a weak/guessable/non-generated
token (including a correctly-shaped repeated-character payload) can never match even if its
digest were provisioned. `generate_settings_admin_service_token()`
produces the `(raw_token, digest)` pair for provisioning. `verify_settings_session` requires the
cookie's expiry to be ASCII decimal digits with a bounded length before integer conversion
(`str.isdigit()` alone accepts non-ASCII digits like `²` that `int()` rejects, and a
multi-thousand-digit value overflows CPython's int-conversion limit — both are rejected with
401, never a 500).

## Intentional

- Service-token auth on the same funnel as the sibling receivables API — not SaaS/JWT user
  auth: this box is single-operator, and a deploy-time service token is the established
  Atlas pattern for "public via funnel but must be authenticated."
- Two credential shapes: a **bearer** for CLI/API callers, and an HttpOnly **session cookie**
  for the browser admin UI (which is served from the same public origin, so it cannot embed
  the raw token). The cookie is signed (not a stored session) — stateless, single-operator.
- Router-level (not per-handler) so the whole class is closed and future routes inherit it.
- Fail-closed to 503 when unconfigured/placeholder (never open) via the shared validator.

## Deferred

- **Browser login-form UI (immediate follow-up slice):** the `.tsx` Settings components +
  a shared credentialed fetch helper + a token-entry "unlock" form + surfacing 401/503 instead
  of an endless loader (Codex review #3). Split from this PR at the Python/`.tsx` line so the
  backend gate lands cleanly; deployed together with this PR so the admin UI never breaks.
- **Broader hardening in #2335:** authenticated ntfy *access tokens* so the alert **topic**
  is no longer the sole secret; optionally move remaining destination fields out of the
  mutable sets (mirroring `leads_ntfy_url`).

## Parked hardening

- None.

## Verification

- `.venv/bin/python -m pytest tests/test_settings_auth.py -q` → 22 passed (auth + session matrix).
- `maturity_sweep.py atlas_brain/api --baseline ...` → ratchet gate passed (no new brittleness).
- Pre-fix reachability proof: `curl https://atlas-brain.tailc7bd29.ts.net/api/v1/settings/notifications`
  → HTTP 200 (unauthenticated). Post-deploy: same request → 401.
- Post-merge deploy: provision BOTH `ATLAS_SETTINGS_ADMIN_TOKEN_SHA256` (digest) and the
  INDEPENDENT `ATLAS_SETTINGS_ADMIN_SESSION_SECRET` (from `generate_settings_admin_session_secret()`,
  >=32 chars — required for the cookie/login path) in the runtime `.env`
  (the digest from `generate_settings_admin_service_token()`); restart `atlas-api.service`;
  re-verify the public GET now returns 401 and an authed request (bearer or a fresh session
  cookie) 200.
- **HARD co-deploy requirement (do NOT deploy this PR to prod alone):** once this router
  dependency is live, the settings routes return 503 (digest unconfigured) or 401 (no browser
  credential), so the admin Settings UI is unusable until the follow-up login-form slice
  (**#2343**) ships. Deploy #2342 and #2343 together in the same window and provision the token
  then. Merging #2342 alone is safe (no runtime effect until deployed); deploying it alone is
  not. This is the intentional slice boundary (backend / `.tsx` split), tracked in #2343.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/__init__.py` | 2 |
| `atlas_brain/api/settings.py` | 13 |
| `atlas_brain/api/settings_auth.py` | 224 |
| `atlas_brain/api/settings_session.py` | 77 |
| `atlas_brain/config.py` | 19 |
| `plans/PR-EOM-Settings-Admin-Auth.md` | 238 |
| `tests/test_settings_auth.py` | 280 |
| **Total** | **853** |
