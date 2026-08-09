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
  unauthenticated mutation surface exposed on the public funnel**. There are SEVEN
  unauthenticated PATCH handlers (voice/email/daily/intelligence/llm/notifications/
  integrations), all persisting to `.env.local`, and multiple write **destination/host**
  fields (`ntfy_url`, `imap_host`, `ollama_url`, `asr_url`, `ha_url`, `mqtt_host`,
  `default_from`) plus DoS toggles (`alerts_enabled=false`). Fixing one field is a symptom
  patch; the class is "the router has no auth."
- **Fix at the most-upstream correct point:** a **router-level** auth dependency on the
  `/settings` router closes all seven handlers, every field category, and the GET reads
  (which leak `imap_host` etc.) at once — not one field or one endpoint at a time.
- **Correct fix must touch/change:** add a fail-closed bearer service-token dependency
  (mirroring the receivables API, `atlas_brain/api/invoicing/auth.py`), apply it
  router-level in `atlas_brain/api/settings.py`, add the deploy-only digest config
  (`ATLAS_SETTINGS_ADMIN_TOKEN_SHA256`), and update the only legitimate caller — the local
  admin UI (`atlas-ui/src/components/Settings/*`) — to send the token. Fail-closed: no/blank/
  malformed digest ⇒ 503 (unavailable), never open.
- **Must not change:** no behavior of any settings handler's logic (only the auth gate);
  no other router; the funnel config stays (narrowing it would break the intended MCP/OAuth/
  Stripe surfaces — fix belongs in the app). The broader "topic is the sole secret /
  authenticated ntfy tokens" work stays deferred in #2335.

## Scope (this PR)

Ownership lane: eom-security/settings-auth
Slice phase: Vertical slice

1. Add `SettingsAdminConfig` (`ATLAS_SETTINGS_ADMIN_TOKEN_SHA256`, digest-only) + wire it
   into `Settings`.
2. Add `require_settings_admin` (fail-closed bearer dep) and apply it router-level to the
   `/settings` router.
3. Update the local admin UI (a shared `settingsFetch` + the 7 Settings components) to send
   the bearer token.
4. Add the auth-gate test matrix (503 unconfigured/malformed, 401 no/bad bearer, pass on the
   right token, every route gated).

### Review Contract

- Acceptance criteria:
  - Unconfigured/blank/malformed server digest ⇒ every `/settings` route returns **503**,
    even with a valid-looking bearer (fail-closed) — settled by
    `tests/test_settings_auth.py::test_unconfigured_digest_returns_503_even_with_a_bearer` +
    `::test_malformed_server_digest_returns_503`.
  - Configured but no/malformed bearer ⇒ **401** — `::test_configured_but_no_bearer_returns_401`
    + `::test_configured_bad_authorization_returns_401`.
  - Correct bearer ⇒ reaches the handler (**200** on GET) — `::test_correct_bearer_passes_the_gate`.
  - A mutation (`PATCH`, repoint a destination) is blocked without the bearer — **401**,
    no write — `::test_patch_mutation_is_blocked_without_a_bearer`.
  - The gate is **router-level**, covering ALL `/settings/*` routes (≥7), not just
    notifications — `::test_every_settings_route_is_gated`.
  - Non-ascii bearer does not 500 — `::test_non_ascii_bearer_does_not_crash_returns_401`.
- Reachability proof: real entrypoint `PATCH/GET https://atlas-brain.tailc7bd29.ts.net/api/v1/settings/*`
  (public funnel → 127.0.0.1:8012). Observable effect: unauthenticated request → 401/503
  (was 200). Verified pre-fix with a live `GET → 200`.
- Affected surfaces: `atlas_brain/api/settings.py` (router dep), `atlas_brain/api/settings_auth.py`
  (new dep), `atlas_brain/config.py` (`SettingsAdminConfig`), the 7 `atlas-ui` Settings
  components + `settingsApi.ts`.
- Risk areas: (1) breaking the legitimate admin UI (mitigated — the UI now sends the token;
  it is local-dev-only, never on the funnel); (2) fail-open on misconfiguration (guarded —
  fail-closed 503); (3) timing/constant-time compare (uses `hmac.compare_digest`).
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R3 (security/auth —
  THE point), R5 (backward compatibility — the UI caller updated in lockstep), R9
  (guard boundary probe — this IS an auth guard; both sides probed: valid-token→200,
  no/wrong/non-ascii→401, unconfigured/malformed digest→503, PATCH-without-token→no write),
  R11 (dependencies/config — new token env), R12 (deployment safety — token provisioning +
  restart), R13 (defect class — the auth-gate closure below).

### Boundary-change enumeration

**Closure declaration (auth-gate inventory).** The set of routes that require the settings
admin token is **CLOSED** and **code-owned (DERIVED** from a single router-level
`dependencies=[Depends(require_settings_admin)]` on the `/settings` router, not a
per-endpoint list): membership is *every* route registered on that router (currently 14
GET/PATCH across 7 sections), and any route added to the router in future inherits the gate
automatically. **Outside-the-set default: deny.** A request that does not present a valid
bearer, OR a server with no valid digest configured, is refused (401 / 503) — the router
never serves an unauthenticated caller. The safe/expensive side is chosen deliberately: a
missing token makes the admin API *unavailable* (503) rather than open, because the failure
mode being prevented is exactly "config drift silently re-opens a public mutation surface."
- Guard-relevant fields: `SettingsAdminConfig.token_sha256` (deploy-only digest; empty or
  non-`[0-9a-f]{64}` ⇒ 503), the `Authorization` bearer header.
- Caller × input shape: valid digest + correct bearer → allow; valid digest + missing/wrong/
  non-bearer/non-ascii → 401; blank/malformed digest → 503 (any bearer).

### Deployed-config probing

- Deployed/default config values: `token_sha256` defaults to `""` ⇒ **fail-closed 503**.
  The deploy provisions `ATLAS_SETTINGS_ADMIN_TOKEN_SHA256` (SHA-256 of a raw token kept
  only on the caller); the admin UI gets the raw token via `VITE_SETTINGS_ADMIN_TOKEN`.
- Explicit value probe: configured digest + correct bearer → 200 (`::test_correct_bearer_passes_the_gate`).
- Absent value probe: empty digest → 503 (`::test_unconfigured_digest_returns_503_even_with_a_bearer`).
- Malformed value probe: non-SHA-256 digest → 503 (`::test_malformed_server_digest_returns_503`).
- Side-effect ordering: auth is a router dependency, so it runs BEFORE any handler body —
  no settings mutation or `.env.local` write can occur before the auth decision.

### Files touched

- `atlas-ui/src/components/Settings/DailyIntelligenceSettings.tsx`
- `atlas-ui/src/components/Settings/EmailSettings.tsx`
- `atlas-ui/src/components/Settings/IntegrationSettings.tsx`
- `atlas-ui/src/components/Settings/LLMSettings.tsx`
- `atlas-ui/src/components/Settings/NewsIntelligenceSettings.tsx`
- `atlas-ui/src/components/Settings/NotificationSettings.tsx`
- `atlas-ui/src/components/Settings/VoiceSettings.tsx`
- `atlas-ui/src/components/Settings/settingsApi.ts`
- `atlas_brain/api/settings.py`
- `atlas_brain/api/settings_auth.py`
- `atlas_brain/config.py`
- `plans/PR-EOM-Settings-Admin-Auth.md`
- `tests/test_settings_auth.py`

## Mechanism

`require_settings_admin` (`atlas_brain/api/settings_auth.py`) reads the deploy-only digest
from `settings.settings_admin.token_sha256` via an overrideable dependency. If the digest is
absent or not a lowercase 64-hex SHA-256 it raises 503 (fail-closed — a misconfigured server
is unavailable, not open). Otherwise it requires an `Authorization: Bearer <token>` header,
SHA-256s the presented token, and `hmac.compare_digest`s it against the configured digest;
mismatch/missing/non-ascii ⇒ 401. It is attached with
`APIRouter(prefix="/settings", dependencies=[Depends(require_settings_admin)])`, so it gates
every route (reads and writes) before any handler runs. The admin UI's `settingsFetch`
wrapper injects the bearer (from `VITE_SETTINGS_ADMIN_TOKEN`) into every settings request.

## Intentional

- Service-token (bearer) auth, mirroring the sibling receivables API on the same funnel —
  not SaaS/JWT user auth: this box is single-operator, and a deploy-time service token is the
  established Atlas pattern for "public via funnel but must be authenticated."
- Router-level (not per-handler) so the whole class is closed and future routes inherit it.
- Fail-closed to 503 when unconfigured (never open) — the whole point is that config drift
  cannot re-open the surface.
- The raw token lives in the local UI's env only; the UI is served on localhost dev servers,
  never on the funnel, so the token is not publicly exposed.

## Deferred

- Broader hardening in **#2335**: authenticated ntfy *access tokens* so the alert **topic**
  is no longer the sole secret (removes it from URL/logs, fixes the sibling healthcheck/
  paid-deflection channels). Also optional defense-in-depth: move remaining destination
  fields out of the mutable sets (mirroring `leads_ntfy_url`). Not required now — the
  router-level auth already closes the exposure.

## Parked hardening

- None.

## Verification

- `/.venv/bin/python -m pytest tests/test_settings_auth.py -q` → 13 passed (auth matrix).
- `maturity_sweep.py atlas_brain/api --min-score 8` → ratchet gate passed.
- Pre-fix reachability proof: `curl https://atlas-brain.tailc7bd29.ts.net/api/v1/settings/notifications`
  → HTTP 200 (unauthenticated). Post-deploy: same request → 401.
- Post-merge deploy: provision `ATLAS_SETTINGS_ADMIN_TOKEN_SHA256` in the runtime `.env`
  (SHA-256 of a fresh raw token) + `VITE_SETTINGS_ADMIN_TOKEN` in the atlas-ui env; restart
  `atlas-api.service`; re-verify the public GET now returns 401 and an authed request 200.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-ui/src/components/Settings/DailyIntelligenceSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/EmailSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/IntegrationSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/LLMSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/NewsIntelligenceSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/NotificationSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/VoiceSettings.tsx` | 5 |
| `atlas-ui/src/components/Settings/settingsApi.ts` | 34 |
| `atlas_brain/api/settings.py` | 13 |
| `atlas_brain/api/settings_auth.py` | 56 |
| `atlas_brain/config.py` | 17 |
| `plans/PR-EOM-Settings-Admin-Auth.md` | 187 |
| `tests/test_settings_auth.py` | 110 |
| **Total** | **452** |
