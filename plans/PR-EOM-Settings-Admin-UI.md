# PR-EOM-Settings-Admin-UI

## Why this slice exists

Co-deploy partner for **#2342** (merged, squash `ca6c32b11`), which put the public
`/api/v1/settings/*` router behind a fail-closed gate accepting **either** a bearer token
**or** an HttpOnly session cookie. #2342 deliberately split the browser half out at the
Python/`.tsx` line: today the 7 atlas-ui Settings tabs fetch the settings API with **no
auth**, so the moment #2342 is *deployed* those tabs get 401/503 and the admin Settings UI
breaks. This slice (**#2343**) builds the browser login flow — the operator authenticates
once (paste the admin token → `POST /api/v1/settings/session` → HttpOnly cookie → the
existing same-origin fetches carry it automatically) — and closes the deferred Codex #3
endless-loader item from #2342.

atlas-ui lives in the ATLAS monorepo (`atlas_brain/main.py` serves `atlas-ui/dist`), so this
is a full ATLAS PR through the same gates as #2342.

### Problem-derived contract

- Root cause: #2342 closed the backend hole but the browser cannot present the credential.
  The built UI is served from the SAME public funnel origin as the API, so a build-time
  `VITE_*` secret would be inlined into publicly-fetchable JS — the browser cannot hold the
  raw token. The correct browser credential is an **HttpOnly, JS-unreadable session cookie**
  obtained by proving possession of the token once. Without a login flow, deploying #2342
  makes every Settings tab dead (401/503). A secondary, pre-existing defect: the three
  "Style-A" Settings forms (`IntegrationSettings`, `LLMSettings`, `NotificationSettings`)
  gate render on `!form` alone, so a load error leaves `form` null forever and they spin an
  infinite loader with the error hidden in an unrendered footer.
- Correct fix must touch/change: add a token-exchange client (`settingsApi.ts`:
  login/logout/probe, mapping 200/401/503) that stores **no** token and needs **no**
  `credentials:` option (the cookie is same-origin + HttpOnly, carried automatically); add a
  token-entry `SettingsLogin.tsx` that discards the token after exchange; gate `SettingsModal`
  on an on-mount probe (`checking` → `need-login`/`unavailable`/`error` → `authed`) with a
  logout control; fix the three Style-A loaders to surface the load error instead of spinning.
  Add a dev-only `cookie_insecure` config so the Secure cookie can be dropped over
  `http://localhost` to make the flow testable in dev (default OFF ⇒ Secure ON). Stand up a
  vitest + testing-library harness (none exists) and wire `npm run test` into the atlas-ui CI
  leg.
- Must not change: the #2342 backend auth semantics (bearer/cookie/HMAC/digest-binding/
  fail-closed 503) — this slice only adds the `cookie_insecure` toggle on the cookie's
  `Secure` attribute; the prod default (Secure ON); no token is ever stored client-side; the
  four "Style-B" Settings forms (Voice/Email/Daily/News) already surface load errors and are
  left alone; the pre-existing `atlas-ui checks` `npm audit` reds (transitive
  brace-expansion/nanoid/postcss) are not in scope; `atlas_brain/main.py`'s dist-serving mount
  and the funnel config are untouched.

## Scope (this PR)

Ownership lane: eom-security/settings-admin-ui
Slice phase: Vertical slice
Max files: 19

1. Add the browser session-cookie login flow (settingsApi + SettingsLogin + SettingsModal
   auth gate + logout) so the admin Settings UI authenticates against the #2342 gate.
2. Fix the Style-A infinite-loader-on-load-error bug in the three affected Settings forms.
3. Add the env-gated `cookie_insecure` dev toggle (default OFF) so the flow is dev-testable.
4. Add a vitest + testing-library harness and tests, wired into the atlas-ui CI leg.

### Review Contract

- Acceptance criteria:
  - The `cookie_insecure` opt-in drops `Secure` (local http dev) but keeps HttpOnly +
    SameSite=Strict, and the default keeps `Secure` — settled by
    `tests/test_settings_auth.py::test_cookie_insecure_flag_drops_secure_for_local_dev_only`
    (both sides probed). All other #2342 cookie/auth semantics unchanged — full matrix stays
    green at `tests/test_settings_auth.py` (49 passed).
  - `loginSettings` maps 200→`ok`, 401→`invalid`, 503→`unavailable`, other/throw→`error`, and
    POSTs `{token}` JSON to `/api/v1/settings/session` — settled by
    `src/components/Settings/settingsApi.test.ts` (loginSettings cases).
  - `probeSettingsAuth` maps 200→`authed`, 401→`need-login`, 503→`unavailable`, other/throw→
    `error`, GETting the probe URL; `logoutSettings` DELETEs and swallows errors — settled by
    `settingsApi.test.ts` (probe + logout cases).
  - `SettingsLogin` calls `onAuthed` only on success, shows "Invalid admin token." on 401 and
    the not-configured message on 503, and keeps the submit button disabled until a non-blank
    token is entered — settled by `src/components/Settings/SettingsLogin.test.tsx`.
  - `SettingsModal` renders the tabs only when the probe reports `authed`, shows the login form
    on `need-login`, the not-configured banner on `unavailable`, the unreachable banner when
    the probe throws, logs in from the form to reveal the tabs, and returns to the login form
    on logout — settled by `src/components/Settings/SettingsModal.test.tsx`.
  - The three Style-A forms render the load-error banner (not the infinite spinner) when a load
    fails — settled by code inspection (`IntegrationSettings.tsx`/`LLMSettings.tsx`/
    `NotificationSettings.tsx`, the `if (!form) { if (status && !status.ok) … }` branch) and
    the passing `npm run build` typecheck; the four Style-B forms already surface load errors
    (`EmailSettings.tsx:266-271`, `VoiceSettings.tsx:294-298`) and are unchanged.
  - The harness gates CI: `npm run test` runs in the atlas-ui leg of
    `.github/workflows/npm_package_checks.yml`; `npm run build` (tsc -b + vite build) stays
    green with test files excluded from the app tsconfig.
- Reachability proof: real entrypoint is the atlas-ui Settings modal served from
  `atlas-brain.tailc7bd29.ts.net` (same funnel as #2342). Observable effect: after #2342+#2343
  deploy, opening Settings shows a login form; pasting the admin token exchanges it for the
  HttpOnly cookie and the tabs load; without a session, tabs stay gated. Local dev proof:
  `ATLAS_SETTINGS_ADMIN_COOKIE_INSECURE=1` + `npm run dev` exercises the full flow over http.
- Affected surfaces: `atlas-ui/src/components/Settings/{settingsApi.ts, SettingsLogin.tsx,
  SettingsModal.tsx, IntegrationSettings.tsx, LLMSettings.tsx, NotificationSettings.tsx}`,
  the vitest harness (`atlas-ui/vitest.config.ts`, `atlas-ui/src/test/setup.ts`,
  `atlas-ui/package.json`, `atlas-ui/tsconfig.app.json`), the CI leg
  (`.github/workflows/npm_package_checks.yml`), and the
  backend cookie toggle (`atlas_brain/config.py`, `atlas_brain/api/settings_session.py`).
- Risk areas: (1) fail-open cookie transport — the dev toggle could ship enabled and drop
  Secure in prod (guarded — default False, prod leaves it unset, test proves both sides);
  (2) token leakage — the browser must never persist the token (guarded — `SettingsLogin`
  discards it after exchange, no storage, cookie is HttpOnly); (3) the app build regressing on
  the new test tooling (guarded — test files excluded from `tsc -b`; `npm run build` verified
  green); (4) mid-session cookie expiry not re-gating the modal (accepted — see Deferred).
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R3 (security/auth —
  cookie transport + no-token-storage), R5 (backward compatibility — Settings tabs now require
  a session; the intended break co-deployed with #2342), R9 (frontend behavior — the login UI
  and Settings forms handle loading/need-login/unavailable/error/authed states, with vitest
  coverage of each), R11 (dependencies — new devDep test harness), R12 (deployment safety —
  co-deploy + env provisioning), R13 (defect class — the Style-A infinite-loader class).

### Boundary-change enumeration

- Boundary path/seam: the session cookie's `Secure` attribute in
  `create_settings_session` (`atlas_brain/api/settings_session.py`) is now gated by
  `SettingsAdminConfig.cookie_insecure`. Membership: `Secure` is set unless the deploy
  explicitly opts into the LOCAL-DEV insecure flag. Outside-the-set default: **Secure ON**
  (the safe side) — an absent/unset/false flag keeps the hardened cookie.
- Replaced-path behaviors: previously `secure=True` unconditionally; now
  `secure=not config.cookie_insecure`. HttpOnly, SameSite=Strict, Path, Max-Age, and the HMAC
  minting are unchanged. No other cookie attribute is conditional.
- Guard-relevant fields: `SettingsAdminConfig.cookie_insecure` (bool, default False; env
  `ATLAS_SETTINGS_ADMIN_COOKIE_INSECURE`). Falsy/absent/"false" ⇒ Secure ON.
- Caller x input shape: flag unset/False (prod, default) → cookie carries `Secure`; flag True
  (local dev only) → cookie omits `Secure`, retains HttpOnly + SameSite=Strict. Both sides
  asserted in `test_cookie_insecure_flag_drops_secure_for_local_dev_only`.

### Deployed-config probing

- Deployed/default config values: `cookie_insecure` defaults to `False` ⇒ **Secure ON** in
  prod. The prod deploy leaves `ATLAS_SETTINGS_ADMIN_COOKIE_INSECURE` unset; only local dev
  sets it to exercise the flow over `http://localhost` via the Vite proxy.
- Explicit value probe: flag `True` ⇒ Set-Cookie has no `Secure`
  (`test_cookie_insecure_flag_drops_secure_for_local_dev_only`).
- Absent value probe: default (flag off) ⇒ Set-Cookie carries `Secure` (same test, second
  side).
- Default-session/default-context probe: N/A — no session/tenant context on this toggle.
- Side-effect ordering: the cookie is only ever set by the existing #2342 login handler AFTER
  the token is validated; this change alters only the `Secure` attribute of an
  already-authorized cookie, never the auth decision or its ordering.

### Files touched

- `.github/workflows/npm_package_checks.yml`
- `atlas-ui/package-lock.json`
- `atlas-ui/package.json`
- `atlas-ui/src/components/Settings/IntegrationSettings.tsx`
- `atlas-ui/src/components/Settings/LLMSettings.tsx`
- `atlas-ui/src/components/Settings/NotificationSettings.tsx`
- `atlas-ui/src/components/Settings/SettingsLogin.test.tsx`
- `atlas-ui/src/components/Settings/SettingsLogin.tsx`
- `atlas-ui/src/components/Settings/SettingsModal.test.tsx`
- `atlas-ui/src/components/Settings/SettingsModal.tsx`
- `atlas-ui/src/components/Settings/settingsApi.test.ts`
- `atlas-ui/src/components/Settings/settingsApi.ts`
- `atlas-ui/src/test/setup.ts`
- `atlas-ui/tsconfig.app.json`
- `atlas-ui/vitest.config.ts`
- `atlas_brain/api/settings_session.py`
- `atlas_brain/config.py`
- `plans/PR-EOM-Settings-Admin-UI.md`
- `tests/test_settings_auth.py`

## Mechanism

`settingsApi.ts` is a thin, secret-free client: `loginSettings(token)` POSTs `{token}` to
`/api/v1/settings/session` and maps the #2342 status codes to a `LoginResult`;
`probeSettingsAuth()` GETs a gated route and maps to an `AuthState`; `logoutSettings()` DELETEs
the session (best-effort). Because the #2342 cookie is HttpOnly + same-origin, the browser
carries it automatically on every existing Settings fetch — no `credentials:` option and no
client-side token storage are needed. `SettingsModal` runs `probeSettingsAuth()` on mount and
renders a state machine: `checking` (spinner) → `need-login` (`<SettingsLogin>`) /
`unavailable` (not-configured banner) / `error` (unreachable banner) / `authed` (the existing
tab bar + forms, plus a header logout that DELETEs the session and returns to `need-login`).
`SettingsLogin` exchanges the pasted token via `loginSettings`, flips the modal to `authed` on
success, surfaces the mapped error otherwise, and discards the token. The three Style-A forms
gain a `status && !status.ok` branch inside `if (!form)` so a load failure shows a red banner
instead of an endless spinner. The backend `cookie_insecure` toggle changes only
`secure=True` → `secure=not config.cookie_insecure` so the login flow can be exercised over
http in local dev; prod leaves it unset (Secure ON). The vitest harness
(`vitest.config.ts` jsdom + `src/test/setup.ts` jest-dom) runs `*.test.{ts,tsx}` under `src`;
those files are excluded from `atlas-ui/tsconfig.app.json` so `tsc -b`/`vite build` are unaffected.

## Intentional

- Modal-level login gate rather than a per-form credentialed-fetch wrapper: it sidesteps the
  two different Settings-form loader styles and gives one login surface for all seven tabs.
- Probe-on-mount (not a global auth store): the modal is the only settings surface; a local
  `AuthState` is simpler than app-wide state and matches the single-operator model.
- `cookie_insecure` is a deploy-time env toggle (default False), mirroring the Atlas pattern
  of dev-only opt-ins; it only relaxes cookie transport for local http, never the auth check.
- vitest + testing-library (not Playwright/e2e): the logic under test is status-code mapping
  and render-state branching — unit/component scope is the right altitude and adds no browser
  infra to CI.

## Deferred

- Mid-session cookie expiry (12h TTL) does not re-trigger the modal login gate — the probe
  runs once on mount, so a tab fetch after expiry shows that tab's own error banner (now
  including the fixed Style-A path), and the operator reopens the modal to re-auth. Acceptable
  for a single-operator admin panel; a re-gate-on-401 interceptor is a follow-up if it bites.
- Broader #2335 hardening (authenticated ntfy access tokens) stays deferred there.
- The `atlas-ui checks` `npm audit --audit-level=high` leg stays red on pre-existing transitive
  advisories (brace-expansion via eslint→minimatch, nanoid, postcss); it is not a required
  merge gate and fixing deps is out of scope (tracked-elsewhere hygiene).

Parked hardening: none.

## Verification

- `python -m pytest tests/test_settings_auth.py -q` → 49 passed (adds the Secure-flag both-sides
  test; full #2342 matrix still green).
- `cd atlas-ui && npx vitest run` → 23 passed across settingsApi / SettingsLogin / SettingsModal.
- `cd atlas-ui && npm run build` (`tsc -b && vite build`) → green with test files excluded.
- Manual (local dev): `ATLAS_SETTINGS_ADMIN_COOKIE_INSECURE=1` + both admin env vars +
  `npm run dev` → open Settings → login → tabs load → PATCH a setting → logout → login
  reappears; bad token → "Invalid admin token."; unconfigured server → not-configured banner.
- **HARD co-deploy requirement:** ship #2343 with #2342. Provision
  `ATLAS_SETTINGS_ADMIN_TOKEN_SHA256` + `ATLAS_SETTINGS_ADMIN_SESSION_SECRET` (from the #2342
  generators), restart `atlas-api.service`, then verify the public GET is 401 and an authed
  browser session reaches the tabs.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/npm_package_checks.yml` | 1 |
| `atlas-ui/package-lock.json` | 1245 |
| `atlas-ui/package.json` | 8 |
| `atlas-ui/src/components/Settings/IntegrationSettings.tsx` | 11 |
| `atlas-ui/src/components/Settings/LLMSettings.tsx` | 11 |
| `atlas-ui/src/components/Settings/NotificationSettings.tsx` | 11 |
| `atlas-ui/src/components/Settings/SettingsLogin.test.tsx` | 72 |
| `atlas-ui/src/components/Settings/SettingsLogin.tsx` | 75 |
| `atlas-ui/src/components/Settings/SettingsModal.test.tsx` | 87 |
| `atlas-ui/src/components/Settings/SettingsModal.tsx` | 114 |
| `atlas-ui/src/components/Settings/settingsApi.test.ts` | 97 |
| `atlas-ui/src/components/Settings/settingsApi.ts` | 56 |
| `atlas-ui/src/test/setup.ts` | 10 |
| `atlas-ui/tsconfig.app.json` | 3 |
| `atlas-ui/vitest.config.ts` | 16 |
| `atlas_brain/api/settings_session.py` | 4 |
| `atlas_brain/config.py` | 2 |
| `plans/PR-EOM-Settings-Admin-UI.md` | 207 |
| `tests/test_settings_auth.py` | 23 |
| **Total** | **2053** |
