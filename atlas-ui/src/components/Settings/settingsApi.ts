// Client helpers for the settings-admin session-cookie auth (#2343 / #2335).
//
// The /api/v1/settings/* router is fail-closed behind a bearer OR an HttpOnly
// session cookie. The browser cannot hold the raw token (the built UI is served
// from the same public origin, so a build-time secret would leak into the JS
// bundle). Instead the operator POSTs the token ONCE to /settings/session, which
// returns an HttpOnly, Secure, SameSite=Strict cookie. Because that cookie is
// same-origin and HttpOnly, the existing same-origin fetches carry it
// automatically — no `credentials:` option and NO client-side token storage are
// needed here, and the token is never retained after login.

const SESSION_URL = '/api/v1/settings/session';
// Any gated GET works as an auth probe; notifications is the lightest.
const PROBE_URL = '/api/v1/settings/notifications';

export type LoginResult = 'ok' | 'invalid' | 'unavailable' | 'error';
export type AuthState = 'authed' | 'need-login' | 'unavailable' | 'error';

/** Exchange the admin token for a session cookie. The token is not stored. */
export async function loginSettings(token: string): Promise<LoginResult> {
  try {
    const r = await fetch(SESSION_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });
    if (r.ok) return 'ok';
    if (r.status === 401) return 'invalid';
    if (r.status === 503) return 'unavailable';
    return 'error';
  } catch {
    return 'error';
  }
}

/** Clear the session cookie (idempotent; needs no auth). */
export async function logoutSettings(): Promise<void> {
  try {
    await fetch(SESSION_URL, { method: 'DELETE' });
  } catch {
    /* logout is best-effort */
  }
}

/** Probe whether the current session (cookie) can reach the gated settings API. */
export async function probeSettingsAuth(): Promise<AuthState> {
  try {
    const r = await fetch(PROBE_URL);
    if (r.ok) return 'authed';
    if (r.status === 401) return 'need-login';
    if (r.status === 503) return 'unavailable';
    return 'error';
  } catch {
    return 'error';
  }
}
