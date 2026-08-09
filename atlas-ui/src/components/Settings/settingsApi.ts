// Shared client for the /api/v1/settings admin router (#2335).
//
// That router is now fail-closed: every route requires a deploy-time bearer
// service token, because it is reachable from the public Tailscale Funnel and
// mutates outbound destinations (ntfy/imap/ollama/asr/ha/mqtt) + persists to
// .env.local. This local admin UI supplies the raw token via the build/dev-time
// env var VITE_SETTINGS_ADMIN_TOKEN. The UI is served only on localhost dev
// servers (never on the funnel), so the token is not publicly exposed.
//
// If the var is unset the header is simply omitted and the API answers 401/503,
// making the "you must configure the token" state obvious rather than silent.

const SETTINGS_ADMIN_TOKEN = (
  import.meta.env as Record<string, string | undefined>
).VITE_SETTINGS_ADMIN_TOKEN;

/** Authorization header for the settings admin API, or {} when unconfigured. */
export function settingsAuthHeaders(): Record<string, string> {
  return SETTINGS_ADMIN_TOKEN
    ? { Authorization: `Bearer ${SETTINGS_ADMIN_TOKEN}` }
    : {};
}

/** fetch() that always carries the settings-admin bearer token (GET and PATCH). */
export function settingsFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers ?? {});
  if (SETTINGS_ADMIN_TOKEN && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${SETTINGS_ADMIN_TOKEN}`);
  }
  return fetch(input, { ...init, headers });
}
