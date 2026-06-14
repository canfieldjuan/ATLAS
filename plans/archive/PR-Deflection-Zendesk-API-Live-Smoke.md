# PR-Deflection-Zendesk-API-Live-Smoke

## Why this slice exists

#1527 added the tenant-scoped Zendesk full-thread export route, the portfolio
credential flow wires that export into private Blob and submit, and #1563 made
the full-thread JSON submit boundary match the bounded tempfile CSV path. The
remaining deferred validation gap is operator-run proof for the actual portfolio
Zendesk API mode. The existing live-smoke script proves direct private-Blob CSV,
local full-thread JSON fixture, and portfolio submit-route mode, but not the
credential-backed Zendesk API route handler that exports from ATLAS, stores the
artifact privately, then submits it.

This slice adds that smoke mode at the existing smoke-script seam. It does not
change the production route behavior; it gives the operator a one-command path
to exercise the already-built Zendesk API funnel with real deployed env, and
tests the request/auth/preflight projection without live network calls in CI.
The slice is slightly over the 400 LOC soft cap because the Codex review found
two related start-time validation gaps after review; the fix and regression
coverage belong in this PR because they protect the new Zendesk API smoke mode's
preflight gate.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Extend `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs` with a
   Zendesk API route-handler mode.
2. Validate the mode requires the route access token and rejects file/blob
   options that belong to the other smoke modes.
3. Project the smoke output as `portfolio_zendesk_api_route` without exposing
   ATLAS tokens, Blob tokens, Zendesk export access tokens, Blob URLs, or raw
   ticket artifacts.
4. Add enrolled shell-test coverage through the existing
   `test:deflection-upload-shell` script.

### Review Contract
- Acceptance criteria:
  - [ ] `--zendesk-api` / `ATLAS_DEFLECTION_ZENDESK_API_SMOKE=1` selects the
        credential-backed portfolio route, not direct Blob submit.
  - [ ] The smoke request sends `Authorization: Bearer <route access token>` to
        the portfolio route handler and a JSON body with company/contact,
        bounded limit, and `start_time`.
  - [ ] The mode preflights missing access token, local/non-HTTPS ATLAS URLs,
        invalid account id, invalid limit/start time, and incompatible
        file/blob/route-handler options.
  - [ ] Success output includes only status/request/result/account/base-host
        fields; failure output uses static error codes and does not include
        route token, service token, Blob token, Blob URL, or ticket artifact.
  - [ ] Existing CSV, full-thread fixture, and submit-route smoke modes keep
        their current behavior.
- Affected surfaces: portfolio deflection submit live-smoke script and its
  enrolled shell test.
- Risk areas: accidentally making live smoke hit local hosts, leaking secrets in
  output, confusing direct Blob mode with Zendesk API mode, and stale frontend
  CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R9, R10, R12, R14.

### Files touched

- `plans/PR-Deflection-Zendesk-API-Live-Smoke.md`
- `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The live-smoke script imports the existing `zendesk-export-submit` route
handler. A new boolean option (`--zendesk-api` or
`ATLAS_DEFLECTION_ZENDESK_API_SMOKE=1`) selects that handler instead of
`submitPrivateBlob` or the generic submit route. The script extends
`withRouteEnv` to include `ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN`, builds
the same public JSON payload the portfolio page sends, and calls the handler
with `Authorization: Bearer <access token>`.

The route response is projected through the same small output envelope pattern
as the existing submit-route smoke. It records whether the smoke passed, source
mode, HTTP status, request id, account id, result path, optional static error,
optional upstream ATLAS status, and deployed base host. It never echoes raw
route responses, ticket artifacts, Blob URLs, or configured secrets.

## Intentional

- No production route changes. The route/proxy already has unit coverage; this
  slice validates the operator smoke entry point.
- No new npm script. The existing enrolled `test:deflection-upload-shell` script
  covers the new mode, so there is no new frontend CI enrollment surface.
- No live Zendesk call in CI. Tests inject a fake handler and assert request
  shape, preflight behavior, and sanitized projection.
- This avoids #1564's docs/proof-framing files to prevent cross-session
  conflicts.

## Deferred

- Browser-automated hosted smoke using the operator's deployed portfolio URL and
  real Zendesk trial credentials.
- Optional export progress/polling UX if live Zendesk exports are too slow for a
  single request.

Parked hardening: none.

## Verification

- `cd portfolio-ui && npm run test:deflection-upload-shell` -- 40 passed.
- `cd portfolio-ui && npm run smoke:deflection-submit-live -- --preflight-only --zendesk-api --base-url https://atlas.example.com --token test-token --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 --blob-token blob-token --zendesk-export-access-token route-token` -- passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Zendesk-API-Live-Smoke.md --check` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Zendesk-API-Live-Smoke.md` | 115 |
| `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs` | 101 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 199 |
| **Total** | **415** |
