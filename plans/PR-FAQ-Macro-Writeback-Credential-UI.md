# PR-FAQ-Macro-Writeback-Credential-UI

## Why this slice exists

PR-FAQ-Macro-Writeback-Credential-API added authenticated, tenant-scoped
Zendesk credential routes, but operators still have no dashboard control to
provision those credentials. Without a UI, FAQ macro writeback remains blocked
on direct API calls or manual database work even though the backend path exists.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add typed Content Ops frontend wrappers for listing, adding/rotating, and
   revoking Zendesk credentials through the host API.
2. Add a compact credential management card to the Content Ops new-run page.
3. Keep token handling write-only in the UI: users can enter a new token, but
   the saved list only shows display-safe fields returned by the API.
4. Add a focused Node test for the route wrappers and UI wiring.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Credential-UI.md` - plan for this slice.
- `atlas-intel-ui/package.json` - focused credential UI test script.
- `atlas-intel-ui/scripts/content-ops-zendesk-credentials-ui.test.mjs` - API wrapper and UI source contract tests.
- `atlas-intel-ui/src/api/contentOps.ts` - Zendesk credential DTOs and route wrappers.
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` - credential management card on the new-run page.

## Mechanism

The frontend API adapter keeps the backend contract in snake case and calls the
routes added by PR-FAQ-Macro-Writeback-Credential-API:

- `GET /content-ops/zendesk-credentials`
- `POST /content-ops/zendesk-credentials`
- `DELETE /content-ops/zendesk-credentials/{credential_id}`

`ContentOpsNewRun` loads the display-safe credential list with `useApiData`,
renders any active credentials, and provides fields for Zendesk email, API
token, subdomain, base URL, and optional label. Save calls the add/rotate route,
clears the plaintext token field after success, and refreshes the credential
list. Revoke calls the delete route for the selected credential and refreshes
the list. Backend authorization remains load-bearing: list is available to
authenticated Content Ops users, while write/revoke failures surface as API
errors in the card.

## Intentional

- This slice does not add macro publish controls. It only provisions the tenant
  credential needed before publish can be productized.
- This slice does not try to validate Zendesk credentials client-side. Backend
  storage validation and future publish/reconcile failures remain the source of
  truth.
- This slice does not show the API token after save. The UI only displays the
  token prefix returned by the API.
- This slice does not add client-side role gating. The backend already enforces
  owner/admin writes, and showing the server error keeps the frontend simple
  until a shared role-capability surface exists.

## Deferred

- `PR-FAQ-Macro-Writeback-Publish-UI`: review UI action that publishes selected
  FAQ entries to Zendesk macros using the saved credential.
- Future PR: client-side capability/role hints if the dashboard gets a shared
  tenant-permission surface.

Parked hardening: none

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-zendesk-credentials - 4 passed.
- npm --prefix atlas-intel-ui run build - passed.
- npm --prefix atlas-intel-ui run lint - passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-credential-ui.md - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| API wrapper types/functions | ~60 |
| New-run credential card | ~290 |
| Focused test script/package hook | ~175 |
| **Total** | **~615** |

This is over the 400 LOC soft cap because the first usable credential UI needs
typed wrappers, visible list/add/revoke controls, and a test that proves the
frontend is wired to the backend routes from the prior slice.
