# PR-EOM-Onboarding-Office-Followup

## Why this slice exists

Juan approved the next EOM public-onboarding vertical slice: office staff need
to see active customer onboarding links, Juan needs to revoke one safely, and
Juan needs to recover a Tracker-local handoff that did not finish in Atlas.
The existing private revoke and recovery commands exist, but the canonical
office UI cannot determine which `sent` drafts still have an **issued** token.
`sent` is email evidence, not current token truth: a token may already be
redeemed or revoked.

This Atlas dependency supplies the one safe source projection required by the
Tracker relay and Website Leads UI. The companion Tracker and Website PRs are
ordered after this API contract is available.

### Problem-derived contract

- Root cause: `eom_onboarding_email_drafts.status = 'sent'` describes email
  delivery, while `eom_public_onboarding_tokens.status` owns whether a link is
  currently usable. The existing draft list therefore cannot identify active
  links for a safe revoke control; no issued-token list is exposed to the
  office boundary.
- Correct fix must touch/change:
  1. Add a service-and-actor authenticated Atlas read projection that joins the
     existing token, draft, and contact records and admits only
     `token.status = 'issued'` rows.
  2. Return exactly the office fields needed to identify and revoke that
     record: `draftId`, `contactId`, `fullName`, `recipientEmail`, `issuedAt`,
     and fixed `status: 'issued'`. Do not return the token ID, raw bearer,
     HMAC/signing material, approval key, or a link URL.
  3. Advertise the new read plus the existing private link-revoke and handoff
     recovery routes through the derived capability manifest, so downstream
     deployments can fail closed instead of rendering an action that 404s.
  4. Prove the route through its registered FastAPI entrypoint and the provider
     projection against disposable PostgreSQL state containing issued,
     redeemed, and revoked tokens.
- Must not change:
  1. Do not alter public session, tracker-context, finalization, token parsing,
     HMAC rotation, issuance, email transport, draft state, customer handoff,
     or revocation/recovery mutation semantics.
  2. Do not mint, resend, replace, expose, store, log, or return a raw bearer
     or customer-facing onboarding URL.
  3. Do not add a migration, configuration setting, table, dependency, or
     production-data change. Payroll, QR/GPS, Home Base, scheduling, billing,
     and lead-stage vocabulary remain outside this slice.

## Scope (this PR)

Ownership lane: eom-public-onboarding
Slice phase: Vertical slice

1. Add the issued-token-only Atlas office projection and derived capability
   manifest entries that the Tracker can relay.
2. Add focused provider and FastAPI tests proving its authorization, closed
   response shape, token-state filter, and capability reachability.
3. Leave Tracker and Website rendering/actions to their coordinated follow-up
   PRs; this API remains additive and independently deployable.

### Review Contract

- Acceptance criteria:
  - [x] `GET /eom-funnel/public-onboarding/issued-links` requires the existing
    service credential and office actor before invoking the CRM provider, and
    returns a bounded `links` list of only issued-token projections; settled by
    `tests/test_eom_public_onboarding.py` ASGI route tests.
  - [x] The provider projection joins the existing token/draft/contact rows,
    selects only `status = 'issued'`, orders newest issued evidence first, and
    does not select or serialize raw token material; settled by
    `tests/test_eom_lead_conversion_integration.py` disposable-Postgres test.
  - [x] Issued-link list/revoke/recovery capabilities are advertised only when
    their exact registered routes exist; settled by
    `tests/test_eom_funnel_capability_manifest.py`.
  - [x] Existing public browser and private recovery routes retain their
    response/error behavior; settled by focused public-onboarding route tests
    and the cold diff reconstruction.
- Reachability proof: ASGI tests call the registered issued-links route and
  observe the safe JSON response; the provider test persists token states and
  observes that only the issued record is returned.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/services/crm_provider.py`, the EOM capability manifest, and
  focused public-onboarding/provider tests.
- Risk areas: service/actor authorization, browser/private data separation,
  capability deployment skew, token lifecycle truth, and response compatibility.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: new private office read
  `GET /eom-funnel/public-onboarding/issued-links`; derived capability manifest
  gains this read and the pre-existing revoke/recovery routes.
- Replaced-path behaviors: none. Existing draft listing retains its email-draft
  semantics; existing public and private mutation paths are untouched.
- Guard-relevant fields: service bearer, actor headers, bounded `limit`, and
  durable `token.status`. No raw bearer is an admitted request or response
  field at this seam.
- Caller x input shape: authenticated Tracker service plus valid office actor
  gets only issued-link metadata; absent service/actor rejects before provider
  access; issued/redeemed/revoked durable rows return issued only.

### Deployed-config probing

- Deployed/default config values: this read and the existing private repair
  routes use the configured service/actor boundary and remain available when
  public issuance is disabled, so prior issued tokens stay operable.
- Explicit value probe: valid service and actor credentials return a safe
  issued-link page.
- Absent value probe: absent service credential or actor header rejects before
  provider access.
- Default-session/default-context probe: no public browser token, public URL,
  or HMAC configuration is read by this office projection.
- Side-effect ordering: the route validates service/actor/limit before the
  provider call; the provider performs a single read and makes no write.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Onboarding-Office-Followup.md`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_public_onboarding.py`

## Mechanism

`DatabaseCRMProvider.list_eom_public_onboarding_issued_links` reads the durable
`eom_public_onboarding_tokens` state and joins its draft/contact identity. It
projects no token identifier or bearer, returns only rows whose status is
`issued`, and orders by `issued_at DESC, token.id DESC`. The router validates
the closed model and exposes it behind the existing EOM service and actor
dependencies. The capability map derives three controls from their registered
routes: issued-link list, issued-link revoke, and public-handoff recovery.

The later Tracker relay validates the closed Atlas response before exposing it
to the Website. The later Website uses the `draftId` only for the existing
Juan-gated revoke command and uses Tracker's separate local-reservation queue
for recovery; this endpoint neither sends a replacement nor changes a token.

## Intentional

- The response omits token IDs as well as raw tokens. A revoke action already
  addresses the durable record by `draftId`; the browser does not need another
  correlator.
- The new list does not reuse `sent` drafts. Email-delivery state and usable
  token state are intentionally distinct.
- The limit is bounded and no cursor is added in this small, operational queue,
  matching the existing Tracker recovery list rather than broadening this slice
  into a generic history browser.
- The list/revoke/recovery routes remain usable after issuance is paused; they
  repair already-issued records and do not mint a link.

## Deferred

- Tracker PR: authenticated relay, deployment proofs, and capability-gated
  calls to the existing revoke/recovery commands.
- Website PR: bilingual Leads panels, all-admin read visibility, Juan-only
  actions, and confirmation dialogs.
- Replacement-link issuance, email resend, expiry/reminders, history browsing,
  additional recovery metadata, and dashboard metrics remain future work.

Parking predicate: park UI polish, link-history browsing, replacement delivery,
and non-safety observability additions. Keep incorrect token-state filtering,
raw bearer disclosure, missing office authorization, or a capability that
advertises an absent route in scope because each would make the operator action
unsafe or misleading.

Parked hardening: none.

## Verification

- `python -m compileall -q atlas_brain/eom_api/funnel.py
  atlas_brain/services/crm_provider.py tests/test_eom_public_onboarding.py
  tests/test_eom_funnel_capability_manifest.py
  tests/test_eom_lead_conversion_integration.py` — passed.
- `python -m pytest -q tests/test_eom_public_onboarding.py
  tests/test_eom_funnel_capability_manifest.py
  tests/test_eom_lead_conversion_integration.py -k 'public_onboarding or
  onboarding_draft_list_projection or capability'` — 48 passed, 13 skipped,
  78 deselected.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=<isolated local PostgreSQL container>
  python -m pytest -q tests/test_eom_lead_conversion_integration.py -k
  public_onboarding_issued_link_projection_excludes_terminal_tokens -rs` —
  1 passed, 90 deselected. The temporary disposable container was removed
  afterward.
- ruff check atlas_brain/eom_api/funnel.py
  atlas_brain/services/crm_provider.py tests/test_eom_public_onboarding.py
  tests/test_eom_funnel_capability_manifest.py
  tests/test_eom_lead_conversion_integration.py — passed.
- `git diff --check` — passed.
- Cold diff audit: route/provider/manifest/test changes trace only to the
  Problem-derived contract. No public lifecycle mutation, token/bearer field,
  migration, configuration, or unrelated product surface changed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 72 |
| `atlas_brain/services/crm_provider.py` | 40 |
| `plans/PR-EOM-Onboarding-Office-Followup.md` | 204 |
| `tests/test_eom_funnel_capability_manifest.py` | 3 |
| `tests/test_eom_lead_conversion_integration.py` | 59 |
| `tests/test_eom_public_onboarding.py` | 69 |
| **Total** | **447** |
