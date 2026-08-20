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

### Review-remediation rationale

The review findings expose one cross-repository contract defect, not unrelated
polish: the office queue must represent links that remain publicly authorized,
must page beyond the first response, and must let the downstream relay prove
the exact registered controls it may invoke. Atlas, Tracker, and Website each
change only their part of that same contract. The larger seven-file Atlas
budget is indivisible because the provider, public response, route capability
projection, CI enrollment, and focused route/provider tests must agree before
the Tracker can safely consume it.

### Problem-derived contract

- Root cause: `eom_onboarding_email_drafts.status = 'sent'` describes email
  delivery, while `eom_public_onboarding_tokens.status` plus the currently
  accepted signing-key fingerprints own whether a link is currently usable.
  The existing draft list therefore cannot identify active links for a safe
  revoke control; the initial issued-token projection also admitted retired
  signing-key rows and exposed only a first page.
- Correct fix must touch/change:
  1. Add a service-and-actor authenticated Atlas read projection that joins the
     existing token, draft, and contact records and admits only `issued` rows
     signed by the current primary or configured previous public-onboarding
     key. It must fail closed when public onboarding authority is unavailable.
  2. Return exactly the office fields needed to identify and revoke that
     record: `draftId`, `contactId`, `fullName`, `recipientEmail`, `issuedAt`,
     and fixed `status: 'issued'`. Do not return the token ID, raw bearer,
     HMAC/signing material, approval key, or a link URL.
  3. Page the bounded queue with the existing opaque cursor grammar and stable
     newest-first keyset ordering, without returning a token identifier.
  4. Advertise the new read plus the existing private link-revoke and handoff
     recovery routes through the derived capability manifest *and* their
     registered method/path signatures, so downstream deployments derive rather
     than copy the decision set.
  5. Prove the route through its registered FastAPI entrypoint and the provider
     projection against disposable PostgreSQL state containing current-key,
     previous-key, retired-key, redeemed, and revoked tokens.
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
Max files: 7

1. Add the publicly-authorized, keyset-paged Atlas office projection and
   derived capability route signatures that the Tracker can relay.
2. Add focused provider and FastAPI tests proving authorization, closed
   response shape, current/previous-key authority filtering, cursor behavior,
   capability reachability, and CI enrollment.
3. Leave Tracker and Website rendering/actions to their coordinated follow-up
   PRs; this API remains additive and independently deployable.

### Review Contract

- Acceptance criteria:
  - [x] `GET /eom-funnel/public-onboarding/issued-links` requires the existing
    service credential and office actor before invoking the CRM provider, and
    returns a bounded, keyset-paged `links` list of only publicly-authorized
    issued-token projections; settled by
    `tests/test_eom_public_onboarding.py` ASGI route tests.
  - [x] The provider projection joins the existing token/draft/contact rows,
    selects only `status = 'issued'` under an accepted current/previous signing
    key, orders newest issued evidence first with an opaque next cursor, and
    does not select or serialize raw token material; settled by
    `tests/test_eom_lead_conversion_integration.py` disposable-Postgres test.
  - [x] Issued-link list/revoke/recovery capabilities and route signatures are
    advertised only when their exact registered routes exist; settled by
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

### Fix-loop disposition preflight

Before a later remote update, classify every current thread as `fixed-in`,
`waived-*`, or `not-applicable` with file/test evidence; rerun the guard after
that disposition is added. No generic `defer` disposition is permitted.

### Boundary-change enumeration

- Boundary path/seam: new private office read
  `GET /eom-funnel/public-onboarding/issued-links`; derived capability manifest
  gains this read and the pre-existing revoke/recovery routes, with canonical
  registered method/path signatures for each advertised capability.
- Replaced-path behaviors: none. Existing draft listing retains its email-draft
  semantics; existing public and private mutation paths are untouched.
- Guard-relevant fields: service bearer, actor headers, bounded `limit`, opaque
  cursor, durable `token.status`, and accepted signing-key fingerprints. No raw
  bearer is an admitted request or response field at this seam.
- Caller x input shape: authenticated Tracker service plus valid office actor
  gets only issued-link metadata; absent service/actor rejects before provider
  access; current/previous-key issued rows return issued only; retired-key,
  redeemed, and revoked rows do not enter the active queue.

### Capability-set closure declaration

- Closed decision set: `GET /eom-funnel/public-onboarding/issued-links`,
  `POST /eom-funnel/onboarding-drafts/{draft_id}/revoke-link`, and
  `POST /eom-funnel/public-onboarding/recover`.
- Canonical source: `served_capabilities()` filtered through
  `_CAPABILITY_ROUTES` in `atlas_brain/eom_api/funnel.py`; the response derives
  both capability names and method/path signatures from that exact registry.
- Deployment default: absent, malformed, or nonmatching route signatures are
  unavailable downstream. The Tracker must fail closed rather than treating a
  copied capability spelling as evidence that an action is deployed.

### Deployed-config probing

- Deployed/default config values: the list route requires valid public
  onboarding authority because it asserts that a link remains usable now.
  Existing private revoke/recovery routes remain usable when issuance is
  paused, because they repair already-issued records rather than mint one.
- Explicit value probe: valid service and actor credentials return a safe
  issued-link page.
- Absent value probe: absent service credential or actor header rejects before
  provider access.
- Default-session/default-context probe: the office route derives only HMAC
  fingerprints from the configured primary/previous secret; it returns neither
  the secret, a public browser token, nor a public URL.
- Side-effect ordering: the route validates service/actor/limit before the
  provider call; the provider performs a single read and makes no write.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Onboarding-Office-Followup.md`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_public_onboarding.py`

## Mechanism

`DatabaseCRMProvider.list_eom_public_onboarding_issued_links` reads the durable
`eom_public_onboarding_tokens` state and joins its draft/contact identity. It
projects no token identifier or bearer, returns only `issued` rows whose
signing-key fingerprint is current or configured as previous, and pages with
`issued_at DESC, draft_id DESC` through the existing opaque cursor grammar. The
router derives fingerprints from the configured public-onboarding authority,
validates the closed model, and exposes it behind the existing EOM service and
actor dependencies. The capability map derives three controls and their
method/path signatures from registered routes: issued-link list, issued-link
revoke, and public-handoff recovery.

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
- The limit remains bounded. A cursor is added only to make the active queue
  complete; it does not turn this slice into a token history browser.
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
  tests/test_eom_funnel_capability_manifest.py` — 49 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=<isolated local PostgreSQL container>
  python -m pytest -q tests/test_eom_lead_conversion_integration.py -k
  'public_onboarding_issued_link_projection'` — 2 passed, 90 deselected. The
  temporary disposable container was removed
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
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 1 |
| `atlas_brain/eom_api/funnel.py` | 156 |
| `atlas_brain/services/crm_provider.py` | 55 |
| `plans/PR-EOM-Onboarding-Office-Followup.md` | 247 |
| `tests/test_eom_funnel_capability_manifest.py` | 24 |
| `tests/test_eom_lead_conversion_integration.py` | 160 |
| `tests/test_eom_public_onboarding.py` | 182 |
| **Total** | **825** |
