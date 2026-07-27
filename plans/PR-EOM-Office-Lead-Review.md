# PR-EOM-Office-Lead-Review

## Why this slice exists

Juan's completed-estimate approval command is now merged in both systems, but
there is still no office-facing path from a real Atlas lead to that command. The
portal cannot show the lead queue or prefill the existing Customer/Site form,
and the tracker cannot read the private Atlas funnel service. This slice is the
smallest operator-visible path that makes the office-controlled decision usable:
review an active EOM lead, open the existing operational form with its identity
prefilled, and let the already-enforced Juan-only command create the linked
Customer and initial Site.

This plan is the canonical cross-repository contract for three independently
deployable PRs: this Atlas read API, an `eom-timetracker` authenticated proxy
and retry endpoint, and the EOM Website portal view. It follows the operator
decision recorded on #2188 that booking does not promote a lead and only Juan
may approve conversion. It deliberately replaces neither the existing
Customer/Site onboarding surface nor the merged customer-handoff transaction.

### Problem-derived contract

- Root cause: Atlas persists public intake as an EOM `lead/new` contact, but
  intentionally returns no CRM identifier to the browser
  (`atlas_brain/api/leads.py:278-300`, `atlas_brain/api/leads.py:379-382`). Its
  current private funnel router exposes only the tracker-to-Atlas finalization
  POST (`atlas_brain/eom_api/funnel.py:59-90`). The tracker consequently has a
  Juan-only `POST /api/admin/funnel/approve-estimate` command that requires a
  caller-supplied Atlas contact ID and completed Customer/Site payload
  (`eom-timetracker/backend/time_tracker_api.py:9519-9622`), while the portal
  has no Leads tab (`Effingham_Office_Maids_Website/portal.html:1055-1085`) and
  its existing form posts the generic Customer route
  (`Effingham_Office_Maids_Website/customer-onboarding.js:2215-2245`). The
  safe backend command therefore exists but an office employee cannot reach it
  from the product without manually constructing an internal API request.
- Correct fix must touch/change:
  1. Atlas must add one authenticated, read-only route below the existing
     `/eom-funnel` service boundary. It must derive its candidate set directly
     from `contacts`: active, `effingham_maids`, `lead`, `new`; project an
     explicit bounded identity/readiness field set; include keyset cursor
     pagination with continuation metadata so the limit is neither a hard
     truncation nor unstable while staff approve leads; require the existing
     tracker bearer and actor headers; and perform no lifecycle,
     interaction, appointment, or Customer/Site write. The provider already has
     the needed scoped lead filters
     (`atlas_brain/services/crm_provider.py:1064-1139`) and the existing
     bearer/actor guard is the trust boundary
     (`atlas_brain/eom_api/funnel_auth.py:91-127`).
  2. The tracker must add an admin-session-authenticated review proxy that
     keeps the Atlas bearer server-side, returns the derived lead projection
     plus a server-derived `canApprove` flag, and exposes a Juan-only retry for
     a durable pending handoff. The retry must reuse the stored local operation
     and idempotency key rather than accept a new Customer/Site payload. The
     existing handoff table is already the recovery owner and the current
     approval path persists it before calling Atlas
     (`eom-timetracker/backend/time_tracker_api.py:9349-9415`,
     `eom-timetracker/backend/time_tracker_api.py:9557-9609`).
  3. The portal must add an admin Leads review view and reuse the existing
     Customer/Site field renderer and validation rather than make a second
     estimate/customer form (`Effingham_Office_Maids_Website/customer-onboarding.js:790-904`).
     Selecting an eligible lead opens that form prefilled from the Atlas
     projection and submits only to the existing
     `/api/admin/funnel/approve-estimate` command. A pending-finalization card
     must call the tracker retry endpoint, so a network interruption does not
     invite staff to create a second Customer.
- Must not change: public website intake and its success envelope; inbound
  identity/receipt semantics; lead ownership or `contact_type`/`lead_stage`;
  the merged Atlas handoff transaction and tracker Customer/Site schema;
  generic CRM/MCP contact APIs; estimate booking, Google Calendar, imported
  schedules, field work, jobs, QR, payroll, receivables, first clean,
  card-on-file, customer emails, advertising attribution, and public
  self-service onboarding. The adjacent Atlas calendar-receipts PR #2195,
  scoped-Gmail PR #2200, and the unopen estimate-booking branch are excluded.
  That branch's `new -> estimate_booked` transition is incompatible with the
  current first-time handoff eligibility check
  (`atlas_brain/services/crm_provider.py:1728-1744`) and is not part of this
  slice.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Permit the authenticated EOM tracker service to read a bounded, current
   office review queue of Atlas-owned `lead/new` contacts without exposing the
   CRM to the browser or changing any lead state.
2. Add the Atlas half only: the authenticated, bounded EOM new-lead read route,
   provider projection, and HTTP/real-PostgreSQL proof. The tracker and portal
   PRs consume the same existing handoff API; their code changes remain in
   their own repositories.
3. Archive the already-merged `PR-EOM-Office-Conversion-Handoff` plan as the
   required Atlas merged-PR housekeeping. It is a documented teardown action,
   not a product behavior change.
4. Shrink the unit-gate known-failures baseline only for stale entries proven
   by current-head CI to pass. This is ratchet housekeeping required by the
   gate; it does not change product behavior or add new baseline entries.

Max files: 8

### Review Contract

1. With the enabled generated funnel credential plus both actor headers,
   `GET /api/v1/eom-funnel/leads` returns only the explicit public projection
   for active EOM `lead/new` contacts and exposes continuation metadata
   (`limit`, `cursor`, `hasMore`, `nextCursor`) for bounded keyset pagination.
   HTTP route tests and a real-PostgreSQL provider test settle both the
   boundary and the query outcome.
2. The projection is closed and explicit: `contactId`, `fullName`, `email`,
   `phone`, `address`, `source`, and `createdAt`. A test proves unrelated
   contact columns and interaction/attribution metadata are absent from the
   response.
3. EOM customers, inactive/archived contacts, non-EOM contacts, and leads at a
   different stage are absent from the query result; a route request with an
   absent/malformed actor, invalid/disabled bearer, out-of-range limit, or
   malformed cursor fails before the provider list call. Focused route tests
   settle those outcomes.
4. The new read route performs no writes: the real-PostgreSQL test observes
   unchanged contacts, lifecycle-event count, and handoff count after a
   successful list request.
5. The previously merged plan is moved only to `plans/archive/` and the plans
   index is regenerated; the product diff contains no unrelated calendar,
   receivables, or generic CRM change.
6. The unit-gate baseline only shrinks: no baseline growth is allowed, and only
   the nine CI-reported stale node IDs are removed.
- Reachability proof: `tests/test_eom_lead_conversion.py` calls the real FastAPI
  route with its real dependencies overridden only at the database provider;
  `tests/test_eom_lead_conversion_integration.py` runs the provider projection
  against disposable PostgreSQL and observes the returned rows, paged keyset
  cursor result, and unchanged table counts.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/services/crm_provider.py`, the existing funnel-auth dependency,
  its focused route/integration tests, and Atlas plan archival.
- Risk areas: service-token/actor admission; EOM tenant and lifecycle filtering;
  customer/lead PII projection; accidental lifecycle write; caller compatibility;
  full-app route enrollment; and deployment ordering with the tracker proxy.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Boundary-change enumeration

The existing private service route gains a read admission path.

- Boundary path/seam: `GET /api/v1/eom-funnel/leads` →
  `require_eom_funnel_api` → `require_eom_funnel_actor` → explicit
  `DatabaseCRMProvider` projection.
- Replaced-path behaviors: before this slice there is no list route; browser
  access remains impossible because the portal calls only the tracker proxy,
  which holds the bearer server-side. The existing POST handoff route and
  public intake route are preserved.
- Guard-relevant fields: bearer digest; `X-EOM-Actor`; `X-EOM-Actor-ID`; bounded
  optional limit; optional opaque cursor over `(created_at, id)`; and database
  `business_context_id`, `status`, `contact_type`, and `lead_stage`.
- Caller x input shape:
  - tracker proxy + valid bearer/positive actor ID → explicit candidate page
    with continuation metadata;
  - tracker proxy + disabled/invalid bearer → 503/401 before the provider;
  - tracker proxy + absent/malformed actor → 422 before the provider;
  - any row outside the derived candidate predicate → absent, not a fallback;
  - public browser + no bearer → 401 and no data.

Closure declaration: the candidate set is **CLOSED / DERIVED** from the four
database predicates above, not a maintained name/source list. The response
field set and pagination metadata are **CLOSED / ENUMERATED** in the serializer.
Any row or field not admitted by those definitions is excluded by default.

### Deployed-config probing

- Deployed/default config values: this route uses the existing full-Atlas
  `ATLAS_EOM_FUNNEL_API_ENABLED` and
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` configuration; no new setting or
  Render service is introduced. The concrete deployed values are
  could-not-determine from committed code and must remain server-only.
- Explicit value probe: a generated `eomf_v1_` token whose digest is configured
  permits the list request with valid actor headers.
- Absent value probe: disabled API returns 503 and blank/mismatched bearer
  returns 401 before the provider list method.
- Default-session/default-context probe: missing/nonpositive actor ID and blank
  actor return 422; rows with NULL/non-EOM context are excluded.
- Side-effect ordering: authentication and candidate filtering finish before
  serialization; the read service issues no state mutation, event append,
  appointment write, or external call.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/INDEX.md`
- `plans/PR-EOM-Office-Lead-Review.md`
- `plans/archive/PR-EOM-Office-Conversion-Handoff.md`
- `tests/unit_gate_baseline.txt`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

Atlas serializes no operational estimate facts. The private route validates the
same server-to-server credential and tracker actor evidence already required by
the handoff callback, then calls a provider method whose SQL selects only the
derived lead cohort and explicit projection. The tracker companion will proxy
that response through its existing admin session and retain the bearer on its
server. The portal companion will render the queue, then hand the selected
identity to a new mode of the existing Customer/Site controller. That mode calls
the existing Juan-only approval endpoint, which remains the only creator of the
Customer, Site, local handoff row, Atlas lifecycle transition, and retryable
finalization.

## Intentional

- This is a review queue, not the earlier multi-stage pipeline board. Booking,
  loss, first clean, and email stages are intentionally absent because current
  office policy only needs `lead/new` until Juan approves a completed estimate.
- The router does not expose the initial web-form message, UTM/click data,
  interaction metadata, notes, billing data, or operational rate/schedule
  fields. The completed estimate and existing Customer/Site form remain their
  owners.
- The Atlas route does not record a read event. The tracker retains its existing
  authenticated access log for office commands; adding a new audit subsystem is
  not required to make this path usable.
- The query orders newest lead first. A separate day-to-day triage/prioritizing
  workflow may later use `lead_owner`/`next_follow_up_at`, but it does not alter
  the approval eligibility cohort here.

## Deferred

- Tracker companion: admin-session review proxy, `canApprove` derivation, and
  stored-operation retry endpoint; it does not change Atlas credentials or the
  customer handoff transaction.
- Website companion: Leads tab plus a prefilled mode of the existing
  Customer/Site form, with client-side stale-response handling and portal tests.
- A declined/non-customer outcome with explicit reopen; estimate booking,
  reschedule/cancel, Calendar projection; first-clean, payments/card collection,
  customer email; attribution reporting; historical Customer linking; and
  multi-site initial approval stay separate commands.

Parking predicate: this slice parks product stages and workflow automation that
do not block an authenticated office worker from reviewing a current `lead/new`
record, creating exactly one Customer/Site through the existing approval
command, or retrying an already-reserved handoff. Any flaw that leaks the
service bearer, exposes an out-of-cohort contact, bypasses Juan-only approval,
or permits a second Customer/Site from a retry is inline-blocking.

Parked hardening: none.

## Verification

- Before implementation: exact route/provider, tracker proxy/retry, portal
  form/controller, and default-tab code paths were read from all three
  repositories; no code was written until this contract was complete.
- Atlas focused tests: pytest for the lead-review/funnel selection passed with
  `8 passed, 1 skipped, 46 deselected`.
- Atlas syntax/checks: py_compile for the changed Python modules and focused
  tests passed; `git diff --check` passed.
- Atlas plan gates: plan shape, files-touched, and diff-size audits passed for
  `plans/PR-EOM-Office-Lead-Review.md` against `origin/main`.
- Unit-gate ratchet: current-head CI reported nine stale baseline entries and
  no regressions; only those nine node IDs were removed. The local growth guard
  passed against the `origin/main` baseline.
- Tracker companion verification is recorded in its own PR: backend FastAPI
  tests against the PostgreSQL fixture cover authenticated proxy, non-approver,
  and retry cases.
- Website companion verification is recorded in its own PR: Node/JSDOM
  controller tests and `node --check` cover the Leads tab and funnel approval
  form integration.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 69 |
| `atlas_brain/services/crm_provider.py` | 37 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-Office-Lead-Review.md` | 285 |
| `plans/archive/PR-EOM-Office-Conversion-Handoff.md` | 0 |
| `tests/unit_gate_baseline.txt` | 9 |
| `tests/test_eom_lead_conversion.py` | 174 |
| `tests/test_eom_lead_conversion_integration.py` | 123 |
| **Total** | **700** |
