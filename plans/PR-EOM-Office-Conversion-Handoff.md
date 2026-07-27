# PR-EOM-Office-Conversion-Handoff

## Why this slice exists

This is the next EOM funnel slice after merged inbound preservation. A completed
estimate normally gives Juan the customer name/contact/address plus the initial
site's per-visit price, service scope, and preferred cleaning schedule. Juan
approves that completed estimate in the office; it should create the operational
Customer and first Site there, not turn a field action or Google Calendar entry
into a customer.

### Problem-derived contract

- Root cause: the only current EOM lifecycle record is the Atlas contact, while
  the completed-estimate facts that make someone a real customer belong to the
  time tracker Customer/Site model. Generic CRM writes are correctly frozen
  after #2216, but there is no idempotent, office-authenticated handoff joining
  one approved EOM lead to one tracker Customer and initial Site. Splitting a
  blind CRM promotion from later operational creation would recreate the
  duplicate-system and partial-retry debt this funnel is meant to remove.
- Correct fix must touch/change: make the time tracker the command owner,
  because it authenticates the named office employee and owns Customer/Site,
  rates, and service schedule. Its Juan-only approval command must create or
  return exactly one Customer draft and one initial Site keyed by the Atlas
  contact and a stable approval key. It then calls an Atlas service-authenticated
  finalization endpoint. Atlas must verify the active EOM `lead/new` contact,
  write one durable handoff record containing only cross-system IDs and actor
  evidence, atomically transition that contact to `customer` only after the
  tracker write exists, and append the lifecycle event. Both sides must return
  the prior result for the same key and make the tracker-to-Atlas finalization
  retryable after a network failure.
- Must not change: public website intake; inbound identity/receipt semantics
  from #2216; field/Google Calendar imports; estimate appointment booking;
  generic non-EOM CRM; payroll, QR, jobs, first-clean, card collection,
  receivables, customer-facing website onboarding, and EOM Website PR #70.
  Atlas must not store the estimate price, rate type, frequency, schedule, or
  service scope as a second operational source of truth. A declined estimate,
  reschedule/cancel, additional sites, self-service onboarding, and later
  customer updates are separate commands.
- Deployment/data-authority extension: the public intake is served by the
  full Atlas application and writes its primary CRM database, while the slim
  EOM Render candidate is configured with a different managed database. There
  is no replication path between those stores, so enabling the callback there
  would make every real web lead appear absent at approval time. Separately,
  the first funnel auth boundary accepts any printable, non-placeholder token
  of a minimum length; that is not a provisioning invariant and permits
  human-chosen, guessable credentials.
- Correct extension: mount the authenticated handoff router on the same full
  Atlas API aggregate that mounts public lead intake, so both commands use the
  canonical `DatabaseCRMProvider` pool. Validate an enabled generated
  `eomf_v1_` bearer during that full application's startup and reject startup
  when that enabled route lacks an initialized primary pool or its contacts,
  lifecycle, and handoff relations; keep the separate EOM Render candidate's
  funnel switch and migrations disabled. Real PostgreSQL proof must run the
  actual intake workhorse and finalization on one provider/database; route
  proof must show both routes are mounted through the full aggregate. The token
  remains server-only in the full Atlas process and tracker configuration.
- Must still not change: the existing full Atlas migration behavior, the EOM
  Render receivables candidate, public intake semantics or browser-held
  credentials, tracker approval authority and Customer/Site behavior, non-EOM
  CRM behavior, or operational estimate facts.
- Replay/closure extension: completion is proven by the immutable handoff row
  plus its matching `customer_approved` lifecycle event, but the first replay
  check also required that the customer's mutable status remain `active`.
  A legitimate later inactive/archive update therefore broke an identical
  recovery retry. The plan also enumerated several membership-dependent sets
  without declaring their source or default, allowing future reviews to find
  one unbound member at a time.
- Correct extension: first-time finalization continues to require an active
  EOM `lead/new`, while a same-key replay validates only immutable completion
  evidence and the permanent EOM/customer transition, not later mutable status.
  Declare every decision-driving set and make malformed/unlisted wire and
  credential values reject at their existing choke points.
- Must still not change: generic status updates; existing inactive/archive
  semantics; first-time approval eligibility; or the completion evidence the
  replay already requires.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Add the Atlas half of the office-approved customer handoff: dedicated
   service authentication, a durable idempotency/link record, a constrained
   `lead/new -> customer` finalization transaction, and an append-only
   lifecycle event. The companion time-tracker PR owns the authenticated Juan
   approval endpoint and Customer/Site transaction; neither half is published
   as merge-ready until the other has the matching contract and proof.
2. Add real PostgreSQL and HTTP-boundary proof for duplicate and overlapping
   finalization, foreign/non-lead/non-new rejection before writes, immutable
   tracker links, tracker-first/Atlas-retry recovery, and the invariant that no
   Atlas lifecycle promotion exists without a matching tracker Customer/Site
   identifier.
3. Serve the private route through the full Atlas API aggregate that already
   owns public intake contacts. Its enabled configuration is validated at full
   Atlas startup and fails closed without the initialized primary contacts,
   lifecycle, and handoff schema; the separate EOM Render candidate remains
   unable to serve the handoff route or run its handoff migrations. The raw
   bearer never enters source control or a browser.
4. Make the Atlas handoff row a complete, exclusive finalization record: only
   the finalization transaction may insert it; a replay verifies the promoted
   EOM customer plus matching lifecycle event; and each tracker Customer and
   Site ID is unique to one Atlas contact.
5. Record and rerun the merged time-tracker recovery proof: after an Atlas
   failure, the office retry reuses its persisted Customer, Site, and approval
   key rather than creating another operational record.
6. Prove the data-authority boundary with a real PostgreSQL public-intake then
   finalization path on one injected provider, prove the full API aggregate
   mounts both routes, and enroll the full-app files in the EOM pipeline gate.
7. Preserve the idempotent approval result after a later customer deactivation
   or archive without weakening first-time active-lead validation, and bind the
   plan's decision-driving sets to explicit closure declarations.

### Review Contract

1. A tracker request from the configured Juan employee ID creates or returns
   one Customer draft and one initial Site for one Atlas contact; the completed
   estimate's price, rate type, service scope, and preferred schedule are
   stored only with that Customer/Site. The companion's database integration
   proof settles this local transaction.
2. The Atlas finalization route accepts only its dedicated enabled service token
   and a validated tracker-issued handoff payload. Missing/invalid token,
   oversized transport IDs, blank/overflowing actor evidence, foreign contact,
   inactive contact, non-lead, or a lead not in `new` fails before
   contact/lifecycle/handoff writes. HTTP tests settle the transport boundary;
   real PostgreSQL fixtures prove rejected contact, lifecycle, and handoff rows
   remain unchanged.
3. The Atlas handoff table has unique contact, approval-key, tracker-Customer,
   and tracker-Site ownership invariants. Its database triggers allow insert
   only from the marked finalization transaction and reject update, delete, and
   truncate mutations. Transaction-scoped, sorted advisory locks for the key,
   contact, Customer, and Site serialize all admitted callbacks before any
   handoff row is read; a same-key replay verifies the customer transition and
   matching lifecycle event before returning the original result, while a
   conflicting key/contact/external-ID pair is rejected. Real PostgreSQL tests
   settle mutation rejection, malformed replay rejection, and overlapping
   same-key/same-contact, same-key/different-contact, and different-key/
   same-tracker-ID callbacks.
4. Atlas moves `contact_type` from `lead` to `customer` and clears the lead
   stage only in the finalization transaction after storing the tracker link.
   Generic provider/MCP paths remain unable to make that lifecycle change;
   integration tests settle the transition and blocked bypasses.
5. If the tracker has committed its Customer/Site but the Atlas call fails, its
   persisted approval operation retries the same request/key. Atlas recovers
   exactly one finalization; no second Customer/Site or lifecycle event is
   produced. A later generic inactive/archive update does not invalidate that
   completed replay, while a first-time inactive lead still fails before writes.
   Cross-repository contract tests and real-PostgreSQL status-transition proof
   settle the recovery rule.
6. `atlas_brain.main:lifespan` validates any enabled funnel configuration and
   requires its initialized primary pool to expose contacts, lifecycle, and
   handoff relations before serving; its `/api/v1` aggregate mounts both the
   public intake and authenticated handoff routers. The separate EOM Render
   candidate leaves the funnel and migration switches disabled. A real
   PostgreSQL fixture runs the actual intake workhorse then finalization on one
   injected provider; route and configuration tests settle the
   shared-authority deployment shape. An enabled configuration accepts only a
   fresh `eomf_v1_` generated bearer with the required random payload;
   parametrized tests settle malformed, short, repeated, and otherwise
   non-generated credentials before request handling.
- Reachability proof: authenticated time-tracker admin request -> configured
  Juan-only guard -> durable tracker Customer/Site operation -> Atlas EOM
  full application's service-authenticated finalization -> Atlas
  handoff/lifecycle rows. Public website intake reaches that same full
  application first, so its contact is directly available to finalization. The
  aggregate route test, real PostgreSQL intake-to-handoff proof, and merged
  time-tracker recovery test settle the persisted request/key contract. A live
  tracker-to-Atlas smoke after setting the full application's secret is a
  post-provisioning deployment verification, not a claim made by this source
  slice.
- Affected surfaces: full Atlas API aggregation/startup validation, EOM funnel
  router and service auth, CRM lifecycle transition, handoff migration, EOM
  pipeline CI, and the companion time-tracker admin API/customer-onboarding
  flow.
- Risk areas: mistaken actor authority, credential exposure, contact tenant
  escape, duplicate Customer/Site creation, tracker/Atlas partial failure,
  lifecycle bypass, and operational-fact duplication.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R13,
  R14.

### Closure declarations

- **CLOSED — authoritative lead store.** `atlas_brain.main:app` includes the
  `atlas_brain.api` aggregate, and that aggregate includes both `leads_router`
  and `eom_funnel_router`. Their production dependencies resolve the same
  `DatabaseCRMProvider`/configured Atlas pool; no copy or replication store is
  admitted by this command.
- **CLOSED — intake/handoff fixture migration stems.** The real PostgreSQL
  fixture applies `035_contacts`, `256_contact_interaction_dedupe`, `346` lead
  pipeline, `351` lifecycle events, `352` inbound delivery receipts, and `353`
  handoffs. Those are the exact direct tables/trigger prerequisites of the
  public-intake workhorse plus finalization; production migration ownership
  remains the existing full Atlas runner.
- **CLOSED — handoff ownership/lock domains.** The four domains are approval
  key, contact, tracker Customer, and tracker Site, exactly the immutable
  handoff record's four unique ownership columns in migration 353. No other
  external identity participates in this command.
- **CLOSED — handoff wire fields.** `EOMCustomerHandoffRequest` owns exactly
  contact UUID plus tracker Customer/Site IDs; the approval key and actor fields
  are dedicated headers. `extra="forbid"` makes unlisted body fields impossible
  to admit, including operational rate/schedule data.
- **DEFAULTED — funnel bearer and approval-key grammar.** The positive
  recognizers are the generated `eomf_v1_` token format and the approval-key
  regex in `funnel_auth.py`/`funnel.py`; every unrecognized, malformed, short,
  weak, or novel value rejects before CRM work. The hostile test values are
  representative evidence, not an open allowlist.
- **CLOSED — EOM pipeline path inventory.** The two workflow `on.paths` lists
  are the canonical current handoff surface: full entrypoint/API aggregate,
  public intake, auth/router, finalizer, migration, and the test modules run by
  that workflow's only test command. A new dependency is out of the closed
  inventory until the same PR adds it to both filters and the shared-authority
  proof.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/api/__init__.py`
- `atlas_brain/eom_api/auth.py`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_auth.py`
- `atlas_brain/main.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_lead_conversion.py`
- `atlas_brain/storage/migrations/353_eom_customer_handoffs.sql`
- `plans/PR-EOM-Office-Conversion-Handoff.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

The browser never calls Atlas. The time tracker verifies the existing employee
session, compares its stable employee ID with a deployment-configured Juan
approver ID, validates the completed-estimate fields, and writes its Customer,
initial Site, and approval operation in one local transaction. `atlas_contact_id`
and a per-approval idempotency key make that result recoverable after a retry.

The tracker sends only the opaque contact/customer/site IDs, approval key, and
trusted actor evidence to Atlas using a dedicated server-side service token.
The full Atlas API aggregate that already receives `POST /api/v1/leads/intake`
also receives the handoff route, so both resolve the configured primary CRM
pool. In one Atlas transaction finalization takes sorted transaction-scoped
advisory locks for the approval key, contact, tracker Customer, and tracker
Site, then reads the canonical handoff rows. A waiting callback rechecks the
canonical row plus its customer transition and matching lifecycle event after
the winning transaction commits: it returns an identical result only for that
completed state or rejects the conflicting/incomplete payload. Atlas then
validates EOM ownership and `lead/new`, records the immutable exclusive
handoff, changes the contact to `customer` with no lead stage, and appends the
matching lifecycle event. Atlas never receives or persists the per-visit price,
schedule, frequency, or service details.

The full application's startup validates any enabled `eomf_v1_` bearer made by
the `secrets.token_urlsafe(32)` provisioning primitive and checks its own
initialized pool for the contacts, lifecycle, and handoff relations before it
serves the private route. The existing full migration runner owns production
schema application; the separate EOM Render candidate remains for its deferred
receivables path and does not enable this funnel or create a second CRM copy.
The tracker holds the matching server-only bearer value.

The finalization provider defaults to the configured Atlas database pool, but
accepts a transaction-capable pool only through its constructor. That keeps the
real-PostgreSQL proof on the production finalization path without mocking the
global storage accessor or changing runtime pool selection.

## Intentional

- Customer/Site creation precedes Atlas finalization because the tracker owns
  the actual customer, site, rate, and schedule. A failed callback is visible
  as a retryable pending handoff, not silently inferred from an Atlas status.
- Juan-only authority is enforced where the end-user session is verifiable: the
  time tracker. Atlas additionally accepts only its dedicated server token and
  records the actor evidence, but does not mistake a caller-controlled display
  name for authentication.
- The handoff implementation stays in `atlas_brain.eom_api`, but its router is
  mounted by the full `atlas_brain.api` aggregate. That is the only process
  that already owns public EOM lead intake and the authoritative CRM database.
- One initial Site is created on approval. A multi-site customer is an explicit
  later office action, not an estimate-time guess.
- The EOM Render candidate remains disabled for funnel traffic. The operator
  enables the generated bearer only on the full Atlas process that handles
  public intake, with the same server-side value in the tracker. The tracker's
  `ATLAS_FUNNEL_BASE_URL` must be that process's `/api/v1` base, not the EOM
  Render candidate; neither service exposes the bearer to a browser or commits
  it.
- The provider's injected pool is a test/adapter seam only. Normal EOM request
  handling instantiates `DatabaseCRMProvider()` with no argument and continues
  to resolve the configured Atlas pool.

## Deferred

- A declined/non-customer outcome and explicit reopen command.
- Canonical estimate booking, reschedule/cancel, and calendar projection.
- Customer-draft completion, first-clean, card collection, emails, and
  attribution reporting.
- Backfill/linking for existing Customers and more than one initial Site.

Parked hardening: none.

## Verification

- Before implementation: create and approve the matching time-tracker contract
  and its clean worktree; do not code the Atlas half against an unreviewed
  companion shape.
- Before push: focused Atlas route/service/real-PostgreSQL tests, the matching
  companion recovery proof, plan synchronization, local review, and the
  unit-gate ratchet.
- Focused full-route, service, real-PostgreSQL, and EOM profile tests ran
  against the local PostgreSQL test URL — 48 passed.
- The merged time-tracker recovery proof ran from
  `eom-timetracker@origin/main`: `python -m pytest
  test_office_conversion_handoff.py -k
  recovers_after_atlas_failure_without_duplicate_customer_or_site -q` — 1
  passed. It proves a 503 from Atlas leaves one persisted Customer/Site/key and
  the retry finalizes that same operation.
- The exact EOM pipeline command in
  `.github/workflows/atlas_eom_lead_pipeline_checks.yml` ran against the local
  PostgreSQL test URL — 263 passed.
- The exact Atlas API maturity ratchet with its CI sensitive-glob set passed
  with no new brittleness above baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 30 |
| `atlas_brain/api/__init__.py` | 2 |
| `atlas_brain/eom_api/auth.py` | 45 |
| `atlas_brain/eom_api/config.py` | 20 |
| `atlas_brain/eom_api/funnel.py` | 90 |
| `atlas_brain/eom_api/funnel_auth.py` | 100 |
| `atlas_brain/main.py` | 35 |
| `atlas_brain/services/crm_provider.py` | 237 |
| `atlas_brain/services/eom_lead_conversion.py` | 44 |
| `atlas_brain/storage/migrations/353_eom_customer_handoffs.sql` | 62 |
| `plans/PR-EOM-Office-Conversion-Handoff.md` | 339 |
| `tests/test_eom_lead_conversion.py` | 313 |
| `tests/test_eom_lead_conversion_integration.py` | 571 |
| **Total** | **1888** |
