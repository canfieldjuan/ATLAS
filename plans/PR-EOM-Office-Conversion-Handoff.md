# PR-EOM-Office-Conversion-Handoff

## Why this slice exists

This is the next EOM funnel slice after merged inbound preservation. A completed
estimate normally gives Juan the customer name/contact/address plus the initial
site's per-visit price, service scope, and preferred cleaning schedule. Juan
approves that completed estimate in the office; it should create the operational
Customer and first Site there, not turn a field action or Google Calendar entry
into a customer.

This exceeds the normal diff budget because the contract is one atomic
cross-system operation: the tracker command, Atlas auth boundary, full-app
mount/startup guard, immutable database record, transaction ordering, and both
HTTP/real-PostgreSQL proofs must agree before either side can safely be enabled.
Shipping those surfaces separately would either expose a callback with no
authoritative store, accept a credential without a trust anchor, or persist a
handoff row without a proven finalization transition.

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
- Review-repair extension: the enabled-route repair still inferred credential
  provenance from the shape of an operator-provided raw bearer, and the
  handoff-table insert trigger treated a caller-settable PostgreSQL setting as
  authorization. Neither is a real authority boundary: a patterned bearer can
  look random, and any database caller able to issue `set_config` can create an
  otherwise incomplete immutable row. The route proof also inspected router
  metadata and a synthetic app instead of the full application's actual HTTP
  entrypoint, while the current PR body retained a superseded slim-Render
  deployment description.
- Correct repair: provision one generated raw bearer only to the tracker and
  store only its SHA-256 digest in the full Atlas process; validate the digest
  at startup and compare the request's digest in constant time. Make a handoff
  insert admissible only after the same transaction has established the exact
  EOM customer transition and matching lifecycle evidence, so a caller cannot
  reserve a handoff slot with a self-set marker. Exercise every new HTTP
  rejection choke point, use the real `/api/v1` aggregate in the route proof,
  and exercise the bounded enabled-startup preflight that the full lifespan
  invokes after database initialization. Record the actual full-Atlas
  deployment, feature-disable rollback, and why the coupled transaction,
  migration, auth, and proof surfaces cannot be split safely.
- Must still not change: the public intake payload, existing tracker approval
  command or Customer/Site schema, the public route shape, generic CRM writes,
  the full Atlas migration runner, or which process owns operational estimate
  facts. A database superuser remains able to administer any table; this slice
  must prevent ordinary application/NocoDB-style inserts from manufacturing an
  incomplete handoff record, not claim to constrain database administration.
- Review-repair extension 2: the handoff table was owned by the same `atlas`
  database login configured for NocoDB. That login can disable the admission
  and immutability triggers, so trigger checks alone did not constrain the
  ordinary office database surface. The full-app HTTP test also sent a request
  without starting the app lifespan, while a separate test only called the
  startup helper directly.
- Correct repair: transfer protected handoff objects to a non-login guard role,
  retain only explicit runtime DML grants, and provision NocoDB with its own
  least-privilege login. That login may retain ordinary CRM-table access but
  receives no lifecycle-event or handoff mutation/trigger-control privilege;
  the real PostgreSQL fixture must prove those denials. Start the actual full
  application's lifespan under a bounded test configuration, run the
  authenticated callback while it is live, and prove the enabled preflight has
  completed before that request is served.
- Must still not change: normal NocoDB CRM visibility or non-EOM editing,
  database-superuser administration, public intake behavior, tracker
  Customer/Site ownership, generic CRM behavior, and optional runtime startup
  services. This slice changes the protected-table privilege boundary and the
  NocoDB connection identity only.
- Review-repair extension 3: the distinct NocoDB login was still granted
  table-wide `UPDATE` on `contacts`, although the EOM ownership/type/stage
  transition restriction exists only in `DatabaseCRMProvider`. A direct NocoDB
  edit could therefore bypass the finalization evidence. Separately, the
  migration runner executes a migration and records it in `schema_migrations`
  as separate autocommit operations. Migration 354 revokes the non-superuser
  executor's temporary guard-role membership in its SQL, so a process failure
  in that interval leaves an applied privilege state that the same executor
  cannot rerun or record.
- Correct repair: make the migration runner support an explicit,
  migration-authored atomic-bookkeeping marker. Only 354 opts in; its privilege
  changes and migration-ledger insert run in one PostgreSQL transaction, so an
  interruption rolls both back and a commit leaves the migration recorded.
  Keep NocoDB's ordinary `contacts` visibility and safe CRM edits, but grant
  `INSERT`/`UPDATE` only for an explicit non-lifecycle column set that excludes
  `business_context_id`, `contact_type`, and `lead_stage`. Prove the role is
  denied each protected direct mutation while ordinary contact editing remains
  available, and prove a forced bookkeeping failure rolls back the ownership
  and membership changes.
- Must still not change: migration execution for every unmarked migration,
  including `CREATE INDEX CONCURRENTLY` migrations; the handoff/lifecycle
  table guards; the full Atlas startup/data-store guard; tracker retry and
  Customer/Site behavior; or permitted non-lifecycle NocoDB CRM operations.

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
8. Replace raw-token format inference with generated-token digest provisioning,
   make the database admission rule derive from the completed transition rather
   than a caller-settable GUC, and prove the full aggregate's HTTP and startup
   choke points. Update the deployment/rollback and diff-size records to match
   the actual full-Atlas implementation.
9. Move the handoff table and its trigger functions under a non-login database
   guard role, give the NocoDB service its distinct least-privilege login, and
   prove that login cannot mutate the protected handoff/lifecycle records or
   alter trigger state. Run an authenticated callback inside the actual app
   lifespan after the enabled authoritative-store preflight completes.
10. Close the remaining privilege and crash-consistency gaps: restrict the
    NocoDB role to explicit non-lifecycle contact columns, and execute only
    migration 354 plus its `schema_migrations` record in one rollback-safe
    database transaction. Prove both the protected-column denials and the
    forced bookkeeping-failure rollback on PostgreSQL without changing the
    execution model for unmarked migrations.

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
   only after its matching finalization transition and reject update, delete,
   and truncate mutations. A non-login guard role owns the protected table and
   trigger functions; the NocoDB role cannot alter triggers or mutate protected
   handoff/lifecycle records. Transaction-scoped, sorted advisory locks for the key,
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
   integration tests settle the transition and blocked bypasses. The distinct
   NocoDB role has only explicit safe-column `INSERT`/`UPDATE` grants on
   `contacts`, so it cannot directly write `business_context_id`,
   `contact_type`, or `lead_stage`; real PostgreSQL tests settle the three
   direct denials and a permitted non-lifecycle edit.
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
   generated tracker bearer only by its Atlas-only SHA-256 provisioning digest;
   parametrized tests settle missing, malformed, placeholder, and otherwise
   non-provisioned digests before request handling. A bounded real-lifespan
   test proves that preflight completes before the authenticated callback is
   served; it does not launch optional production services.
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
- The migration runner executes only a migration declaring the
  `atlas: atomic-bookkeeping` marker and its `schema_migrations` bookkeeping
  in one PostgreSQL transaction. Migration 354 declares that marker; a forced
  ledger-insert failure leaves neither guard-object ownership nor temporary
  role-membership revocation committed, while a successful run records 354
  before the executor loses that membership. Existing unmarked migrations,
  including concurrent-index migrations, retain their autocommit execution
  model. Real PostgreSQL and runner tests settle the distinction.
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
  handoffs plus `354` privilege ownership. Those are the exact direct tables/
  trigger/role prerequisites of the
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
- **DEFAULTED — funnel credential digest and approval-key grammar.** The
  positive recognizers are a lowercase SHA-256 digest emitted by
  `generate_eom_funnel_service_token()` and the approval-key regex in
  `funnel_auth.py`/`funnel.py`; every missing, malformed, placeholder, or novel
  configuration value rejects before CRM work. The hostile test values are
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
- `atlas_brain/storage/migrations/354_eom_customer_handoff_privileges.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `docker-compose.yml`
- `plans/PR-EOM-Office-Conversion-Handoff.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_migrations_runner.py`

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

The provisioning helper creates an `eomf_v1_` tracker bearer with
`secrets.token_urlsafe(32)`, sends that raw value only to the tracker, and
stores only its SHA-256 digest in the full Atlas process. Atlas validates that
digest at startup, compares the supplied bearer digest in constant time, and
checks its own initialized pool for the contacts, lifecycle, and handoff
relations before it serves the private route. The existing full migration
runner owns production schema application; the separate EOM Render candidate
remains for its deferred receivables path and does not enable this funnel or
create a second CRM copy.

The protected handoff table and its trigger functions are owned by a dedicated
non-login database role. Atlas retains only the grants needed to run the
finalization transaction. NocoDB uses a separate login that can keep normal
CRM-table access but has no grant on the handoff or lifecycle evidence tables,
so it cannot manufacture a finalization or disable its guards. The deployment
must first have a database administrator provision both roles, the NocoDB
login password, and a temporary admin membership for the non-superuser Atlas
migration executor. Migration 354 transfers ownership and revokes that
temporary membership before the application serves. The Compose profile never
falls back to the Atlas service login.

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
  enables only the generated bearer digest on the full Atlas process that
  handles public intake, with the matching raw value in the tracker. The
  tracker's `ATLAS_FUNNEL_BASE_URL` must be that process's `/api/v1` base, not
  the EOM Render candidate; neither service exposes the bearer to a browser or
  commits it.
- The provider's injected pool is a test/adapter seam only. Normal EOM request
  handling instantiates `DatabaseCRMProvider()` with no argument and continues
  to resolve the configured Atlas pool.
- The full-lifespan proof uses the production settings model with optional
  external services disabled; it still executes the application lifespan's
  database-init, enabled-funnel preflight, and request-serving order.

## Deferred

- A declined/non-customer outcome and explicit reopen command.
- Canonical estimate booking, reschedule/cancel, and calendar projection.
- Customer-draft completion, first-clean, card collection, emails, and
  attribution reporting.
- Backfill/linking for existing Customers and more than one initial Site.

Parking predicate: this vertical slice does not park a flaw that can create an
unauthorized handoff, cross-store 404, partial customer transition, or unsafe
deployment rollback; those are blockers to enabling the command. Broader
database-administrator policy, customer card collection, and later customer
commands stay separate because they do not alter this command's safe path.

Parked hardening: none against that predicate.

### Rollback

1. Disable `ATLAS_EOM_FUNNEL_API_ENABLED` on the full Atlas service and remove
   the tracker `ATLAS_FUNNEL_BASE_URL`/raw bearer pair, then redeploy both
   processes. The endpoint returns its existing fail-closed `503`; public lead
   intake and the tracker Customer/Site records remain intact.
2. Retain migration 353 and all handoff/lifecycle rows. They are immutable audit
   evidence and must not be dropped or truncated as a feature-disable step.
   Retain migration 354 and the non-login guard/NocoDB roles as well; rolling
   back a feature flag must not restore trigger-control or protected-table
   mutation privileges.
3. A future schema removal requires a dedicated retention/migration plan after
   proving there are no retained handoff rows that must remain auditable; it is
   not a rollback action for this production command.

## Verification

- Before implementation: create and approve the matching time-tracker contract
  and its clean worktree; do not code the Atlas half against an unreviewed
  companion shape.
- Before push: focused Atlas route/service/real-PostgreSQL tests, the matching
  companion recovery proof, plan synchronization, local review, and the
  unit-gate ratchet.
- Focused full-route and EOM profile tests ran — 51 passed.
- The merged time-tracker recovery proof ran from
  `eom-timetracker@origin/main`: `python -m pytest
  test_office_conversion_handoff.py -k
  recovers_after_atlas_failure_without_duplicate_customer_or_site -q` — 1
  passed. It proves a 503 from Atlas leaves one persisted Customer/Site/key and
  the retry finalizes that same operation.
- The exact EOM pipeline command in
  `.github/workflows/atlas_eom_lead_pipeline_checks.yml` ran against the local
  PostgreSQL test URL — 278 passed.
- The exact Atlas API maturity ratchet with its CI sensitive-glob set passed
  with no new brittleness above baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 34 |
| `atlas_brain/api/__init__.py` | 2 |
| `atlas_brain/eom_api/auth.py` | 78 |
| `atlas_brain/eom_api/config.py` | 23 |
| `atlas_brain/eom_api/funnel.py` | 90 |
| `atlas_brain/eom_api/funnel_auth.py` | 127 |
| `atlas_brain/main.py` | 86 |
| `atlas_brain/services/crm_provider.py` | 234 |
| `atlas_brain/services/eom_lead_conversion.py` | 44 |
| `atlas_brain/storage/migrations/353_eom_customer_handoffs.sql` | 84 |
| `atlas_brain/storage/migrations/354_eom_customer_handoff_privileges.sql` | 152 |
| `atlas_brain/storage/migrations/__init__.py` | 30 |
| `docker-compose.yml` | 5 |
| `plans/PR-EOM-Office-Conversion-Handoff.md` | 493 |
| `tests/test_eom_lead_conversion.py` | 499 |
| `tests/test_eom_lead_conversion_integration.py` | 838 |
| `tests/test_migrations_runner.py` | 56 |
| **Total** | **2875** |
