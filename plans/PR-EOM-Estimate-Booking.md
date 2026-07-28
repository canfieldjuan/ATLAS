# PR-EOM-Estimate-Booking

## Why this slice exists

Juan's current office flow is lead/prospect -> phone call -> scheduled
estimate -> Juan qualifies the lead as customer/non-customer. Atlas now
preserves public intake as EOM leads and exposes the office review queue, but
there is still no durable Atlas-owned command for the estimate-booking step
between "new lead" and "approve customer." The parked
`claude/pr-eom-estimate-booking` branch attempted this before the #2242 review
queue landed; current main now owns the private `/api/v1/eom-funnel` boundary,
so this slice rebuilds the booking proof on that boundary instead of reviving a
second funnel API.

Diff-budget override: the durable operation model, the existing-funnel HTTP
entrypoint, the Calendar event-id recovery seam, the #2242 approval
compatibility change, and the real-PostgreSQL proof are one indivisible
vertical slice. Splitting the operation table from its only route, or the route
from its stage/approval compatibility proof, would publish an uncallable or
approval-breaking intermediate state.

### Problem-derived contract

- Root cause: a staff estimate booking is currently an external calendar action
  with no Atlas operation identity. A retry after an interrupted calendar call
  can duplicate the estimate or leave the CRM without a durable "this lead has
  an estimate booked" state. Current Atlas approval also admits only
  `lead/new`, while the next funnel stage must not make a booked lead disappear
  from Juan's approval queue.
- Correct fix must touch/change: extend the existing authenticated
  `/api/v1/eom-funnel` service boundary with one estimate-booking command;
  accept actor evidence plus an idempotency key; persist one booking operation
  before any calendar side effect; derive one deterministic Google event ID
  from that operation; recover a lost response by reusing that event ID; create
  or resume one linked Atlas appointment; transition only an active EOM
  `lead/new` contact to `lead/estimate_booked`; append the lifecycle event; and
  keep booked leads visible/approvable by teaching the review queue and customer
  handoff finalizer that `estimate_booked` is still a lead state Juan may
  convert. The fix must also fence customer approval while a booking operation
  is pending, keep the browser/NocoDB role from writing Atlas-owned booking
  links, fail enabled startup when the booking schema is absent, and keep the
  review-queue index aligned with every approval-reachable lead stage. A
  pending/retryable booking operation must also fence direct contact state
  mutations that would strand a projected Calendar event before Atlas can
  complete the operation.
- Must not change: public website intake; attribution receipt semantics; the
  tracker Customer/Site schema or handoff payload; generic CRM/MCP APIs;
  non-EOM scheduling; imported recurring schedules; calendar import receipts
  in #2195; first-clean, Stripe/card, receivables, onboarding emails, customer
  emails, declined/non-customer outcomes, reschedule/cancel, or public
  self-service onboarding. The future tracker/portal booking form is a
  companion slice; this PR only proves the Atlas private service contract.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Add the Atlas side of durable EOM estimate booking below the existing
   private `eom-funnel` service boundary.
2. Preserve #2242 customer approval compatibility by keeping
   `estimate_booked` leads in the review/approval path until a later
   companion UI can display the richer stage.
3. Add focused HTTP/service/calendar tests plus migration/compile/plan proof.
4. Close reviewer-found lifecycle, privilege, readiness, and index gaps without
   widening into UI/onboarding/payment work.

### Review Contract

- Acceptance criteria:
  1. `POST /api/v1/eom-funnel/leads/{contact_id}/estimate-bookings` with a
     valid funnel bearer, valid actor headers, valid idempotency key, and
     timezone-aware booking time whose computed end time is representable
     returns one booking result and records one
     operation, one appointment link, one deterministic calendar event ID, one
     `estimate_booked` lifecycle event, and a `lead_stage='estimate_booked'`
     contact transition.
  2. A same-key retry with the same booking payload returns the persisted
     operation/appointment/event without another calendar call; the same key
     with different booking details returns 409 before any second calendar
     call.
  3. A different idempotency key for a lead that already has a non-terminal
     booking operation returns 409 before any calendar call. If Calendar
     permanently rejects the command before appointment projection and the
     deterministic event is proven absent or cancelled, the operation becomes
     terminal and a corrected command with a new key can proceed; unreconciled
     Calendar auth/config failures stay retryable so a stale external success
     cannot be released into a duplicate new-key booking.
  4. Non-EOM, inactive, non-lead, and non-`new` contacts are rejected before the
     calendar adapter is called. The only replay exception is the persisted
     same-key operation.
  5. A Calendar 409 for the deterministic event ID is recovered by fetching the
     same event ID and completing the same operation. Transient calendar
     failures mark the operation retryable and return a surfaced error.
     Permanent Calendar failures first fetch the deterministic event ID: a live
     recovered event completes the same operation, an absent/cancelled event
     marks the operation terminal so the lead can be corrected with a new key,
     and an unverifiable auth/config state remains retryable rather than
     releasing the lead.
  6. The lead review queue includes active EOM `lead/new` and
     `lead/estimate_booked` contacts, so a booked lead stays reachable to Juan.
     The serialized public projection remains backward compatible with the
     tracker parser.
  7. Customer approval accepts active EOM leads in `new` or `estimate_booked`,
     records the lifecycle `from_stage` actually converted, ignores historical
     terminal `calendar_rejected` attempts after a corrected booking completes,
     and still rejects customers, inactive contacts, non-EOM contacts, every
     other lead stage, and any contact with an incomplete non-terminal
     estimate-booking operation.
  8. The NocoDB/browser role cannot write the appointment booking-operation link;
     the link remains Atlas-service-owned while ordinary appointment edit fields
     stay writable. While an estimate-booking operation is unfinished,
     runtime/NocoDB contact state mutations that would archive/delete/retype/
     restage/move the lead are rejected, while ordinary CRM edits and the
     booking service's own `new -> estimate_booked` completion transition
     remain allowed.
  9. Enabled full-app startup fails closed unless the estimate-booking operation
     table, appointment link column, validated appointment-link foreign key,
     valid appointment-link unique index, enabled contact-state trigger,
     handoff/lifecycle tables, owner roles, and browser-role privilege
     boundaries are present.
  10. The established-table appointment-link uniqueness index and lead-review
      queue replacement index are built/replaced concurrently; the
      appointment-link column and foreign key are added outside migration 356's
      larger ordinary batch, the FK is added `NOT VALID` and then validated with
      PostgreSQL's low-lock validation path, migration retry rebuilds a stale
      invalid appointment-link index left by an interrupted concurrent build,
      and the review-queue predicate covers both `new` and `estimate_booked`.
  11. Startup/auth config remains the existing funnel bearer boundary; no raw
     service token is added, no browser route is added, the estimate Calendar
     ID cannot exceed the persisted 512-character column limit, and
     disabled/invalid funnel auth rejects before the booking service runs.
- Execution model and invariant for criteria 1-5, 7-8:
  - PostgreSQL is the serialization authority for Atlas state. The service uses
    short PostgreSQL `READ COMMITTED` transactions with `SELECT ... FOR UPDATE`
    on the operation/contact rows and unique
    indexes on active contact operations, contact/idempotency keys, Calendar
    event IDs, and appointment operation links. The application invariant is:
    for each active lead, at most one non-terminal booking operation may own a
    deterministic Calendar event ID and at most one Atlas appointment; every
    admitted retry either observes that same operation or fails before Calendar.
  - Calendar is a side effect outside the SQL transaction, so projection
    linearizes at the operation-row lease update that writes a fresh
    `projection_token`. Only a current, unexpired token holder may issue the
    Calendar write, mark projection failure, or complete the operation; expired
    holders fail before external side effects after a later holder reclaims the
    row. Every projection attempt uses the persisted Calendar ID/event ID; a
    live recovered event completes the same operation, transient failures leave
    the operation retryable without an appointment, and terminal-eligible
    failures reconcile the deterministic event ID before releasing the lead. A
    live recovered event completes the same operation, an absent/cancelled event
    terminally releases the lead for a corrected key, and an unverifiable
    auth/config state stays retryable rather than permitting a second live
    Calendar event under a new key.
  - Completion linearizes inside one transaction that locks contact before
    operation, inserts/resumes the unique appointment link, performs the only
    allowed `new -> estimate_booked` contact transition, appends lifecycle
    evidence, clears the projection token, and marks the operation completed.
    Customer approval and direct contact-state mutation use the same contact
    lock/trigger boundary, so they cannot invalidate an incomplete booking
    between Calendar success and Atlas completion.
  - Evidence under this model: `tests/test_eom_estimate_booking.py` proves
    same-key replay/different-payload rejection, same-key recheck after the
    contact lock, Calendar live-event recovery, cancelled-conflict terminal
    classification, and lock-order expectations;
    `tests/test_eom_estimate_booking_integration.py` proves the real PostgreSQL
    operation/calendar/appointment/lifecycle chain, corrected-key recovery,
    stale projection-token holder rejection, stale external-event reconciliation
    before terminal release, stale-holder Calendar write rejection after lease
    reclaim, unverifiable terminal failure downgrade to retryable, pending
    approval fence, NocoDB operation-link denial, and the contact-state trigger
    fence including a browser-role temporary-table shadow attempt.
- Reachability proof: FastAPI route tests call the real `eom-funnel` router,
  integration tests exercise the booking service against disposable
  PostgreSQL, and calendar adapter tests observe the deterministic event-id
  request/recovery behavior through `CalendarTool.create_event`.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/eom_api/config.py`, `atlas_brain/main.py`,
  `atlas_brain/services/eom_lead_booking.py`,
  `atlas_brain/services/crm_provider.py`, `atlas_brain/tools/calendar.py`,
  additive/repair migrations, focused route/service/calendar/readiness tests,
  and the EOM lead pipeline CI selector.
- Risk areas: authenticated admission, actor headers, idempotent replay,
  duplicate calendar events under retry, DB/calendar partial failure, stage
  compatibility with #2242 approval, tenant/type/stage authorization, NocoDB
  column privilege scope, startup schema readiness, review-queue index
  selectivity, and staged deployment before the tracker/portal companion exists.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `POST /api/v1/eom-funnel/leads/{contact_id}/estimate-bookings`
  reuses `require_eom_funnel_api` and `require_eom_funnel_actor`, then enters
  the booking service. Existing `GET /api/v1/eom-funnel/leads` widens its
  admitted stage predicate from `new` to `new OR estimate_booked`, and
  customer handoff now rejects incomplete booking operations before approval.
- Replaced-path behaviors: before this slice Atlas has no private estimate
  booking command. Calendar-only/manual booking remains outside this endpoint.
  Customer handoff remains a separate endpoint but accepts the new booked
  stage.
- Guard-relevant fields: bearer digest, `X-EOM-Actor`, `X-EOM-Actor-ID`,
  `Idempotency-Key`, `contact_id`, timezone-aware `startTime`,
  `durationMinutes`, active EOM tenant, `contact_type='lead'`, allowed
  `lead_stage`, operation fingerprint, persisted estimate Calendar ID capped at
  512 characters, deterministic calendar event ID, service-owned appointment
  operation link, pending booking operation status, projection token, and
  protected contact state fields (`business_context_id`, `contact_type`,
  `lead_stage`, `status`, and DELETE).
- Caller x input shape: tracker service + valid bearer + valid actor + same
  key/same payload -> idempotent replay; same key/different payload -> 409;
  valid auth + non-EOM/non-new lead -> 404/409 before calendar; browser/no
  bearer -> 401/503 and no data; malformed time/key/body -> 422.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: this route uses the existing
  `ATLAS_EOM_FUNNEL_API_ENABLED` and
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` boundary, plus a new optional
  `ATLAS_EOM_FUNNEL_ESTIMATE_CALENDAR_ID` defaulting to `primary`. Committed
  code cannot determine deployed values; the raw bearer remains caller-side.
- Explicit value probe: tests configure a generated `eomf_v1_` bearer digest
  and a non-default estimate calendar ID, then observe that the calendar
  request uses that ID.
- Absent value probe: disabled API returns 503 and blank/mismatched bearer
  returns 401 before the booking dependency is invoked.
- Default-session/default-context probe: missing/nonpositive actor ID, blank
  actor, and overflowed start/duration combinations return 422; overlong
  estimate Calendar IDs are rejected by settings validation before enabled
  startup can serve; rows with NULL/non-EOM context are rejected before
  calendar.
- Side-effect ordering: operation row is persisted before calendar; calendar
  projection is leased outside the SQL transaction with a projection token;
  stale holders cannot issue Calendar writes, mark failure, or complete once the
  lease has expired or been reclaimed; appointment/stage/lifecycle completion
  locks contact before operation in the same order as customer approval; customer
  approval and direct contact-state mutation are fenced until any operation for
  the contact reaches `completed` or a permanent `calendar_rejected` terminal
  state releases the lead for correction.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/main.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_lead_booking.py`
- `atlas_brain/storage/migrations/356_eom_lead_estimate_booking_operations.sql`
- `atlas_brain/storage/migrations/357_eom_lead_review_queue_booked_index.sql`
- `atlas_brain/storage/migrations/358_eom_estimate_booking_appointment_link_index.sql`
- `atlas_brain/tools/calendar.py`
- `plans/PR-EOM-Estimate-Booking.md`
- `tests/test_eom_estimate_booking.py`
- `tests/test_eom_estimate_booking_integration.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_migrations_runner.py`

## Mechanism

The booking route validates the existing funnel bearer/actor headers and a
bounded idempotency key. The service opens a transaction, loads any same-key
operation, locks the contact, rechecks the same idempotency key after the
contact lock, and either returns the persisted same-fingerprint operation or
creates a new operation row for an active EOM `lead/new` contact. The operation
UUID derives a Google-safe event ID, so retries target the same external event.

Calendar projection is leased with a fresh projection token so concurrent
same-key retries do not both own the operation. A 409 for the deterministic
event ID is recovered by fetching that same event and treating a non-cancelled
event as the lost prior success; a cancelled recovered event is terminal rather
than retryable. Immediately before the Calendar write, the service refreshes and
proves an unexpired current-token lease; stale holders fail before the external
side effect. Transient Calendar failures remain retryable. Before any terminal
Calendar failure releases the lead for a new key, the service fetches the
deterministic event ID: live recovery completes the original operation,
absent/cancelled recovery permits `calendar_rejected`, and an unverifiable
auth/config state stays `calendar_failed`. Failure marking and completion both
require the current projection token, so an expired projection holder cannot
clobber a reclaimed operation. Once projection succeeds, Atlas inserts or
resumes one appointment linked by operation ID, transitions the contact to
`estimate_booked`, records the lifecycle event, and marks the operation
completed. The existing lead review and approval paths are widened only enough
to keep booked leads visible and convertible, and approval ignores historical
terminal attempts after a corrected booking completes.

Migration 356 also installs a contact trigger for EOM leads with unfinished
booking operations. The security-definer trigger pins its search path to
`pg_catalog, pg_temp` and resolves the operation table through the trigger
table's schema, so a browser-role temporary table cannot shadow the authority
lookup. It blocks direct archive/delete/type/stage/tenant changes from runtime
or NocoDB while the operation is pending/projecting/retryable, but allows
ordinary CRM edits, a permanent `calendar_rejected` correction path, and the
booking service's own completion transition from `new` to `estimate_booked`.
Migration 358 adds the nullable appointment operation-link column outside the
larger operation-table batch, adds its foreign key as `NOT VALID`, validates it
with PostgreSQL's low-lock validation path, then drops the named appointment
operation-link index concurrently before rebuilding it concurrently on the
established `appointments` table. A retry repairs any invalid relation left by
an interrupted concurrent build. The enabled startup preflight requires the
validated FK and valid/ready unique partial index before the funnel can serve.

## Intentional

- This is not a tracker/portal UI PR. Atlas proves the private service contract;
  browser reachability waits for the companion repository slice.
- The command books the first estimate only. Reschedule/cancel/lost-lead and
  completed-estimate semantics are separate commands.
- No payment/card/onboarding gate is added here. Juan explicitly deferred card
  collection until after the first clean.
- The review projection stays backward compatible for the merged tracker parser;
  richer stage labels belong to the companion UI slice.

## Deferred

- Companion PR: time-tracker admin proxy plus portal booking form/retry UI that
  calls this Atlas route with the existing server-side bearer.
- Declined/non-customer outcome, reschedule/cancel, completed-estimate
  evidence, first-clean, Stripe/card collection after first clean, onboarding
  automation, customer emails, and attribution reporting.
- Historical/customer matching beyond the explicit lead contact ID.

Parking predicate: this slice parks workflow stages and UI decisions that do
not block Atlas from durably recording exactly one first estimate booking for
an active EOM lead, projecting exactly one calendar event for retry, preserving
approval reachability, and rejecting out-of-cohort contacts before side effects.

Parked hardening: none.

## Verification

- Command: `python -m py_compile atlas_brain/eom_api/funnel.py
  atlas_brain/main.py atlas_brain/services/crm_provider.py
  atlas_brain/services/eom_lead_booking.py atlas_brain/tools/calendar.py
  tests/test_eom_estimate_booking.py tests/test_eom_estimate_booking_integration.py
  tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py
  tests/test_migrations_runner.py`; result: passed.
- Command: `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:55432/atlas_migration_tests python -m pytest tests/test_crm_read_scoping.py tests/test_eom_complaints_integration.py tests/test_eom_contacts_api_tenant_scope.py tests/test_eom_estimate_booking.py tests/test_eom_estimate_booking_integration.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_lead_pipeline_integration.py tests/test_eom_mailbox_context_binding.py tests/test_eom_lead_ingress.py tests/test_eom_recurring_appointments_integration.py tests/test_eom_scoped_gmail_credentials.py tests/test_eom_scoped_gmail_hardening.py tests/test_eom_sent_email_tenant_scope.py tests/test_leads_intake.py tests/test_migrations_runner.py -q`;
  result: 321 passed, 1 third-party `pynvml` deprecation warning against a
  disposable local Postgres 16 container.
- Command: `python -m pytest tests/test_eom_estimate_booking.py
  tests/test_eom_lead_conversion.py tests/test_migrations_runner.py -q`;
  result: 74 passed, 2 skipped, 1
  third-party `pynvml` deprecation warning.
- Command: `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:55432/atlas_migration_tests python -m pytest tests/test_eom_estimate_booking_integration.py tests/test_eom_lead_conversion_integration.py tests/test_migrations_runner.py -q`;
  result: 51 passed, 1 third-party `pynvml` deprecation warning.
- Command: `git diff --check`; result: passed.
- Command: `python scripts/audit_plan_doc.py
  plans/PR-EOM-Estimate-Booking.md`; result: passed after amend.
- Command: `python scripts/audit_plan_doc_files_touched.py
  plans/PR-EOM-Estimate-Booking.md origin/main`; result: passed after amend.
- Command: `python scripts/audit_plan_doc_diff_size.py
  plans/PR-EOM-Estimate-Booking.md origin/main`; result: passed after amend.
- Command: `python scripts/sync_pr_plan.py
  plans/PR-EOM-Estimate-Booking.md origin/main --check`; result: passed after
  amend.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 16 |
| `atlas_brain/eom_api/config.py` | 8 |
| `atlas_brain/eom_api/funnel.py` | 121 |
| `atlas_brain/main.py` | 81 |
| `atlas_brain/services/crm_provider.py` | 46 |
| `atlas_brain/services/eom_lead_booking.py` | 646 |
| `atlas_brain/storage/migrations/356_eom_lead_estimate_booking_operations.sql` | 169 |
| `atlas_brain/storage/migrations/357_eom_lead_review_queue_booked_index.sql` | 15 |
| `atlas_brain/storage/migrations/358_eom_estimate_booking_appointment_link_index.sql` | 39 |
| `atlas_brain/tools/calendar.py` | 132 |
| `plans/PR-EOM-Estimate-Booking.md` | 385 |
| `tests/test_eom_estimate_booking.py` | 496 |
| `tests/test_eom_estimate_booking_integration.py` | 927 |
| `tests/test_eom_lead_conversion.py` | 41 |
| `tests/test_eom_lead_conversion_integration.py` | 68 |
| `tests/test_migrations_runner.py` | 250 |
| **Total** | **3440** |
