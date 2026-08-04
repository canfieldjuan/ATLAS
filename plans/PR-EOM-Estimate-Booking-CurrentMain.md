# PR-EOM-Estimate-Booking-CurrentMain

## Why this slice exists

The stale estimate-booking PR shell was too far behind current `main` and
conflicted with the now-canonical EOM funnel runtime. Juan's current operating
decision is that Customer/Site creation happens later in the office after he
approves the estimate; the missing step before that is an office-owned way to
schedule the estimate itself without promoting the lead into a customer.

Diff-budget justification: this is over the 400 LOC soft cap because the
durable vertical is indivisible at the Calendar boundary. The route admission,
CRM lifecycle prepare/complete pair, deterministic Calendar ID behavior, booked
lead review compatibility, later handoff compatibility, immutable prepared
Calendar snapshot, booking-key ownership index, and regression tests must land
together; splitting any one of those would either expose an unwired route,
permit duplicate Calendar events on retry, create mutable-contact replay drift,
or leave booked leads unable to become customers.

### Problem-derived contract

- Root cause: Atlas currently has a private EOM lead review endpoint and a later
  customer-handoff endpoint, but no durable middle transition for "Juan booked
  an estimate." Scheduling that estimate is an external Calendar side effect; if
  the route records only the Calendar result, retries can duplicate calendar
  events; if retries recompute Calendar fields from mutable contact state, the
  deterministic-ID replay can reject the event that the original request
  actually created; and if booking promotes the contact straight to customer it
  violates the office-owned approval boundary.
- Correct fix must touch/change: add one private EOM funnel booking command for
  an existing active EOM lead; validate the operator, idempotency key, contact,
  and aware start/end times before any Calendar write; record immutable
  lifecycle evidence before and after the Calendar write; persist and replay
  the prepared Calendar summary/start/end/location/description/calendar ID/event
  ID from the lifecycle request event; use a deterministic Calendar event ID for
  safe retries; move the lead only from `new` to `estimate_booked`; keep
  `estimate_booked` leads visible to office review; index booking operation-key
  lookups with `operation_key` as the leading indexed column; and allow the
  existing customer handoff to approve either `new` or `estimate_booked` leads
  only after any prepared booking reaches a terminal booked/failed state, while
  preserving its idempotency and tracker-ID uniqueness guarantees. Only Calendar
  failures that prove no write occurred must be terminal evidence that allow a
  corrected new operation; transport failures, 5xx responses, and
  conflict-verification failures are ambiguous Calendar state and must block
  until reconciliation.
- Must not change: do not create or update tracker Customer/Site records; do not
  add Stripe/card-on-file, receivables, first-clean, reschedule, cancel,
  customer email, or website intake behavior; do not change public lead-intake
  payloads; do not change generic CRM/MCP customer creation semantics; do not
  change SMS/call retry hardening; do not expose lifecycle authority through
  NocoDB or browser-editable CRM columns.

## Scope (this PR)

Ownership lane: eom-lead-funnel-estimate-booking
Slice phase: vertical slice

1. Add `POST /api/v1/eom-funnel/leads/{contact_id}/estimate-bookings` for the
   tracker/server-side office surface to book one estimate for one active EOM
   lead.
2. Record `estimate_booking_requested`, `estimate_booking_calendar_failed`,
   `estimate_booking_calendar_ambiguous`, and `estimate_booked` events in the
   existing immutable EOM lifecycle ledger, with the operation key and
   deterministic Calendar event ID as retry evidence.
3. Keep booked leads in the review projection and make customer handoff accept
   `lead/new` and `lead/estimate_booked` as office-approvable lead states,
   while rejecting handoff during pending or ambiguous estimate booking.
4. Add HTTP and provider-level tests for boundary validation, idempotent retry,
   conflict rejection, non-promotion on Calendar failure, and later customer
   approval from `estimate_booked`.
5. Add the booking-key lookup index and keep the integration test migration
   setup aligned with the appointment columns read by CRM context queries.

### Review Contract

- Acceptance criteria:
  - [ ] `POST /api/v1/eom-funnel/leads/{contact_id}/estimate-bookings` rejects
        unauthenticated, malformed actor, bad idempotency-key, naive datetime,
        and `end <= start` requests before CRM or Calendar calls, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] A valid booking request calls the CRM prepare step before Calendar, uses
        the prepared deterministic `expected_calendar_event_id` in the Calendar
        create call, then completes CRM only when Calendar returns that same ID,
        settled by `tests/test_eom_lead_conversion.py`.
  - [ ] Calendar failures that prove no write occurred record
        `estimate_booking_calendar_failed`, returns an error without completing
        the booking or promoting the lead, and admits a corrected new booking
        key for the same lead, settled by `tests/test_eom_lead_conversion.py`
        and `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] `CalendarTool.create_event` propagates real Google HTTP response status
        into `ToolResult.data.status_code` before the booking service classifies
        a no-write-proven 4xx response, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] Ambiguous Calendar state, including transport failure, 5xx response,
        and a deterministic-ID 409 whose fetched event no longer matches the
        requested booking, records `estimate_booking_calendar_ambiguous` and
        remains fail-closed until reconciliation, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] A late same-key ambiguity marker cannot override a completed
        `estimate_booked` operation, and historical same-operation ambiguity
        rows do not block completed replay or customer handoff, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] Same-key retry uses the originally prepared Calendar event snapshot
        instead of mutable contact fields, settled by
        `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] Replaying the same idempotency key and same payload returns the existing
        booking without creating a second lifecycle transition or second
        Calendar event, including after the booked lead is later handed off to a
        customer, settled locally by `tests/test_eom_lead_conversion.py` and on
        real Postgres by `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] Reusing a booking key across contacts, or reusing a booking key or lead
        with a different active booking payload, rejects with 409 before
        another Calendar write, settled by `tests/test_eom_lead_conversion.py`
        and `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] `DatabaseCRMProvider.list_eom_new_lead_review_items` returns active EOM
        leads in `new` or `estimate_booked`, and the response exposes
        `leadStage`, settled by `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py`.
  - [ ] `DatabaseCRMProvider.finalize_eom_customer_handoff` approves an active
        EOM lead from either `new` or `estimate_booked` and records the actual
        prior stage in the customer-approved lifecycle event, but rejects while
        a prepared estimate booking is pending or ambiguous, settled by
        `tests/test_eom_lead_conversion_integration.py`.
  - [ ] `CalendarTool.create_event(..., event_id=...)` includes that ID in the
        Google Calendar create body and treats a matching 409 conflict as an
        idempotent success only after verifying active status, summary, start,
        end, location, and description, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] The global booking-key ownership lookup has an additive partial index
        with `operation_key` as the leading column, settled by
        `tests/test_migrations_runner.py`.
- Reachability proof: the real FastAPI route
  `POST /api/v1/eom-funnel/leads/{contact_id}/estimate-bookings` is exercised
  with auth headers in `tests/test_eom_lead_conversion.py`, asserting both the
  JSON response and the fake CRM/Calendar calls.
- Affected surfaces: private EOM funnel API, `DatabaseCRMProvider` EOM lead
  lifecycle methods, existing EOM lifecycle ledger, Calendar event creation
  helper, lead review projection, customer handoff from booked leads, migration
  for the review-queue index predicate.
- Risk areas: external side-effect retry/idempotency, lead-stage compatibility,
  authorization boundary, migration/index deploy safety, mutable contact drift
  during retry, response-shape compatibility for the private tracker client.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: private EOM funnel route admission for
  `/eom-funnel/leads/{contact_id}/estimate-bookings`; `CalendarTool.create_event`
  deterministic-ID handling; `DatabaseCRMProvider` lead-stage/lifecycle
  predicates for review, booking, terminal failure, and handoff.
- Replaced-path behaviors: no prior booking route exists; handoff previously
  admitted only `lead_stage = 'new'`; Calendar create previously sent no
  caller-supplied event ID.
- Guard-relevant fields: bearer token digest, `X-EOM-Actor`,
  `X-EOM-Actor-ID`, `Idempotency-Key`, `contact_id`, `scheduled_start`,
  `scheduled_end`, `calendar_id`, `notes`, lead `business_context_id`,
  `contact_type`, `lead_stage`, `status`.
- Caller x input shape: server-side tracker/office caller sends snake_case JSON
  with aware datetimes and service-auth headers; browser/public website callers
  are not admitted.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: existing disabled-by-default EOM funnel API
  token config is unchanged.
- Explicit value probe: route tests use an enabled
  `EOMFunnelConfig(api_enabled=True, service_token_sha256=<generated digest>)`.
- Absent value probe: existing funnel auth tests remain the coverage for
  disabled/missing token config.
- Default-session/default-context probe: route tests call the isolated router
  and full app with explicit dependency overrides; no ambient operator context
  is trusted.
- Side-effect ordering: CRM prepare must complete before Calendar create;
  CRM completion must not run unless Calendar returns the prepared deterministic
  ID; only Calendar failures that prove no write occurred are marked terminal;
  transport/5xx/conflict-verification ambiguity is marked for reconciliation;
  tests assert the call order and failure branches.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_estimate_booking.py`
- `atlas_brain/storage/migrations/356_eom_lead_review_queue_booked_stage.sql`
- `atlas_brain/storage/migrations/357_eom_estimate_booking_operation_key_index.sql`
- `atlas_brain/tools/calendar.py`
- `plans/PR-EOM-Estimate-Booking-CurrentMain.md`
- `render.eom.yaml`
- `requirements.eom.txt`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_lead_pipeline_integration.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migrations_runner.py`

## Mechanism

The route validates the private EOM service bearer, actor headers, operation
key, contact UUID, and aware start/end window. It delegates to a small service
that asks the CRM provider to prepare a booking. The provider locks the contact
and operation key, verifies an active EOM lead, rejects conflicting existing
bookings, and appends `estimate_booking_requested` lifecycle evidence that
contains the deterministic Calendar event ID and the immutable prepared Calendar
summary/start/end/location/description/calendar ID/event ID. The service then
calls Calendar with that prepared snapshot, not freshly recomputed contact
fields. Only a matching Calendar result is allowed to complete the CRM
transition to `lead_stage = 'estimate_booked'` and append `estimate_booked`
lifecycle evidence. Completed same-key replay is validated against the
operation metadata before applying lead-only admission checks, so a delayed
retry remains idempotent after a later customer handoff.

The Calendar helper keeps existing callers unchanged. When a deterministic ID is
provided, it includes that ID in the Google event body; if Google reports a
duplicate ID, the helper fetches the existing event and returns it as the
idempotent result only when the active event still matches the requested
summary, start, end, location, and description. A mismatched duplicate is an
ambiguous Calendar state, not a completed booking.

Calendar create failures that prove no write occurred append
`estimate_booking_calendar_failed`, which terminates that operation key and
allows a corrected new booking key for the same lead. Transport failures, 5xx
responses, and deterministic-conflict verification failures append
`estimate_booking_calendar_ambiguous` instead, because the Calendar event may
already exist. A completed `estimate_booked` event dominates late same-operation
ambiguity: the ambiguity marker no-ops after completion, and lifecycle readers
return completed replay / allow handoff when booked evidence already exists for
that key. Handoff rereads lifecycle events under the contact transaction and
rejects while any booking is pending or ambiguous without a matching completion,
preventing Customer/Site approval from racing the external Calendar side effect.

Migration `357_eom_estimate_booking_operation_key_index.sql` adds an additive
partial index with `operation_key` as the leading column for estimate-booking
event types, matching the provider's cross-contact ownership lookup without
changing ledger uniqueness or append-only behavior.

Migration rollback for the review-queue index is explicit: drop
`idx_contacts_eom_lead_review_queue` concurrently, then recreate it concurrently
with the prior predicate `lead_stage = 'new'`. Roll-forward safety is also
explicit: if older application code runs while the widened index remains, the
old SQL query still filters `lead_stage = 'new'`; the widened partial index does
not widen old query results.

## Intentional

- This slice does not write tracker Customer/Site rows. Juan approves the
  estimate and creates the customer/site in the office after the estimate
  produces real service/rate/schedule data.
- This slice does not add a separate booking table. The existing immutable EOM
  lifecycle ledger already provides operation-key uniqueness and audit evidence;
  using it avoids another owned table before the workflow proves it needs one.
- This slice does not implement reschedule/cancel. Those need their own
  lifecycle semantics and should not be smuggled into the first booking command.

## Deferred

- Estimate reschedule/cancel/rebook lifecycle and Calendar update/delete
  semantics.
- Tracker UI wiring for collecting estimate date/time from the office screen.
- Customer/Site onboarding after estimate approval.
- First-clean/card-on-file automation after Customer/Site approval.
- Ambiguous-booking calendar reconciliation surface. Transport/5xx ambiguity
  is an absorbing state in this slice (fail-closed by design); the follow-up
  either ships an operator reconciliation command or admits deterministic-ID
  same-key retry for transport/5xx ambiguity, which the 409 fetch-verify path
  makes provably duplicate-safe. Pre-request failures (TOOL_DISABLED /
  NOT_CONFIGURED) are already terminal-failed in this slice, so misconfigured
  boots cannot wedge a lead.

Parking predicate: hardening that is narrower than one estimate-booking request,
or that requires a new table/subsystem/dependency beyond the lifecycle ledger,
is parked unless it can duplicate Calendar events, promote a lead incorrectly,
or expose lifecycle authority outside the private service boundary.

Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/services/eom_estimate_booking.py atlas_brain/services/crm_provider.py atlas_brain/tools/calendar.py` -- passed.
- `python -m py_compile atlas_brain/services/eom_estimate_booking.py atlas_brain/tools/calendar.py tests/test_eom_lead_conversion.py` -- passed.
- `python -m pytest tests/test_eom_lead_conversion.py -q` -- 66 passed, 1 warning.
- `pytest -q tests/test_eom_lead_conversion.py tests/test_migrations_runner.py tests/test_eom_render_profile.py` -- 149 passed, 1 skipped, 1 warning.
- `pytest -q tests/test_eom_lead_conversion_integration.py tests/test_eom_lead_pipeline_integration.py` -- 1 passed, 40 skipped locally because `ATLAS_MIGRATION_TEST_DATABASE_URL` is not configured.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:<disposable-port>/atlas_migration_tests python -m pytest tests/test_crm_read_scoping.py tests/test_eom_complaints_integration.py tests/test_eom_contacts_api_tenant_scope.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_lead_pipeline_integration.py tests/test_eom_mailbox_context_binding.py tests/test_eom_lead_ingress.py tests/test_eom_recurring_appointments_integration.py tests/test_eom_scoped_gmail_credentials.py tests/test_eom_scoped_gmail_hardening.py tests/test_eom_sent_email_tenant_scope.py tests/test_leads_intake.py tests/test_migrations_runner.py -q` -- 333 passed, 1 warning against disposable Postgres 16.
- `pytest -q tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_and_receivables_auth` -- 1 passed.
- `python scripts/maturity_sweep.py atlas_brain/tools --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_tools.json` -- ratchet gate passed; `calendar.py` remains at baseline score 15.
- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/eom_estimate_booking.py atlas_brain/services/crm_provider.py atlas_brain/tools/calendar.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_render_profile.py tests/test_migrations_runner.py` -- passed.
- `python -m pytest -q tests/test_audit_plan_doc.py tests/test_audit_plan_code_consistency.py tests/test_audit_pr_plan_presence.py tests/test_check_diff_budget.py` -- 103 passed.
- `python scripts/check_boundary_change_enumeration.py --base origin/main --strict` -- OK.
- `python scripts/check_deployed_config_probing.py --base origin/main --strict` -- OK.
- `python scripts/maturity_sweep.py atlas_brain/eom_api --tests-root tests --top 10` -- informational; changed `funnel.py` score 3 (`NO_RAISES_TESTS`) and broader pre-existing directory findings.
- `python scripts/maturity_sweep.py atlas_brain/services --tests-root tests --top 10` -- informational; changed `crm_provider.py` score 97 among broad pre-existing service debt.
- `python scripts/maturity_sweep.py atlas_brain/tools --tests-root tests --top 10` -- informational; changed `calendar.py` score 15 with pre-existing swallowed-exception/prefetch findings at lines outside this slice.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 8 |
| `atlas_brain/eom_api/funnel.py` | 96 |
| `atlas_brain/services/crm_provider.py` | 843 |
| `atlas_brain/services/eom_estimate_booking.py` | 262 |
| `atlas_brain/storage/migrations/356_eom_lead_review_queue_booked_stage.sql` | 27 |
| `atlas_brain/storage/migrations/357_eom_estimate_booking_operation_key_index.sql` | 18 |
| `atlas_brain/tools/calendar.py` | 119 |
| `plans/PR-EOM-Estimate-Booking-CurrentMain.md` | 483 |
| `render.eom.yaml` | 10 |
| `requirements.eom.txt` | 1 |
| `tests/test_eom_lead_conversion.py` | 849 |
| `tests/test_eom_lead_conversion_integration.py` | 847 |
| `tests/test_eom_lead_pipeline_integration.py` | 7 |
| `tests/test_eom_render_profile.py` | 15 |
| `tests/test_migrations_runner.py` | 47 |
| **Total** | **3632** |

## Cold diff reconstruction

Gaps first: no contract gaps found in the current diff. The real-Postgres
EOM lead pipeline proof now ran locally against disposable Postgres 16, covering
the CI lane that previously failed on estimate-booking metadata, typed Calendar
outcome marker inserts, and the completed/ambiguous/failed precedence ladder.

Change-by-change reconstruction against the contract:

- The private funnel API now defines an estimate-booking request model with
  forbidden extra fields, timezone-aware `scheduled_start`/`scheduled_end`, a
  positive time window, optional bounded `calendar_id`, and optional bounded
  `notes`. This traces to the contract's route/body validation requirement.
  Citation: `atlas_brain/eom_api/funnel.py:47`.
- The lead review response now exposes `leadStage`, so the private office queue
  can distinguish `new` from `estimate_booked` leads. This traces to the
  contract's booked-lead review compatibility requirement. Citation:
  `atlas_brain/eom_api/funnel.py:81`.
- The new route `POST /eom-funnel/leads/{contact_id}/estimate-bookings` uses
  the existing private bearer and actor dependencies, the existing idempotency
  header validator, the CRM dependency, and a lazy Calendar dependency. It
  delegates to the booking service and maps booking/provider errors to HTTP
  errors. This traces to the contract's private operator-visible booking
  command. Citation: `atlas_brain/eom_api/funnel.py:203`.
- The booking service derives a deterministic Google-safe event ID from the
  contact and idempotency key, resolves an omitted request calendar to the
  configured CalendarTool default before CRM preparation, requires CRM
  prepare/complete/ambiguous/failed lifecycle hooks, prepares CRM first, uses
  the prepared Calendar snapshot for create/retry, returns a closed JSON-safe
  shape for completed idempotent replay, marks deterministic-ID conflicts and
  indeterminate Calendar failures ambiguous, records only no-write-proven
  Calendar failures as terminal failed attempts, refuses to complete CRM if
  Calendar returns a different ID, and otherwise completes CRM. This traces to
  the contract's duplicate-safe Calendar side-effect ordering. Citations:
  `atlas_brain/services/eom_estimate_booking.py:51`,
  `atlas_brain/services/eom_estimate_booking.py:70`,
  `atlas_brain/services/eom_estimate_booking.py:99`,
  `atlas_brain/services/eom_estimate_booking.py:112`,
  `atlas_brain/services/eom_estimate_booking.py:155`,
  `atlas_brain/services/eom_estimate_booking.py:167`.
- `DatabaseCRMProvider.list_eom_new_lead_review_items` now selects `lead_stage`
  and filters active EOM leads by `lead_stage IN ('new', 'estimate_booked')`.
  This traces to keeping booked estimates visible until Juan approves the
  Customer/Site handoff. Citation: `atlas_brain/services/crm_provider.py:1212`.
- `DatabaseCRMProvider.prepare_eom_estimate_booking` locks the contact and
  operation key, enforces booking-key ownership across contacts, validates
  active EOM lead ownership/type/stage for new or pending operations, treats
  failed attempts as terminal,
  rejects pending/booked/ambiguous conflicting operations, compares replay
  payload metadata, returns completed same-key replay before lead-only
  admission checks, persists the immutable Calendar event snapshot, and writes
  `estimate_booking_requested` before Calendar is called. This traces to the
  contract's durable prepare/idempotency requirement. Citations:
  `atlas_brain/services/crm_provider.py:1221`,
  `atlas_brain/services/crm_provider.py:1307`,
  `atlas_brain/services/crm_provider.py:1475`,
  `atlas_brain/services/crm_provider.py:1530`.
- `DatabaseCRMProvider.mark_eom_estimate_booking_calendar_ambiguous` records an
  immutable ambiguity event when Calendar returns an unexpected ID, without
  promoting the lead. This traces to the contract's "do not release/complete on
  unexpected Calendar success" requirement. Citation:
  `atlas_brain/services/crm_provider.py:1496`.
- `DatabaseCRMProvider.mark_eom_estimate_booking_calendar_failed` records a
  definitive Calendar failure as terminal lifecycle evidence for that operation
  key. This traces to allowing a corrected new booking operation without
  deleting or mutating the failed attempt. Citation:
  `atlas_brain/services/crm_provider.py:1535`.
- `DatabaseCRMProvider.complete_eom_estimate_booking` rejects mismatched Calendar
  IDs, rejects ambiguous or failed prepared attempts, verifies the prepared
  request metadata, updates only `lead/new` to `lead/estimate_booked`, and
  appends the `estimate_booked` lifecycle event with Calendar metadata. This
  traces to the contract's middle lifecycle transition. Citations:
  `atlas_brain/services/crm_provider.py:1584`,
  `atlas_brain/services/crm_provider.py:1679`.
- `DatabaseCRMProvider.finalize_eom_customer_handoff` now accepts prior stage
  `new` or `estimate_booked`, rejects handoff while a booking is pending or
  ambiguous, updates only the current prior stage, and records that actual prior
  stage in the `customer_approved` lifecycle event. This traces to allowing
  booked estimates to become customers later without changing tracker
  Customer/Site ownership or racing Calendar. Citations:
  `atlas_brain/services/crm_provider.py:2198`,
  `atlas_brain/services/crm_provider.py:2356`,
  `atlas_brain/services/crm_provider.py:2435`.
- `CalendarTool.create_event` now accepts optional `event_id`, includes it in
  the Google Calendar body only when provided, treats a 409 duplicate as
  idempotent success only when the fetched active event matches the requested
  summary/start/end/location/description, and returns `IDEMPOTENCY_CONFLICT`
  otherwise. HTTP status/auth failures now expose the Calendar request phase and
  status code so the booking service can distinguish no-write-proven create
  failures from indeterminate post-conflict verification failures. Existing
  callers that omit `event_id` keep the old request body. This traces to the
  contract's Calendar retry requirement without changing general Calendar
  callers. Citations:
  `atlas_brain/tools/calendar.py:50`,
  `atlas_brain/tools/calendar.py:597`, `atlas_brain/tools/calendar.py:629`,
  `atlas_brain/tools/calendar.py:648`, `atlas_brain/tools/calendar.py:661`,
  `atlas_brain/tools/calendar.py:713`.
- `DatabaseCRMProvider.mark_eom_estimate_booking_calendar_ambiguous` now inserts
  only when the same operation key is not already booked, and provider readers
  let same-key `estimate_booked` dominate ambiguity. This traces to the
  contract's completed-replay guarantee under overlapping Calendar retries.
  Citations: `atlas_brain/services/crm_provider.py:1450`,
  `atlas_brain/services/crm_provider.py:1592`,
  `atlas_brain/services/crm_provider.py:1747`,
  `atlas_brain/services/crm_provider.py:2380`.
- Migration `356_eom_lead_review_queue_booked_stage.sql` replaces the review
  queue index predicate so it matches the new provider filter for `new` and
  `estimate_booked` leads, and documents the exact concurrent rollback plus the
  roll-forward safety claim. This traces to the contract's review-queue
  compatibility and migration rollback requirements. Citations:
  `atlas_brain/storage/migrations/356_eom_lead_review_queue_booked_stage.sql:7`,
  `atlas_brain/storage/migrations/356_eom_lead_review_queue_booked_stage.sql:27`.
- Migration `357_eom_estimate_booking_operation_key_index.sql` adds an
  additive partial index on `(operation_key, contact_id, event_type)` for
  estimate-booking lifecycle events. This traces to the cross-contact
  booking-key ownership lookup and avoids a lifecycle-ledger scan. Citation:
  `atlas_brain/storage/migrations/357_eom_estimate_booking_operation_key_index.sql:10`.
- Route tests cover valid prepare-Calendar-complete ordering, Calendar failure
  classification into ambiguous vs no-write-proven terminal failure, configured
  default Calendar ID resolution when the payload omits a calendar,
  post-conflict auth ambiguity, immutable prepared Calendar snapshot reuse,
  unexpected Calendar ID ambiguity, deterministic-ID 409 ambiguity, completed
  JSON-safe replay without another Calendar call, provider conflict without
  Calendar, bad body rejection, HTTP guard rejection, deterministic Calendar
  helper verification, and real Calendar HTTP status/request-phase propagation
  into the failure classifier. This traces to the route, side-effect ordering,
  idempotency, and guard criteria. Citations:
  `tests/test_eom_lead_conversion.py:444`,
  `tests/test_eom_lead_conversion.py:498`,
  `tests/test_eom_lead_conversion.py:658`,
  `tests/test_eom_lead_conversion.py:775`,
  `tests/test_eom_lead_conversion.py:966`,
  `tests/test_eom_lead_conversion.py:1004`.
- Real-Postgres tests, when the migration DB URL is configured, cover booked
  leads staying visible in the review projection, one requested/booked
  lifecycle pair on replay, immutable prepared Calendar snapshot persistence,
  late ambiguity marker suppression after completion, same-key replay despite a
  historical ambiguity row after booking, same-key replay after customer
  handoff, approval from `estimate_booked`, pending booking handoff rejection
  plus corrected-operation admission after a failed terminal event, and
  cross-contact booking-key ownership. This traces to the provider lifecycle and
  handoff compatibility criteria. Citations:
  `tests/test_eom_lead_conversion_integration.py:190`,
  `tests/test_eom_lead_conversion_integration.py:421`,
  `tests/test_eom_lead_conversion_integration.py:629`,
  `tests/test_eom_lead_conversion_integration.py:722`.
- EOM lead-pipeline integration fixtures now apply
  `348_appointment_operating_fields.sql` before lifecycle/contact code queries
  appointment operating columns, matching the production migration dependency
  and fixing the CI `appointments.recurrence_interval` failure. Citation:
  `tests/test_eom_lead_pipeline_integration.py:52`.
- Slim Render profile tests now assert the new route is exposed without loading
  the full API/config/reasoning stack. This traces to preserving the slim EOM
  runtime boundary. Citation: `tests/test_eom_render_profile.py:220`.
- Migration tests assert the new index SQL uses the provider-compatible
  `lead_stage IN ('new', 'estimate_booked')` predicate and includes rollback
  evidence. This traces to the migration/index acceptance criterion. Citation:
  `tests/test_migrations_runner.py:169`.

Scope check:

- Everything changed traces to the contract: private EOM funnel API,
  `DatabaseCRMProvider` EOM lifecycle methods, existing lifecycle ledger usage,
  Calendar deterministic create support, review index migration/rollback
  evidence, and tests.
- Everything the contract required appears in the diff: booking route, auth/body
  guards, CRM prepare/complete ordering, deterministic Calendar ID, definitive
  failure terminal events, ambiguous Calendar blocking, booked-lead review
  visibility, handoff from `estimate_booked`, pending handoff rejection,
  cross-contact booking-key ownership, and regression tests.
- No declared out-of-scope module moved: no tracker Customer/Site creation,
  Stripe/card-on-file, receivables, first-clean, reschedule/cancel, website
  intake payload, generic CRM/MCP customer creation, or SMS/call retry behavior
  is changed.
