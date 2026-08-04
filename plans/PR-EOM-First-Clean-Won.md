# PR-EOM-First-Clean-Won

## Why this slice exists

The funnel go-live arc (#2275) has a durable estimate booking (A1, #2272) and
a later customer/site handoff, but nothing in between records the decision
that actually wins the business: the office books the lead's FIRST CLEANING.
Today no `won` stage exists anywhere in the stage set, so the tracker cannot
distinguish "estimate happened" from "client accepted and is scheduled", and
the onboarding email Juan approves in the next slice (A3) has no queue to
read from. Office-controlled conversion is the standing decision (issue
#2188, 2026-07-26): the funnel routes ARE the explicit office actions, so
booking the first clean must be a private office command that advances
`lead_stage` to `won` and enqueues one reviewable draft -- nothing sends.

Diff-budget justification: this is over the 400 LOC soft cap because a stage
that nothing can safely enter is worse than no stage. The `won` transition,
the review-queue and handoff admission widening, the draft table with its
single-send contract, the two index-predicate migrations, and the regression
tests must land together; splitting them would either strand `won` leads
outside the review queue and handoff, ship a booked first clean without its
draft evidence, or leave A3 to invent the draft schema ad hoc. The booking
engine itself is NOT duplicated: A1's adversarially hardened engine is
parametrized by a frozen family config, and the untouched A1 test suite is
the no-regression proof for that refactor.

### Problem-derived contract

- Root cause: the EOM funnel has no durable transition for "the office booked
  the first cleaning". The stage set (`new`, `estimate_booked`) is hardcoded
  in the review-queue index predicate, the review-queue query, and the
  booking/handoff admission guards, so a first-clean command cannot be added
  without widening those admission boundaries in one slice. Separately, the
  onboarding email A3 will send needs a queue whose claim semantics are
  race-free; the invoicing pending-drafts pattern A3 was told to clone
  read-checks-then-updates without a status guard, which double-sends under
  two concurrent approvers, so the new draft table must bake the atomic
  single-send contract into its schema instead of inheriting that race.
- Correct fix must touch/change: parametrize the existing estimate-booking
  engine (prepare/complete/markers, deterministic Calendar IDs, execution
  lock, prepared-snapshot replay, terminal/ambiguous classification) over a
  frozen booking-family config so the first-clean family reuses every proven
  invariant; add the four first-clean lifecycle event types and the `won`
  target stage; keep the execution-lock namespace shared across families so
  the round-4 handoff fence and round-5 single-connection scope stay
  byte-identical; widen the review queue and customer handoff to admit
  `won`; enqueue exactly one pending onboarding draft inside the
  won-completion transaction, snapshotting recipient/subject/body at enqueue
  and recording `blocker='no_email'` instead of silently skipping; make
  draft replays no-ops via UNIQUE(operation_key); document and test the
  `UPDATE ... WHERE status = 'pending' RETURNING` single-send claim; and
  recreate the two partial indexes whose predicates enumerate stages/event
  types, using the replay-safe drop-then-create ritual from migrations
  355/356/357.
- Must not change: do not send any email (A3 owns approval and send); do not
  add the approval/claim surface itself, the tokenized customer link, or any
  tracker Customer/Site behavior; do not change the estimate-booking public
  contract, its lifecycle semantics, or its tests; do not change public
  lead-intake payloads, generic CRM/MCP semantics, or NocoDB grants; do not
  reuse the Gmail-reply-shaped `email_drafts` table.

## Scope (this PR)

Ownership lane: eom-lead-funnel-first-clean
Slice phase: vertical slice

1. Add `POST /api/v1/eom-funnel/leads/{contact_id}/first-clean-bookings` for
   the tracker/server-side office surface to book one first cleaning for one
   active EOM lead, reusing the estimate-booking request model, auth
   dependencies, idempotency-key dependency, and response conventions.
2. Parametrize the booking engine in `crm_provider.py` and
   `eom_estimate_booking.py` over a frozen `_EOMBookingFamily` /
   `_EOMBookingServiceBinding` pair; the existing estimate methods become
   thin bindings with unchanged signatures and behavior, and the first-clean
   family binds events `first_clean_booking_requested` /
   `first_clean_booked` / `first_clean_booking_calendar_failed` /
   `first_clean_booking_calendar_ambiguous`, admission from
   `new`/`estimate_booked`, target stage `won`, deterministic Calendar ID
   prefix `eomfcl`, and the draft-enqueue completion side effect.
3. Enforce cross-family booking-key ownership (a key belongs to one family
   forever) and cross-family blocking (an unsettled operation of either
   family blocks the other; a completed estimate booking never blocks the
   first clean -- that is the normal funnel path).
4. Widen the review queue and customer handoff to admit `won` leads, with
   the handoff fence and blocking predicates covering all eight booking
   event types.
5. Create `eom_onboarding_email_drafts` (migration 360) with claim
   ownership modeled separately from delivery (pending -> sending -> sent,
   readiness predicate built into the atomic claim), UNIQUE operation_key
   replay idempotency, and a partial unique index allowing at most one
   live draft per contact; enqueue the rendered `onboarding_welcome`
   template inside the won-completion transaction, resolving the recipient
   through the review queue's latest-intake projection; gate funnel
   startup on the table so deployments ahead of the migration fail closed.
6. Recreate the review-queue index (migration 358) and the booking
   operation-key index (migration 359) with widened predicates using the
   355/356/357 replay-safe ritual.
7. Add HTTP, provider, migration-shape, and real-Postgres tests for the
   first-clean lifecycle, draft semantics, claim contract, fence coverage,
   status-flip settlement, and cross-family rules.

### Review Contract

- Acceptance criteria:
  - [ ] `POST /api/v1/eom-funnel/leads/{contact_id}/first-clean-bookings`
        rejects unauthenticated, malformed actor, bad idempotency-key, and
        malformed body requests before CRM or Calendar calls, sharing the
        estimate route's boundary table, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] A valid first-clean request runs prepare -> Calendar -> complete in
        order under the shared execution lock, uses the deterministic
        `eomfcl` event ID, and returns `lead_stage='won'`,
        `status='first_clean_booked'`, and `onboarding_draft_id`, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] The estimate booking methods keep their exact signatures and
        behavior after the family parametrization, settled by the untouched
        pre-existing estimate tests in `tests/test_eom_lead_conversion.py`
        and `tests/test_eom_lead_conversion_integration.py`.
  - [ ] First-clean Calendar failures classify exactly like estimate
        failures: pre-request and auth-phase failures record the terminal
        first-clean failed marker; indeterminate failures and completion
        rejections record the first-clean ambiguous marker, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] Completion moves the lead to `won` and inserts exactly one
        `status='pending'` draft row in the same transaction; prepare and
        complete replays are idempotent and report the same
        `onboarding_draft_id`; a contact with no email gets
        `blocker='no_email'` instead of being skipped, settled by
        `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The draft recipient resolves through the same latest-intake
        projection the office review queue shows (ingress keeps
        re-submitted addresses in the web_form interaction metadata, not
        the contact column), settled by `tests/test_eom_lead_conversion.py`
        and `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The draft claim contract models ownership separately from
        delivery: claiming flips exactly one `pending` row to `sending`
        under the built-in readiness predicate (blocked or recipient-less
        rows are never claimable), two concurrent sessions settle to one
        winner, delivery confirms `sending -> sent` only after transport
        acceptance, and the partial unique index refuses a second live
        draft per contact, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The shared funnel startup guard fails closed until migration 360
        provisions the draft table and its required columns, so a
        deployment ahead of the migration cannot admit the first-clean
        route and wedge completions on undefined_table, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured and
        `tests/test_eom_render_profile.py`.
  - [ ] A booking key belongs to one family permanently; an unsettled
        operation of either family blocks the other family's prepare; a
        completed estimate booking admits the first-clean prepare, and a
        booked outcome dominates that operation's own historical ambiguity
        marker when the other family scans it, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The review queue returns `won` leads and customer handoff approves
        from `won`, recording `from_stage='won'` in the customer-approved
        lifecycle event, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The handoff fence covers an in-flight first-clean execution through
        the shared `eom-estimate-booking:execution:<key>` namespace, settled
        by `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] A mid-execution status flip does not orphan the Calendar event: the
        first-clean completion still records `won` plus the draft row, while
        review and handoff keep rejecting the inactive contact, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] Migrations 358 and 359 follow the replay-safe concurrent
        drop-then-create ritual with rollback evidence, and migration 360 is
        additive with the single-send claim documented verbatim, settled by
        `tests/test_migrations_runner.py`.
- Reachability proof: the real FastAPI route
  `POST /api/v1/eom-funnel/leads/{contact_id}/first-clean-bookings` is
  exercised with auth headers in `tests/test_eom_lead_conversion.py`,
  asserting the JSON response and the fake CRM/Calendar calls.
- Affected surfaces: private EOM funnel API, `DatabaseCRMProvider` EOM
  booking/review/handoff methods, EOM lifecycle ledger event set, onboarding
  email template registry, review-queue and operation-key index predicates,
  new `eom_onboarding_email_drafts` table.
- Risk areas: refactor regression in the hardened booking engine,
  cross-family admission mistakes, draft double-enqueue or double-send,
  migration/index deploy safety, stage-set compatibility for review and
  handoff, response-shape compatibility for the private tracker client.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: atlas_brain/eom_api/funnel.py
  - Replaced-path behaviors: no prior first-clean route exists; the estimate
    route, its request model, and its validators are byte-identical.
  - Guard-relevant fields: bearer token digest, `X-EOM-Actor`,
    `X-EOM-Actor-ID`, `Idempotency-Key`, `contact_id`, `scheduled_start`,
    `scheduled_end`, `calendar_id`, `notes` -- all through the same
    `EOMEstimateBookingRequest` model and `_approval_key_dependency` the
    estimate route uses.
  - Caller x input shape: the same server-side tracker/office caller shape
    as estimate bookings; both routes share the RFC 3339 timestamp boundary
    and reject malformed input with 422 before CRM or Calendar calls, with
    the boundary tables parametrized over both routes.
- Boundary path/seam: `DatabaseCRMProvider` booking-family admission,
  cross-family ownership, review-queue, and handoff predicates in
  atlas_brain/services/crm_provider.py.
  - Replaced-path behaviors: booking admission previously hardcoded the
    estimate event types and `new -> estimate_booked`; the booking-key
    ownership scan previously fetched only estimate event types; the review
    queue and handoff previously admitted `('new', 'estimate_booked')`; the
    handoff fence and blocking predicates previously enumerated only the
    four estimate event types.
  - Guard-relevant fields: lead `business_context_id`, `contact_type`,
    `lead_stage`, `status`, lifecycle `operation_key`/`event_type` over the
    eight booking event types, family admission stages
    (estimate: `new`/`estimate_booked`; first clean:
    `new`/`estimate_booked`/`won`), the shared
    `eom-estimate-booking:execution:<booking_key>` advisory lock key.
  - Caller x input shape: the booking service is the only caller of the
    family-bound prepare/complete/markers and resolves them on the
    execution-scoped provider; a booking key found under another family's
    event types rejects 409 for every caller; unsettled operations of either
    family block the other family's prepare while completed estimate
    bookings admit first-clean prepare, and a booked outcome dominates that
    operation's own historical ambiguous/failed markers (same precedence
    ladder as completion writers and handoff), so a reconciled estimate with
    a stale ambiguity row cannot wedge the first clean; handoff and review
    read all eight event types and the widened stage set; the draft
    recipient resolves through the same latest-intake projection the review
    queue shows before falling back to the contact column.
- Boundary path/seam: atlas_brain/eom_api/funnel_store.py
  - Replaced-path behaviors: the shared funnel readiness guard previously
    admitted the funnel surface once contacts, lifecycle events, and
    handoffs were provisioned; a deployment missing migration 360 would
    have started, let Calendar creation succeed, and then wedged the
    first-clean completion ambiguous on undefined_table.
  - Guard-relevant fields: `to_regclass('eom_onboarding_email_drafts')`
    plus the draft columns the enqueue and claim contract require
    (`contact_id`, `operation_key`, `status`, `recipient_email`, `blocker`,
    `subject`, `body`).
  - Caller x input shape: the same enabled-funnel startup preflight in
    `main.py`/`main_eom.py`; the guard fails closed with the existing
    controlled RuntimeError until migration 360 is applied, exactly as it
    already does for the lifecycle and handoff relations; disabled funnels
    still skip the probe entirely.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: existing disabled-by-default EOM funnel
  API token config is unchanged; no new config keys are added.
- Explicit value probe: route tests use an enabled
  `EOMFunnelConfig(api_enabled=True, service_token_sha256=<generated digest>)`.
- Absent value probe: existing funnel auth tests remain the coverage for
  disabled/missing token config; the first-clean route shares the same
  dependency chain and its guard table asserts the 503 disabled path.
- Default-session/default-context probe: route tests call the isolated
  router with explicit dependency overrides; no ambient operator context is
  trusted.
- Side-effect ordering: CRM prepare must complete before Calendar create;
  CRM completion is reached only when Calendar returns the prepared
  deterministic ID; the draft insert happens inside the same transaction as
  the `won` stage update and `first_clean_booked` ledger event; draft
  send-side effects are deferred to the A3 approval surface whose claim
  statement is documented in migration 360; tests assert the call order,
  the failure branches, and the single-claim settlement.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_store.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_estimate_booking.py`
- `atlas_brain/storage/migrations/358_eom_lead_review_queue_won_stage.sql`
- `atlas_brain/storage/migrations/359_eom_booking_operation_key_index_first_clean.sql`
- `atlas_brain/storage/migrations/360_eom_onboarding_email_drafts.sql`
- `atlas_brain/templates/email/__init__.py`
- `atlas_brain/templates/email/onboarding_welcome.py`
- `plans/PR-EOM-First-Clean-Won.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migrations_runner.py`

## Mechanism

The booking engine keeps one implementation and gains a second binding. A
frozen module-level `_EOMBookingFamily` dataclass in `crm_provider.py` names
each family's four lifecycle event types, admission stages, already-booked
stage, target stage, Calendar summary prefix, and whether completion
enqueues the onboarding draft. `_ESTIMATE_BOOKING_FAMILY` reproduces current
behavior exactly; `_FIRST_CLEAN_BOOKING_FAMILY` admits
`new`/`estimate_booked`, targets `won`, and enqueues the draft. The four
public estimate methods delegate to family-parametrized internals with
unchanged signatures, and the four `*_first_clean_*` methods are the new
bindings. Event-type SQL literals become bind parameters throughout.

Booking-key ownership is global and family-scoped: prepare fetches this
key's events across ALL eight event types; a key whose events belong to the
other family rejects 409 regardless of settlement, so a key can never
migrate between families. The other-operation blocking predicate treats an
operation as blocking when it is booked in this family, ambiguous in either
family, or requested-but-unsettled in either family -- which makes an
unsettled estimate block the first clean (and vice versa) while a completed
estimate booking admits it, because completed estimate -> first clean is the
normal funnel path.

The execution-lock namespace stays `eom-estimate-booking:execution:<key>`
for BOTH families (operation keys are globally unique across contacts and
now across families). This keeps the handoff fence probe and the
single-pooled-connection execution scope from A1 byte-identical; the lock
docstring documents the cross-family namespace and the historical prefix.

Completion for the first-clean family updates `lead_stage` to `won` from
the family's completion stages, appends `first_clean_booked` with the
actual from-stage, and calls the draft-enqueue helper on the same
transaction connection: rendered subject/body from
`format_onboarding_welcome`, recipient resolved through the same
latest-intake projection the review queue shows (ingress leaves
contacts.email unchanged on re-submission, so the contact column alone can
be stale) with the contact email as fallback, and `blocker='no_email'`
with a NULL recipient when neither exists (never silently skip -- A3
surfaces the blocker like invoicing's needs_hours). The insert is
`ON CONFLICT (operation_key) DO NOTHING` with a fallback SELECT, so a
won-replay reports the existing draft id without a second row; a booked
first clean without a draft row is impossible because they commit or roll
back together. The shared funnel startup guard now also requires the
draft table and its claim columns, so a deployment ahead of migration 360
fails closed at startup instead of wedging completions on
undefined_table.

`eom_estimate_booking.py` mirrors the shape with a frozen
`_EOMBookingServiceBinding` naming the deterministic-ID prefix (`eomest`
kept for estimates, `eomfcl` for first cleans), the booked status, and the
four CRM callable names. `schedule_eom_estimate_booking` keeps its public
contract and `schedule_eom_first_clean_booking` is a sibling binding over
the same `_schedule_eom_booking` / `_run_eom_booking` internals, so the
prepared-snapshot reuse, terminal/ambiguous classification, expected-ID
verification, and completion-rejection ambiguous backstop apply to both
families without duplication.

The route is a verbatim sibling of the estimate route: same request model,
same auth and idempotency dependencies, same 201/200-idempotent envelope,
plus `onboarding_draft_id` passed through the closed result shape.

Migration 358 recreates the review-queue partial index with
`lead_stage IN ('new', 'estimate_booked', 'won')` and migration 359
recreates the operation-key index with all eight booking event types; both
use the concurrent drop-then-create ritual because a canceled concurrent
build leaves an INVALID same-named index that IF NOT EXISTS would record as
applied. Migration 358's header also documents the application-rollback
data step: old code admits only `new`/`estimate_booked`, so reverting the
application runs the documented UPDATE that returns won leads to
`estimate_booked` -- stage state only; the append-only ledger keeps the
`first_clean_booked` evidence, replays still return the booked outcome,
and the booked-operation guard still refuses a second first-clean booking.

Migration 360 creates `eom_onboarding_email_drafts` with claim ownership
modeled separately from delivery: the status ladder is
`pending -> sending -> sent` (plus `revoked`), the atomic claim documented
in the header flips exactly one `pending` row to `sending` and carries the
readiness predicate inline (`AND blocker IS NULL AND recipient_email IS
NOT NULL`, so a blocked or recipient-less draft is never claimable),
delivery is confirmed `sending -> sent` only after the transport accepts
with the draft id as the transport idempotency key, and a row stuck in
`sending` is operator reconciliation evidence rather than silently
retryable. The partial unique index covers `('pending', 'sending')` so at
most one live draft exists per contact. The two-session integration test
proves the single-winner claim, the blocked-claim refusal, and the
separate delivery confirmation.

## Intentional

- This slice does not send email and does not add the approval surface. A3
  reads the pending drafts, resolves blockers, and claims with the
  documented single-send statement; shipping the queue and its contract now
  is what lets A3 parallelize safely.
- The draft table is new instead of reusing `email_drafts` (migration 029),
  which is Gmail-reply-shaped (`gmail_message_id NOT NULL`) and cannot
  represent an unsent draft.
- The single-send contract lives in the schema and migration header rather
  than cloning the invoicing pending-drafts pattern, because the invoicing
  read-check-update path double-sends under two concurrent approvers; A2
  fixes that contract at the root for the new queue instead of propagating
  it.
- The execution-lock namespace intentionally keeps the historical
  `eom-estimate-booking:` prefix for the first-clean family. A per-family
  namespace would silently exempt first-clean executions from the existing
  handoff fence probe; byte-compatibility is the safety property, and the
  docstring now names the cross-family scope.
- The booking engine is parametrized rather than duplicated. Copying ~500
  lines of concurrency-hardened code guarantees drift between families; the
  frozen-config refactor keeps one implementation, and the untouched A1
  estimate tests prove the estimate behavior is unchanged.
- `won` leads stay `contact_type='lead'`. Customer/Site creation remains the
  office handoff's job (issue #2188); the first clean being booked does not
  make the contact a customer record.

## Deferred

- A3: the office approval queue surface (list pending drafts, resolve
  `no_email` blockers, claim-and-send with the documented statement, revoke).
- A4: the tokenized customer-facing link.
- Estimate or first-clean reschedule/cancel lifecycle and Calendar
  update/delete semantics.
- Draft content editing before approval; the enqueue snapshot is the A3
  review payload for now.
- Ambiguous-booking calendar reconciliation surface (unchanged from A1's
  deferral; the first-clean family inherits the same absorbing fail-closed
  ambiguity semantics).
- Tracker UI wiring for the first-clean booking screen.

Parking predicate: hardening narrower than one first-clean booking request,
or requiring a new subsystem beyond the lifecycle ledger and the draft
table, is parked unless it can duplicate Calendar events, double-send the
onboarding email, promote a lead incorrectly, or expose lifecycle authority
outside the private service boundary.

Parked hardening: none.

## Verification

Local environment note: torch and the pinned dependency set are unavailable
in this container, so a fixed set of full-app/MCP tests fails collection or
import locally; the identical failures reproduce on the unmodified base via
`git stash`, and the `eom-lead-pipeline` CI lane runs the full stack with
pinned requirements.

- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py atlas_brain/services/eom_estimate_booking.py atlas_brain/templates/email/__init__.py atlas_brain/templates/email/onboarding_welcome.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_migrations_runner.py` -- passed.
- ASCII scan of every touched Python file -- no non-ASCII bytes.
- `python -m pytest tests/test_eom_lead_conversion.py tests/test_migrations_runner.py -q` -- 143 passed, 1 skipped; 3 pre-existing torch-import failures.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/atlas_migration_tests python -m pytest tests/test_eom_lead_conversion_integration.py -q` -- 31 passed against disposable Postgres 16, including the first-clean lifecycle, draft claim-contract two-session proof, first-clean fence, status-flip settlement, and cross-family ownership; 3 pre-existing torch-import failures.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/atlas_migration_tests python -m pytest tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_mailbox_context_binding.py tests/test_eom_lead_ingress.py tests/test_eom_scoped_gmail_credentials.py tests/test_eom_scoped_gmail_hardening.py tests/test_migrations_runner.py -q` -- 253 passed; 12 environment-only failures (torch full-app trio in two files, MCP/pydantic version drift) reproduced identically on the unmodified base via `git stash` (212 passed there; this diff adds 41 passing tests and no new failures).
- `python -m pytest -q tests/test_audit_plan_doc.py tests/test_audit_plan_code_consistency.py tests/test_audit_pr_plan_presence.py tests/test_check_diff_budget.py` -- 103 passed.
- `python scripts/check_boundary_change_enumeration.py --base origin/main --strict` -- OK.
- `python scripts/check_deployed_config_probing.py --base origin/main --strict` -- OK.
- `python -m pytest tests/test_eom_render_profile.py::test_eom_profile_import_does_not_load_full_api_package -q` -- 1 passed; the slim profile exposes the first-clean route.
- `python -m pytest tests/test_eom_render_profile.py -q` -- 56 passed; 5 environment-only failures (llm-registry deps, receivables subprocess env) reproduced identically on an origin/main worktree.

Round-1 Codex reconciliation reverification (same environment caveats):

- `python -m pytest tests/test_eom_lead_conversion.py tests/test_migrations_runner.py -q` -- 145 passed, 1 skipped; the pre-existing torch-import trio.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/atlas_migration_tests python -m pytest tests/test_eom_lead_conversion_integration.py -q` -- 33 passed against disposable Postgres 16, adding the latest-intake recipient proof, the two-phase claim/confirm proof with blocked-claim refusal, the drafts-relation startup-guard admission case, and the cross-family mixed-marker regression; the pre-existing torch-import trio.
- `python -m pytest tests/test_eom_render_profile.py::test_shared_eom_funnel_datastore_guard_keeps_missing_relations_in_verdict tests/test_eom_render_profile.py::test_eom_profile_import_does_not_load_full_api_package -q` -- 2 passed.
- `python scripts/maturity_sweep.py atlas_brain/templates --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_templates.json --min-score 8` (with the workflow's sensitive globs) -- ratchet gate passed; `onboarding_welcome.py` scores 6, below the ratchet threshold.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 10 |
| `atlas_brain/eom_api/funnel.py` | 46 |
| `atlas_brain/eom_api/funnel_store.py` | 26 |
| `atlas_brain/services/crm_provider.py` | 689 |
| `atlas_brain/services/eom_estimate_booking.py` | 156 |
| `atlas_brain/storage/migrations/358_eom_lead_review_queue_won_stage.sql` | 48 |
| `atlas_brain/storage/migrations/359_eom_booking_operation_key_index_first_clean.sql` | 46 |
| `atlas_brain/storage/migrations/360_eom_onboarding_email_drafts.sql` | 80 |
| `atlas_brain/templates/email/__init__.py` | 5 |
| `atlas_brain/templates/email/onboarding_welcome.py` | 61 |
| `plans/PR-EOM-First-Clean-Won.md` | 560 |
| `tests/test_eom_lead_conversion.py` | 433 |
| `tests/test_eom_lead_conversion_integration.py` | 898 |
| `tests/test_eom_render_profile.py` | 6 |
| `tests/test_migrations_runner.py` | 140 |
| **Total** | **3204** |

## Cold diff reconstruction

Gaps first: no contract gaps found in the current diff. The real-Postgres
proof ran locally against disposable Postgres 16 with migration 360 applied
in the schema fixture, covering the draft transaction, claim contract, and
fence semantics the CI lane will re-run with pinned requirements.

Change-by-change reconstruction against the contract:

- The frozen `_EOMBookingFamily` config and its two instances carry every
  family-specific constant the engine needs; the union event-type sets feed
  the cross-family scans. This traces to the contract's
  parametrize-not-duplicate requirement. Citations:
  `atlas_brain/services/crm_provider.py:44`,
  `atlas_brain/services/crm_provider.py:85`,
  `atlas_brain/services/crm_provider.py:98`.
- `prepare_eom_first_clean_booking` binds the shared `_prepare_eom_booking`,
  which fetches key events across all eight event types, enforces
  cross-family key ownership and cross-family blocking, admits
  `new`/`estimate_booked`, and reports `onboarding_draft_id` on completed
  replay. Citations: `atlas_brain/services/crm_provider.py:1512`,
  `atlas_brain/services/crm_provider.py:1541`.
- The first-clean markers bind the shared ambiguous/failed internals with
  family event types as bind parameters, on the unchanged
  `eom-estimate-booking:` lock prefixes. Citations:
  `atlas_brain/services/crm_provider.py:1891`,
  `atlas_brain/services/crm_provider.py:1987`.
- `complete_eom_first_clean_booking` binds `_complete_eom_booking`, which
  updates the widened completion stages to `won`, appends
  `first_clean_booked` with the actual from-stage, and calls
  `_enqueue_eom_onboarding_email_draft` on the same transaction connection;
  the helper renders the template, snapshots the recipient, records the
  `no_email` blocker, and is replay-idempotent via
  `ON CONFLICT (operation_key) DO NOTHING` plus a fallback SELECT.
  Citations: `atlas_brain/services/crm_provider.py:2078`,
  `atlas_brain/services/crm_provider.py:2106`,
  `atlas_brain/services/crm_provider.py:2302`.
- The execution lock docstring documents the shared cross-family namespace;
  handoff fetches all eight event types, fences in-flight executions, and
  admits `won`; the review queue filters
  `lead_stage IN ('new', 'estimate_booked', 'won')`. Citations:
  `atlas_brain/services/crm_provider.py:1424`,
  `atlas_brain/services/crm_provider.py:2807`,
  `atlas_brain/services/crm_provider.py:1249`.
- The service module gains `EOMFirstCleanBooking`, the
  `_EOMBookingServiceBinding` config, the `eomfcl` deterministic-ID helper,
  and `schedule_eom_first_clean_booking` over the shared
  `_schedule_eom_booking` / `_run_eom_booking` internals; the estimate
  wrapper keeps its public contract. Citations:
  `atlas_brain/services/eom_estimate_booking.py:40`,
  `atlas_brain/services/eom_estimate_booking.py:45`,
  `atlas_brain/services/eom_estimate_booking.py:99`,
  `atlas_brain/services/eom_estimate_booking.py:255`,
  `atlas_brain/services/eom_estimate_booking.py:266`,
  `atlas_brain/services/eom_estimate_booking.py:303`.
- The route is a sibling of the estimate route with identical guards and
  envelope. Citation: `atlas_brain/eom_api/funnel.py:272`.
- The onboarding template renders a price-free, clock-promise-free welcome
  with the shared business constants and is eagerly exported. Citations:
  `atlas_brain/templates/email/onboarding_welcome.py:52`,
  `atlas_brain/templates/email/__init__.py:19`.
- Migrations 358/359 recreate the two widened partial indexes with the
  replay-safe ritual and rollback evidence; migration 360 creates the draft
  table with the documented single-send claim and one-pending-per-contact
  index. Citations:
  `atlas_brain/storage/migrations/358_eom_lead_review_queue_won_stage.sql:26`,
  `atlas_brain/storage/migrations/359_eom_booking_operation_key_index_first_clean.sql:33`,
  `atlas_brain/storage/migrations/360_eom_onboarding_email_drafts.sql:10`.
- Unit tests cover the first-clean happy path with `onboarding_draft_id`,
  shared-lock scoping, execution-scoped binding resolution, terminal
  classification, completion-rejection ambiguity, idempotent replay, the
  draft-enqueue recipient/blocker matrix, and both-route boundary
  parametrization. Citations: `tests/test_eom_lead_conversion.py:1207`,
  `tests/test_eom_lead_conversion.py:1266`,
  `tests/test_eom_lead_conversion.py:1289`,
  `tests/test_eom_lead_conversion.py:1329`,
  `tests/test_eom_lead_conversion.py:1362`,
  `tests/test_eom_lead_conversion.py:1392`,
  `tests/test_eom_lead_conversion.py:1435`,
  `tests/test_eom_lead_conversion.py:1493`.
- Real-Postgres tests cover the full first-clean lifecycle into `won` with
  the transactional draft, replay identity, review-queue visibility, and
  handoff from `won`; the no-email blocker and one-pending index; the
  two-session claim contract; the fence over an in-flight first-clean
  execution; status-flip settlement; and cross-family ownership/blocking.
  Citations: `tests/test_eom_lead_conversion_integration.py:2216`,
  `tests/test_eom_lead_conversion_integration.py:2447`,
  `tests/test_eom_lead_conversion_integration.py:2634`,
  `tests/test_eom_lead_conversion_integration.py:2743`,
  `tests/test_eom_lead_conversion_integration.py:2841`,
  `tests/test_eom_lead_conversion_integration.py:2938`.
- Migration-shape tests assert the 358/359 ritual and predicates and the
  360 schema/claim documentation. Citations:
  `tests/test_migrations_runner.py:277`,
  `tests/test_migrations_runner.py:311`,
  `tests/test_migrations_runner.py:358`.
- The slim Render profile test now asserts the first-clean route is exposed
  without loading the full API/config/reasoning stack, preserving the slim
  EOM runtime boundary. Citation: `tests/test_eom_render_profile.py:225`.
- Round-1 Codex reconciliation: the estimate wrappers (complete + both
  markers) keep their original typed keyword-only signatures and the
  first-clean siblings match; the other-operation predicate lets a booked
  outcome dominate that operation's own historical ambiguity (with the
  cross-family mixed-marker regression); the draft recipient resolves
  through the review queue's latest-intake projection with an
  ingress-shaped integration proof; the shared funnel startup guard
  requires the drafts relation and claim columns; migration 360 models
  claim ownership separately from delivery (pending -> sending -> sent,
  readiness predicate inline, live-draft unique index) with the two-phase
  proof; migration 358 documents the application-rollback data step for
  persisted won leads. Citations:
  `atlas_brain/services/crm_provider.py:1851`,
  `atlas_brain/services/crm_provider.py:1668`,
  `atlas_brain/services/crm_provider.py:2302`,
  `atlas_brain/eom_api/funnel_store.py:26`,
  `atlas_brain/storage/migrations/360_eom_onboarding_email_drafts.sql:13`,
  `atlas_brain/storage/migrations/358_eom_lead_review_queue_won_stage.sql:26`,
  `tests/test_eom_lead_conversion_integration.py:2539`.

Scope check:

- Everything changed traces to the contract: booking-family
  parametrization, first-clean route/service/provider bindings, stage and
  index widening, draft table and enqueue, template, workflow path filters,
  and tests.
- Everything the contract required appears in the diff: the `won`
  transition, cross-family rules, shared lock namespace, transactional
  draft with blocker semantics, single-send claim documentation and proof,
  review/handoff admission of `won`, replay-safe migrations, and regression
  tests.
- No declared out-of-scope module moved: no email sending, no approval
  surface, no tokenized link, no tracker Customer/Site behavior, no
  estimate-contract change, no public intake change, no NocoDB grant
  change.
