# PR-EOM-Won-Lead-Loss-Teardown

## Why this slice exists

The authorized production rehearsal for public onboarding stopped before it
created a synthetic lead. A completed rehearsal necessarily creates a `won`
lead, a pending onboarding draft, and a first-clean Calendar event. Current
Atlas deliberately refuses to lose that lead because it cannot safely remove
those two effects (Atlas issue #2292). Shipping this safety prerequisite first
lets a later rehearsal clean up its synthetic record rather than leaving live
customer, email, or calendar debris in production.

This is a `Production hardening` slice because it fixes a concrete production
safety blocker exposed by that attempted vertical proof. It does not add a new
buyer-facing flow.

The initial implementation correctly introduced a durable prepare/delete/
complete path, but review found three safety gaps in that new protocol: a
legacy pre-won `lead_lost` key could be mistaken for a won-loss key, competing
writers trusted only the lifetime lock rather than durable unfinished
cancellation evidence, and a persisted credential-relative Calendar alias
could make a later `404` refer to a different Google principal. These are
root-cause fixes to the protocol, not UI or recovery-workflow changes.

### Diff-budget exception

This slice is expected to exceed Atlas' 400-LOC soft target. The externally
visible safety guarantee is indivisible: the private route, canonical Calendar
boundary, authoritative CRM prepare/complete writer, the two existing
competing writers, and the existing contact-writer inventory proof must land
together, with real-Postgres proof of their shared lock. Splitting them would
either leave the currently blocked rehearsal blocked, or worse, expose a
partial route that can lose a lead without the atomic draft/Calendar teardown
or silently add an unreviewed `contacts` writer. The implementation is
deliberately constrained to existing ledger, advisory-lock, Calendar, and
contact-write-boundary components rather than adding a new table, worker, or
framework.

### Problem-derived contract

- Root cause: `DatabaseCRMProvider.mark_eom_lead_lost` owns only an internal
  `lead_stage` transaction. A `won` lead additionally owns an external
  first-clean Calendar event and a draft that the approval path can advance to
  `sending`; no durable prepare/cancel/complete protocol currently coordinates
  those effects. Admitting `won` to the existing transaction would therefore
  allow either an orphaned appointment or a welcome send for a lost lead. The
  first implementation of that protocol also treated a key as won-loss-owned
  based only on an allowed event-type set, treated a released execution lock as
  evidence of a settled cancellation, and treated the credential-relative
  `primary` alias as an exact Calendar identifier. `primary` is also the
  current configured booking default, so a loss-time-only rejection would make
  the newly admitted path unusable for ordinary first cleans. Those are unsafe
  defaults: old pre-won evidence, a crash/uncertain delete, or a rotated Google
  principal can make the next action describe the wrong external state.
- Correct fix must touch/change:
  1. Add a narrow, durable won-loss orchestration service that reuses the
     existing PostgreSQL lifecycle ledger and advisory-lock component: prepare
     immutable first-clean cancellation evidence, cancel the exact persisted
     Calendar event, then complete the lost transition.
  2. Add authoritative CRM-provider prepare/complete methods. The completion
     transaction must validate the prepared first-clean evidence, revoke only a
     pending onboarding draft, transition the contact from `won` to `lost`, and
     append cancellation and `lead_lost` evidence atomically. A `sending`,
     `sent`, missing, mismatched, or issued-public-link state must block with a
     caller-correctable conflict rather than silently changing it.
  3. Add the existing Calendar tool's delete operation with a closed result
     shape. It must use the persisted `(calendar_id, event_id)`, treat an
     already-absent event as an idempotent cancellation, and surface any
     uncertain external result without promoting the lead to `lost`.
  4. Fence the existing onboarding-draft claim and customer-handoff paths on
     the same contact execution lock before their first mutation. Thus an
     in-flight won-loss cancellation cannot race a draft claim or a handoff;
     after a successful completion, a claim sees `revoked` and a handoff sees
     `lost`. After acquiring that lock, each writer must also reject a durable
     `first_clean_cancellation_requested` record that lacks matching completed
     evidence, so executor exit cannot turn unfinished external work into an
     admission.
  5. Route the existing private `POST /eom-funnel/leads/{contact_id}/lost`
     entrypoint through this orchestration only when the authoritative provider
     reports a won-loss preparation. Keep its current direct provider path for
     `new` and `estimate_booked` leads.
  6. Add route, Calendar-tool, and real-Postgres integration proof for success,
     replay, send-state rejection, uncertain-delete recovery, and the shared
     execution fence.
  7. Register the required `contacts` transition writer in the existing
     contact-write-boundary inventory and update its exact provider-writer
     count. This keeps the new loss completion reviewable by the repository's
     existing writer policy rather than silently expanding its allow-list.
  8. Treat an operation key carrying a legacy/pre-won `lead_lost` row as a
     collision before any Calendar delete. A key is won-loss-owned only when
     its `lead_lost` evidence is the paired `won -> lost` terminal result.
  9. For a new first-clean booking, resolve the requested/default Calendar ID
     through the current OAuth principal before any CRM prepare write, persist
     the returned concrete ID in the existing booking metadata, and create the
     event on that same ID. At teardown, reject legacy relative `primary`
     records and preflight that the current principal can still resolve the
     persisted exact ID before treating an event 404/410 as an idempotent
     absence. This must be read-only Calendar work and reuse the existing
     metadata field; it adds no schema or configuration surface.
- Must not change:
  1. The existing `new` and `estimate_booked` loss/reopen state machine,
     response shape, reasons, lifecycle evidence, and idempotency behavior.
  2. Estimate booking behavior, deterministic IDs, and the first-clean
     prepare/create/complete transaction contract. First-clean booking gains
     only the read-only Calendar-identity preflight required to persist the
     concrete existing `calendar_id`; no new Calendar event or identity record
     is introduced.
  3. Customer/Site creation, Tracker handoff payloads, public onboarding token
     format/issuance/redemption, onboarding email copy/transport, and all
     payroll, QR/GPS, Home Base, receivables, Website, and tracker lanes.
  4. The public-product shape: no new UI, email, data field, API route, schema,
     migration, dependency, configuration setting, or background worker.
  5. Reopening a newly lost `won` lead. Restoring `won` would falsely resurrect
     a cancelled appointment and revoked draft; the existing reopen admission
     remains intentionally limited to pre-won stages.
  6. The existing `tests/unit_gate_baseline.txt` entries and unrelated
     monthly-invoice test behavior. Those environment-bound known failures are
     neither evidence for nor a dependency of won-lead loss safety.

## Scope (this PR)

Ownership lane: eom-public-onboarding-lifecycle-safety
Slice phase: Production hardening

Max files: 10

1. Permit the existing lost-lead route to dispose of a `won` EOM lead only
   through a persisted cancellation operation that removes its exact persisted
   first-clean event and atomically revokes its still-pending draft with the
   final loss; reject legacy relative Calendar IDs and reused pre-won loss keys
   before the external call. New first-clean bookings resolve and persist a
   concrete Calendar ID so normal default-configured bookings remain eligible.
2. Use the current append-only lifecycle ledger, database advisory locks, and
   Calendar tool rather than a new table, migration, queue, worker, or generic
   saga framework.
3. Preserve the direct pre-won loss path and make competing approval/handoff
   writers wait behind the won-loss execution decision, then reject durable
   incomplete cancellation evidence before their first mutation.
4. Add focused HTTP, Calendar-boundary, and real-Postgres tests that prove the
   existing route is wired and that the state transition is safe under retries
   and the admitted concurrent writer interleavings.
5. Record the one new provider-owned `UPDATE contacts` statement in the
   existing contact-write inventory without changing the guard's admission
   policy or any other writer.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_estimate_booking.py`
- `atlas_brain/services/eom_won_lead_loss.py`
- `atlas_brain/tools/calendar.py`
- `plans/PR-EOM-Won-Lead-Loss-Teardown.md`
- `tests/contact_write_boundary/baseline.json`
- `tests/test_contact_write_boundary.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

### Review Contract

1. `POST /eom-funnel/leads/{contact_id}/lost` on a won lead calls
   `eom_won_lead_loss` and returns the existing closed lost response only after
   `tests/test_eom_lead_conversion.py::test_private_mark_lead_lost_runs_won_teardown`
   observes the persisted first-clean identifiers sent to Calendar and the
   completion result.
2. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_cancels_first_clean_and_revokes_pending_draft`
   proves with a real PostgreSQL schema that one successful operation leaves
   the contact `lost`, the draft `revoked`, one cancellation ledger record,
   one `lead_lost` record, and no second Calendar delete on idempotent replay.
3. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_blocks_sending_or_sent_draft_before_calendar_delete`
   proves that each unsafe delivery state returns 409 while the contact remains
   `won`, the draft remains unchanged, and the Calendar fake receives no
   deletion call.
4. `tests/test_eom_lead_conversion.py::test_won_lead_loss_retries_an_uncertain_delete_without_false_loss`
   proves an uncertain Calendar deletion leaves the lead/draft unchanged and
   that replaying the same operation can complete exactly once after a
   determinate idempotent delete result.
5. The execution model is one Postgres session advisory lock keyed by contact
   across `prepare -> Calendar DELETE -> complete`; the draft-claim and
   handoff writers acquire the same key before their first mutation. Its
   invariant is: in every admitted interleaving, exactly one writer observes a
   terminal contact/draft state, and no writer can claim/send or finalize a
   contact after this operation commits `lost`. It is settled by
   `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_execution_fences_claim_and_handoff`.
6. `tests/test_eom_lead_conversion.py::test_calendar_delete_event_treats_an_absent_event_as_idempotent_cancellation`
   proves the Calendar boundary considers the exact persisted event absent
   after a 404/410 response, not a false successful event creation or loss.
7. `tests/test_contact_write_boundary.py::test_repository_has_exactly_the_known_insert_sites`
   and the existing contact-write checker against
   `tests/contact_write_boundary/baseline.json` prove the new provider-owned
   transition is the reviewed eleventh `UPDATE contacts` writer and the
   committed inventory exactly matches the tree.
8. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_rejects_reused_prewon_lost_key_before_calendar_delete`
   proves each finite pre-won loss stage rejects its historical key while the
   lead remains won and Calendar receives no delete.
9. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_fences_durable_unsettled_cancellation_from_claim_and_handoff`
   proves a persisted requested-but-uncompleted cancellation blocks both draft
   claim variants and customer handoff after the executor lock is no longer
   held.
10. `tests/test_eom_lead_conversion.py::test_private_first_clean_booking_resolves_and_persists_concrete_calendar_identity`
    and its real-Postgres companion prove a new default/relative first-clean
    request resolves to the Calendar API's concrete ID before CRM preparation,
    Calendar creation, and lifecycle persistence.
11. `tests/test_eom_lead_conversion.py::test_calendar_delete_event_rejects_an_unresolvable_calendar_before_delete`
    proves delete checks current access to the exact persisted calendar before
    considering an event 404/410 idempotent.
12. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_rejects_relative_calendar_identifier_before_delete`
    proves a legacy `primary` booking cannot convert a 404 from another
    credential principal into a lost lead; it is rejected before Calendar.

- Reachability proof: authenticated private funnel route -> service
  orchestration -> CRM lifecycle state + Calendar delete result; the route test
  exercises the real FastAPI entrypoint, while the integration test verifies
  the provider's persisted state through real Postgres. Calendar remains the
  one intentionally faked third-party boundary.
- Affected surfaces: existing private lost-lead route; CRM lifecycle ledger and
  onboarding-draft claim/handoff writers; Calendar tool delete boundary; EOM
  unit and integration tests; existing contact-write inventory proof.
- Risk areas: customer-facing email delivery, Calendar appointment loss,
  idempotency-key replay, crash/retry between external and DB effects,
  customer-handoff race, public onboarding token state, and legacy pre-won
  behavior.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R14.

### Boundary-change enumeration

#### Boundary path/seam: private lost-lead admission

- Replaced-path behaviors: `new`/`estimate_booked` still call the existing
  direct CRM loss writer; `won` no longer reaches that writer directly and is
  admitted only through prepared first-clean cancellation plus completion.
- Guard-relevant fields: locked EOM business/contact type/status/stage;
  Idempotency-Key ownership and replay evidence; first-clean booking operation,
  persisted calendar ID/event ID; onboarding draft status and ID; active public
  onboarding token state; Calendar delete result.
- Caller x input shape:
  - private route + valid pre-won lead -> preserved direct loss;
  - private route + won/pending-draft/booked-event -> prepared cancellation;
  - private route + won/sending-or-sent/missing-or-mismatched evidence -> 409,
    no Calendar call and no loss mutation;
  - private route + won/reused pre-won `lead_lost` key -> 409, no Calendar call
    and no loss mutation;
  - first-clean booking + configured/requested relative Calendar ID -> one
    read-only identity resolution, then CRM prepare/create/complete only with
    the returned concrete ID;
  - private route + won/relative `primary` Calendar evidence -> 409, no
    Calendar call and no loss mutation;
  - private route + same loss key after successful completion -> closed replay;
  - draft approval or office/public handoff + active won-loss execution -> wait,
    then read the terminal revoked/lost state;
  - draft approval or office/public handoff + requested-but-uncompleted
    cancellation evidence after executor exit -> 409 before any mutation.

#### Boundary path/seam: Calendar delete result

- Replaced-path behaviors: no EOM path currently deletes a Calendar event; the
  new path sends one DELETE only for the prepared persisted first-clean pair.
- Guard-relevant fields: Calendar enabled/credential configuration, requested
  and resolved `calendar_id`, persisted `calendar_id`, `event_id`, HTTP status,
  and whether a request may have been issued.
- Caller x input shape:
  - new first-clean `primary`/configured alias -> Calendar list identity lookup
    returns a concrete ID before prepare/create/complete;
  - accessible persisted concrete Calendar ID + 200/204/404/410 -> the
    idempotent delete result, eligible for DB completion;
  - credential-relative `primary` persisted Calendar ID -> reject before DELETE
    because a later credential principal makes a not-found result ambiguous;
  - unresolvable/mismatched persisted Calendar ID -> no DELETE and no lost
    transition, because its not-found result cannot prove the old event absent;
  - disabled/auth/network/other API outcome -> no lost transition; durable
    request evidence permits the same operation to retry the exact delete.

### Deployed-config probing

N/A - this slice adds no setting or fallback. It resolves the current Calendar
tool's deployed/default ID to the API's concrete ID while booking a first clean,
persists that resolved value, and never substitutes today's configured default
for an already-recorded ID.

- Side-effect ordering: all EOM/contact/draft/token admissions and the durable
  cancellation-request write happen before Calendar DELETE. The contact
  `won -> lost`, draft `pending -> revoked`, cancellation completion evidence,
  and `lead_lost` evidence happen only in the post-delete database transaction.

## Mechanism

`eom_won_lead_loss` takes the existing contact-keyed PostgreSQL advisory lock
for the whole external execution. It asks the authoritative CRM provider to
prepare the request. For a pre-won lead, preparation deliberately selects the
unchanged direct writer. For a won lead, preparation locks and validates the
contact, booked first-clean ledger row, pending/revoked draft, and operation
key; it writes append-only `first_clean_cancellation_requested` evidence with
the persisted Calendar pair and returns that pair.

Before a new first-clean CRM prepare, the booking service resolves the selected
Calendar ID through the current OAuth principal. It persists and creates only
against that concrete returned ID. The loss service invokes DELETE using that
immutable pair only when its persisted Calendar ID is concrete rather than
credential-relative. The Calendar tool first verifies that the current
principal can resolve that exact ID; only then can 404/410 mean the exact event
is already absent. A failure records only cancellation-attempt evidence; it
never changes lead/draft state. A retry with the same Idempotency-Key reuses the
prepared evidence and repeats the idempotent DELETE. A legacy `primary` record
is a conflict before DELETE, not evidence that a later credential's primary
calendar is absent.

After a determinate delete, CRM completion revalidates every prepared fact in
one PostgreSQL transaction. It revokes a pending draft (or accepts an already
revoked one), rejects `sending`/`sent`/issued-link state, changes `won` to
`lost`, and appends cancellation-complete plus lost lifecycle evidence in that
same transaction. Existing claim and customer-handoff writers acquire the same
execution advisory key before their first update and reject any
requested-but-uncompleted cancellation after acquiring it, so neither an
in-flight executor nor its crash/uncertain-result residue can cross the
external cancellation window.

Execution model: PostgreSQL session advisory locking is the selected
closed-surface component. One lock spans the only non-transactional Calendar
step; transaction-scoped acquisitions by claim/handoff serialize behind it.
The ledger provides restart evidence, and the fixed Calendar pair makes DELETE
replay idempotent. Assumption: after Calendar's identity lookup confirms access
to the exact recorded calendar, Google Calendar's delete endpoint applies a
DELETE to that `(calendar_id, event_id)` pair and its 404/410 response means
that pair is absent. No new lease, queue, clock, retry worker, or table is
introduced.

The resulting `UPDATE contacts SET lead_stage = 'lost'` is still inside the
canonical `DatabaseCRMProvider`, the only existing allowed contact-writer
module. The committed writer inventory and exact-count test change alongside
that statement, so the write cannot appear as an unreviewed exception.

## Intentional

- No attempt is made to reopen a loss from `won`; re-opening cannot truthfully
  restore the cancelled appointment and revoked draft.
- `sending` and `sent` remain hard conflicts. A stale-send recovery already
  exists and must be reconciled before an operator asks to lose this lead;
  this PR does not reinterpret delivery evidence.
- The external Calendar call is not placed inside a database transaction.
  Holding an existing advisory execution lock plus append-only prepare evidence
  gives a retry-safe boundary without a long database transaction or a new
  durable-workflow subsystem.
- The provider's existing direct `mark_eom_lead_lost` keeps excluding `won`.
  Only the routed orchestration can establish the required external proof; this
  prevents a future internal caller from accidentally bypassing it.
- New first-clean rows persist a concrete Calendar ID obtained through the
  active OAuth principal. Existing first-clean rows with the relative `primary`
  alias are deliberately not deleted by this route. Rejecting them before
  Calendar is safer than assuming a rotated refresh token still identifies the
  historical account.
- No generic Calendar-provider rewrite: the existing portal uses
  `CalendarTool`, so identity resolution and delete verification stay at that
  canonical boundary only.
- The contact-write guard's policy does not broaden: only its reviewed
  inventory and exact count acknowledge the new statement in the already
  approved provider module.

## Deferred

Parking predicate: broader reconciliation UX, automated retry scheduling,
Calendar-event discovery beyond the immutable booking evidence, and any new
customer-facing wording are parked unless required to prevent an unsafe loss
transition in this route.

- A later operator-facing reconciliation/status UI can surface durable failed
  cancellation attempts. This slice leaves the existing 409/502 response and
  ledger evidence as the operational recovery path.
- A later dedicated onboarding lifecycle slice can define whether a fully sent
  welcome followed by a cancelled first clean has a supported customer-service
  disposition; this PR blocks it safely.
- A later operator reconciliation flow can repair legacy first-clean rows that
  still carry `primary`; this slice intentionally refuses their irreversible
  delete rather than guessing which historical Calendar principal owned them.

Parked hardening: none within the stated predicate.

## Verification

- `pytest -q tests/test_eom_lead_conversion.py` -> 219 passed (local; one
  pre-existing `pynvml` deprecation warning).
- `ATLAS_MIGRATION_TEST_DATABASE_URL=<isolated-local-test-db> pytest -q
  tests/test_eom_lead_conversion_integration.py` -> 104 passed against an
  isolated temporary PostgreSQL 16 container (local; container removed after
  the run).
- Focused route, Calendar boundary, success/replay, unsafe-delivery,
  uncertain-delete, execution-fence, reused-key, relative-calendar, and
  concrete-calendar-identity cases passed before the full suites.
- Full `ruff check` on the changed files reports only four pre-existing `F841`
  violations in unchanged `atlas_brain/tools/calendar.py` blocks inherited from
  the base. The scoped check with that known baseline code excluded passes; no
  new lint finding is suppressed.
- `python -m compileall -q ...` and `git diff --check` -> passed.
- `pytest -q tests/test_contact_write_boundary.py` -> 65 passed (local).
- `python scripts/check_contact_write_boundary.py --baseline
  tests/contact_write_boundary/baseline.json` -> passed; 47 contact writes
  are inside approved modules or recorded in the reviewed inventory (local).
- Pending before push: the wrapper-owned Atlas local PR review, which will run
  through `scripts/push_pr.sh` exactly once with the final PR body and its
  isolated unit-gate database environment.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 12 |
| `atlas_brain/services/crm_provider.py` | 927 |
| `atlas_brain/services/eom_estimate_booking.py` | 87 |
| `atlas_brain/services/eom_won_lead_loss.py` | 120 |
| `atlas_brain/tools/calendar.py` | 235 |
| `plans/PR-EOM-Won-Lead-Loss-Teardown.md` | 415 |
| `tests/contact_write_boundary/baseline.json` | 1 |
| `tests/test_contact_write_boundary.py` | 8 |
| `tests/test_eom_lead_conversion.py` | 380 |
| `tests/test_eom_lead_conversion_integration.py` | 734 |
| **Total** | **2919** |
