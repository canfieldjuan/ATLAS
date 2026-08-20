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
  allow either an orphaned appointment or a welcome send for a lost lead.
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
     `lost`.
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
- Must not change:
  1. The existing `new` and `estimate_booked` loss/reopen state machine,
     response shape, reasons, lifecycle evidence, and idempotency behavior.
  2. First-clean and estimate booking creation, their deterministic IDs, their
     current prepare/create/complete protocol, and any Calendar event other
     than the persisted event for the won lead being lost.
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

Max files: 9

1. Permit the existing lost-lead route to dispose of a `won` EOM lead only
   through a persisted cancellation operation that removes its exact first-clean
   event and atomically revokes its still-pending draft with the final loss.
2. Use the current append-only lifecycle ledger, database advisory locks, and
   Calendar tool rather than a new table, migration, queue, worker, or generic
   saga framework.
3. Preserve the direct pre-won loss path and make competing approval/handoff
   writers wait behind the won-loss execution decision.
4. Add focused HTTP, Calendar-boundary, and real-Postgres tests that prove the
   existing route is wired and that the state transition is safe under retries
   and the admitted concurrent writer interleavings.
5. Record the one new provider-owned `UPDATE contacts` statement in the
   existing contact-write inventory without changing the guard's admission
   policy or any other writer.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
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
  - private route + same loss key after successful completion -> closed replay;
  - draft approval or office/public handoff + active won-loss execution -> wait,
    then read the terminal revoked/lost state.

#### Boundary path/seam: Calendar delete result

- Replaced-path behaviors: no EOM path currently deletes a Calendar event; the
  new path sends one DELETE only for the prepared persisted first-clean pair.
- Guard-relevant fields: Calendar enabled/credential configuration,
  `calendar_id`, `event_id`, HTTP status, and whether a request may have been
  issued.
- Caller x input shape:
  - 200/204/404/410 -> determinate cancellation, eligible for DB completion;
  - disabled/auth/network/other API outcome -> no lost transition; durable
    request evidence permits the same operation to retry the exact delete.

### Deployed-config probing

N/A - this slice adds no setting or fallback. It uses the current Calendar
tool's deployed/default configuration and the calendar ID persisted by the
first-clean booking; it never substitutes today's configured default for that
recorded ID.

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

The service invokes the Calendar tool DELETE using that immutable pair. The
tool makes 404/410 an idempotent absence result, invalidates its cache, and
returns a closed failure result for all uncertain outcomes. A failure records
only cancellation-attempt evidence; it never changes lead/draft state. A retry
with the same Idempotency-Key reuses the prepared evidence and repeats the
idempotent DELETE.

After a determinate delete, CRM completion revalidates every prepared fact in
one PostgreSQL transaction. It revokes a pending draft (or accepts an already
revoked one), rejects `sending`/`sent`/issued-link state, changes `won` to
`lost`, and appends cancellation-complete plus lost lifecycle evidence in that
same transaction. Existing claim and customer-handoff writers acquire the same
execution advisory key before their first update, so they cannot cross the
external cancellation window.

Execution model: PostgreSQL session advisory locking is the selected
closed-surface component. One lock spans the only non-transactional Calendar
step; transaction-scoped acquisitions by claim/handoff serialize behind it.
The ledger provides restart evidence, and the fixed Calendar pair makes DELETE
replay idempotent. Assumption: Google Calendar's delete endpoint applies a
DELETE to that exact recorded `(calendar_id, event_id)` pair and its 404/410
response means that pair is absent. No new lease, queue, clock, retry worker,
or table is introduced.

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
- No generic Calendar-provider rewrite: the existing portal uses
  `CalendarTool`, so delete support is added at that canonical boundary only.
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

Parked hardening: none within the stated predicate.

## Verification

- `pytest -q tests/test_eom_lead_conversion.py` -> 215 passed (local).
- `ATLAS_MIGRATION_TEST_DATABASE_URL=<isolated-local-test-db> pytest -q
  tests/test_eom_lead_conversion_integration.py` -> 98 passed against an
  isolated temporary PostgreSQL 16 container (local; container removed after
  the run).
- Focused route, Calendar boundary, success/replay, unsafe-delivery,
  uncertain-delete, and execution-fence cases passed before the full suites.
- Standard Ruff lint passed for
  `atlas_brain/services/eom_won_lead_loss.py`; targeted `F`/`E9` lint
  passed on the other changed Python files.
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
| `atlas_brain/services/crm_provider.py` | 821 |
| `atlas_brain/services/eom_won_lead_loss.py` | 120 |
| `atlas_brain/tools/calendar.py` | 114 |
| `plans/PR-EOM-Won-Lead-Loss-Teardown.md` | 324 |
| `tests/contact_write_boundary/baseline.json` | 1 |
| `tests/test_contact_write_boundary.py` | 8 |
| `tests/test_eom_lead_conversion.py` | 211 |
| `tests/test_eom_lead_conversion_integration.py` | 372 |
| **Total** | **1983** |
