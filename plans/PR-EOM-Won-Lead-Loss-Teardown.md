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
complete path, but review found six safety gaps in that new protocol: a legacy
pre-won `lead_lost` key could be mistaken for a won-loss key, competing writers
trusted only the lifetime lock rather than durable unfinished cancellation
evidence, generic provider contact-status writers could bypass that fence, a
persisted credential-relative Calendar alias could make a later `404` refer to
a different Google principal, a second key could start work while the first
key remained unsettled, and the deployed NocoDB role could directly change a
contact's `status` without the provider fence. The direct-NocoDB hard-delete
subclaim is overstated: the immutable lifecycle foreign key is `ON DELETE
RESTRICT`; status mutation is nevertheless reachable and unsafe. These are
    root-cause fixes to the protocol and its database boundary, not UI or
    recovery-workflow changes. A later review also found that a completed
    first-clean request explicitly submitted as `primary` reaches a live
    identity lookup before its stored booked replay, allowing a Calendar outage
    to turn a completed idempotency key into an error. The EOM pipeline
    workflow likewise omits the new won-loss service and migration from both
    path filters, so a future isolated change could skip its safety suite.

### Diff-budget exception

This slice is expected to exceed Atlas' 400-LOC soft target. The externally
visible safety guarantee is indivisible: the private route, canonical Calendar
boundary, authoritative CRM prepare/complete writer, competing draft/handoff
and generic status writers, and the existing contact-writer inventory proof
must land together, with real-Postgres proof of their shared lock. Splitting
them would either leave the currently blocked rehearsal blocked, or worse,
expose a partial route that can lose a lead without the atomic draft/Calendar
teardown or silently add an unreviewed `contacts` writer. The implementation
is deliberately constrained to existing ledger, advisory-lock, Calendar, and
    contact-write-boundary components plus one additive direct-SQL trigger. It
adds no table, worker, queue, generic saga framework, or product surface.

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
  `primary` alias as an exact Calendar identifier. It additionally checked
  cancellation state only under the caller's own operation key, so a different
  key could start a second external cancellation while the first remained
  unresolved. `primary` is also the
  current configured booking default, so a loss-time-only rejection would make
  the newly admitted path unusable for ordinary first cleans. Those are unsafe
  defaults: old pre-won evidence, a crash/uncertain delete, or a rotated Google
  principal can make the next action describe the wrong external state. The
  generic `delete_contact` archive and `update_contact` status writers also
  mutate a won contact without taking the shared execution lock or consulting
  that durable evidence; either can make completion reject only after the
  external event has been removed. The direct `atlas_nocodb` role has a
  column-level `status` grant and bypasses those provider methods. Its direct
  delete is already blocked by the existing lifecycle FK, but its direct status
  write can still produce the same completion-after-delete failure.
  The repair pass must also preserve every existing lifecycle-table consumer:
  its generic durable-fence probe currently orders by `lifecycle_sequence`, a
  column introduced only by migration 363, even though valid integration
  schemas use the base migration 351 table. Because the probe only decides
  whether any unresolved record exists, that newer ordering is neither needed
  nor safe. Finally, `delete_contact` bypasses its provider's injected pool for
  the global pool; the resulting real-PostgreSQL proof must monkeypatch a
  first-party storage module even when the provider was deliberately given an
  isolated pool, which the maturity ratchet correctly rejects. The Calendar
  delete proof also becomes stale if its first identity lookup succeeds under
  one OAuth access token, a DELETE returns 401, and the forced refresh reloads
  a rotated principal from the token store: a retry 404/410 under that new
  principal means neither absence nor a safe completion unless the exact
  persisted calendar is re-resolved first. Finally, migration 386 installs a
  persistent SECURITY DEFINER function and trigger but needs an explicit
  forward-only operational policy so a rollback cannot strand unresolved
  cancellation evidence behind a retained database fence. Even after the
  401-specific repair, the first identity lookup and first DELETE independently
  acquire authorization headers; a near-expiry refresh between them can reload
  a different principal, so the Calendar proof must be bound to the exact
    header that DELETE uses. The initial NocoDB-only trigger also leaves the
    operational import and portal-sync scripts outside the durable fence even
    though their matched-row updates can change a won lead's `contact_type` or
    `status`, both facts canonical completion requires after the external
    delete. A completed same-key first-clean booking submitted as `primary`
    still takes the live identity branch before its stored booked outcome, so a
    Calendar outage can break replay without any unfinished work. That repair
    initially persists only the resolved concrete ID, however, so it cannot
    prove that an original completed request used `primary`: a concrete booking
    could change its retry to `primary` and bypass the immutable Calendar-ID
    comparison. Finally, the EOM workflow path filters do not name the new
    won-loss service or migration, so their tests can be skipped by a future
    isolated change.
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
  10. At the generic provider status-mutation boundary, acquire the same
      contact execution lock and assert the durable cancellation fence before
      the first `UPDATE contacts`. This must cover both `delete_contact`'s
      archive update and `update_contact` requests that contain `status`, so
      every caller of those provider methods is protected without changing an
      MCP command signature, response shape, or authorization model.
  11. Before a currently won lead can prepare a new loss operation, reject a
      requested-but-uncompleted cancellation owned by a different operation
      key. The original key remains the sole retry owner; the check must use
      the existing contact lock and append-only ledger before any Calendar
      call, not a key-local lookup or a cleanup side effect.
  12. Add one additive database trigger that rejects every direct SQL
      `status`/`contact_type` update or delete of a won EOM lead with
      requested-but-uncompleted cancellation evidence before that mutation. The
      trigger must safely read the protected ledger, leave canonical Atlas
      completion and ordinary non-state direct edits intact, and leave existing
      role grants and table/column shape unchanged.
  13. Make the durable cancellation-fence query use only columns present in
      the base lifecycle migration. It needs only an unresolved-record
      predicate, so it must not depend on `lifecycle_sequence` or expand a
      fixture schema merely to make a generic status/archive mutation work.
  14. Make `delete_contact` acquire its store through the provider's existing
      `_get_pool` accessor, just like the other provider mutations. The
      accessor must retain its configured-global fallback, while an explicitly
      injected provider pool stays intact so real-PostgreSQL proof does not
      mock a first-party storage dependency.
  15. Bind every `CalendarTool.delete_event` exact-calendar proof to the same
      authorization header that makes DELETE. After a DELETE 401, repeat that
      paired refresh/identity/delete sequence. A refreshed-identity failure or
      mismatch must remain a non-success result, and no independently acquired
      header may turn an earlier identity proof into a 404/410 completion.
  16. Record migration 386 as a forward-only fence: before an application
      downgrade, resolve every requested-but-uncompleted cancellation through
      the current protocol; retain its immutable lifecycle evidence and do not
      automatically remove the trigger or SECURITY DEFINER function. A
      destructive database rollback, if ever separately approved after that
      zero-unresolved precondition, must drop the trigger before its function
      and still preserve the ledger evidence.
  17. Make migration 386 enforce its existing unresolved-cancellation predicate
      for every direct SQL caller that updates `status` or `contact_type`, or
      deletes a won EOM lead. The canonical completion changes only
      `lead_stage`, and ordinary non-state direct edits remain outside the
      trigger, so this closes the evidenced import/portal bypass without a
      role-grant, script, or lifecycle-schema redesign.
  18. For a completed same-key first-clean booking that was explicitly submitted
      as `primary`, normalize that request only to its persisted concrete
      Calendar ID for the existing immutable-payload comparison, then return
      the stored booked/draft replay before the service can call Calendar. An
      unfinished request must retain its current identity preflight, and a
      different start/end/notes/event payload must still conflict.
  19. Enroll `eom_won_lead_loss.py` and migration 386 in both existing EOM
      pipeline path filters, and prove that enrollment in the already-run EOM
      unit test file. Keep the existing workflow job and its test command
      unchanged.
  20. Persist the original requested Calendar identifier in the existing
      first-clean request metadata before the resolved concrete target replaces
      it. Normalize a completed `primary` retry only when that immutable field
      proves the original request used `primary`; a concrete-original booking
      retried as `primary` must conflict before any Calendar call. Legacy rows
      without that field retain the existing identity-preflight path.
- Must not change:
  1. The existing `new` and `estimate_booked` loss/reopen state machine,
     response shape, reasons, lifecycle evidence, and idempotency behavior.
  2. Estimate booking behavior, deterministic IDs, and the first-clean
     prepare/create/complete transaction contract. First-clean booking gains
     only the read-only Calendar-identity preflight required to persist the
     concrete existing `calendar_id`. A completed exact-key `primary` replay
     returns its stored result without another Calendar call; no new Calendar
     event or identity record is introduced.
  3. Customer/Site creation, Tracker handoff payloads, public onboarding token
     format/issuance/redemption, onboarding email copy/transport, and all
     payroll, QR/GPS, Home Base, receivables, Website, and tracker lanes.
  4. The public-product shape: no new UI, email, API field or route,
     table/column shape, dependency, configuration setting, or background
     worker. The existing opaque lifecycle metadata may retain the original
     Calendar identifier required to prove a replay invariant; it is not a
     schema, role-grant, or product-surface change.
  5. Reopening a newly lost `won` lead. Restoring `won` would falsely resurrect
     a cancelled appointment and revoked draft; the existing reopen admission
     remains intentionally limited to pre-won stages.
  6. The existing `tests/unit_gate_baseline.txt` entries and unrelated
     monthly-invoice test behavior. Those environment-bound known failures are
     neither evidence for nor a dependency of won-lead loss safety.
  7. Generic contact updates that do not request `status`, and all MCP command
     signatures, copy, tenant checks, and authorization behavior. The provider
     fence is the shared mutation boundary; this slice does not create a
     second MCP-specific lifecycle policy.
  8. The maturity-sweep baseline and its rules. This repair removes the
     newly-introduced internal mocks instead of accepting or recalibrating
     them, and it does not alter any required-check policy, workflow job, or
     workflow test command.
  9. Normal single-credential Calendar deletion semantics. The extra identity
     proof runs only after a rejected DELETE has forced credential refresh; it
     does not broaden accepted response codes or add another external action
     on the ordinary successful/absent path.
  10. The import and portal-sync scripts, their matching rules, data payloads,
      receipts, and command interfaces. The database boundary—not a duplicate
      script-specific protocol—must reject only a conflicting state mutation
      while durable cancellation evidence is unresolved.

## Scope (this PR)

Ownership lane: eom-public-onboarding-lifecycle-safety
Slice phase: Production hardening

Max files: 12

1. Permit the existing lost-lead route to dispose of a `won` EOM lead only
   through a persisted cancellation operation that removes its exact persisted
   first-clean event and atomically revokes its still-pending draft with the
   final loss; reject legacy relative Calendar IDs, reused pre-won loss keys,
   and a different key's unresolved cancellation before the external call. New
   first-clean bookings resolve and persist a concrete Calendar ID so normal
   default-configured bookings remain eligible.
2. Use the current append-only lifecycle ledger, database advisory locks, and
   Calendar tool rather than a new table, queue, worker, or generic saga
   framework. One additive direct-SQL trigger migration is required because
   direct database callers bypass the provider boundary.
3. Preserve the direct pre-won loss path and make competing approval/handoff
   and generic status/archive writers wait behind the won-loss execution
   decision, then reject durable incomplete cancellation evidence before their
   first mutation. Direct SQL `status`/`contact_type`/delete paths receive the
   equivalent durable fence at the database boundary.
4. Add focused HTTP, Calendar-boundary, workflow-enrollment, and real-Postgres
   tests that prove the existing route is wired and that the state transition is
   safe under retries and the admitted concurrent writer interleavings.
5. Record the one new provider-owned `UPDATE contacts` statement in the
   existing contact-write inventory without changing the guard's admission
   policy or any other writer.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_estimate_booking.py`
- `atlas_brain/services/eom_won_lead_loss.py`
- `atlas_brain/storage/migrations/386_eom_won_loss_nocodb_fence.sql`
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
4. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_rejects_second_key_while_cancellation_is_unsettled`
   proves an uncertain Calendar deletion leaves the lead/draft unchanged and
   that replaying the same operation can complete exactly once after a
   determinate idempotent delete result.
5. The execution model is one Postgres session advisory lock keyed by contact
   across `prepare -> Calendar DELETE -> complete`; the draft-claim, handoff,
   and generic status/archive writers acquire the same key before their first
   mutation. Its invariant is: in every admitted interleaving, no writer can
   mutate the contact/draft while Calendar cancellation is in flight; after
   completion, existing terminal-state rules decide its own outcome. It is
   settled by
   `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_execution_fences_claim_handoff_and_status_writers`.
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
13. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_fences_durable_cancellation_from_generic_contact_status_writes`
    proves both provider-owned archive and generic status updates reject a
    requested-but-uncompleted cancellation before changing the contact.
14. `tests/test_eom_lead_conversion_integration.py::test_won_lead_loss_rejects_second_key_while_cancellation_is_unsettled`
    proves a second operation key is rejected after the first key records an
    uncertain Calendar delete, before a second Calendar call; retry of the
    original key remains the only admitted recovery.
15. `tests/test_eom_lead_conversion_integration.py::test_nocodb_cannot_mutate_won_lead_with_unsettled_cancellation`
    proves, through real NocoDB and ordinary application database roles plus
    the applied additive migration, that direct completion-sensitive mutations
    reject while an EOM won-loss cancellation is unsettled, while ordinary
    non-state edits and canonical application completion remain unaffected.
16. `tests/test_eom_lead_conversion_integration.py::test_completed_explicit_primary_first_clean_replay_skips_identity_lookup`
    proves a completed explicit-`primary` first-clean key returns its stored
    result through a simulated Calendar identity outage, without another
    identity lookup or event creation, while a changed immutable payload still
    returns 409.
17. `tests/test_eom_lead_conversion.py::test_eom_lead_pipeline_workflow_enrolls_won_loss_runtime_paths`
    proves both the pull-request and main-push filters enroll the won-loss
    service and migration, while the workflow keeps running its existing EOM
    test command.
18. `tests/test_eom_lead_conversion_integration.py::test_completed_concrete_first_clean_rejects_primary_replay`
    proves the request metadata retains a concrete original Calendar ID and a
    same-key `primary` retry returns 409 before identity resolution or event
    creation.

- Reachability proof: authenticated private funnel route -> service
  orchestration -> CRM lifecycle state + Calendar delete result; the route test
  exercises the real FastAPI entrypoint, while the integration test verifies
  the provider's persisted state through real Postgres. Calendar remains the
  one intentionally faked third-party boundary.
- Affected surfaces: existing private lost-lead route; CRM lifecycle ledger;
  onboarding-draft claim/handoff and generic contact-status writers; Calendar
  tool delete boundary; EOM unit and integration tests; EOM pipeline path
  enrollment; existing contact-write inventory proof.
- Risk areas: customer-facing email delivery, Calendar appointment loss,
  idempotency-key replay, crash/retry between external and DB effects,
  customer-handoff and generic contact-status races, public onboarding token
  state, and legacy pre-won behavior.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R14.

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
  - private route + won/different operation key while another cancellation is
    requested but not completed -> 409, no second Calendar call and no loss
    mutation; only the original key can retry;
  - first-clean booking + configured/requested relative Calendar ID -> one
    read-only identity resolution, then CRM prepare/create/complete only with
    the returned concrete ID;
  - completed first-clean booking + exact same key explicitly resubmitted as
    `primary` -> immutable stored replay, no Calendar identity lookup or event
    creation; a changed immutable payload still conflicts;
  - private route + won/relative `primary` Calendar evidence -> 409, no
    Calendar call and no loss mutation;
  - private route + same loss key after successful completion -> closed replay;
  - draft approval or office/public handoff + active won-loss execution -> wait,
    then read the terminal revoked/lost state;
  - draft approval or office/public handoff + requested-but-uncompleted
    cancellation evidence after executor exit -> 409 before any mutation.
  - generic archive or status update + requested-but-uncompleted cancellation
    evidence -> 409 before `UPDATE contacts`; all other generic update fields
    retain their existing path.
  - direct SQL status/contact-type update or delete + requested-but-uncompleted
    cancellation evidence for a won EOM lead -> database exception before the
    direct mutation; ordinary non-state direct CRM fields retain their existing
    path.

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
key. Before it accepts a new key, it rejects any different operation key with
requested-but-uncompleted cancellation evidence for that contact; it writes
append-only `first_clean_cancellation_requested` evidence with the persisted
Calendar pair and returns that pair.

Before a new or unfinished first-clean CRM prepare, the booking service resolves
the selected Calendar ID through the current OAuth principal. It persists and
creates only against that concrete returned ID, while the existing request
metadata retains the original requested identifier. A completed exact-key
request normalizes `primary` to the immutable stored ID only when that metadata
proves the original request used `primary`; a concrete-original request changed
to `primary` conflicts before Calendar. The admitted primary replay returns its
persisted result without contacting Calendar; different immutable fields still
conflict. The loss service invokes DELETE using that
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
same transaction. Existing claim and customer-handoff writers, plus generic
provider `status`/archive writes, acquire the same execution advisory key
before their first update and reject any requested-but-uncompleted cancellation
after acquiring it, so neither an in-flight executor nor its
crash/uncertain-result residue can cross the external cancellation window.

Direct SQL callers are not callers of the provider, so an additive database
trigger gives their direct `status`/`contact_type`/delete paths the same durable
admission. It securely consults the append-only evidence and rejects only a won
EOM contact with an unresolved cancellation. Direct hard deletion is already
disallowed by the lifecycle
foreign key; the trigger makes the intended cancellation fence explicit and
blocks status before completion could observe a changed contact. Atlas's
canonical completion runs under its application session and remains the sole
writer that can settle the prepared operation.

The durable-fence predicate is intentionally compatible with the lifecycle
table as created by migration 351: it only asks whether an unresolved request
exists and orders, when needed, by base-table timestamps and ID rather than the
later migration-363 sequence. `delete_contact` likewise takes its database
handle from the provider's established accessor, so an explicitly supplied
transaction-capable pool is honored while production still falls back to the
configured global pool.

The Calendar identity proof must belong to the credential that makes the
decision. Its exact-calendar lookup and DELETE use one acquired header; a
DELETE 401 repeats that paired sequence with a forced refresh. Only then can a
retry 404/410 mean that the persisted event is absent. A failed or mismatched
lookup leaves the durable cancellation unresolved. Migration 386 is
forward-only: the current protocol first settles every requested cancellation
before an application downgrade, while the trigger/function and append-only
evidence remain in place. There is no automatic destructive rollback. If a
separately approved database recovery is required after the zero-unresolved
precondition, it drops the trigger before its function and preserves the
lifecycle ledger.

The same migration is the narrow boundary for every direct SQL caller, not
only the NocoDB login: while a won EOM lead has requested-but-uncompleted
cancellation evidence, it rejects only `status`/`contact_type` updates and
deletes. The canonical completion updates `lead_stage` and inserts completion
evidence in its existing transaction, while ordinary notes/tags/contact-detail
edits are not blocked. This lets the import and portal-sync scripts retain
their existing commands and matching logic without allowing a post-Calendar
state mutation to strand the protocol.

Execution model: PostgreSQL session advisory locking is the selected
closed-surface component. One lock spans the only non-transactional Calendar
step; transaction-scoped acquisitions by claim/handoff/status writers serialize
behind it. The ledger provides restart evidence, and the fixed Calendar pair
makes DELETE replay idempotent. Assumption: after Calendar's identity lookup
confirms access to the exact recorded calendar, Google Calendar's delete
endpoint applies a DELETE to that `(calendar_id, event_id)` pair and its
404/410 response means that pair is absent. No new lease, queue, clock, retry
worker, or table is introduced.

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
- No MCP command rewrite: all archive and generic-status callers meet the same
  provider-owned fence, so command signatures, response shape, tenant checks,
  and authorization stay unchanged.
- No direct-writer capability rewrite: ordinary direct CRM edits retain their
  existing grants and behavior. The additive trigger is limited to direct
  `status`/`contact_type`/delete mutations of a won EOM lead with unresolved
  cancellation evidence; it adds neither a role membership nor a new bypass
  token.
- Migration 386 is intentionally not rolled back with an application deploy.
  The current build must reconcile unresolved cancellation records first; the
  trigger/function remain intact and immutable lifecycle evidence is retained.
  A destructive database recovery is a separately approved, trigger-before-
  function operation only after that zero-unresolved condition.
- The generalized trigger does not turn into a direct-writer framework: it
  guards only the three existing completion-sensitive mutation forms and does
  not change script-specific behavior, credentials, grants, or ordinary direct
  contact-detail edits.
- The EOM pipeline workflow keeps its existing job and test command. Its two
  path filters merely name the won-loss service and migration so an isolated
  safety change cannot skip the already-selected tests.
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

- `pytest -q tests/test_eom_lead_conversion.py` -> 224 passed (local; one
  pre-existing `pynvml` deprecation warning).
- `ATLAS_MIGRATION_TEST_DATABASE_URL=<isolated-local-test-db> pytest -q
  tests/test_eom_lead_conversion_integration.py` -> 107 passed against an
  isolated temporary PostgreSQL 16 container (local; container removed after
  the run).
- Focused route, Calendar boundary, success/replay, unsafe-delivery,
  uncertain-delete/second-key, execution-fence, direct-NocoDB fence,
  reused-key, relative-calendar, concrete-calendar-identity, and
  generic-status-fence cases passed before the full suites.
- Full `ruff check` on the changed files reports only four pre-existing `F841`
  violations in unchanged `atlas_brain/tools/calendar.py` blocks inherited from
  the base. The scoped check with that known baseline code excluded passes; no
  new lint finding is suppressed.
- `python -m compileall -q ...` and `git diff --check` -> passed.
- `pytest -q tests/test_contact_write_boundary.py` -> 65 passed (local).
- `python scripts/check_contact_write_boundary.py --baseline
  tests/contact_write_boundary/baseline.json` -> passed; 47 contact writes
  are inside approved modules or recorded in the reviewed inventory (local).
- The EOM lead-pipeline integration fixtures that stop at migration 352 must
  continue to archive without a `lifecycle_sequence` lookup failure; the
  existing concurrent-archive and delivery-replay cases prove that boundary.
- The maturity sweep must remain at the existing storage baseline. The two
  won-loss real-PostgreSQL tests use injected provider pools directly rather
  than monkeypatching `atlas_brain.storage.database.get_db_pool`.
- Every DELETE header must be the one that made the exact-calendar proof. Unit
  proof covers a header refresh between initial identity and DELETE, a DELETE
  401 refresh, a failed refreshed identity (no second DELETE/no completion),
  and a matching refreshed identity (retry 404/410 may be absent).
- The direct-NocoDB migration proof must show that current-protocol completion
  clears the unresolved predicate before ordinary NocoDB status edits resume;
  that is the executable counterpart of the forward-only recovery policy.
- The same real-PostgreSQL proof must show that an ordinary application-role
  direct `status`/`contact_type` mutation is rejected both during a live
  cancellation and after an uncertain executor exit, then resumes only after
  current-protocol completion.
- The completed explicit-`primary` same-key proof must pass with a Calendar
  identity failure fake and show no replay-time lookup/create; its changed-notes
  variant must still return the existing conflict. A concrete-original booking
  retried as `primary` must likewise conflict before lookup/create. The EOM unit
  suite also pins the service and migration in both pipeline path filters.
- Pending before push: the wrapper-owned Atlas local PR review, which will run
  through `scripts/push_pr.sh` exactly once with the final PR body and its
  isolated unit-gate database environment.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 4 |
| `atlas_brain/eom_api/funnel.py` | 12 |
| `atlas_brain/services/crm_provider.py` | 1060 |
| `atlas_brain/services/eom_estimate_booking.py` | 96 |
| `atlas_brain/services/eom_won_lead_loss.py` | 120 |
| `atlas_brain/storage/migrations/386_eom_won_loss_nocodb_fence.sql` | 68 |
| `atlas_brain/tools/calendar.py` | 284 |
| `plans/PR-EOM-Won-Lead-Loss-Teardown.md` | 687 |
| `tests/contact_write_boundary/baseline.json` | 1 |
| `tests/test_contact_write_boundary.py` | 8 |
| `tests/test_eom_lead_conversion.py` | 551 |
| `tests/test_eom_lead_conversion_integration.py` | 1180 |
| **Total** | **4071** |
