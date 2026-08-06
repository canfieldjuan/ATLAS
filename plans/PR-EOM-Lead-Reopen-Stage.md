# PR-EOM-Lead-Reopen-Stage

## Why this slice exists

Arc #2275 is now past S0/A1/A2/A3 and the end-to-end lost-lead companion
landed in Atlas #2291 plus tracker #124 plus website #102. During that slice,
the reviewer identified a real follow-up: #2293. Reopening a lead that was lost
from `estimate_booked` currently returns it to `new`, while its completed
estimate-booking lifecycle evidence remains in the ledger. The booking-family
guard then refuses another estimate booking under a different key, so the lead
is active again but one normal operator path is wedged.

This is a small, related vertical fix before the larger A4/T2 onboarding work.
T3b/T3c from tracker #54 remains operator-gated until the production linkage
audit summary proves there are no duplicate `atlas_contact_id` groups, so this
slice deliberately does not add the tracker unique index or onboarding
completion endpoint.

Diff-budget override: review feedback made this slice exceed the 400-LOC soft
cap, but the overage is indivisible from the lost/reopen boundary being fixed.
The code change, the shared restorable-stage source, the committed-generation
ordering proof, and the missing/unsafe evidence rejection proofs all validate
one transactional lifecycle seam; splitting them would publish either behavior
without its required proof or proof without the behavior it guards.

### Problem-derived contract

- Root cause: `reopen_eom_lead` treats every reopen as `lost -> new`, even
  though `lead_lost` already records the stage the lead came from. For leads
  lost from `estimate_booked`, restoring `new` disagrees with the durable
  booking ledger: the old `estimate_booked` row still means that booking family
  has already completed, so a later attempt to book a fresh estimate can be
  blocked by existing lifecycle evidence. The reopen operation is not a true
  inverse of the loss it is undoing.
- Correct fix must touch/change: the Atlas CRM transaction that reopens a lead
  must recover the latest applicable `lead_lost.from_stage` for that lead,
  order current/future loss evidence by a persisted transition generation
  established after the contact lock, validate it against the stages that are
  safe to restore, update `contacts.lead_stage` to that restored stage, and
  write the `lead_reopened` lifecycle event with `to_stage` equal to the
  restored stage. The private funnel route/service response must surface that
  stage, and tests must prove both the route response and the real-Postgres
  transaction for at least `estimate_booked` and `new`.
- Verification cleanup must touch/change: pre-existing ruff findings in files
  touched by this slice may be mechanically cleaned only when they do not alter
  runtime behavior, so the focused lint command can prove the changed files are
  clean.
- Must not change: do not change mark-lost admission (`won` remains excluded;
  #2292 owns draft/calendar teardown), do not change the reason-code vocabulary
  or website/tracker request shape, do not weaken idempotency/cross-contact key
  guards, do not alter booking-family fences, do not add customer/site creation
  or tokenized onboarding behavior, and do not touch tracker #54/T3b/T2 or
  website #59/W2 surfaces.

## Scope (this PR)

Ownership lane: eom/funnel-go-live
Slice phase: vertical slice

1. Change only reopen semantics for EOM lost leads: the restored stage comes
   from the latest `lead_lost.from_stage` instead of hard-coded `new`.
2. Add focused proof that a lead lost from `estimate_booked` reopens to
   `estimate_booked`, while a lead lost from `new` still reopens to `new`.
3. Add focused proof that reopen reads lifecycle chronology rather than UUID
   or transaction-start timestamp ordering, and that missing or unsafe
   `lead_lost.from_stage` evidence fails closed without changing the contact or
   appending `lead_reopened`.
4. Mechanically clean pre-existing lint-only issues in touched files if required
   by the focused verification command; no behavior may change outside the
   reopen path.

### Review Contract

- Acceptance criteria:
  - `DatabaseCRMProvider.reopen_eom_lead` restores the latest matching
    `lead_lost.from_stage` and records `lead_reopened.to_stage` with the same
    restored stage.
  - A lost-from-`estimate_booked` lead reopens to `estimate_booked` in a
    real-Postgres integration test; a lost-from-`new` lead still reopens to
    `new`.
  - `mark_eom_lead_lost` writes a persisted `transition_generation` after the
    contact lock, and reopen orders generated loss evidence by that generation
    before falling back to legacy timestamps.
  - A two-cycle ledger regression proves the latest loss is selected by
    committed transition generation, not random UUID sort order or PostgreSQL
    `NOW()` transaction-start timestamps.
  - Missing lost-stage evidence and unsafe lost-stage evidence both return 409
    without changing the contact or appending a `lead_reopened` lifecycle row.
  - A replay of the same reopen key remains idempotent only while the lead is
    still at the stage that key restored; if the lead is lost again, the
    existing 409 replay guard still fires.
  - The private route response reports the restored `lead_stage`, so the tracker
    and website see the same stage the CRM row now holds.
  - Mark-lost admission and `won` exclusion remain unchanged.
  - Any non-reopen edits are lint-only cleanups that remove unused locals or
    unnecessary syntax without changing SQL text, branch conditions, or runtime
    state.
- Reachability proof: the real private funnel entrypoint
  `POST /api/v1/eom-funnel/leads/{contact_id}/reopen` is exercised through the
  ASGI route test, and the observable output is the JSON `lead_stage`; the CRM
  transaction is exercised against disposable PostgreSQL, and the observable
  output/state is `contacts.lead_stage` plus the `lead_reopened` lifecycle row.
- Affected surfaces: `atlas_brain/services/crm_provider.py`
  (`reopen_eom_lead` only), the private route fake/test expectations in
  `tests/test_eom_lead_conversion.py`, and the lost/reopen real-Postgres tests
  in `tests/test_eom_lead_conversion_integration.py`.
- Risk areas: lifecycle replay correctness, restoring unsafe/stale stages,
  keeping the `won` teardown deferral intact, and preserving existing
  idempotency/cross-contact guard behavior.
- Reviewer rules triggered: R1, R2, R4, R5, R8, R10, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `DatabaseCRMProvider.reopen_eom_lead` admission/result
  boundary for lost EOM leads.
- Replaced-path behaviors: replaces hard-coded `lost -> new` with
  `lost -> <latest lead_lost.from_stage>` for the lead being reopened.
- Restorable-stage closure: the set is deliberately closed over the stages
  admitted by `mark_eom_lead_lost`, and both loss admission and reopen restore
  validation read the same code-owned `_EOM_LOST_RESTORABLE_STAGES` tuple. Any
  future stage added to one side must move through that shared source instead
  of adding a second local literal.
- Guard-relevant fields: `contacts.business_context_id`, `contact_type`,
  `lead_stage`, `status`; `eom_lead_lifecycle_events.event_type`,
  `from_stage`, `to_stage`, `operation_key`, `contact_id`.
- Caller x input shape: unchanged; existing callers still POST only
  `Idempotency-Key` plus actor headers.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no config/env/default changes.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: the row lock, operation-key guard, contact active check,
  guarded `UPDATE contacts`, and lifecycle insert stay inside the existing
  transaction; only the target stage changes.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Lead-Reopen-Stage.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

Inside the existing `reopen_eom_lead` transaction, after the contact row is
locked and proven to be an active lost EOM lead, query the latest
`lead_lost` lifecycle row for that contact by `transition_generation`, not UUID
primary-key ordering and not PostgreSQL `NOW()` timestamps. `mark_eom_lead_lost`
computes that generation after the contact lock and stores it in lifecycle
metadata. Legacy rows without the generation still fall back to the existing
timestamps. The selected row's `from_stage` is the stage the loss operation
displaced. Reopen updates `contacts.lead_stage` from `lost` to that restored
stage and records `lead_reopened` with `to_stage` equal to the same value.
Idempotent replays validate against that same restored stage before returning
200.

## Intentional

- This slice restores only stages admitted by the existing lost flow
  (`new`, `estimate_booked`). `won` remains excluded from mark-lost, so no
  `won` restore path is needed until #2292 owns onboarding-draft revoke and
  first-clean cancellation.
- This slice does not delete or void old booking lifecycle rows. Restoring
  `estimate_booked` is the least destructive inverse: the prior estimate
  booking remains true evidence and the office can continue from that stage.

## Deferred

- #2292: losing a `won` lead still needs atomic onboarding-draft revoke and
  first-clean cancellation before `won` can enter mark-lost admission.
- #2295: generation-guard stale lost/reopen replay metadata across
  lost -> reopen -> relose cycles.
- A4/T2/W2 from #2275 remain deferred until the tracker #54 T3b audit/index
  gate is safe to continue.

Parking predicate: defer hardening only when it changes a different lifecycle
owner or adds a new side-effect surface instead of proving the current
lost/reopen inverse. Under that predicate, parked hardening is none: chronology
ordering, committed transition-generation ordering, unsafe/missing evidence
rejection, and stage-set closure are all inside this slice and covered here.

## Verification

- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py` — PASS.
- `python -m pytest tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py -k "lost or reopen" -q -rs` — PASS locally: 4 passed, 2 skipped, 212 deselected. The skipped tests are real-Postgres integration tests gated on `ATLAS_MIGRATION_TEST_DATABASE_URL`; CI supplies the database lane.
- `python -m ruff check atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py` — PASS.
- `git diff --check` — PASS.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 2 |
| `atlas_brain/services/crm_provider.py` | 108 |
| `plans/PR-EOM-Lead-Reopen-Stage.md` | 209 |
| `tests/test_eom_lead_conversion.py` | 8 |
| `tests/test_eom_lead_conversion_integration.py` | 196 |
| **Total** | **523** |
