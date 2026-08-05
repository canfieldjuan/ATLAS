# PR-EOM-Lead-Lost

## Why this slice exists

The EOM funnel shipped the happy path only (`new → estimate_booked → won → customer`). The office has no way to disposition a lead that will not convert — a spam/bot "lead", or one that got an estimate but didn't buy — so dead leads sit in the review queue forever. `lost` was documented in the funnel plans (#2188 "+ lost at any point"; Website #59 board "(+ lost)") but never built. Issue #2289.

### Problem-derived contract

The transition must be **endpoint-driven** (EOM stage changes are already forced through the funnel service by `crm_provider.py:_validate_eom_transition`, and NocoDB is revoked from writing `lead_stage` by migration 354). It must record **why** a lead was lost, be **idempotent**, be **reversible** (re-open), and must **not** let a lost lead still receive a calendar event from an in-flight booking. It must add no new stored side-table and no schema DDL.

## Scope (this PR)

Ownership lane: eom-lead-funnel-disposition
Slice phase: vertical slice

- `atlas_brain/eom_api/funnel.py`: `POST /eom-funnel/leads/{contact_id}/lost` (reason-carrying) and `POST /eom-funnel/leads/{contact_id}/reopen`, mirroring the `customer-handoffs` route (no calendar). New `EOMLeadLostRequest` (`reason_code` enum + optional `note`, `extra="forbid"`).
- `atlas_brain/services/eom_lead_conversion.py`: `EOMLeadLost`/`EOMLeadReopen` dataclasses + `mark_eom_lead_lost`/`reopen_eom_lead` delegates.
- `atlas_brain/services/crm_provider.py`: two single-transaction CRM methods structured like `finalize_eom_customer_handoff` — contact-row-first advisory lock order, in-flight booking fence, guarded compare-and-set on `lead_stage`, and an idempotent lifecycle INSERT.
- Unit + real-Postgres integration tests.

### Files touched
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/eom_lead_conversion.py`
- `atlas_brain/services/crm_provider.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `plans/PR-EOM-Lead-Lost.md`

### Review Contract

- Reviewer rules triggered: R2, R8.
- **R2 (tests / failure branches):** unit tests cover the reason model (valid/invalid code, blank-note→NULL), the actor forwarding, and reopen; the integration test proves the stage flip, the reason on the ledger, idempotent replay (exactly one event), and reopen — against real Postgres.
- **R8 (concurrency / durability):** the transitions take a transaction-scoped `pg_advisory_xact_lock` on `eom-lead-lost:contact:<id>` and lock the contact row `FOR UPDATE` **before** the lifecycle rows — the same contact-before-lifecycle order the booking/handoff paths use — so a concurrent booking/handoff cannot deadlock (40P01). The in-flight booking fence **mirrors** the handoff: both the execution-lock probe (`pg_try_advisory_xact_lock('eom-estimate-booking:execution:<key>')`) **and** the subsequent requested/ambiguous-not-terminal block, so a lost transition cannot race an outstanding Calendar call **or** strand a booking whose executor died mid-flight. The lost/reopen idempotency keys are rejected when reused across a different contact, matching the booking/handoff paths, and an idempotent replay is validated against the current stage (a replay after the opposite transition is a 409, not a stale success).

### Boundary-change enumeration
No module/ownership boundary changes: new routes register on the existing `eom_funnel_router`; new service delegates + CRM methods sit beside their siblings. No new imports across layers beyond the existing funnel service surface.

### Deployed-config probing
No env/config or deployed-config change. No new secret, blueprint slot, or migration; `event_type` is bare `VARCHAR(64)` so `lead_lost`/`lead_reopened` need no DDL.

## Mechanism

`mark_eom_lead_lost` runs one transaction: acquire the sorted advisory locks, look up any prior `lead_lost` event under this `operation_key` (idempotent replay), reject a key already owned by another contact, lock the contact `FOR UPDATE`, reject non-EOM/non-lead/inactive, admit only `lead_stage IN ('new','estimate_booked')`, fence any unsettled booking, then `UPDATE contacts SET lead_stage='lost'` guarded on the observed `from_stage`, and INSERT a `lead_lost` lifecycle row carrying `reason` (the note), `metadata.lost_reason_code`, `metadata.lost_by_employee_id`, and `actor='employee:{id}:{name}'`. `reopen_eom_lead` is the inverse (`lost → new`, `lead_reopened`) and requires the contact still be `active`. Idempotency is the lifecycle unique index `(contact_id, event_type, operation_key)`; the per-action key means retries dedup while a re-marked-after-reopen (new key) is allowed.

## Intentional
- The reason **code** goes to `metadata.lost_reason_code` (structured, for reporting); the free-text **note** goes to the existing `reason TEXT` column. A whitespace-only note is stored as `NULL`.
- Admission is `('new','estimate_booked')`; a lost lead naturally drops out of `list_eom_new_lead_review_items` (its predicate excludes `lost`). `'lost'` is deliberately **not** added to the review-queue partial index (358).
- `'won'` is **excluded** from admission. A won lead already booked a first clean and enqueued an onboarding welcome draft, and `claim_eom_onboarding_draft` gates only on the contact being an active `effingham_maids` contact (not on `lead_stage`) — so losing a won lead without atomically revoking that draft would let the welcome still be sent. Neither of #2289's two cases is `won`; losing a won lead is deferred (see Deferred).
- A retry of the **same** operation key is an idempotent no-op, validated against the current stage (a replay after the opposite transition is a 409, not a stale success). Marking an already-lost lead under a **different** key — or reopening an already-active lead under a different key — is a **409 conflict**, not a keyless no-op, so no operation key is ever reported successful without a durable replay row behind it.
- Reopen restores stage `new` this slice; a lead reopened from `estimate_booked` keeps its `estimate_booked` lifecycle row, so a *new* estimate booking under a different key is refused by `_other_operation_blocks` (`crm_provider.py:1683`) — the office path forward is the customer handoff, which admits `new` (`crm_provider.py:3379`). Restoring the pre-loss stage is deferred to #2293.

## Deferred
- **Losing a `won` lead** (excluded from admission above): needs to atomically revoke the pending onboarding welcome draft and cancel the booked first clean in the same transaction — a distinct slice with its own draft/calendar teardown. Filed as a follow-up.
- Tracker proxy routes and the Website "Mark lost" reason-picker + Undo are follow-up PRs in their repos (this PR is the Atlas foundation they call).
- The stale-lead nudge (#2188 item 5) is a separate slice.
- A "Lost" bucket for browsing/re-opening lost leads is #59 board work; re-open's near-term surface is the Website Undo.

## Verification

`make api-test` scope, `tests/test_eom_lead_conversion.py` + `tests/test_eom_lead_conversion_integration.py` (the latter needs `ATLAS_MIGRATION_TEST_DATABASE_URL`). Both proven against real Postgres: **4 unit + 2 integration green** (6 passed, 212 deselected). The second integration test is the guards case — `won` refused, an unreconciled requested booking refused, cross-contact key reuse refused, replay-after-reopen a 409, and reopen refusing an inactive contact. Manual: mark a lead lost from the portal → it leaves the board; `SELECT lead_stage` = `lost`; the lifecycle row shows code + note + actor; reopen returns it to `new`.

## Estimated diff size

| Change | LOC |
|---|---|
| funnel.py routes + model | ~85 |
| crm_provider.py two transactions + review-hardening | ~357 |
| service delegates | ~50 |
| unit + integration tests | ~315 |
| this plan doc | ~85 |
| **Total** | **~890** |
