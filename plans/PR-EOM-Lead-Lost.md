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

### Review Contract

- Reviewer rules triggered: R2, R8.
- **R2 (tests / failure branches):** unit tests cover the reason model (valid/invalid code, blank-note→NULL), the actor forwarding, and reopen; the integration test proves the stage flip, the reason on the ledger, idempotent replay (exactly one event), and reopen — against real Postgres.
- **R8 (concurrency / durability):** the transitions take a transaction-scoped `pg_advisory_xact_lock` on `eom-lead-lost:contact:<id>` and lock the contact row `FOR UPDATE` **before** the lifecycle rows — the same contact-before-lifecycle order the booking/handoff paths use — so a concurrent booking/handoff cannot deadlock (40P01). The in-flight booking fence (`pg_try_advisory_xact_lock('eom-estimate-booking:execution:<key>')`) is copied verbatim from the handoff so a lost transition cannot race an outstanding Calendar call.

### Boundary-change enumeration
No module/ownership boundary changes: new routes register on the existing `eom_funnel_router`; new service delegates + CRM methods sit beside their siblings. No new imports across layers beyond the existing funnel service surface.

### Deployed-config probing
No env/config or deployed-config change. No new secret, blueprint slot, or migration; `event_type` is bare `VARCHAR(64)` so `lead_lost`/`lead_reopened` need no DDL.

## Mechanism

`mark_eom_lead_lost` runs one transaction: acquire the sorted advisory locks, look up any prior `lead_lost` event under this `operation_key` (idempotent replay), lock the contact `FOR UPDATE`, reject non-EOM/non-lead/inactive, admit only `lead_stage IN ('new','estimate_booked','won')`, fence any unsettled booking, then `UPDATE contacts SET lead_stage='lost'` guarded on the observed `from_stage`, and INSERT a `lead_lost` lifecycle row carrying `reason` (the note), `metadata.lost_reason_code`, `metadata.lost_by_employee_id`, and `actor='employee:{id}:{name}'`. `reopen_eom_lead` is the inverse (`lost → new`, `lead_reopened`). Idempotency is the lifecycle unique index `(contact_id, event_type, operation_key)`; the per-action key means retries dedup while a re-marked-after-reopen (new key) is allowed.

## Intentional
- The reason **code** goes to `metadata.lost_reason_code` (structured, for reporting); the free-text **note** goes to the existing `reason TEXT` column. A whitespace-only note is stored as `NULL`.
- Admission is `('new','estimate_booked','won')`; a lost lead naturally drops out of `list_eom_new_lead_review_items` (its predicate excludes `lost`). `'lost'` is deliberately **not** added to the review-queue partial index (358).
- Already-lost (marked again under a different key) and already-reopened (`new`) are treated as idempotent no-ops rather than errors.

## Deferred
- Tracker proxy routes and the Website "Mark lost" reason-picker + Undo are follow-up PRs in their repos (this PR is the Atlas foundation they call).
- The stale-lead nudge (#2188 item 5) is a separate slice.
- A "Lost" bucket for browsing/re-opening lost leads is #59 board work; re-open's near-term surface is the Website Undo.

## Verification

`make api-test` scope, `tests/test_eom_lead_conversion.py` + `tests/test_eom_lead_conversion_integration.py` (the latter needs `ATLAS_MIGRATION_TEST_DATABASE_URL`). Both proven locally against real Postgres: 4 unit + 1 integration green. Manual: mark a lead lost from the portal → it leaves the board; `SELECT lead_stage` = `lost`; the lifecycle row shows code + note + actor; reopen returns it to `new`.

## Estimated diff size

| Change | LOC |
|---|---|
| funnel.py routes + model | ~85 |
| crm_provider.py two transactions | ~251 |
| service delegates | ~50 |
| unit + integration tests | ~217 |
| this plan doc | ~72 |
| **Total** | **~675** |
