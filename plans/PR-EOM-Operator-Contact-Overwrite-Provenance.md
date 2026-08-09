# PR-EOM-Operator-Contact-Overwrite-Provenance

## Why this slice exists

Observed live on 2026-08-08, immediately after Slice 0C shipped (website #110,
eom-timetracker#149, website#155). Reconciling tracker customer 5 through
`POST /eom-funnel/operator-contacts` produced `contact_updated`, not
`contact_created`: the operator boundary matched an existing `calendar_import`
contact on phone last-10 and overwrote its `full_name`.

The lifecycle event recorded `changed_fields: ["full_name", "phone"]` and
nothing else, so the name that was replaced could not be recovered — Atlas has
no contact history table, and the row itself had already been updated. The
operator could be told only that a name changed, never what it had been.

Folded into website issue #158 (Slice 0C deferred hardening) at operator
direction.

### Problem-derived contract

- Root cause: `_write_lifecycle_event` in
  `DatabaseCRMProvider.mutate_eom_operator_contact_atomic` records which fields
  an update changed but not their prior values. Because this boundary is
  create-OR-return, a create carrying no `contactId` can resolve to an existing
  contact and rewrite its identity as operator intent, so the discarded value
  exists nowhere after the UPDATE commits.
- Correct fix must touch/change: the lifecycle event metadata written on
  `contact_updated`, so the overwritten values are captured inside the same
  transaction as the UPDATE that discards them.
- Must not change: the mutation's matching, idempotency, provenance-metadata, or
  conflict semantics; the HTTP contract of
  `POST /eom-funnel/operator-contacts`; `contact_created` events, which have no
  prior value by definition; the `eom_lead_lifecycle_events` schema, which
  already stores `metadata jsonb`.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Record `previous_values` — the pre-update value of each changed field — in
   the `contact_updated` lifecycle event metadata.
2. Add the tracker-contract test pinning the exact request body
   eom-timetracker sends, since `extra="forbid"` makes any cross-repo drift a
   silent production 422 that neither repo's own tests would catch.

### Review Contract

- Acceptance criteria:
  1. An operator update records the prior value of every changed field —
     settled by
     `tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_updates_exact_match_and_claims_legacy`,
     which asserts `previous_values == {"email": None, "phone": "1 (217) 555-0100"}`.
  2. A create-shaped command that matches an existing contact by phone records
     the identity it overwrote — settled by
     `tests/test_eom_lead_conversion_integration.py::test_operator_contact_records_the_identity_it_overwrote_on_a_phone_match`,
     which asserts `previous_values["full_name"] == "Cal Import Label"`. This
     reproduces the live 2026-08-08 case.
  3. The body eom-timetracker sends is accepted, in both the full-identity and
     name-only shapes — settled by
     `tests/test_eom_lead_conversion.py::test_operator_contact_accepts_the_tracker_customer_create_body`.
  4. `contact_created` events carry neither `previous_values` nor
     `changed_fields` — settled by the direct absence assertions in
     `tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_creates_replays_and_records_actor`.
     Asserted directly because the surrounding create assertions would all stay
     green if a regression began attaching overwritten values to creates.
- Reachability proof: `POST /api/v1/eom-funnel/operator-contacts` →
  `mutate_eom_operator_contact` → `mutate_eom_operator_contact_atomic` →
  `_write_lifecycle_event`. Observable effect: a row in
  `eom_lead_lifecycle_events` whose `metadata` now carries `previous_values`.
  This is the live path the tracker's customer create and reconcile routes call.
- Affected surfaces: `atlas_brain/services/crm_provider.py`
  (`mutate_eom_operator_contact_atomic` only); `eom_lead_lifecycle_events` rows
  written by that path. No caller reads `metadata.changed_fields` today, so no
  consumer contract moves.
- Risk areas: metadata size growth on updates touching `notes` (max 4000) or
  `address` (max 2000); JSON-serializability of stored values; accidental change
  to the create path, which has no `previous` row.
- Reviewer rules triggered: R1 (problem-derived contract), R2 (reachability),
  R5 (test evidence), R12 (data/provenance).

### Boundary-change enumeration

N/A - no boundary change. This adds a field to an audit record written inside
the existing transaction. Matching, admission, idempotency, and conflict
behavior are untouched; the diff adds no branch to any guard.

### Deployed-config probing

N/A - no guard/config boundary change. No env var, feature flag, or default
value is read or added.

### Files touched

- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Operator-Contact-Overwrite-Provenance.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

`_write_lifecycle_event` already computes `changed` by comparing the `previous`
row against the written `contact` for each field the command carried. The diff
reuses that same comparison and additionally stores
`previous_values = {key: previous.get(key) for key in changed}`.

It runs inside the same transaction as the UPDATE, so the captured values are
exactly the ones being discarded — there is no window in which another writer
can move the row between the read and the record.

Only fields the command actually carried are considered, and only those that
actually changed, so an update that rewrites nothing writes an empty mapping
rather than a snapshot of the whole contact. The create path passes
`previous=None` and is untouched.

## Intentional

- Values are stored whole rather than truncated. Truncating an audit value to
  bound row size would defeat the purpose of recording it; the fields are
  already length-capped upstream by `EOMOperatorContactRequest`.
- Stored on the event rather than in a new contact-history table. A history
  table is the more complete answer, but it is a schema and retention decision
  well beyond this fix, and the event is already the audit record for this
  boundary and is already append-only (migration 351 blocks UPDATE/DELETE).
- Prior values are PII of the same kind the event already carries (actor) and
  the `contacts` row itself holds, so this does not widen the data class of the
  table.

## Deferred

- A contact history table covering every writer, not just this boundary.
- Backfilling the identity overwritten on 2026-08-08 for contact
  `53efdc2c-e605-4c3b-bbd3-521a1af9a0cf`; the prior value is already
  unrecoverable, which is the reason for this slice.
- The reconcile identity race and reservation recovery path, tracked in website
  issue #158.

Parked hardening: none.

## Verification

- Pending before push:
  - `pytest tests/test_eom_lead_conversion_integration.py tests/test_eom_lead_conversion.py tests/test_eom_lead_ingress.py`
    against a throwaway `postgres:16` (never the local `atlas` database) —
    Result: pass, 301 passed.
  - Negative control: both new assertions fail with the provider change stashed
    — Result: pass, 2 failed as expected.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/crm_provider.py` | 15 |
| `plans/PR-EOM-Operator-Contact-Overwrite-Provenance.md` | 155 |
| `tests/test_eom_lead_conversion.py` | 67 |
| `tests/test_eom_lead_conversion_integration.py` | 71 |
| **Total** | **308** |
