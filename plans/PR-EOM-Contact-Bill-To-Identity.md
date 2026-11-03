# PR-EOM-Contact-Bill-To-Identity

## Why this slice exists

Slice A of the operator-accepted EOM customer bill-to identity contract
(eom-timetracker `EOM_CUSTOMER_BILL_TO_IDENTITY_CONTRACT.md`, eom-timetracker
PR 280, accepted 2026-11-03).

Observed 2026-11-03: Milano & Grunloh Engineers was converted from a lead to a
commercial Customer and its business name and billing address were saved in
the tracker. The linked Atlas contact still has the lead's person name and no
address, so its first monthly invoice (generated 2026-12-01) would be billed to
a person with no address. The tracker side (slice B) will send the bill-to
identity; this slice gives Atlas somewhere correct to store it and a way to
refuse stale writes.

### Problem-derived contract

- Root cause: Atlas has no bill-to address. `contacts.address` already serves a
  different concept: it is the `location` of EOM booking calendar events
  (`atlas_brain/services/crm_provider.py:2854`) and also the monthly invoice
  bill-to (`atlas_brain/autonomous/tasks/monthly_invoice_generation.py:413`).
  Writing a billing office into it would move cleaners' booking locations;
  leaving it empty leaves invoices without an address. Separately, the
  operator boundary has no ordering for identity writes, so a retried stale
  tracker write could undo a newer one.
- Correct fix must touch/change: an additive `contacts.billing_address` column;
  the operator mutation boundary (`EOM_OPERATOR_CONTACT_FIELDS`,
  `EOMOperatorContactRequest`, the atomic provider mutation) admitting
  `billingAddress` and an `identityRevision` fence stored in
  `contacts.bill_to_revision`; a semantic capability name so the tracker can
  detect both; monthly invoice generation preferring `billing_address` over
  `address`.
- Must not change: booking event locations (`address`), any existing invoice,
  the invoicing MCP `create_invoice` (it takes the address from its caller),
  commercial billing-run invoices, request fingerprints of requests that send
  neither new field, the tracker, the website, and every other contact writer.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Migration `400_eom_contact_bill_to_identity.sql`: nullable
   `contacts.billing_address` (at most 500 characters) and nullable
   `contacts.bill_to_revision` (positive), each enforced by a CHECK.
2. Operator boundary admits `billingAddress` (absent = untouched; null or blank
   clears it, only on an edit that names `contactId`) and `identityRevision`
   (positive integer, only with `contactId`).
3. Revision fence in the atomic provider mutation: an identity write applies
   only when its revision is greater than the stored `bill_to_revision`, and
   stores it in the same UPDATE.
4. Capability `contact.bill_to_identity` on the operator-contacts route; the
   operator-contact response carries `billingAddress`.
5. Monthly invoice bill-to address: `billing_address` when set, else `address`.

### Review Contract

- Acceptance criteria:
  - Migration 400 is additive: it only adds two nullable columns and two CHECK
    constraints (`atlas_brain/storage/migrations/400_eom_contact_bill_to_identity.sql`);
    the integration schema applies it and existing rows read NULL in both.
  - An edit with `contactId` plus `billingAddress` changes `billing_address`
    and leaves `address` untouched --
    `tests/test_eom_lead_conversion_integration.py::test_operator_contact_billing_address_edit_leaves_address_untouched`.
  - Absent `billingAddress` never reaches the command; present null with
    `contactId` reaches it as None; null or blank without `contactId` is 422
    before the provider; over 500 characters is 422 --
    `tests/test_eom_lead_conversion.py::test_operator_contact_route_billing_address_*`.
  - Revision fence, execution model (AGENTS 3k.4): every write that names
    `contactId` holds the `eom-operator-contact:contact:{id}` advisory lock and
    the row lock for its whole transaction, so writes to one contact apply one
    at a time; the idempotency receipt check runs before the fence, so a retry
    under its original key replays its receipt. Invariant: `bill_to_revision`
    only grows, and a write carrying `identityRevision` changes the row only if
    its revision is greater than the stored one at the moment it holds the
    lock. Evidence:
    `tests/test_eom_lead_conversion_integration.py::test_operator_contact_identity_revision_fence`
    (stored 5: revision 6 applies and stores 6; revisions 6, 5, 4 under new
    keys return 409 and leave the row unchanged; the revision-6 retry under its
    original key returns its receipt).
  - `identityRevision` without `contactId`, zero, negative, string, boolean, or
    above the BIGINT range is 422 before the provider --
    `tests/test_eom_lead_conversion.py::test_operator_contact_route_rejects_invalid_identity_revision`.
  - A command that sends neither new field keeps the exact pre-change request
    fingerprint, so in-flight retries across the deploy still replay --
    `tests/test_eom_lead_conversion.py::test_operator_contact_fingerprint_unchanged_without_bill_to_fields`.
  - Create through the operator boundary stores `billingAddress`
    (`_insert_contact_row` names the column only when it is supplied) --
    `tests/test_eom_lead_conversion_integration.py::test_operator_contact_create_stores_billing_address`.
  - `contact.bill_to_identity` maps to `POST /eom-funnel/operator-contacts`,
    shares availability with `contact.operator_mutation`, and is served --
    `tests/test_eom_funnel_capability_manifest.py::test_bill_to_identity_capability_versions_the_operator_mutation_semantics`.
  - Monthly invoice bill-to: `billing_address` when it is a non-blank string,
    else `address` --
    `tests/test_monthly_invoice_generation.py::test_contact_bill_to_address_*`.
  - Booking event `location` still reads `address` at
    `atlas_brain/services/crm_provider.py:2854` (unchanged by this diff; cold
    diff reconstruction confirms).
- Reachability proof: an ASGI request to the mounted
  `POST /eom-funnel/operator-contacts` route carries `billingAddress` and
  `identityRevision` into the command and returns `billingAddress`; the
  integration tests run the real `DatabaseCRMProvider` against migration 400.
- Affected surfaces: `atlas_brain/eom_api/funnel.py` (request model, field
  map, response item, capability map), `atlas_brain/services/eom_crm_mutations.py`
  (field set, limits, command), `atlas_brain/services/crm_provider.py`
  (atomic mutation, contact insert), migration 400, monthly invoice generation.
- Risk areas: fingerprint drift for in-flight retries; create-or-return
  clearing a matched contact's billing address; the fence admitting an equal
  revision; a pending migration breaking contact inserts for callers that never
  send `billing_address`; booking locations moving.
- Reviewer rules triggered: R1, R2, R3, R5, R7, R10, R12, R13, R14.

### Guard-class closure declaration

- Member set: `EOM_OPERATOR_CONTACT_FIELDS` gains `billing_address`; the
  capability map gains `contact.bill_to_identity`; the command gains
  `identity_revision`.
- Membership is **CLOSED**: the contract's bill-to identity is exactly name
  (existing `full_name`) and address (new `billing_address`); the fence token
  is exactly one integer. Billing email and an "Attn" name are deferred by the
  contract.
- Membership source is **ENUMERATED** in `EOM_OPERATOR_CONTACT_FIELDS` and the
  request model; advertised capabilities stay **DERIVED** from registered
  routes by `served_capabilities()`.
- Out-of-set default is **REJECT** at the boundary (unknown fields 422 via
  `extra="forbid"` and `_normalize_fields`) and **OMIT** for capabilities.

### Boundary-change enumeration

- Boundary path/seam: `EOMOperatorContactRequest` -> `_operator_contact_fields`
  -> `EOMOperatorContactMutation.from_raw` -> `mutate_eom_operator_contact_atomic`.
- Replaced-path behaviors: requests without the new fields behave exactly as
  before, including their fingerprint. `billingAddress` is a new admitted
  field; `identityRevision` is a new admitted fence.
- Guard-relevant fields: `billingAddress` (presence, null/blank, length,
  `contactId`), `identityRevision` (type, sign, range, `contactId`), stored
  `bill_to_revision`.
- Caller x input shape: current tracker (sends neither) x unchanged; slice-B
  tracker x edit with both; any caller x null `billingAddress` without
  `contactId` (422); any caller x stale revision (409).

### Deployed-config probing

N/A - no environment or config value participates. The fence and the field are
decided by request content and the stored row only.

### Files touched

- TODO: run `python scripts/sync_pr_plan.py plans/PR-EOM-Contact-Bill-To-Identity.md` after implementation.

## Mechanism

The migration adds `billing_address TEXT` and `bill_to_revision BIGINT`, both
nullable, with `chk_contacts_billing_address_length` (at most 500 characters)
and `chk_contacts_bill_to_revision_positive`.

`billing_address` joins `EOM_OPERATOR_CONTACT_FIELDS` with a 500-character
limit, so it rides the existing normalization, update, and lifecycle-event
paths (`changed_fields`, `previous_values`, `cleared_fields`). `from_raw`
rejects a null `billing_address` and any `identity_revision` when no
`contact_id` is named: a create-or-return request can resolve to a contact it
merely matched, and must not clear or fence that contact. The request
fingerprint adds `identity_revision` only when it is present, so older
requests hash exactly as before.

In the atomic mutation, the existing receipt lookup runs first (a retry under
its key replays). `_update_existing` then compares `identity_revision` with the
row's `bill_to_revision` under the locks already held for a `contactId` edit,
raises 409 when it is not greater, and otherwise writes `bill_to_revision`
with the field updates. `_insert_contact_row` takes `billing_address` as a
keyword (like `customer_type`) and names the column only when it is supplied,
so contact inserts from callers that never send it do not depend on migration
400.

Monthly invoice generation reads the bill-to address through
`_contact_bill_to_address(contact)`.

## Intentional

- A caller-supplied fence token rather than an Atlas-owned sequence like
  migration 367's `customer_type_revision`. 367 orders Atlas evidence for the
  tracker to read; this fences the tracker's own writes into Atlas, and the
  tracker is the only bill-to writer under the contract, so its reservation
  sequence is the natural order. Compare-and-set on an Atlas version would also
  work but would make the tracker read before every write.
- An equal revision under a new key is refused, not treated as a no-op: two
  different reservations must never share a revision, so equality means a
  stale or malformed request.
- `customer_address` on already-created invoices is a snapshot and is not
  rewritten.

## Deferred

- Tracker slice B (identity saga on PATCH, conversion, repair, supersede, Site
  route guard) and the website pending-state slice, per the contract's order.
- Billing email as the invoice recipient and an "Attn" line, per the contract.
- Commercial billing-run invoices carry no address
  (`atlas_brain/services/commercial_billing_approvals.py:914`); listed as
  deferred in the contract.

Parking predicate: anything that does not change whether a linked Customer's
bill-to name and address reach Atlas and its invoices stays parked.

Parked hardening: none.

## Verification

- Pending before push: focused unit tests, integration tests in CI, ruff,
  ASCII check, guard-class closure audit, plan sync.

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **0** |
