# PR-EOM-Operator-Contact-Match-Return

## Why this slice exists

Slice A of the operator-accepted EOM customer create match contract
(eom-timetracker `EOM_CUSTOMER_CREATE_MATCH_CONTRACT.md`, eom-timetracker
PR 287, accepted 2026-11-04; eom-timetracker issue 282).

A tracker Customer create calls `POST /eom-funnel/operator-contacts` without
`contactId`. When that create-or-return matches an existing contact by email
or phone, `_update_existing` writes the name, email and phone it carried
before the tracker learns that another Customer already holds the contact.
The tracker then refuses the create, but the other Customer's invoice name and
recipient email in Atlas have already changed. The same write happens on
tracker reconcile and on a Leads-page customer create. This slice lets a
caller ask Atlas to return a matched contact unchanged; the tracker (slice B)
then decides, with its own ownership check, whether to link it.

Diff budget: about 750 lines against the 400-line target. About 60 are
production code (boundary field, provider branch, capability, index
migration); the rest is this plan and tests on both sides of the new guard,
most of them database-backed proof that a match writes nothing. The slice is
indivisible: the field, the receipt, and the index must ship together, or a
`"return"` request either has no receipt to replay or makes every operator
mutation scan the lifecycle ledger.

### Problem-derived contract

- Root cause: the create-or-return writes a matched contact before the only
  check that knows who holds it. Atlas cannot make that check: it records a
  tracker Customer only on handoff rows, so contacts linked by a tracker
  create, reconcile, or legacy backfill have no holder record here. The
  tracker can make it, but only after Atlas returns the contact id.
- Correct fix must touch/change: the operator boundary admitting a closed
  `onMatch` choice on create-or-return only (`EOMOperatorContactRequest`,
  `EOMOperatorContactMutation.from_raw`); the atomic provider mutation writing
  nothing on a `"return"` match while keeping today's refusals; a receipt for
  that outcome so retries replay; the receipt-lookup index covering it; a
  semantic capability name so the tracker can detect it.
- Must not change: any request that does not send `onMatch` (behavior,
  receipt, and fingerprint), creates, `contactId` edits, the bill-to fence,
  the tracker, the website, and every other contact writer.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 10

1. `onMatch` on the operator-contacts route: closed enum `"apply"` |
   `"return"`; absent means `"apply"`; null or any other value is 422; any
   `onMatch` with `contactId` is 422.
2. Provider: a `"return"` request that matches one contact runs today's
   refusal checks, writes no `contacts` column, and records a
   `contact_matched` lifecycle event as its receipt. No match still creates.
3. `contact_matched` joins `EOM_OPERATOR_CONTACT_EVENT_TYPES`, so the receipt
   lookup replays it; the request fingerprint includes `onMatch` only when it
   is `"return"`.
4. Migration `401_eom_operator_contact_matched_receipt_index.sql` recreates
   migration 364's receipt-lookup index over all three receipt types.
5. Capability `contact.match_without_write` on the operator-contacts route.

### Review Contract

- Acceptance criteria:
  - A `"return"` create that matches one contact by email, or by phone
    last-10, leaves every column of that row identical, creates no contact,
    returns operation `contact_matched` for that contact id, and records
    exactly one `contact_matched` event with the operation key, empty
    `changed_fields`/`previous_values`, and an unchanged lead stage --
    `tests/test_eom_lead_conversion_integration.py::test_operator_contact_on_match_return_leaves_the_matched_contact_unchanged`.
  - A `"return"` create with no match creates the contact as today --
    `...::test_operator_contact_on_match_return_creates_when_nothing_matches`.
  - Execution model (AGENTS 3k.4): a `"return"` request takes the same
    advisory locks as any operator mutation (operation key, phone, email,
    source ref) and locks the candidate rows `FOR UPDATE`, so it serializes
    with any write to a contact it matches; the receipt lookup runs first, so
    a retry under its key replays. Invariant: one operation key has exactly
    one receipt and one meaning. Evidence:
    `...::test_operator_contact_on_match_return_replays_and_refuses_a_changed_key`
    (the replay returns the same `contact_matched` receipt with no second
    event; the same key sent with `"apply"` returns 409 and changes nothing).
  - Under `"return"`, several matches and a contact-type mismatch return
    today's 409, leave every contact unchanged, and record no receipt --
    `...::test_operator_contact_on_match_return_keeps_todays_refusals`.
  - Admission: `onMatch` reaches the command as sent; absent is `"apply"`;
    null, `""`, `"RETURN"`, `"skip"`, `true`, and `1` are 422; `"apply"` or
    `"return"` with `contactId` is 422; each refusal happens before the
    provider runs -- `tests/test_eom_lead_conversion.py::test_operator_contact_route_*on_match*`.
  - Fingerprint: absent and `"apply"` hash identically, and the pre-change
    hand-built hash still matches an absent request; `"return"` hashes
    differently --
    `tests/test_eom_lead_conversion.py::test_operator_contact_fingerprint_changes_only_for_on_match_return`
    and `::test_operator_contact_fingerprint_unchanged_without_bill_to_fields`.
  - The index predicate in migration 401 lists exactly
    `EOM_OPERATOR_CONTACT_EVENT_TYPES`, and the file runs as two
    statements, drop then create --
    `tests/test_migrations_runner.py::test_operator_contact_receipt_index_lists_exactly_the_receipt_types`.
  - `contact.match_without_write` maps to the operator-contacts route, shares
    availability with `contact.operator_mutation`, and is served --
    `tests/test_eom_funnel_capability_manifest.py::test_match_without_write_capability_versions_the_operator_mutation_semantics`.
- Reachability proof: ASGI requests to the mounted
  `POST /eom-funnel/operator-contacts` carry `onMatch` into the command; the
  integration tests run the real `DatabaseCRMProvider` against migrations
  035-400.
- Affected surfaces: `atlas_brain/eom_api/funnel.py` (request model, route,
  capability map), `atlas_brain/services/eom_crm_mutations.py` (event types,
  on-match type, command, fingerprint), `atlas_brain/services/crm_provider.py`
  (atomic mutation), migration 401, the write-boundary audit's comment.
- Risk areas: a `"return"` match that still writes (provenance, business
  context, `updated_at`); fingerprint drift for in-flight retries; a matched
  receipt the lookup cannot find; a looser match under `"return"`; an
  unindexed receipt lookup.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R12, R13, R14.

### Guard-class closure declaration

- Member set: the `onMatch` values; `EOM_OPERATOR_CONTACT_EVENT_TYPES` gains
  `contact_matched`; the capability map gains `contact.match_without_write`.
- Membership is **CLOSED**: a matched contact is either written (`"apply"`)
  or left unchanged (`"return"`); the contract defines no third outcome.
- Membership source is **ENUMERATED** once in `EOMOperatorOnMatch` (a
  `Literal`); the route model uses that type and `from_raw` derives its value
  tuple from it with `get_args`. The receipt index predicate is checked
  against `EOM_OPERATOR_CONTACT_EVENT_TYPES` itself. Advertised capabilities
  stay **DERIVED** from registered routes by `served_capabilities()`.
- Out-of-set default is **REJECT** (422) at the boundary, and **OMIT** for
  capabilities.

### Boundary-change enumeration

- Boundary path/seam: `EOMOperatorContactRequest` ->
  `EOMOperatorContactMutation.from_raw` -> `mutate_eom_operator_contact_atomic`.
- Replaced-path behaviors: none. `onMatch` absent keeps the existing create,
  update, and refusal paths and their fingerprints.
- Guard-relevant fields: `onMatch` (presence, value, type, `contactId`).
- Caller x input shape: current tracker (never sends it) x unchanged;
  slice-B tracker x create or reconcile with `"return"`; Leads-page customer
  create (slice B) x `"return"`; any caller x `onMatch` with `contactId`
  (422); any caller x null or unknown value (422).

### Deployed-config probing

N/A - no environment or config value participates. The outcome is decided by
request content and the stored rows only.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_crm_mutations.py`
- `atlas_brain/storage/migrations/401_eom_operator_contact_matched_receipt_index.sql`
- `plans/PR-EOM-Operator-Contact-Match-Return.md`
- `scripts/eom_write_boundary_audit.py`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_migrations_runner.py`

## Mechanism

`EOMOperatorOnMatch = Literal["apply", "return"]` is the single owner of the
choice. The request model types `onMatch` with it (default `"apply"`), so null
and unknown values fail schema validation. The route forwards the value only
when it was sent, and `from_raw` refuses any sent `onMatch` beside
`contactId`, then normalizes absent to `"apply"`.

In the atomic mutation nothing changes before target resolution: the locks,
the receipt lookup, and `_resolve_target` (with its several-matches refusal)
run as today. On a resolved target with `"return"`, the provider calls
`_assert_contact_type_matches` (today's archived, editability, and
contact-type refusals) and then records the receipt with the target as both
the previous and the current row, so the event's change lists are empty and
its stage is unchanged. `_update_existing` is not called, so no column,
provenance entry, business context, or `updated_at` is written.

Migration 401 drops and recreates `idx_eom_lead_lifecycle_operator_contact_operation_key`
with the predicate widened to the three receipt types; the runner executes the
two statements one at a time outside a transaction.

## Intentional

- A distinct `contact_matched` receipt rather than a `contact_updated` event
  with an empty change set: the write-boundary audit counts created/updated
  events as accounting for operator writes, and a match is not one. Its
  comment now says so.
- The event keeps empty `changed_fields`/`previous_values` instead of omitting
  them, so every operator receipt has the same metadata shape.
- `onMatch` beside `contactId` is refused even as `"apply"`, so one field
  cannot mean "edit this contact" and "decide on a match" at once.
- An archived contact is never a create-or-return candidate (every match leg
  filters it), so it has no `"return"` case.

## Deferred

- Tracker slice B (Customer create, reconcile, and Leads-page customer create
  send `onMatch: "return"` behind the capability; the Leads page refuses a
  match with 409 `funnel_contact_exists`), per the contract's order.
- The contract's read-only past-damage audit, after A and B deploy.
- The pre-existing unused `EOM_BUSINESS_CONTEXT_ID` import in
  `eom_crm_mutations.py` (ruff F401 on `origin/main` too).

Parking predicate: anything that does not change whether a create-or-return
can write a contact it merely matched stays parked.

Parked hardening: none.

## Verification

- Passed locally: the new route, fingerprint, capability, and migration tests
  (14), run with every database setting pointed at a closed port so no test
  can reach the live database.
- Passed locally: every test in `tests/test_eom_lead_conversion.py`,
  `tests/test_eom_funnel_capability_manifest.py`, and
  `tests/test_migrations_runner.py` selected by
  `-k "on_match or fingerprint or operator_contact or capability or receipt_index"`.
- Skipped locally, proven in CI: the six new database-backed tests in
  `tests/test_eom_lead_conversion_integration.py`. No disposable database is
  prepared on this machine, and the runbook routes that proof to GitHub
  Actions.
- Passed locally: `ruff check` adds no findings over `origin/main` on any
  changed file; no new non-ASCII; `python scripts/check_guard_class_closure.py --base origin/main --strict`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 14 |
| `atlas_brain/services/crm_provider.py` | 12 |
| `atlas_brain/services/eom_crm_mutations.py` | 34 |
| `atlas_brain/storage/migrations/401_eom_operator_contact_matched_receipt_index.sql` | 33 |
| `plans/PR-EOM-Operator-Contact-Match-Return.md` | 220 |
| `scripts/eom_write_boundary_audit.py` | 6 |
| `tests/test_eom_funnel_capability_manifest.py` | 22 |
| `tests/test_eom_lead_conversion.py` | 76 |
| `tests/test_eom_lead_conversion_integration.py` | 236 |
| `tests/test_migrations_runner.py` | 43 |
| **Total** | **~700** |
