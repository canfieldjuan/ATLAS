# PR-EOM-Known-Contact-Types

## Why this slice exists

ATLAS #2354 put `customer_type` on the Atlas contact, which is the account
record and the sole write authority for it. eom-timetracker #161 mirrors that
value into the tracker so the portal — which reads customers from the tracker,
not from Atlas — can render it.

**Root cause.** That mirror is populate-on-write-path only. It fills in from the
operator-mutation response, the one Atlas call that returns the contact. Any
path that does not make that call leaves the local row at `unknown` with no way
to correct it, and the reconciliation route then refuses the row because it is
already linked. Three known instances, one cause:

1. **Office estimate approval** — calls `/eom-funnel/customer-handoffs`, whose
   response carries `success`, `contact_id`, `tracker_customer_id`,
   `tracker_site_id`, `handoff_id`, `approval_key`. No contact, no type.
2. **Bulk linkage backfill** — makes no Atlas call at all; it applies an
   operator-supplied customerId → contactId mapping.
3. **A type changed in Atlas after the local row exists** — never learned.

Two of the three have no Atlas response to read from, so a per-path write hook
cannot fix them. The fix has to be a **read**. This is ATLAS #2357.

### Problem-derived contract

- Root cause: the mirror has no refresh, because no read surface reports the
  classification.
- Correct fix must touch/change: the tenant-scoped contact read behind
  `GET /eom-funnel/known-contacts` so it returns the type alongside the id, and
  the response model that carries it.
- Must not change: the tenant predicate (still IN the query, not a filter over
  its result); the attribution rule (no id the caller did not submit may appear
  in any field); the existing `knownContactIds` shape, which the tracker's link
  audit already consumes; identity disclosure — no name, email, phone, address
  or notes; the write boundary — this PR adds no write path.

## Scope (this PR)

Ownership lane: eom-crm/known-contact-types
Slice phase: Vertical slice
Max files: 4

1. `list_known_eom_contact_ids` returns `{id, customer_type}` rows instead of
   bare ids, from the same tenant-scoped query.
2. `EOMKnownContactsResponse` gains `customerTypes`, a mapping keyed by the same
   ids, additive alongside the unchanged `knownContactIds`.
3. Tests for the new field, the dangling case, the attribution filter, and a
   real-PostgreSQL proof that the type read is tenant-scoped.

Not in this PR: the tracker-side refresh that consumes it. That is a separate
eom-timetracker PR and is the other half of #2357.

### Review Contract

- Acceptance criteria:
  - A known contact reports its type — settled by
    `tests/test_eom_link_verification.py::test_the_route_reports_the_type_for_each_known_contact`.
  - A dangling id gets no type at all, and the two fields always agree on their
    key set — settled by `::test_a_dangling_id_gets_no_type_at_all`.
  - The attribution filter governs the types too, not only the id list — settled
    by `::test_types_never_carry_an_id_the_caller_did_not_submit`, which stubs a
    provider returning an id the caller never submitted. Negative control run:
    building the map from the provider's rows instead of the requested ids fails
    it.
  - Another tenant's type is not readable, proven against real PostgreSQL rather
    than a stub — settled by
    `::test_the_type_read_is_tenant_scoped_against_real_postgres`, which seeds a
    `churnsignals` contact with a type beside an `effingham_maids` one and asks
    for both. Negative control run: removing the tenant predicate fails it.
  - No identity data is disclosed — settled by
    `::test_the_route_discloses_no_contact_data_beyond_the_id`, which asserts the
    exact key set and that `fullName`/`email`/`phone`/`address`/`notes` appear
    nowhere in the rendered payload.
  - The existing consumer is unbroken — `knownContactIds` keeps its exact prior
    shape, settled by the unchanged
    `::test_repeated_ids_are_asked_once_and_answered_once` exact-body assertion.
- Reachability proof: `GET /api/v1/eom-funnel/known-contacts` on the aggregate
  app, behind `require_eom_funnel_api` + `require_eom_funnel_actor`, settled by
  `::test_the_real_aggregate_serves_the_route_at_its_deployed_path`.
- Affected surfaces: the known-contacts response model and route body, one
  provider read method, and that route's test file.
- Risk areas: disclosing another tenant's classification; reporting a type for
  an id the caller never asked about; breaking the tracker's existing consumer
  of `knownContactIds`; widening an intentionally minimal route.
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R3
  (security and authorization — this widens what an EOM-scoped credential can
  read), R5 (backward compatibility — an additive response field with an
  existing consumer), R14 (verify against the codebase).

### Boundary-change enumeration

- Boundary path/seam: `GET /eom-funnel/known-contacts` — an existing read
  boundary whose response widens.
- Replaced-path behaviors: none. `knownContactIds`, `checked` and `limit` are
  byte-identical to before; `customerTypes` is new.
- Guard-relevant fields: `contacts.business_context_id` (still in the query) and
  the requested-id set that filters both response fields.
- Caller x input shape: authenticated tracker x {known id, dangling id,
  foreign-tenant id, duplicate ids}; a provider returning an unrequested id.

### Deployed-config probing

N/A - no guard/config boundary change. No environment or config value is read.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Known-Contact-Types.md`
- `tests/test_eom_link_verification.py`

## Mechanism

The provider's query is unchanged except for selecting one more column; the
tenant predicate stays inside it, so a contact in another business context is
absent from the result and cannot leak a type the route never has an id for.

The route builds `customerTypes` from `known_ids` — the list already filtered to
what the caller submitted — rather than from the provider's rows. That is the
same attribution rule the id list uses, applied to the second field, so neither
field can carry an id the other does not.

`knownContactIds` is untouched on purpose. The tracker's link audit consumes it
today; turning a list of ids into a list of objects would break that consumer
for no gain, so the types arrive as a parallel mapping.

## Intentional

- **Widening this route is a disclosure decision, made explicitly.** It was
  introduced id-only. The type is included because it is not personal data — it
  is a classification the operator set — the credential is already EOM-scoped so
  no cross-tenant information is exposed, and the alternative is a mirror that
  can never self-correct.
- **A dangling id gets no type.** Reporting one would be worse than silence,
  because it looks like an answer for a link that does not resolve.
- **`unknown` is returned rather than omitted** for a known contact with no
  classification, matching how the column stores it. Absence means "not a known
  contact"; `unknown` means "known, not yet classified". Collapsing the two
  would make the refresh unable to distinguish them.
- **Rejected: a new endpoint.** The existing route already takes up to 100
  contact ids, is already tenant-scoped, and is already authenticated. A second
  route would duplicate all three and add an auth surface for one field.

## Deferred

- The tracker-side refresh that batches linked contact ids through this route
  and updates the mirror, using the COALESCE semantics already in
  eom-timetracker #161 (an absent value must never clobber a known one). That is
  the other half of #2357.

Parking predicate: hardening is parked when it protects a caller that does not
exist yet, or an input shape this route cannot receive. Nothing qualifies —
every shape the route accepts has a test at this head.

Parked hardening: none.

## Verification

- `tests/test_eom_link_verification.py` against a throwaway `postgres:16`:
  **19 passed**.
- Neighbouring EOM suites for regression, including the tracker-facing lead
  conversion tests: **693 passed, 0 failed**.
- **Negative controls, both run and restored:** removing the tenant predicate
  fails both real-PostgreSQL tenant tests; building `customerTypes` from the
  provider's rows instead of the requested ids fails the attribution test.
- The tenant claim is proven at the database, not through a stub. Every other
  test in the file stubs the provider and so cannot tell a scoped query from an
  unscoped one — and widening this route makes that claim matter more, since a
  leak would now disclose another tenant's account type rather than only the
  fact that an id exists.
- The pre-existing tenant test needed migration 366 added to its schema list,
  because the provider now selects that column.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 41 |
| `atlas_brain/services/crm_provider.py` | 13 |
| `plans/PR-EOM-Known-Contact-Types.md` | 180 |
| `tests/test_eom_link_verification.py` | 159 |
| **Total** | **393** |
