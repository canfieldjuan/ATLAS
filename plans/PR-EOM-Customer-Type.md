# PR-EOM-Customer-Type

## Why this slice exists

Website umbrella #105 makes the portal Leads page the CRM UI. Its Slice 2 (the
create/edit form) cannot be built until an account record can state whether it
is residential or commercial: the form has to offer that selector, and which
billing fields apply follows from the answer. Slice 1 = Req A = website #174.

**Root cause.** There is no customer-level type anywhere in the system. Type
exists only at the SITE level (`locations.location_type`, tracker). The record
that IS the account -- an Atlas contact, in a verified flat model with no
account/person/site hierarchy -- carries no type at all. Two consequences, and
they are the defect:

1. Every consumer must infer type -- from site rows, from a name, or from a
   human remembering. An account with no site has no answer at all.
2. Billing shape is a CONSEQUENCE of customer type, so with no customer-level
   type nothing prevents a residential customer carrying commercial billing
   data, and nothing detects it.

"The Edit Customer form shows billing fields to everyone" is the symptom, not
the cause, and is deliberately NOT fixed here. Hiding fields while the record
still cannot state its own type moves the ambiguity somewhere harder to see.

Measured before building, not assumed: 54 tracker customers (52 active), all 54
already linked to Atlas; site type coverage is 34 Residential + 18 Commercial =
52/52 active with zero ambiguous and zero missing. Atlas's 163 "active
customers" is inflated by `calendar_import` (92) and `email_backfill` (64),
which are not accounts. So the evidence to classify every real account already
exists and no name heuristic is needed.

### Why this slice is indivisible above the 400-LOC budget

763 insertions. The application change is 57 LOC across three files: the
migration, the boundary field plus its normalizer, and one column in the
provider's INSERT. The rest is the backfill (203), its proof (256), the
integration proof (194), and CI registration.

No split leaves both halves reviewable. The migration without the boundary ships
a column nothing can set. The boundary without the migration writes a column
that does not exist. The backfill without its tests is an unreviewed bulk UPDATE
over every real customer account, where a wrong value silently changes billing
behaviour -- the single highest-consequence piece here. What COULD be split was:
the tracker mirror and the Edit Customer UI are separate PRs and are not in this
diff.

### Problem-derived contract

- Root cause: stated above -- no customer-level type, so type is inferred and
  billing shape is unconstrained.
- Correct fix must touch/change: the Atlas contact record (a constrained column,
  enforced by the database and not only by application code); the 0B operator
  mutation boundary, whose field allowlist is closed and would 422 the new field
  otherwise; the provider's explicit INSERT column list; and a backfill that
  classifies from real evidence with a reviewable report.
- Must not change: `contact_type` (lead|customer -- a different axis, not
  overloaded and not inferred from this one); `locations.location_type` (read as
  backfill evidence, otherwise left exactly where it is); billing field storage
  (stays in the tracker `customers` table); the flat model (no account ->
  contact-person -> site hierarchy); any write path outside the canonical
  boundary, with no 0A guard exemption.

## Scope (this PR)

Ownership lane: eom-crm/customer-type
Slice phase: Vertical slice
Max files: 11

1. `customer_type` on the Atlas contact record, `residential | commercial |
   unknown`, defaulting to `unknown`, enforced by a CHECK constraint.
2. The field carried through the 0B operator mutation boundary -- and only
   through it.
3. A backfill fed by tracker site evidence, dry-run by default.

Deliberately not here: the tracker mirror, site type derivation, and the Edit
Customer selector with adaptive billing. Those are PR2 and PR3.

### Review Contract

- Acceptance criteria:
  - The database itself refuses an out-of-set value, not just the application --
    settled by
    `tests/test_eom_lead_conversion_integration.py::test_the_database_refuses_a_customer_type_outside_the_set`,
    which asserts a `CheckViolationError` from a direct UPDATE. Negative control
    run: deleting the CHECK block from migration 366 fails it.
  - The boundary rejects a bad value as 422 before the database sees it, so a
    caller gets a validation error rather than a 500 -- settled by
    `::test_the_boundary_refuses_a_bad_customer_type_before_the_database_sees_it`,
    which also asserts the capitalised `Residential`/`Commercial` the tracker
    stores are accepted. Negative control run: removing the `customer_type`
    branch from `_normalize_fields` fails it.
  - A create that states a type stores it rather than falling back to the column
    default -- settled by `::test_operator_create_persists_customer_type_rather_than_defaulting`.
    Negative control run: removing `customer_type` from `_insert_contact_row`'s
    explicit column list fails it.
  - A create that states no type is `unknown`, never guessed from a
    company-shaped name -- settled by
    `::test_operator_create_without_a_type_is_unknown_not_guessed`.
  - Changing a type records the overwritten value, since no contact history
    table exists -- settled by
    `::test_operator_update_changes_customer_type_and_audits_the_old_value`,
    asserting `previous_values["customer_type"]`.
  - The backfill does not write on a dry run, writes both types on apply, is
    idempotent, refuses another tenant, refuses an unknown contact id, refuses a
    value outside the set, and never overwrites a decision already made --
    settled by the seven behavioural tests in
    `tests/test_backfill_eom_customer_type.py`.
  - The backfill's UPDATE carries its own tenant and never-overwrite guards, as
    a check-then-act race guard independent of the Python pre-checks -- settled
    by `::test_the_apply_statement_itself_refuses_wrong_tenant_and_already_set`.
    Negative control run: neutering the WHERE to a tautology (keeping `$3` bound
    so it is not an arity artifact) fails that test and only that test.
  - The new write site is recorded in the 0A inventory rather than exempted --
    settled by the single added line in
    `tests/contact_write_boundary/baseline.json`.
  - The proof actually runs in CI -- settled by
    `.github/workflows/atlas_eom_lead_pipeline_checks.yml` (pytest argument plus
    both path-filter blocks) and by adding 366 to `_prepare_schema`'s migration
    list, without which every operator write in that suite fails.
- Reachability proof: real entrypoint is `POST /api/v1/eom-funnel/operator-contacts`
  with `customerType` in the body, behind `require_eom_funnel_api` +
  `require_eom_funnel_actor`. Observable effect is `contacts.customer_type` and
  the `contact_updated` lifecycle event.
- Affected surfaces: the contacts table, the operator mutation contract, the
  provider's contact INSERT, the funnel request/response models, one new
  operator script.
- Risk areas: a create silently defaulting the caller's value; a value the
  boundary accepts but the CHECK rejects (500 instead of 422); the backfill
  crossing tenants or stomping a manual correction; a new contact write site
  escaping the 0A inventory.
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R3
  (security and authorization -- the field is settable only behind the
  authenticated funnel boundary and the backfill is tenant-scoped), R4 (data and
  migration safety -- additive migration plus a bulk UPDATE over live accounts),
  R5 (backward compatibility -- new request and response fields), R7 (input
  bounds -- a closed value set), R12 (deployment safety and CI enrollment), R14
  (verify against the codebase).

### Boundary-change enumeration

- Boundary path/seam: `_normalize_fields` in
  `atlas_brain/services/eom_crm_mutations.py` -- a closed field allowlist that
  gains a member, plus a new per-field validator.
- Replaced-path behaviors: none. `customer_type` was previously rejected as "not
  an operator contact field"; nothing else changes shape.
- Guard-relevant fields: `customer_type` -- membership of
  `EOM_CUSTOMER_TYPES`, case, surrounding whitespace, blank, non-string; and
  `contacts.business_context_id` in the backfill's UPDATE.
- Caller x input shape: operator create x {stated type, absent type}; operator
  update x {same value, changed value}; boundary x {`residential`,
  `Residential`, `  COMMERCIAL  `, `unknown`, `bogus`, `""`, `"   "`, `None`,
  `3`}; backfill x {valid pair, wrong tenant, unknown id, out-of-set value,
  already-decided row, re-run}.

### Member-set closure: `EOM_CUSTOMER_TYPES`

- Open or closed: CLOSED. Adding a member is a migration (the CHECK) plus a code
  change, deliberately, because billing behaviour keys off these values.
- Where membership comes from: the tuple at
  `atlas_brain/services/eom_crm_mutations.py:44`, which must stay identical to
  `chk_contacts_customer_type` in migration 366. The CHECK is the enforcement;
  the tuple only decides what the boundary admits, so a value accepted here and
  refused there would surface as a 500 rather than a 422.
- Unlisted values: refused with 422 at the boundary and by the constraint at the
  database. `unknown` is a real member, not a NULL substitute -- a contact whose
  type was never established is a distinct, honest state.
- Drift policy: the funnel request model deliberately does NOT restate the set
  as a `Literal`; that would be a third copy and would reject the capitalised
  values the tracker stores before the case-folding normalizer runs.

### Deployed-config probing

N/A - no guard/config boundary change. No environment or config value is read;
the value set is a module constant bound to a database constraint.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_crm_mutations.py`
- `atlas_brain/storage/migrations/366_contacts_customer_type.sql`
- `plans/PR-EOM-Customer-Type.md`
- `scripts/backfill_eom_customer_type.py`
- `tests/contact_write_boundary/baseline.json`
- `tests/test_backfill_eom_customer_type.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

The migration adds the column with a conforming DEFAULT and then the CHECK. No
`DO`-block atomicity is needed (unlike 365's seed+FK pair): no writer sets the
column until the boundary change ships, so no row can exist in the window that
the constraint would reject -- and if one somehow did, ADD CONSTRAINT validates
existing rows and aborts loudly rather than admitting it.

`_normalize_customer_type` folds case before checking membership. That is not
politeness: the evidence this field is populated from is the tracker's
`location_type`, which stores `Residential` and `Commercial` capitalised, so
rejecting them would refuse the very values the backfill reads. Blank is refused
rather than mapped to `unknown` -- every other text field here maps blank to
NULL meaning "clear it", and that mapping would let an empty form field silently
downgrade a commercial account, a value change disguised as a no-op.

The provider's UPDATE path builds its SET clause from the caller's fields and so
carries a new column for free. Its INSERT does not: the column list is explicit,
so an omitted column is written from its DEFAULT and the caller's value is lost
with no error. `customer_type` is therefore named explicitly at
`crm_provider.py:854,886`.

The backfill takes its mapping as a FILE rather than reaching into the tracker.
An Atlas maintenance script should not carry Render credentials for a second
datastore, and a file-driven script can be proven end-to-end in CI, which one
that dials production cannot. The tracker query that produces the mapping is in
the script's docstring; its `HAVING` clause emits nothing for a customer whose
sites disagree, so mixed-type accounts stay `unknown` rather than being guessed.

## Intentional

- **`unknown` is a member of the set, not a NULL.** A contact whose type has
  never been established is a distinct state, and the ~650 calendar/email import
  rows are not accounts at all, so `unknown` is the correct answer for them.
- **The CHECK, not just the validator.** Application validation is code a future
  writer can bypass; the constraint is not.
- **The backfill never overwrites a non-`unknown` value**, so a re-run is a
  no-op and an operator's later correction in the CRM survives a stale mapping
  file. A disagreement is reported and exits non-zero rather than being applied.
- **Rejected: classifying by name.** "AKRA Builders" reads commercial and "Anna
  McClellan" residential, but a wrong confident value puts a residential
  customer into commercial billing. Site evidence covers 52/52 active accounts
  unambiguously, so the heuristic buys nothing and risks real damage.
- **Rejected: a `Literal` on the request model.** See the closure section --
  a third copy of the value set, and it would reject the tracker's capitalised
  values.

## Deferred

- PR2 (eom-timetracker): mirror `customer_type` for ops reads, derive site type
  from the parent customer, and enforce the billing rule server-side.
- PR3 (website): the Edit Customer selector and adaptive billing fields, gated
  on the 0E capability manifest.
- Website #174 and umbrella #105 stay open until those land.

Parking predicate: hardening is parked when it protects a caller that does not
exist yet, or an input shape this boundary cannot receive. Against that
predicate, nothing is parked -- every shape the boundary accepts has a test at
this head, and the only caller is the operator boundary itself.

Parked hardening: none.

## Verification

- EOM CI test list on a clean throwaway `postgres:16`: **891 passed, 0 failed**.
- Every guard carries a negative control that was run and did fail before
  restore: the CHECK, the boundary normalizer, the provider INSERT column, and
  the backfill's SQL guards.
- **One control initially failed to fail, which is why the suite changed.**
  Removing the backfill's SQL guards left all seven tests green, because the
  Python pre-checks `continue` before the UPDATE is ever issued -- the WHERE
  clause was untested. The guards are not redundant: they are the check-then-act
  race guard between reading the row and writing it. A test that exercises
  `SQL_APPLY` directly was added, and a clean control (neutering the WHERE to a
  tautology while keeping `$3` bound, so the failure is not an arity artifact)
  now fails that test and only that test.
- Migration proven directly at the database: default applied, all three values
  accepted, `bogus` rejected by `chk_contacts_customer_type`, NULL rejected by
  NOT NULL, and a re-run is a no-op that preserves an already-set value.
- The 0A write-boundary guard detected the backfill's UPDATE and it was recorded
  in the inventory as a one-line reviewable diff, not exempted.
- Tests run only against a throwaway container, never the live `atlas` database.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 7 |
| `atlas_brain/eom_api/funnel.py` | 11 |
| `atlas_brain/services/crm_provider.py` | 10 |
| `atlas_brain/services/eom_crm_mutations.py` | 36 |
| `atlas_brain/storage/migrations/366_contacts_customer_type.sql` | 45 |
| `plans/PR-EOM-Customer-Type.md` | 287 |
| `scripts/backfill_eom_customer_type.py` | 235 |
| `tests/contact_write_boundary/baseline.json` | 4 |
| `tests/test_backfill_eom_customer_type.py` | 320 |
| `tests/test_eom_lead_conversion.py` | 77 |
| `tests/test_eom_lead_conversion_integration.py` | 198 |
| **Total** | **1230** |
