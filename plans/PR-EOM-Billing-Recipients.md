# PR-EOM-Billing-Recipients

## Why this slice exists

A commercial customer has no explicit billing-recipient relationship anywhere
in the system.

**Root cause.** Three concepts — the account, its service contact, and its
billing recipient — are approximated by scattered fields that nothing
reconciles:

- The tracker stores primary-contact identity and `billing_name` /
  `billing_email` / `billing_address` on one `customers` row, linked to exactly
  one `atlas_contact_id`.
- Atlas invoices separately store an optional `contact_id` **and** a
  denormalized `customer_email`, and `send_invoice` delivers to
  `inv["customer_email"]` without ever reading the contact.
- Nothing maps `customers.billing_email` into `invoices.customer_email`. There
  is no code path between them at all.

So the recipient of an invoice, the tracker's billing data, and the Atlas
contact association can all describe different things, silently. For a
commercial account whose AP address changes independently of its site contact,
that divergence is the normal case rather than an edge one.

Fixing it needs a relationship the tracker can store and the invoicing path can
resolve. Before the tracker can store one, Atlas has to be able to say which
contacts are permissible recipients and who they are. That is this PR.

### Diff budget

Over the 400-line soft cap. The code is a small fraction of it: the routes plus
the two provider methods and their helpers. The remainder is the plan document
and the tests, both required by this same contract.

This is not indivisible and I will not claim it is -- the detail route could
ship before the list route. The claim is that splitting is worse. The two share
the eligibility predicate and both projection helpers, so separating them either
duplicates that rule or lands a helper whose second consumer arrives next PR,
and each half carries its own copy of the plan document. Total reviewed lines go
up, and the disclosure decision that actually needs scrutiny gets reviewed twice
in halves rather than once whole.

### Problem-derived contract

- Root cause: no modelled billing-recipient relationship, and no read that lets
  a caller validate one.
- Correct fix must touch/change: a narrow Atlas read exposing recipient
  eligibility and recipient identity, behind the billing credential.
- Must not change: the pool that owns EOM contacts — these reads go through the
  funnel CRM provider, because `ReceivablesService.pool` resolves the separate
  global DSN which in the deployed slim topology is a different database;
  `/eom-funnel/known-contacts`, which stays id-and-type only —
  its value is that it discloses nothing, and widening it a second time turns
  link verification into a general contact reader; the EOM funnel credential,
  which gains no billing capability; any write path — this PR is read-only.

## Scope (this PR)

Ownership lane: eom-crm/billing-recipients
Slice phase: Vertical slice
Max files: 10

1. `ReceivablesService.list_billing_recipients` — eligible EOM contacts only.
2. `ReceivablesService.get_billing_recipient` — authoritative per-contact
   verdict.
3. Two GET routes on the receivables router, behind `require_receivables_api`.
4. Tests covering every eligibility branch, the tenant probe, the disclosure
   guarantee, and the credential boundary.
5. Pool lifecycle in `atlas_brain/eom_api/funnel_database.py` and
   `atlas_brain/main_eom.py`: the funnel CRM pool comes up when receivables is
   enabled **and a DSN is configured**, and closes under the same condition.
   Enabling receivables without a DSN must not stop a profile that never
   touches billing recipients from booting, so the routes fail closed with
   `billing_recipients_unavailable` (503) instead. Canonical-database
   admission is owed by whoever OPENS the pool, so
   `validate_eom_funnel_canonical_crm_config` now applies to the receivables
   path too — gating it on the funnel flag alone would open a configured DSN
   unadmitted, and a non-canonical Atlas database holding `effingham_maids`
   contacts would then be readable by the receivables bearer.
   `/receivables/ready` reports the contact pool as well, and distinguishes
   two states rather than collapsing them: **unconfigured** (no funnel DSN --
   a supported deployment that simply does not use billing recipients, so
   readiness still passes; failing it would take invoicing out over a
   capability it never calls) and **unavailable** (a DSN IS configured but the
   pool cannot serve the two queries, including a reachable but partially
   migrated database -- an initialized pool proves a connection opened, not
   that `contacts` carries the columns both queries name). Only the second
   fails readiness.
6. Enrolment of that test file in `.github/workflows/atlas_invoicing_checks.yml`
   — both path-filter blocks and the pytest arguments. The fifth file exists
   because the test skips without `ATLAS_RECEIVABLES_TEST_DATABASE_URL`, which
   only that workflow provisions; without it the only proof of the SQL
   eligibility logic would never run in CI.

Not in this PR: `billing_contact_id` on the tracker (1B), the resolver (1C),
invoice-creation integration (1D), and the caller-supplied-email refusal (1E).
**1E is the closure slice** — 1A–1D build correct data beside a still-live
divergence path.

### Review Contract

- Acceptance criteria:
  - An eligible contact reports its name and email — settled by
    `tests/test_eom_billing_recipients.py::test_the_billing_projection_answers_every_eligibility_case`.
  - Every ineligible cause (`archived`, `inactive`, absent email, blank email,
    missing id) returns `eligible: false` with `displayName` and `email` **null**
    and exactly five keys — same test. Negative control run: returning the row's
    name/email on refusal fails it with "an ineligible verdict leaked a name".
  - A contact under another tenant is byte-identical to a missing one — same
    test. Negative control run: removing the tenant predicate fails it.
  - The list offers eligible rows only — same test, asserting all five
    ineligible kinds are absent.
  - The funnel credential does not open this route, and a wrong bearer is 401 —
    settled by `::test_the_routes_sit_behind_the_receivables_credential`.
  - An ineligible verdict is 200 with a reason, not 404 — settled by
    `::test_an_ineligible_verdict_is_200_with_a_reason_not_404`.
  - The public reason set excludes `wrong_tenant` — settled by
    `::test_wrong_tenant_is_not_a_public_reason`.
- Reachability proof: `GET /api/v1/receivables/billing-recipients` and
  `/{contact_id}` on the receivables router, behind `require_receivables_api`.
- Affected surfaces: the receivables service and router; the funnel CRM pool's
  lifecycle and its canonical-database admission; app startup and shutdown in
  `atlas_brain/main_eom.py`; `/receivables/ready`; **configuration and
  deployment** (four environment variables, tabulated under Deployed-config
  probing); plus the new test file and one assertion in the render-profile
  suite.
- Risk areas: disclosing contact identity to a credential that should not have
  it; leaking a name or address on an ineligible verdict; making the route a
  cross-tenant existence oracle; admitting a contact that cannot actually
  receive mail.
- Reviewer rules triggered: R2 (test evidence), R3 (security/authorization —
  this is a new disclosure), R5 (backward compatibility — additive only),
  **R11 (config/env fallback — the admission and startup predicates now read
  four environment values, and the canonical-database gate applies to a second
  trigger)**, **R12 (deployment safety — a startup gate that can refuse to boot,
  and a readiness endpoint that can now fail)**, R14 (verify against the
  codebase).

### Boundary-change enumeration

Written for two read-only routes and never revisited as later rounds added an
eligibility rule, a normalization rule, and a startup admission gate. One group
per seam, each with what it replaced and the caller x input classes that reach
it.

**Seam 1 — recipient eligibility** (`DatabaseCRMProvider.get_billing_recipient`,
`.list_billing_recipients`)

- Replaced behavior: none existed; this is the new decision.
- Guard-relevant fields: `contacts.business_context_id` (tenant predicate, in
  the SQL), `contacts.status` (allow-listed to the single live value, since the
  column carries no CHECK), `contacts.contact_type` (**customer only** — a lead
  is refused), `contacts.email`.
- Caller x input: receivables credential x {eligible customer, archived,
  inactive, unknown-future status, **lead with a valid address**, absent email,
  blank/tab/newline email, malformed email, foreign-tenant id, missing id}.
  Funnel credential x any -> 401. No credential x any -> 401.
- Disposition: every class above is asserted in
  `::test_the_billing_projection_answers_every_eligibility_case`; the
  cross-tenant class is asserted byte-identical to missing.

**Seam 2 — address validity and representation**
(`normalize_contact_email`, exported from `eom_crm_mutations`)

- Replaced behavior: an SQL regex that was a **second, weaker grammar** — it
  admitted `a@b..com` and `a@.b.com`, and it returned the stored column, which
  `btrim` leaves with Unicode edge whitespace and mixed case. Deleted, not
  tightened.
- Guard-relevant fields: `contacts.email` only.
- Caller x input: both routes x {canonical address, Unicode-padded, ASCII-padded,
  mixed case, empty-label domain, no TLD, no `@`, double `@`, leading/trailing
  dot in local part, whitespace-only, absent}.
- Disposition: `::test_recipient_eligibility_follows_the_canonical_grammar`
  asserts the route verdict **equals** the canonical validator for every class;
  `::test_the_projection_returns_the_canonical_address_not_the_column` asserts
  the emitted address is the canonical form.

**Seam 3 — result completeness under paging** (`list_billing_recipients`)

- Replaced behavior: a single SQL `LIMIT`, which read N candidates rather than N
  recipients and let rejected rows displace eligible ones behind them.
- Guard-relevant fields: `contacts.id` (the cursor — immutable),
  `contacts.full_name` (display order only; **deliberately not the cursor**,
  because it is operator-editable and a rename between pages would skip or
  duplicate).
- Caller x input: {limit within one page, limit requiring several pages, a page
  entirely of rejected rows, a rename between pages, candidates exhausted early,
  scan cap reached}.
- Disposition: `::test_the_list_pages_until_the_requested_limit_is_filled`,
  `::test_paging_survives_a_rename_between_pages`, and
  `::test_the_list_stops_at_the_scan_cap_and_says_so` (which asserts the
  truncation is logged, not silent).

**Seam 4 — configuration admission and pool startup**
(`validate_eom_funnel_canonical_crm_config`, `init_eom_funnel_database`,
`atlas_brain/main_eom.py`, `/receivables/ready`)

- Replaced behavior: admission was demanded only when the funnel API was
  enabled, so the receivables path could open a configured DSN unadmitted.
- Guard-relevant fields: `ATLAS_EOM_FUNNEL_API_ENABLED`,
  `ATLAS_INVOICING_RECEIVABLES_API_ENABLED`,
  `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`,
  `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED` (values tabulated under
  Deployed-config probing).
- Caller x input: the full 2 x 2 x 10-DSN-shape x 2 grid.
- Disposition: `::test_canonical_admission_is_owed_by_whoever_opens_the_pool`
  and `::test_admission_holds_over_the_whole_configuration_grammar` (80 points
  against a rule-derived oracle);
  `::test_initialization_cannot_open_a_pool_admission_would_refuse` executes
  **both** predicates over 40 points and asserts initialization cannot precede
  admission; `::test_readiness_separates_unconfigured_from_unavailable` covers
  the readiness decision in both directions.

**Seam 5 — transport and credential** (two GET routes on the receivables router)

- Replaced behavior: none; additive.
- Caller x input: as Seam 1's credential rows. An ineligible verdict is **200
  with a reason**, not 404 — settled by
  `::test_an_ineligible_verdict_is_200_with_a_reason_not_404`.

### Deployed-config probing

This was written when the PR was two read-only routes and stayed "N/A" after
later rounds turned it into a config boundary. It is one: the diff makes the
admission and startup predicates depend on receivables enablement and the
funnel DSN, mirrors that predicate in `atlas_brain/main_eom.py`, and makes
readiness depend on the same values.

**Deployed / default values** (from repo-owned deployment config,
`render.eom.yaml:25-36`, and the typed defaults in
`atlas_brain/eom_api/config.py`):

| variable | blueprint | code default |
|---|---|---|
| `ATLAS_INVOICING_RECEIVABLES_API_ENABLED` | `"false"` | `False` |
| `ATLAS_EOM_FUNNEL_API_ENABLED` | `"false"` | `False` |
| `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` | `sync: false` (operator-supplied, absent by default) | `""` |
| `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED` | `"false"` | `False` |

So at blueprint defaults the pool never opens, admission is never demanded, and
the billing routes answer 503 — the gate is inert until an operator turns
something on. That is the settling shape, and it is the one the deployed EOM
slim profile starts from.

**Probes.** All three shapes are covered, and the grammar test walks them
exhaustively rather than by example:

- *explicit* — `api_enabled=True`, and `receivables_api_enabled=True` with a
  real DSN: admission demanded, and refused unless confirmed.
- *absent* — no DSN at all (empty, and every whitespace shape): the pool is not
  opened, admission is not demanded, readiness reports `unconfigured`, and the
  routes fail closed with 503 rather than 500.
- *default-session* — the blueprint row above: receivables off, funnel off,
  confirmed false. Nothing opens, nothing raises. Covered by the behavioural
  predicate test below, which walks it as one of its 40 points.

  The real-app probe
  (`::test_the_real_app_fails_closed_when_the_contact_pool_is_unconfigured`)
  is **not** the default session and this plan previously implied it was: it
  sets `ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true` deliberately, because its
  subject is the receivables-on/no-DSN deployment. It asserts
  `dependency_overrides == 0` so the production wiring is what answers.

**No side effect before admission.** `init_eom_funnel_database` opens no pool
until its predicate passes, and `validate_eom_funnel_canonical_crm_config` runs
at startup before it (`atlas_brain/main_eom.py`).

`::test_initialization_cannot_open_a_pool_admission_would_refuse` **executes
both** across all 40 configuration points and asserts the implication that
matters: if initialization opens the pool, admission was owed. An earlier
version of this test only compared the two function bodies for shared tokens,
which settles nothing — negating the receivables flag inside the initializer
keeps every token in place while opening a pool the validator never admitted.
That exact mutation is the test's negative control.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/eom_api/funnel_database.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_crm_mutations.py`
- `plans/PR-EOM-Billing-Recipients.md`
- `tests/test_eom_billing_recipients.py`
- `tests/test_eom_render_profile.py`

## Mechanism

Eligibility is computed **before** the projection is constructed, never after.
`_billing_recipient_projection` takes an already-eligible row and
`_billing_recipient_refusal` takes an id and a reason, so an ineligible answer
has no field to leak. Building a full contact response and blanking it on the
way out would make "never a partial row" depend on every failure branch
remembering to redact.

The tenant predicate is IN the detail query, so a contact belonging to another
tenant returns no row and is reported `not_found` — identical to one that does
not exist. The service never learns the difference, so no later branch, log
line, or error path can betray it.

`status` is allow-listed to the single live value rather than deny-listing the
inactive ones, because the column carries no CHECK constraint and enumerating
would silently admit any future status.

Address usability AND representation are decided by the **canonical**
normalizer (`normalize_contact_email`, exported from `eom_crm_mutations`), not
by a second expression of the grammar. The SQL regex this replaced admitted
`a@b..com` and `a@.b.com`, which the canonical write path rejects.

The normalizer returns the address rather than a verdict, because splitting the
two reintroduces the drift: `btrim(email, $3)` strips only the ASCII blanks it
is handed, while the validator also strips Unicode edge whitespace and
lowercases. A verdict-plus-column design reported ` ap@example.com `
eligible while returning an address nothing can send to, and emitted
`AP@Example.COM` where the canonical write path stores `ap@example.com`.

The list **pages** until the caller's limit is filled with eligible rows. A
single SQL `LIMIT` reads N candidates, not N recipients, so rejected rows
displace eligible ones ordered after them — 500 malformed rows followed by one
valid recipient returned an empty list for `limit=1`. Paging is keyset on
`(full_name, id)` rather than `OFFSET`, because a concurrent write under an
offset can skip or repeat a row. The scan is bounded at
`BILLING_RECIPIENT_MAX_PAGES` (20 x 500 = 10k candidates, far beyond the real
EOM contact count) and **logs a warning when that cap truncates the result**,
since a silently short list reads as "no eligible recipients".

## Intentional

- **Disclosing name and email is a deliberate decision, not a convenience.**
  `eligible: true` alone would validate machines and fail operators: assigning
  a recipient by UUID without seeing where money-related mail goes is how the
  wrong address gets chosen. The projection is exactly five fields — no phone,
  address, notes, tags, source, or lifecycle.
- **Behind the receivables credential, not the funnel one.** This is a billing
  capability and the funnel token is broad.
- **An ineligible verdict is 200, not 404.** "Not an eligible recipient,
  because X" is a domain answer the caller must handle, not a transport
  failure.
- **`wrong_tenant` is not a public reason.** An earlier draft of this contract
  listed it publicly while also requiring callers never receive it — a
  contradiction. Tenant-scoping the query removes the distinction rather than
  computing and suppressing it.
- **Rejected: widening `known-contacts`.** It already gained `customerTypes` in
  #2358. A second widening for identity fields would end its usefulness as a
  route that discloses nothing.

## Deferred

- `billing_contact_id` on the tracker customer, its assignment UI, the
  resolver, invoice-creation integration, and the caller-email refusal — 1B
  through 1E.
- The account/contact-role model. `billing_contact_id` is the specialised
  representation that a future `account_contact` relationship with role
  `billing` can absorb without wasted work.

Parking predicate: hardening is parked when it protects a caller that does not
exist yet, or an input shape this route cannot receive. Nothing qualifies —
every shape the routes accept has a test at this head.

Parked hardening: none.

## Verification

- `tests/test_eom_billing_recipients.py`, `tests/test_receivables.py` and
  `tests/test_eom_render_profile.py` against a throwaway `postgres`:
  **138 passed**.
- The fail-closed path is proven through the REAL app in a subprocess, with
  `dependency_overrides == 0` asserted, because every other route test here
  overrides `_billing_crm_dependency` and therefore could not see the
  production wiring. That is how the first version of the guard passed while
  being unreachable: the app installs the provider factory unconditionally, so
  a check placed after the factory branch never ran.
- Ruff clean on all three changed source files. The repo-wide ruff count is
  identical with these changes stashed, so it is pre-existing.
- **Negative controls, both run and restored:** removing the tenant predicate
  makes a foreign contact distinguishable from a missing one and fails the
  tenant probe; returning the row's name and email on an ineligible verdict
  fails with `an ineligible verdict leaked a name for inactive`.
- The first tenant control was rewritten: its initial form failed with an
  asyncpg type error rather than on the assertion, which would have "passed"
  as a control while proving nothing.
- The admission rule is proven over the whole configuration grammar --
  api_enabled x receivables_enabled x ten DSN shapes (blank, tab, CRLF,
  padded, real) x confirmed, 80 points, against an oracle derived from the
  rule ("admission is owed exactly when the pool is opened") rather than from
  the implementation. The defect was a whole FAMILY of configurations, not one
  case, so a fixture list would not have shown its shape.
- Additional negative controls, all run and restored: moving the availability
  check back after the factory branch fails the real-app test; gating admission
  on `api_enabled` alone fails the receivables-opens-the-pool case; dropping
  `billingRecipients` from readiness fails the real-app test.
- A regression the unit gate caught and this head fixes: initializing the pool
  whenever receivables is enabled made startup raise for a profile with no
  funnel DSN, breaking
  `test_eom_profile_reaches_receivables_ready_through_real_app`.
- Eligibility was checked against live data before being encoded: every EOM
  contact that actually receives invoices is `status='active'` with a usable
  email, so the rule admits the AP addresses this feature exists to reach
  rather than excluding them.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 5 |
| `.github/workflows/atlas_invoicing_checks.yml` | 11 |
| `atlas_brain/eom_api/funnel_database.py` | 58 |
| `atlas_brain/eom_api/receivables.py` | 119 |
| `atlas_brain/main_eom.py` | 8 |
| `atlas_brain/services/crm_provider.py` | 246 |
| `atlas_brain/services/eom_crm_mutations.py` | 27 |
| `plans/PR-EOM-Billing-Recipients.md` | 418 |
| `tests/test_eom_billing_recipients.py` | 892 |
| `tests/test_eom_render_profile.py` | 5 |
| **Total** | **1789** |
