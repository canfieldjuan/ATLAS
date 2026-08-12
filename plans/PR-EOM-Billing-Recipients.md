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
Max files: 7

1. `ReceivablesService.list_billing_recipients` — eligible EOM contacts only.
2. `ReceivablesService.get_billing_recipient` — authoritative per-contact
   verdict.
3. Two GET routes on the receivables router, behind `require_receivables_api`.
4. Tests covering every eligibility branch, the tenant probe, the disclosure
   guarantee, and the credential boundary.
5. Pool lifecycle in `atlas_brain/eom_api/funnel_database.py` and
   `atlas_brain/main_eom.py`: the funnel CRM pool must come up when receivables
   is enabled, not only when the funnel API is, and close under the same
   condition. Reading contacts from the pool that owns them ties this slice to
   that pool's lifecycle.
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
- Affected surfaces: the receivables service and router, plus one new test file.
- Risk areas: disclosing contact identity to a credential that should not have
  it; leaking a name or address on an ineligible verdict; making the route a
  cross-tenant existence oracle; admitting a contact that cannot actually
  receive mail.
- Reviewer rules triggered: R2 (test evidence), R3 (security/authorization —
  this is a new disclosure), R5 (backward compatibility — additive only), R14
  (verify against the codebase).

### Boundary-change enumeration

- Boundary path/seam: two new GET routes on an existing authenticated router.
- Replaced-path behaviors: none. Nothing existing changes shape.
- Guard-relevant fields: `contacts.business_context_id` (in the query),
  `contacts.status`, `contacts.email`.
- Caller x input shape: receivables credential x {eligible id, archived id,
  inactive id, no-email id, blank-email id, foreign-tenant id, missing id};
  funnel credential x any; no credential x any.

### Deployed-config probing

N/A - no guard/config boundary change. The route reuses
`require_receivables_api` and its existing configuration; no environment or
config value is newly read.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/eom_api/funnel_database.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Billing-Recipients.md`
- `tests/test_eom_billing_recipients.py`

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

- `tests/test_eom_billing_recipients.py` plus `tests/test_receivables.py`
  against a throwaway `postgres`: **68 passed**.
- Ruff clean on all three changed source files. The repo-wide ruff count is
  identical with these changes stashed, so it is pre-existing.
- **Negative controls, both run and restored:** removing the tenant predicate
  makes a foreign contact distinguishable from a missing one and fails the
  tenant probe; returning the row's name and email on an ineligible verdict
  fails with `an ineligible verdict leaked a name for inactive`.
- The first tenant control was rewritten: its initial form failed with an
  asyncpg type error rather than on the assertion, which would have "passed"
  as a control while proving nothing.
- Eligibility was checked against live data before being encoded: every EOM
  contact that actually receives invoices is `status='active'` with a usable
  email, so the rule admits the AP addresses this feature exists to reach
  rather than excluding them.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 3 |
| `atlas_brain/eom_api/funnel_database.py` | 26 |
| `atlas_brain/eom_api/receivables.py` | 74 |
| `atlas_brain/main_eom.py` | 8 |
| `atlas_brain/services/crm_provider.py` | 157 |
| `plans/PR-EOM-Billing-Recipients.md` | 224 |
| `tests/test_eom_billing_recipients.py` | 287 |
| **Total** | **779** |
