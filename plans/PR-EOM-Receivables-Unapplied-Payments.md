# PR-EOM-Receivables-Unapplied-Payments

Issue: #2362

## Why this slice exists

Billing & Payments must record a payment for a canonical customer who has no
invoice, such as a residential customer. ATLAS currently requires an invoice
allocation at both versioned provider request models and the shared service layer.
The allocation also acts as the current proof that a supplied customer exists.

This first provider slice is additive and safe before tracker or website
consumers change.

### Problem-derived contract

- Root cause: the full `main:app` provider used by the deployed tracker and
  the slim EOM provider each have a CreatePaymentRequest that rejects [], and
  _normalize_allocations rejects it too, even though customer_payments, its
  event, its idempotency fingerprint, and its payment view already represent
  an unapplied payment.
- Correct fix must change only initial creation: admit missing/empty
  allocations, validate and key-share lock the canonical contact, skip invoice
  work for no invoice IDs, and preserve the existing fingerprint/replay path.
- Must not change: adjustment/MCP allocation requirements, nonempty allocation
  validation, invoice locks/recalculation, payment status/method behavior,
  checks/deposits/returns/voids, migration/schema, Gmail, or UI behavior.

## Scope (this PR)

Ownership lane: eom/receivables
Slice phase: Vertical slice
Max files: 5

1. Allow both POST /api/v1/receivables/payments provider implementations to
   omit allocations or send [].
2. Atomically create an unapplied payment/event for an existing canonical
   contact without querying, allocating to, or recalculating an invoice.
3. Keep same-key retries deduplicated and leave all non-create allocation
   boundaries strict.

### Review Contract

- Acceptance criteria: both provider request models accept missing/empty
  allocations; a valid empty request yields one received check payment, zero
  allocated cents, full unapplied cents, one event, and no invoice writes;
  unknown contacts fail before a parent/event insert; an unchanged retry
  returns its original ID and does not add a payment/event; default
  normalization and adjustments still reject []; legacy invoicing/MCP request
  behavior remains allocation-required.
- Reachability proof: the deployed tracker has a configured server-only token
  and its base URL matches the live Atlas Funnel, which proxies `/api` to the
  full `main:app`; that app includes `api.invoicing.receivables`. Both full and
  slim create-payment functions convert body.allocations and call
  ReceivablesService.create_payment. Focused tests invoke both real
  model/route functions, the service transaction, and an optional real-Postgres
  concurrent replay path.
- Affected surfaces: full and slim EOM create requests, payment creation
  normalizer, zero-invoice contact validation, and receivables tests.
- Risk areas: broad relaxation of adjustment admission, unknown contact,
  invoice balance mutation, retry duplication, and lifecycle regression.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R11, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: CreatePaymentRequest.allocations at POST /payments in
  the full `main:app` and slim EOM provider routers.
- Replaced-path behaviors: create changes one-to-100 rows to omitted/zero-to-100;
  AdjustAllocationsRequest remains one-to-100.
- Guard-relevant fields: allocations, invoice_id, amount_cents,
  total_amount_cents, contact_id, and Idempotency-Key.
- Caller x input shape: a valid EOM create caller sends no allocations or [];
  nonempty rows still require UUID plus strict positive cents.
- Boundary path/seam: _normalize_allocations(..., allow_empty=True) called only
  from create_payment_with_outcome, plus its no-invoice contact lookup.
- Replaced-path behaviors: exact [] is permitted only for initial creation,
  key-share locks an existing contacts.id, then returns no invoice rows to lock.
- Guard-relevant fields: exact list cardinality, contact ID, amount, source,
  and idempotency key.
- Caller x input shape: [] with an existing contact proceeds; absent/unknown
  contact and malformed/non-list/row shapes reject; default normalizer callers
  remain one-or-more.
- Closure declaration: allocation cardinality is CLOSED and DERIVED from
  CreatePaymentRequest plus create_payment_with_outcome: omitted, [], or the
  existing one-to-100 valid rows. Row contents are OPEN but retain UUID,
  positive exact-decimal, no-duplicate, and total-not-exceeded grammar.
  API_PAYMENT_METHODS, statuses, events, invoice states, and default
  normalizer callers are unchanged CLOSED sets.

### Deployed-config probing

- Deployed/default config values: no config, flag, credential, or migration is
  changed. The live `atlas-api.service` runs full `main:app` with
  ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true and a masked present token;
  the deployed tracker has a masked present service token and a base URL that
  matches that service's Funnel `/api/v1` route. Repo-owned render.eom.yaml
  remains an undeployed slim-service candidate with receivables disabled.
- Explicit value probe: tests use [] plus an inserted canonical contact and
  prove one payment/event with no invoice side effect.
- Absent value probe: the request-model test omits allocations and proves [].
- Default-session/default-context probe: isolated model, route, and fake-pool
  tests read no environment or credentials.
- Side-effect ordering: validation and fingerprinting precede the transaction;
  idempotency locks/existing-row checks precede a key-share contact lookup;
  only then can the parent and one payment event insert. A same-key replay
  returns before a second insert.

### Files touched

- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/services/receivables.py`
- `plans/PR-EOM-Receivables-Unapplied-Payments.md`
- `tests/test_receivables.py`

## Mechanism

Both provider request models default allocations to []. Only initial creation
passes allow_empty=True to the existing normalizer; all other callers retain
its strict default. With no invoice IDs, the service key-share locks
contacts.id and raises the domain not-found error if it is absent, then skips
invoice locking/allocation/recalculation. Existing Decimal amounts, parent
insert, payment event, fingerprint, and replay logic are unchanged.

## Intentional

- No migration is needed: customer_payments is already separate from
  invoice_payments and supports unapplied totals.
- The full live provider and the slim future EOM provider remain behaviorally
  equivalent; legacy invoicing/MCP stays allocation-required and unchanged for
  backwards compatibility.
- Customer selection, check metadata, receipts, tracker proxy, and website UI
  belong to later #2362 slices. Payment validity does not infer customer type.

## Deferred

- #2362 Slice 2 tracker proxy; Slice 3 customer-wide website payment entry;
  later receipt, ledger, billing preview/review, Gmail, sent-mail, and Square
  slices.
- #2363 H-01 legacy invoice float-money audit and H-06 duplicate-provider refactor remain separate.

Parking predicate: do not add customer fields, delivery, invoices, UI, or a
schema migration without a later provider-backed slice contract.

Parked hardening: #2363 H-01 and H-06.

## Verification

- pytest -q tests/test_receivables.py: focused tests pass locally; the
  real-Postgres concurrent-replay test is skipped only when
  ATLAS_RECEIVABLES_TEST_DATABASE_URL is absent.
- ruff check on all four changed Python files, git diff --check, plan sync,
  and plan-sync check pass locally.
- Cold reconstruction is recorded in the final PR body; publish through
  push_pr.sh exactly once with that audited body.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/receivables.py` | 2 |
| `atlas_brain/eom_api/receivables.py` | 2 |
| `atlas_brain/services/receivables.py` | 23 |
| `plans/PR-EOM-Receivables-Unapplied-Payments.md` | 166 |
| `tests/test_receivables.py` | 206 |
| **Total** | **399** |
