# PR-EOM-Customer-Ledger-History

## Why this slice exists

The Billing & Payments coordinating issue (#2362) needs a safe ATLAS read
contract before tracker and website can show a customer ledger. The existing
`GET /receivables/payments` endpoint only returns payment rows and only searches
payer name/reference; it has no invoice timeline, receipt-delivery projection,
or authoritative per-customer balance snapshot
(`atlas_brain/services/receivables.py:1017-1070`).

This slice closes that provider gap without changing the established mutation
paths. It follows the deployed receipt-outbox slice #2372 and is intentionally
read-only, so ATLAS remains the financial source of truth and the tracker can
later proxy one canonical query rather than inventing a second ledger.

### Problem-derived contract

- Root cause: ATLAS stores invoice state, customer payments, allocations,
  payment lifecycle state, deposit linkage, and receipt-outbox state, but its
  EOM HTTP read surface exposes only a payment-list projection. A portal cannot
  truthfully assemble customer history from that partial projection without
  becoming a parallel financial reporter.
- Correct fix must touch/change: add one bounded, customer-scoped query on
  `ReceivablesService`; expose the same authenticated route from full and slim
  EOM providers; prove the result through a real local PostgreSQL schema and
  both ASGI entrypoints.
- Must not change: no payment/invoice/deposit/return/void/retry write behavior,
  no invoice schema migration, no MCP tool behavior, no Gmail/receipt transport,
  no canonical-CRM lookup or new customer-facing UI/product wording, and no
  financial records outside isolated local test schemas.

## Scope (this PR)

Ownership lane: eom/customer-ledger
Slice phase: vertical slice
Max files: 5

1. Add `GET /receivables/customers/{contact_id}/ledger` to both the full and
   slim authenticated EOM provider routes. The additive response contains a
   bounded, deterministic combined timeline of invoice and payment entries;
   payment entries include allocations, unapplied cents, deposit/clearing
   state, return/void reasons, and receipt-delivery state.
2. Support the existing finance-safe query shapes: optional case-insensitive
   search over
   payer, payment reference (including check/Square references), receipt number,
   allocated invoice number, and invoice customer/name; optional payment
   status/method and date-range filters; `limit` 1..200 and nonnegative
   `offset`. Payment-only filters return payment entries only. The unfiltered
   balance snapshot remains current ATLAS state, never a filter-dependent or
   reconstructed historical balance.
3. Return `open_invoice_balance_cents` and
   `unapplied_payment_balance_cents` as current source-of-truth snapshots.
   Do not call either a historical running balance: immutable invoice lifecycle
   evidence is tracked as H-08 in #2363.
4. Require the already-deployed receipt-outbox schema for this new receipt-aware
   read contract and map an unavailable schema through the existing controlled
   receivables error boundary. Existing payment-list and mutation routes remain
   unchanged.
5. Prove normal, filtered, paginated, retry/repeated-read, unavailable-schema,
   and both-provider HTTP behavior locally without sending email or modifying
   any non-test financial data.

### Review Contract

- Acceptance criteria:
  - [ ] `ReceivablesService.list_customer_ledger` returns only the requested
    customer's deterministic invoice/payment entries, each payment with
    allocations, active-unapplied cents, deposit/clearing state, reversal
    reason/state, and receipt-delivery projection; settled by the real
    PostgreSQL service test in `tests/test_receivables.py`.
  - [ ] A case-insensitive payer, check/Square reference, receipt number, or invoice
    number search, payment status/method filter, date range, and bounded page
    return only matching entries; settled by the same service test, including
    an unrelated customer's rows.
  - [ ] `open_invoice_balance_cents` and
    `unapplied_payment_balance_cents` reflect current persisted invoice/payment
    state in integer cents; settled by the service test. The response does not
    claim a historical running balance.
  - [ ] A repeated/stale read and a rejected inverted date range create no
    payment, allocation, invoice, receipt, event, or email side effect;
    settled by before/after counts and the validation test.
  - [ ] The full and slim service-authenticated ASGI routes reach the same
    service contract, while an unavailable receipt-aware schema is reported by
    the existing `ReceivablesSchemaUnavailableError` boundary; settled by
    `tests/test_receivables.py`.
  - [ ] Existing `GET /receivables/payments` response semantics, all financial
    mutation routes, and MCP payment behavior remain backward-compatible and
    pass the focused receivables/MCP regression suite.
- Reachability proof: authenticated `GET /api/v1/receivables/customers/<uuid>/ledger`
  through both `atlas_brain.main.app` and `atlas_brain.main_eom.app` returns a
  bounded JSON ledger from a real local PostgreSQL service; no external email
  or financial system is contacted.
- Affected surfaces: receivables service; full and slim EOM HTTP routers;
  service-token authorization (existing dependency); receipt-outbox schema;
  local PostgreSQL/ASGI tests.
- Risk areas: financial-data truthfulness, backward compatibility, read-path
  pagination, schema availability, receipt privacy, no-write/retry behavior,
  performance/bounded loading.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R12, R14.

### Closure declaration

- Ledger entry type is **CLOSED** and **DERIVED** from the two owned source
  branches in the service query: `invoices` derives `invoice`; and
  `customer_payments` derives `payment`. No caller supplies an entry type, so
  an unlisted value cannot enter the response; rows outside those financial
  sources are omitted rather than guessed.
- Payment status and method are not locally enumerated policy sets. They are
  stored financial values compared exactly when a caller supplies a filter;
  an unknown value takes the safe read-only default of matching no payment
  rows. Existing lifecycle writers remain their canonical validators.

### Boundary-change enumeration

The new authenticated route is an additive read admission boundary. It does not
replace an existing route or widen an existing authorization dependency.

- Boundary path/seam: full and slim
  `GET /receivables/customers/{contact_id}/ledger` query parsing before
  `ReceivablesService.list_customer_ledger`. FastAPI rejects zero/out-of-range
  page limits before service invocation; the real ASGI test exercises that
  admission failure as well as the authenticated success path.
- Replaced-path behaviors: none; existing `GET /receivables/payments` retains
  its list/offset/query semantics and all mutations retain their current
  entrypoints.
- Guard-relevant fields: service token (existing router dependency), canonical
  UUID path `contact_id`, optional `search`, `payment_status`,
  `payment_method`, `from_date`, `to_date`, `limit`, and `offset`. FastAPI
  rejects non-UUID dates/limits; the service rejects an inverted date interval;
  unknown historical status/method values safely yield no matching payment rows.
- Caller x input shape:
  - full provider + valid token + valid contact/filter page -> ledger envelope;
  - slim provider + valid token + valid contact/filter page -> same envelope;
  - either provider + absent/invalid token -> existing auth failure before the
    service; malformed UUID/date/limit -> FastAPI 422; inverted dates ->
    controlled domain 422; receipt schema unavailable -> controlled 503;
  - repeated/stale valid GET -> fresh read only, no financial mutation.

### Deployed-config probing

No new environment/config fallback is introduced. Existing
`ATLAS_INVOICING_RECEIVABLES_API_ENABLED` and digest-only service-token
configuration gate both routes through `require_receivables_api`.

- Deployed/default config values: provider deployment already reports the
  receipt-aware receivables route ready; the slice adds no setting.
- Explicit value probe: local ASGI tests use a generated configured digest and
  matching bearer token.
- Absent value probe: existing auth regression tests retain the disabled/missing
  token failures; this slice does not bypass the router dependency.
- Default-session/default-context probe: no session/context fallback exists;
  `contact_id` is required by the path and unknown rows produce an empty ledger.
- Side-effect ordering: all filter/schema validation occurs before read queries;
  the read contract contains no write, queue, mail, or external call.

### Files touched

- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/services/receivables.py`
- `plans/PR-EOM-Customer-Ledger-History.md`
- `tests/test_receivables.py`

## Mechanism

`ReceivablesService` starts one `REPEATABLE READ, READ ONLY` transaction,
requires its receipt-aware schema capability inside that snapshot, then builds
one combined SQL page from invoices and customer payments constrained by the
requested `contact_id`. The union selects only identifiers plus stable ordering
fields; the service hydrates invoice rows and payment rows in bounded set
queries, groups allocations, and reads the one-to-one receipt outbox rows. The
page, nested details, and summary therefore reflect one committed database
snapshot; a concurrent commit falls entirely before or after the read. It
returns nested entry objects so invoice and payment field names cannot be
confused.

Payment references are intentionally the common check/Square reference source;
receipt numbers are queried from the durable outbox; allocated invoice numbers
are queried through `invoice_payments`. Current balance snapshot values use the
existing `invoices.amount_due` and active customer-payment allocations, then
convert exact `Decimal` values to integer cents at the service boundary. No
float is used for new financial calculations.

The route handlers only pass validated path/query fields into the service and
reuse `_call` for controlled domain/schema/database failures. They do not call
canonical CRM, Gmail, MCP, or any mutation.

## Intentional

- The new route is customer-scoped rather than a global free-text ledger scan.
  Canonical customer discovery belongs at the tracker boundary; requiring the
  selected ATLAS contact preserves the existing customer-payment index and
  avoids turning a read endpoint into an unbounded cross-customer report.
- Offset pagination is retained for consistency with existing receivables list
  clients. It is read-only: a stale page can be refreshed but can never create,
  allocate, email, deposit, clear, return, or void money.
- The response exposes a current balance snapshot, not a historical running
  balance. The invoices table has mutable state but no immutable invoice event
  history; inventing one from present-day values would be false financial
  evidence.
- No new migration is needed: all queried payment, allocation, deposit, and
  receipt-outbox data is owned by deployed additive migrations 344/368/369.
- The slice exceeds the normal 400-line soft target because one safe provider
  contract needs its bounded snapshot query and shared payment hydrator, plus a
  real PostgreSQL lifecycle fixture and mounted full/slim ASGI proof. Splitting
  that proof from the contract would weaken the financial no-write and
  entrypoint evidence rather than create an independently deployable slice.

## Deferred

- H-08 / #2363: add an immutable invoice financial-event basis before the UI
  uses the phrase "historical running balance." Discovered and linked from this
  slice in #2362.
- H-09 / #2363: `origin/main` maturity baselines omit existing
  `atlas_brain/api/invoicing/receivables.py` and
  `atlas_brain/templates/email/payment_receipt.py` evidence. The exact sweep
  fails identically on the base, so this finance-safe read slice must not mask
  it with an unrelated baseline rewrite. The follow-up must add direct
  test-discovery evidence and/or make an intentional targeted baseline-refresh
  decision. Discovered by #2373 and linked from #2362.
- Tracker slice: proxy this contract and combine it with canonical active
  customer search; ATLAS deliberately does not create another CRM reader here.
- Website slice: render ledger filters, CSV export from this canonical query,
  accessibility states, and recovery copy.
- Receipt sender slice: transition pending/failed delivery outbox rows only
  after verified transport; this query merely shows the current outbox state.

Parking predicate: park unrelated reporting redesign, global search/index
hardening, historical event-model work, and UI/product-copy choices unless they
block this provider read contract or prove a financial truthfulness/safety risk.

Parked hardening: H-08 is tracked in #2363 because historical balance wording
would otherwise overstate what mutable invoice rows can prove. H-09 is also
parked in #2363 because the maturity baseline/test-discovery defect predates
this slice and needs a dedicated coverage decision.

## Verification

- Passed locally (no GitHub Actions acceptance required by the operator):
  - `ATLAS_RECEIVABLES_TEST_DATABASE_URL=... python -m pytest -q
    tests/test_receivables.py tests/test_eom_payment_receipts.py
    tests/test_eom_billing_recipients.py tests/test_invoice_repository.py` —
    157 passed; the real PostgreSQL test creates only an isolated temporary
    schema and proves receipt/status/search/filter/page/repeated-read/schema
    failure/no-write behavior plus full/slim mounted ASGI reachability.
  - `ATLAS_RECEIVABLES_TEST_DATABASE_URL=... python -m pytest -q
    tests/test_invoicing_readonly_mcp.py tests/test_invoicing_draft_writer_mcp.py`
    — 22 passed, preserving existing MCP behavior.
  - `python -m compileall -q` for each changed Python file; `ruff check` for
    those files; and `bash scripts/check_ascii_python.sh` — passed.
- Before publication, `scripts/push_pr.sh` will run the repository's single
  local PR-review/unit-gate mirror with the same reviewed PR body; its result
  is recorded in the PR and coordinating issue. No real email, external
  financial record, Gmail draft, or production schema is contacted by these
  tests.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/receivables.py` | 27 |
| `atlas_brain/eom_api/receivables.py` | 27 |
| `atlas_brain/services/receivables.py` | 302 |
| `plans/PR-EOM-Customer-Ledger-History.md` | 267 |
| `tests/test_receivables.py` | 293 |
| **Total** | **916** |
