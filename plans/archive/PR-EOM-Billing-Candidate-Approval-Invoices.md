# PR-EOM-Billing-Candidate-Approval-Invoices

## Why this slice exists

Coordinating issue #2362 requires an operator-reviewed commercial candidate to
become an ATLAS financial invoice only after explicit approval. Current main
can retain and reconcile the review snapshot, but it has no approval command.
The legacy monthly task is not reusable: its review mode creates invoices and
PDFs, marks services invoiced, records CRM interactions, and can send mail.

This provider boundary exceeds the normal 400-line target because the
additive migration, one atomic exact-cent writer, authenticated entrypoint,
and isolated-PostgreSQL retry/rollback proof must deploy together. Splitting
those pieces would either publish an unreachable ledger writer or leave a
financial transaction without its schema invariant and failure evidence.

### Problem-derived contract

- Root cause: durable `commercial_billing_runs` rows remain immutable drafts,
  while the only existing invoice writer accepts float-oriented legacy inputs
  and the monthly task carries forbidden delivery and service-marker effects.
- Correct fix must touch/change: add an additive approval/audit record and
  source-scoped invoice idempotency constraint; add a separate exact-cent
  approval service and authenticated receivables endpoint that consumes one
  stored candidate only after current evidence matches its stored fingerprint.
- Must not change: pure preview/run creation and reconciliation; legacy invoice
  CRUD/MCP/monthly task behavior; payment, deposit, receipt, CRM, service
  marker, Gmail, email, PDF, and invoice-sent behavior. An approved invoice
  remains `draft` and no delivery artifact is created here.

## Scope (this PR)

Ownership lane: eom/billing-approved-gmail-drafts
Slice phase: Vertical slice

1. Approve one explicitly named, unblocked current candidate into one exact-cent
   ATLAS draft invoice with a stable source reference and actor/audit record.
2. Refuse stale, blocked, malformed, or UI-fingerprint-mismatched candidates
   before any invoice write; retries and concurrent requests reuse the original
   approval/invoice rather than duplicate financial records.
3. Add focused service, route, migration, and compatibility tests. Gmail/PDF
   artifact creation is intentionally deferred to the next provider slice.

### Review Contract

- Acceptance criteria: `tests/test_commercial_billing_approvals.py` proves an
  eligible commercial snapshot produces a `draft` invoice whose stored money is
  derived exclusively from integer cents; it proves same-key replay does not
  regenerate source evidence or a second invoice; it proves stale/blocked/UI
  mismatch failures attempt no insert; and it proves the route requires the
  existing service token, actor, and idempotency key. The migration test proves
  approval evidence and its invoice relation are restrictive and source scoped.
- Reachability proof: `POST /api/v1/receivables/commercial-billing-runs/{id}/approvals`
  reaches `CommercialBillingApprovalService.approve`; its observable result
  contains the approval and draft invoice identity. Existing UI consumers do
  not call the route until the tracker/UI proxy slice lands.
- Affected surfaces: receivables provider API, a new approval service, invoice
  table (additive source-specific unique index), approval audit table, focused
  tests, and invoice-workflow enrollment.
- Risk areas: stale-source race, duplicate invoices, integer-cent conversion,
  malformed retained snapshots, direct/manual-Square invoice eligibility,
  customer-facing invoice dates/copy, and accidental Gmail/service-marker work.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R10, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: approval admission verifies selected `candidateKey`,
  stored `expectedSourceFingerprint`, a fresh candidate projection, blockers,
  delivery method, canonical commercial identity, exact totals, and line-item
  arithmetic before an invoice insert.
- Replaced-path behaviors: the legacy monthly review path is not called. A new
  source-scoped `eom_commercial_billing` invoice path returns only the durable
  original approval on an unchanged retry.
- Guard-relevant fields: run UUID, candidate key, expected source fingerprint,
  idempotency key, actor, contact UUID, delivery method, line quantities/rates,
  subtotal/tax/total cents, and configured due days.
- Caller x input shape: route Pydantic validation bounds UI request strings;
  service-level tests use stored JSON snapshots and adversarial values so a
  direct caller cannot bypass the router's field grammar.

### Closure Declaration

- Retained candidate snapshot evidence is **OPEN**: it is producer-supplied
  nested JSON, so its keys, text, and container shapes cannot be exhaustively
  enumerated. Membership is **DERIVED** from the candidate service's canonical
  full-snapshot SHA-256 at preview time and recomputed by the approval service
  before it reads invoice fields. Any missing, malformed, or nonmatching
  fingerprint defaults to the safe side—unavailable/rejected with no invoice
  insert—because an unsealed snapshot could otherwise change money or customer
  identity.
- Invoice-capable delivery methods are **CLOSED** to `gmail_pdf` and
  `manual_square`, enumerated from the canonical delivery-method vocabulary in
  `EOMBillingDeliveryMethod`; every other value rejects before the writer,
  because `no_invoice_residential_receipt` and unknown values cannot safely
  create a commercial invoice.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: reuse the established
  `invoicing.auto_invoice_due_days` setting, whose default is 30 and whose
  configuration validation allows only 1 through 365 days.
- Explicit value probe: focused service tests inject a non-default due-days
  value and assert the draft due date derives from it.
- Absent value probe: this setting has a validated default; an invalid injected
  value is rejected before the invoice insert.
- Default-session/default-context probe: the route requires the existing
  receivables token and `X-EOM-Actor`; an absent actor/token reaches no writer.
- Side-effect ordering: idempotent existing approvals are returned before any
  source read; new approvals validate current evidence before the one atomic
  invoice-plus-approval transaction. No Gmail, email, PDF, CRM, or service
  marker call is imported or reachable.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_approvals.py`
- `atlas_brain/storage/migrations/372_commercial_billing_candidate_approvals.sql`
- `plans/PR-EOM-Billing-Candidate-Approval-Invoices.md`
- `tests/test_commercial_billing_approvals.py`

## Mechanism

The service loads the immutable candidate snapshot, uses its fingerprint as the
operator's explicit selection proof, and checks the current pure candidate
projection before a first approval. It recomputes the retained fingerprint
before consuming its fields, converts snapshot cents to Decimal only at the
NUMERIC database boundary, and stores cent-renderable JSON strings for the
draft invoice. Transaction-scoped locks cover the operation key and the
run/candidate pair; a partial unique index on the new source reference backs
the database invariant. The approval row and invoice are inserted atomically,
so a transport retry returns the same invoice without duplicating money.

## Intentional

- One candidate per command is deliberate. It lets later UI batch orchestration
  report a recoverable per-candidate result without turning one failed item into
  an ambiguous multi-invoice transaction.
- The established due-days setting is reused for invoice terms; no customer
  billing preference is inferred from residential/commercial type.
- The new invoice stays `draft`; this slice does not imply Gmail drafting,
  sending, a Square-send action, or a service invoiced marker.

## Deferred

- Durable PDF artifact creation/reuse and per-candidate recovery state: #2362,
  discovered while splitting this provider writer from forbidden legacy monthly
  side effects; track under #2363.
- Scoped no-send Gmail draft creation/recovery, sent-mail reconciliation, and
  manual-Square sent-state action: #2362 subsequent provider slices.
- Batch UI orchestration and tracker/website proxy adoption: #2362 after this
  provider contract is deployed.

Parking predicate: exact selected-candidate approval is the smallest safe
financial writer. Legacy invoice repository float cleanup, broad invoice model
refactors, and delivery work are parked because they do not block this proof.

## Verification

- `python -m pytest tests/test_commercial_billing_approvals.py` — 9 passed,
  2 skipped when no disposable PostgreSQL URL is configured; it covers exact
  cents, stale/blocked/malformed refusal, tamper refusal, retry, Square
  eligibility, route authorization, migration shape, and delivery-import scan.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=... python -m pytest
  tests/test_commercial_billing_approvals.py -q` — 11 passed against a local
  disposable PostgreSQL 16 container, including duplicate reuse and an injected
  approval-audit failure that rolls back the invoice. The container was removed
  after the test; no production records were queried or changed by this test.
- `python -m pytest tests/test_commercial_billing_approvals.py
  tests/test_commercial_billing_runs.py tests/test_commercial_billing_candidates.py
  tests/test_invoice_pdf.py tests/test_receivables.py tests/test_invoice_repository.py
  tests/test_eom_billing_recipients.py tests/test_eom_payment_receipts.py -q`
  — 186 passed, 28 environment-gated skips.
- `python -m ruff check atlas_brain/services/commercial_billing_approvals.py
  atlas_brain/api/invoicing/receivables.py tests/test_commercial_billing_approvals.py`
  and `python -m compileall -q atlas_brain/services/commercial_billing_approvals.py
  atlas_brain/api/invoicing/receivables.py` — passed.
- `python scripts/maturity_sweep_file_lane.py --min-score 8 --baseline
  tests/maturity_sweep/baseline_atlas_brain_api.json
  atlas_brain/api/invoicing/receivables.py
  atlas_brain/services/commercial_billing_approvals.py` — passed the ratchet;
  the receiver's existing score of 28 is unchanged and the new service scored 0.
- A read-only production-schema preflight counted zero existing
  `eom_commercial_billing` source references and zero duplicate source-reference
  groups before introducing the source-specific unique index.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/api/invoicing/receivables.py` | 68 |
| `atlas_brain/services/commercial_billing_approvals.py` | 636 |
| `atlas_brain/storage/migrations/372_commercial_billing_candidate_approvals.sql` | 58 |
| `plans/PR-EOM-Billing-Candidate-Approval-Invoices.md` | 204 |
| `tests/test_commercial_billing_approvals.py` | 612 |
| **Total** | **1585** |
