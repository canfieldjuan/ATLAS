# PR-EOM-Commercial-Invoice-Sent-Reconciliation

## Why this slice exists

The Billing & Payments workspace requires an ATLAS invoice to become sent only
after a manually sent Gmail draft has verifiable sent-mail evidence.  Provider
slice #2383 deliberately creates only a no-send Gmail draft and leaves the
invoice in `draft`; its durable row cannot be interpreted as delivery.  The
legacy `/invoicing/{id}/send` path marks an invoice sent before it attempts an
email, so it is not a safe reconciliation path.  This slice closes that gap
without sending mail or changing the draft-creation behavior.

Diff-budget override: the additive proof/outcome migration, bounded Gmail
metadata reader, durable idempotency state machine, authenticated route, and
real-PostgreSQL failure/concurrency proof are one financial-lifecycle boundary.
Splitting them would either let a provider write happen without durable recovery
evidence or ship a route that cannot safely prove or persist the required state.

### Problem-derived contract

- Root cause: the durable Gmail-draft state proves only intent and a draft
  identity; it has no proof-gated transition from the Gmail Sent mailbox to
  the ATLAS invoice lifecycle.  Reusing the legacy send route would invert the
  required ordering by recording `sent` before external delivery is known.
- Correct fix must touch/change: add an additive ATLAS migration for durable
  reconciliation outcome/idempotency evidence; add a read-only Gmail Sent
  lookup that verifies the stable message and invoice/approval headers; add a
  transactional service that updates the draft reconciliation record and the
  still-draft invoice together only after that evidence; expose the service
  through the existing authenticated receivables boundary; and prove normal,
  missing, malformed, retry, concurrent, and route-auth paths with a real
  temporary PostgreSQL schema and mocked Gmail transport.
- Must not change: no Gmail send, draft create, PDF create, invoice creation,
  payment/allocation, CRM/service evidence, Square workflow, customer-facing
  email/PDF copy, MCP behavior, or browser credentials.  The legacy automatic
  send route remains untouched and is not a dependency of this slice.

## Scope (this PR)

Ownership lane: eom/billing-sent-reconciliation
Slice phase: Vertical slice

1. Add `POST /api/v1/receivables/commercial-billing-approvals/{approval_id}/gmail-draft/reconcile`, protected by the existing receivables token,
   authenticated actor, and idempotency-key contract.
2. Reconcile only a durable, `draft_created` Gmail-PDF draft.  The service
   searches Gmail Sent mail by its stable RFC Message-ID, requires the `SENT`
   label, the expected Message-ID and namespaced approval/invoice headers, and
   a valid Gmail internal timestamp before marking the linked still-draft
   invoice sent via Gmail.
3. When Sent evidence is absent, record `draft_present` or `draft_missing`
   after a read-only draft lookup.  Neither result changes financial state; an
   ambiguous, malformed, or unavailable Gmail response fails closed.
4. Persist each reconciliation request and outcome.  An unchanged completed
   retry returns its original outcome without another Gmail lookup; a retry
   after a read failure remains safe because the Gmail operation was read-only.
5. Prove the route and service with no live Gmail/customer operation.  Tests
   cover the invoice transition, exact retry, missing/deleted draft,
   malformed/ambiguous sent proof, unchanged financial state on failures, and
   concurrent reconciliation requests.

### Review Contract

- Acceptance criteria:
  - [ ] An authenticated request to the real receivables route reaches the
    reconciliation service with its approval UUID, actor, and idempotency key;
    `tests/test_commercial_billing_gmail_drafts.py::test_full_atlas_app_gmail_sent_reconciliation_route_requires_existing_auth` settles it.
  - [ ] A Gmail message carrying the stable Message-ID, `SENT` label, matching
    approval/invoice headers, and valid Gmail `internalDate` atomically stores
    final Gmail identifiers/actor/time and changes only the linked draft invoice
    from `draft` to `sent` via `gmail`; the real-PostgreSQL happy-path test
    settles it.
  - [ ] Missing/draft-present/ambiguous/malformed/unavailable Gmail outcomes
    do not mark the invoice sent; the named real-PostgreSQL and mocked-transport
    tests settle each outcome.
  - [ ] A completed request replay returns its durable outcome without another
    Gmail lookup, while two fresh concurrent requests cannot produce more than
    one invoice lifecycle transition; the idempotency and concurrency tests
    settle the admitted interleavings.
  - [ ] The Gmail adapter uses only `users.messages.list` and
    `users.messages.get` for Sent reconciliation, never `messages.send`; the
    transport test settles the no-send boundary.
- Reachability proof: the full Atlas ASGI app test invokes the real
  `/api/v1/receivables/.../gmail-draft/reconcile` route under existing bearer
  and actor dependencies and observes the service result without exposing PDF
  bytes or contacting Gmail.
- Affected surfaces: receivables authenticated API; ATLAS invoices and durable
  Gmail-draft reconciliation records; Gmail OAuth transport read boundary;
  migration runner; invoicing workflow test enrollment.
- Risk areas: financial lifecycle truth, Gmail evidence ambiguity, replay,
  concurrent operator requests, migration compatibility, authorization, and
  inadvertent email send.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

The changed seam resolves one external Gmail mailbox result into an ATLAS
financial lifecycle transition.  It is deliberately finite and fails closed.

- Boundary path/seam: `CommercialBillingInvoiceGmailSentReconciliationService`
  decides whether one approved Gmail draft has proof that authorizes the invoice
  status transition.
- Replaced-path behaviors: no provider route previously reconciled a manually
  sent Gmail draft; the invoice remained draft.  The legacy send route remains
  outside this boundary because it sends/marks before email delivery.
- Guard-relevant fields: approval UUID; durable draft state; draft RFC
  Message-ID; expected approval/invoice IDs; Gmail message ID/thread ID,
  `SENT` label, metadata headers, and internal timestamp; current linked
  invoice status; actor and idempotency key.
- Caller x input shape:
  - Existing authenticated manager + exact single sent proof + linked draft
    invoice -> persist proof and mark sent.
  - Existing authenticated manager + no sent proof + retained draft -> persist
    `draft_present`, leave invoice draft.
  - Existing authenticated manager + no sent proof + missing draft -> persist
    `draft_missing`, leave invoice draft and expose recovery state.
  - Missing/creating/recovery-required draft, invalid/multiple/mismatched Gmail
    result, pre-sent invoice, or unavailable mailbox -> conflict/unavailable;
    no invoice transition.
  - Same idempotency key + completed result -> replay that result; fresh keys
    serialize final state under the approval lock.

### Deployed-config probing

N/A - no environment/config fallback changes.  Existing Gmail OAuth and
receivables authorization configuration are reused.  The migration records
intent/outcomes before/after only read-only Gmail requests; the invoice write is
inside the final evidence-confirming PostgreSQL transaction.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py`
- `atlas_brain/storage/migrations/375_commercial_billing_invoice_gmail_sent_reconciliation.sql`
- `atlas_brain/tools/gmail.py`
- `plans/PR-EOM-Commercial-Invoice-Sent-Reconciliation.md`
- `tests/test_commercial_billing_gmail_drafts.py`

## Mechanism

The draft service's deterministic RFC Message-ID and namespaced approval and
invoice headers are the stable external identity.  The Gmail transport performs
a bounded `messages.list` search restricted to `SENT`, then fetches metadata
for a unique candidate.  The service validates all expected fields rather than
treating a missing draft as delivery proof.

The service uses PostgreSQL, not a client cache, as the execution component.
It first stores an idempotency operation under the existing per-approval draft
advisory lock.  It releases that transaction before the read-only Gmail calls.
Its final transaction reacquires the same lock and locks the linked invoice:
only an exact proof can write the Gmail final identifiers and change a still
draft invoice to `sent`; every non-proof outcome persists a recovery-visible
observation and leaves invoice state unchanged.  Completed same-key operations
replay their stored outcome.  Fresh-key requests may perform concurrent
read-only lookup, but their final transitions serialize on the approval lock,
and a confirmed sent state cannot be downgraded by a stale observation.

The execution model admits process failure before or after either transaction,
network failure during Gmail reads, same-key retry, fresh-key concurrent retry,
and a concurrent draft create/recovery.  Gmail reads have no financial or
delivery side effect; the durable operation makes every later attempt either
return the completed outcome or re-query safely.  The shared approval lock
prevents a reconciliation from interpreting a still-creating draft as missing,
and the final invoice row lock prevents any interleaving from overwriting an
already-confirmed financial lifecycle transition.  The model assumes Gmail's
metadata API is the evidence source; an unavailable, malformed, zero-match, or
multi-match response never authorizes `sent`.

## Intentional

- Gmail's `messages.send`, draft creation, PDF generation, and browser-facing
  PDF bytes are absent from this slice.  Gmail remains a delivery system; ATLAS
  retains the financial lifecycle state and audit actor/time.
- A deleted/missing Gmail draft is stored as `draft_missing`, not inferred sent.
  The response exposes that recovery state for a later Website view; it does
  not automatically create a replacement draft because an operator must first
  resolve any mailbox-indexing or manual-delivery ambiguity.
- The transition accepts only an invoice still in `draft`, the same state that
  created the approved artifact and Gmail draft.  A legacy/out-of-band status
  update conflicts rather than silently rewriting financial history.
- This is a new delivery/lifecycle slice, not a modification of the legacy
  `/invoicing/{id}/send` behavior or a new customer-visible email/PDF design.

## Deferred

- Website Sent-mail recovery view and an explicit, audited replacement-draft
  operator action after `draft_missing` are deferred to the Billing & Payments
  UI/recovery follow-up and will be linked to #2363 from this slice.
- Manual Square invoice queue/reference recording, Gmail sent-mail polling or
  webhook scheduling, and cross-repository tracker/Website consumers remain
  deferred to their already-tracked Billing & Payments slices (#2363).

Parked hardening: none.

## Verification

- Temporary-PostgreSQL Gmail draft/reconciliation suite: `86 passed`.
- Local `Atlas Invoicing Checks` mirror: approval blockers `2 passed`;
  ledger/PDF/billing suite `428 passed`; MCP/OAuth surface `43 passed`.
- Migration runner regression: `30 passed, 1 skipped`.
- `ruff`, Python compilation, `git diff --check`, and the new sent-mail service
  maturity lane passed with zero service findings.
- Final local PR review passed: body contract, diff-budget override,
  AI-reconciliation record, plan/code consistency, guard closure, and the
  locally escalated full unit gate. Its `160` failing/error nodes matched the
  trusted baseline exactly (zero regressions and zero newly-passing nodes).
- No test contacted Gmail or altered a live financial record; the database
  tests created and dropped only temporary local schemas.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 4 |
| `atlas_brain/api/invoicing/receivables.py` | 65 |
| `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py` | 1100 |
| `atlas_brain/storage/migrations/375_commercial_billing_invoice_gmail_sent_reconciliation.sql` | 120 |
| `atlas_brain/tools/gmail.py` | 211 |
| `plans/PR-EOM-Commercial-Invoice-Sent-Reconciliation.md` | 221 |
| `tests/test_commercial_billing_gmail_drafts.py` | 549 |
| **Total** | **2270** |
