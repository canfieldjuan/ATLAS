# PR-EOM-Manual-Square-Invoice-Queue

## Why this slice exists

The Billing & Payments delivery issue #2362 requires a manual-Square route to
remain financially accurate without adding a Square API. The earlier
delivery-preference and approval slices already allow an explicitly selected
manual_square commercial candidate to create an ATLAS draft invoice. H-15 in
#2363 records the remaining provider gap: there is no durable operator queue,
external Square reference, or audited transition to sent via Square.

Diff-budget override: this exceeds the 400-line soft target because the
additive schema, transaction-owned state machine, mounted authenticated routes,
and disposable-PostgreSQL concurrency/recovery proof are one financial
lifecycle boundary. Splitting it would ship an operator mutation without its
durable evidence, retry recovery, or lifecycle proof.

### Problem-derived contract

- Root cause: an approved manual_square candidate has only a generic ATLAS
  draft invoice. Its policy is stored in invoice metadata, but no provider state
  owns the manual invoice reference or proves that the operator explicitly
  marked that exact draft invoice sent through Square. Reusing the legacy
  /invoicing/{id}/send action is incorrect because it marks an invoice sent
  before attempting email delivery.
- Correct fix must touch/change: add an additive, PostgreSQL-owned
  manual-Square delivery record and idempotent operation audit; derive a
  bounded queue from approved manual_square invoices; add authenticated provider
  routes to record one opaque external Square reference and then explicitly,
  transactionally mark the linked draft invoice sent with sent_via=square; prove
  the real routes and persistence behavior with disposable PostgreSQL and ASGI
  tests.
- Must not change: commercial candidate generation, approval/invoice creation,
  invoice PDFs, Gmail draft/sent reconciliation, the legacy MCP and generic
  invoice-send behavior, payments/allocations, CRM delivery preference policy,
  customer-facing Website shape, tracker ownership, or any real Square, Gmail,
  PDF, email, CRM, payment, or production-record operation.

## Scope (this PR)

Ownership lane: eom/billing-manual-square-queue
Slice phase: Vertical slice

1. Add an additive provider queue for commercial approval rows whose immutable
   ATLAS invoice metadata declares the already-canonical manual_square delivery
   method. An unrecorded row projects as needs_square_invoice; a durable
   reference record projects as reference_recorded; a conflicting legacy
   lifecycle projects visibly as lifecycle_conflict rather than being silently
   called sent.
2. Add two actor- and idempotency-key-protected provider actions: record one
   opaque Square invoice/reference string without changing financial state, and
   explicitly mark only that referenced draft invoice sent via Square. A
   different reference cannot overwrite the first one in this slice.
3. Add bounded queue paging and disposable-PostgreSQL/ASGI tests for normal,
   validation, authorization, retry, concurrent, stale-lifecycle, and recovery
   outcomes. Enroll the new provider contract in the existing invoicing check.

### Review Contract

- Acceptance criteria:
  - [ ] GET /api/v1/receivables/commercial-billing/manual-square-invoices
    returns bounded, paged manual_square approval/invoice rows without creating
    a delivery record or changing an invoice; settled by the real PostgreSQL
    queue test and ASGI route test in
    tests/test_commercial_billing_manual_square_invoices.py.
  - [ ] POST .../{approval_id}/manual-square-invoice-reference accepts one
    authenticated actor and safe opaque reference, stores actor/time/reference
    once, and leaves the invoice draft with no sent_at/sent_via; settled by the
    real PostgreSQL reference/retry assertions.
  - [ ] POST .../{approval_id}/manual-square-invoice/mark-sent can update only
    a referenced manual_square draft to status=sent and sent_via=square; absent
    reference, Gmail/non-Square approval, changed lifecycle, malformed request,
    and idempotency-key reuse with a different payload fail before a financial
    state write; settled by the real PostgreSQL state-machine tests.
  - [ ] Under the selected PostgreSQL transaction model, the operation-key
    advisory lock, approval lock, durable unique keys, and guarded
    UPDATE WHERE status=draft mean every admitted interleaving commits at most
    one immutable external reference and at most one Square sent transition;
    same-key retries replay and all other state races conflict; settled by the
    concurrent real-PostgreSQL test plus the service code.
  - [ ] Existing bearer authorization, X-EOM-Actor, and idempotency-header
    requirements protect both mutations and do not expose an unauthenticated
    financial action; settled by the full ASGI route test.
  - [ ] The service has no Square/Gmail/PDF/email/payment transport path, uses
    no floating-point money conversion, and the migration is additive with a
    documented non-destructive rollback; settled by source/migration contract
    tests and the local invoicing workflow mirror.
- Reachability proof: the full atlas_brain.main:app ASGI tests exercise each
  mounted /api/v1/receivables route with the existing digest bearer dependency
  and observe the provider response; disposable PostgreSQL tests observe the
  committed queue, operation, audit, and invoice rows.
- Affected surfaces: atlas_brain/api/invoicing/receivables.py; a new
  atlas_brain/services provider service; migration 376; the existing Atlas
  invoice/approval tables; the invoicing workflow; provider contract tests.
- Risk areas: financial lifecycle truthfulness, authorization, immutable audit
  evidence, external-reference input safety, migration rollout, backward
  compatibility, pagination, idempotency, concurrent retries, and stale
  invoice state.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R12, R14.

### Closure declaration

- Manual-Square delivery eligibility: CLOSED for this provider boundary;
  membership is DERIVED from the canonical
  EOMBillingDeliveryMethod.MANUAL_SQUARE value already persisted in the approved
  invoice metadata.deliveryMethod. Any different, absent, or malformed delivery
  value is outside the set and is rejected/not queued, because mutating a
  non-Square invoice is financially unsafe.
- Durable queue states and operation kinds: CLOSED and ENUMERATED in migration
  376 because this slice authors their finite state machine. An unrecognized
  stored state or operation kind is rejected as a conflict; a missing durable
  state row is the explicit needs_square_invoice projection, not an inferred
  send.

### Boundary-change enumeration

- Boundary path/seam: the three mounted provider routes named above.
- Replaced-path behaviors: no existing route is replaced. The generic
  invoice-send route remains outside this path; manual Square state is admitted
  only through the new explicit routes.
- Guard-relevant fields: approval UUID; service bearer; authenticated actor;
  idempotency key; operation kind; reference text; linked approval/invoice
  identity; invoice source, metadata delivery method, current invoice status,
  and durable queue state.
- Caller x input shape: bearer-only list reads admit bounded limit/offset;
  reference writes require a valid UUID, bearer, actor, key, and safe
  1-to-256-character opaque reference; mark-sent requires a valid UUID, bearer,
  actor, and key. Missing/invalid fields, wrong delivery policy, stale status,
  and changed idempotency payload are rejection shapes with zero new financial
  writes.

### Deployed-config probing

- Deployed/default config values: no new configuration or fallback is added;
  the existing require_receivables_api digest-bearer configuration remains the
  sole route admission boundary.
- Explicit value probe: ASGI tests provide a generated matching service bearer
  plus an actor and observe the mounted route response.
- Absent value probe: ASGI tests omit bearer, idempotency key, actor, and
  required reference independently and observe 401/422 before service calls.
- Default-session/default-context probe: the existing disabled/missing-bearer
  path is unchanged; this slice adds no session, tenant, or credential default.
- Side-effect ordering: both mutations validate route admission and request
  shape before service execution; service-side policy/lifecycle checks happen
  before any queue, operation, or invoice write, all of which commit atomically
  in one PostgreSQL transaction.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_manual_square_invoices.py`
- `atlas_brain/storage/migrations/376_commercial_billing_manual_square_invoices.sql`
- `plans/PR-EOM-Manual-Square-Invoice-Queue.md`
- `tests/test_commercial_billing_manual_square_invoices.py`

## Mechanism

Migration 376 creates one durable manual-Square record per approved invoice
only after the operator records a Square reference, plus a unique
(source, idempotency_key) operation receipt. A queue read is a left join over
the immutable approval/invoice identity, so historical approved manual-Square
drafts appear as needs_square_invoice without a destructive backfill.

record_reference locks the operation key and approval scope inside one
PostgreSQL transaction, confirms the invoice source and manual_square policy,
and inserts the immutable reference/audit row and operation receipt together.
The invoice remains draft. Repeating an identical reference with another key is
a no-op receipt; a different reference conflicts rather than overwriting
history.

mark_sent uses the same transaction model, requires that durable reference,
locks the linked invoice row, and performs a conditional draft-to-sent
transition with sent_via=square before committing the actor/time audit and
operation receipt. PostgreSQL advisory locks serialize same-key and
same-approval callers; unique constraints and the guarded update close races
with independently deployed writers. There is no external network operation, so
a crashed transaction commits neither lifecycle state nor a success receipt.

Deployment is provider-first and additive: deploy Atlas/migration before a
future tracker proxy or Website queue consumer. Rollback disables/reverts the
new route/service while retaining manual delivery evidence; dropping audit rows
is a separately authorized destructive retention action.

## Intentional

- No Square API integration, polling, fee calculation, payment creation, or
  credential is added. The reference is opaque operator evidence.
- Recording a reference never marks the invoice sent. Only the second, explicit
  actor-audited action can do so.
- The legacy generic invoice-send action is deliberately not reused because it
  marks sent before email delivery and would violate the manual-Square boundary.
- A missing durable record is projected as an actionable queue item rather than
  backfilled; a mismatched legacy invoice lifecycle is visible as a conflict.
- A different external reference is not editable in place. An append-only,
  explicitly reasoned correction action belongs in a later slice.

## Deferred

- eom-timetracker proxy and Website Needs Square invoice UI/recovery states
  consume this provider contract after Atlas deploys; tracked by #2362.
- An explicit, append-only Square-reference correction/reversal action requires
  its own audit/recovery contract; tracked in #2363 H-15 follow-up.
- Square API integration, automated send verification, and payment ingestion
  remain explicitly out of scope for this project and slice.

Parking predicate: consumer/UI decisions, automatic external-system behavior,
and recoverable reference-correction ergonomics are parked by default. A
finding that can create a false financial lifecycle, bypass authorization,
overwrite audit history, or break the documented provider flow is fixed inline.

Parked hardening: none.

## Verification

- Temporary-local-PostgreSQL provider suite: `105 passed`, covering canonical
  queue membership, no-write preview, immutable reference, same-key replay,
  changed-key conflict, recovery reuse, explicit Square transition,
  stale-lifecycle refusal, concurrent same-key serialization, input grammar,
  migration safety, no-transport boundary, and mounted ASGI auth/actor/key
  behavior. It creates and drops only a temporary schema in the local test
  container.
- Local Atlas Invoicing Checks mirror: approval blockers `2 passed`; ledger,
  PDF, payment, billing, and queue suite `533 passed`; MCP/OAuth surface
  `43 passed`.
- Migration-runner regression: `30 passed, 1 skipped`.
- Changed-path `ruff`, Python compilation, Black on new Python modules, and
  `git diff --check` passed. The service maturity lane reports zero findings.
- Pending before push: sync the final plan after this verification note,
  construct the canonical PR body, and run the managed local PR review plus
  its local full-unit baseline comparison.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/api/invoicing/receivables.py` | 113 |
| `atlas_brain/services/commercial_billing_manual_square_invoices.py` | 1006 |
| `atlas_brain/storage/migrations/376_commercial_billing_manual_square_invoices.sql` | 84 |
| `plans/PR-EOM-Manual-Square-Invoice-Queue.md` | 243 |
| `tests/test_commercial_billing_manual_square_invoices.py` | 739 |
| **Total** | **2192** |
