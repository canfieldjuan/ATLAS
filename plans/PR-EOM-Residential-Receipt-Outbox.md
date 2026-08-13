# PR-EOM-Residential-Receipt-Outbox

## Why this slice exists

Billing & Payments coordination issue [#2362](https://github.com/canfieldjuan/ATLAS/issues/2362) requires a residential payment to commit even when email cannot be delivered, while retaining one durable, retry-safe receipt-delivery record.  The current receivables transaction persists a payment, allocations, and a payment event, but has no receipt number, receipt email snapshot, or delivery status.  The current Gmail and Resend send abstractions also do not accept a caller-owned idempotency key, so calling either transport from the payment transaction would make an uncertain HTTP failure indistinguishable from a duplicate customer email.

### Problem-derived contract

- Root cause: the financial payment transaction has no durable, payment-keyed receipt outbox; neither the deployed full EOM route nor the slim EOM profile resolves customer type/email from its authoritative CRM before committing a payment.  The user-required draft-PR workflow also leaked its one-shot draft-consent environment variable into local verification, invalidating the wrapper's own no-consent safety fixtures.
- Correct fix must touch/change: add an additive ledger-side receipt-outbox table and readiness migration; have both EOM payment routes obtain an active canonical customer snapshot from their respective CRM boundary; atomically create one residential receipt record with the payment; render stable receipt contents and expose a minimal delivery projection; test transaction/replay/no-email/failure-before-write behavior; consume draft consent after argv admission before invoking local verification.
- Must not change: payment amounts, allocations, idempotency fingerprints, check/deposit/clearing/return/void lifecycle, generic MCP invoicing behavior, Gmail/Resend transport behavior, or any real customer email delivery.

## Scope (this PR)

Ownership lane: eom/billing-receipts
Slice phase: Vertical slice
Max files: 15

1. `POST /api/v1/receivables/payments` on both the deployed full EOM application and the slim EOM profile will read the active canonical EOM customer before the financial write.  A residential snapshot creates exactly one payment-keyed receipt record in the same ledger transaction: `pending` with a canonical email, or `skipped` with `no_email` when no usable canonical email exists.  Commercial and unknown customers receive no residential receipt record.
2. A payment created with a canonical residential snapshot will add a
   receipt-delivery projection containing receipt number, recipient, and
   status.  The payment itself remains valid and committed when the email is
   absent; this slice deliberately queues only and never calls Gmail or
   Resend.  A retry whose current CRM snapshot is unavailable still returns an
   old financial payment without querying an outbox that may not yet exist.
3. The receipt template will be deterministic from the persisted payment/customer snapshot, include the requested customer/payer/number/amount/method/reference/received date/EOM contact details, and state only that a check was received, never cleared.
4. The full migration runner and the slim EOM startup migration list will install the new additive schema before either receipt-aware route is reachable.  A missing outbox schema makes the new residential receipt path fail closed before it can create a payment; generic service/MCP callers that do not opt into receipt context retain their existing behavior.
5. The existing `scripts/open_pr.sh` draft-consent check remains mandatory at argv admission, but its consent value is removed from the child local-review environment.  This preserves draft authorization while allowing the local suite to independently prove that unauthorised draft flags fail closed.

### Review Contract

- Acceptance criteria:
  - `ReceivablesService.create_payment_with_outcome` inserts the payment, event, allocations, and (when a residential context is supplied) one receipt delivery in one database transaction; a same-key replay returns the original payment and cannot insert a second receipt -- settled by `tests/test_receivables.py` unit and local-Postgres transaction tests.
  - A canonical active residential customer with a valid email yields a `pending` delivery whose persisted subject/body/receipt number contain the payment facts and check-received wording; no-email yields `skipped/no_email` while the payment still exists -- settled by focused service/template tests.
  - The deployed full and slim `POST /api/v1/receivables/payments` routes resolve the active canonical customer before invoking the ledger and return a controlled failure without a ledger write when that canonical read is unavailable or refuses the contact -- settled by full direct-route and slim ASGI tests with recording service/CRM doubles, plus isolated PostgreSQL HTTP entrypoint coverage.
  - `EOM_RECEIVABLES_READINESS_MIGRATIONS` includes the receipt-outbox migration and a fresh, isolated local PostgreSQL schema reaches receivables readiness; removal of the outbox relation is detected only for receipt-enabled use -- settled by `tests/test_receivables.py::test_eom_receivables_readiness_migration_set_builds_ready_schema` and focused readiness tests.
  - This slice has no path to Gmail, Resend, a real customer address, or a transport call; the next send/retry slice must introduce an idempotent transport/reconciliation contract -- settled by diff review and focused test doubles.
  - A user-authorized `--draft` remains accepted by `scripts/open_pr.sh`, while its final local review sees no draft-consent value and therefore cannot mutate the wrapper's no-consent safety fixture behavior -- settled by `tests/test_open_pr_wrapper.py::test_open_pr_consumes_draft_consent_before_local_review`.
- Reachability proof: authenticated full and slim EOM `POST /api/v1/receivables/payments` are exercised through direct/ASGI tests; the isolated full HTTP entrypoint produces a returned receipt-delivery projection and one persisted row in the isolated ledger schema.
- Affected surfaces: full and slim EOM receivables routes, canonical CRM customer projection, receivables service/payment response, full migration runner and EOM startup migration tuple, receipt template, draft-PR local-review environment boundary, isolated migration/service/API tests.
- Risk areas: money transaction atomicity, duplicate retries, active-canonical-customer admission, full-primary versus slim-canonical-pool preflight ordering, email/privacy exposure, migration rollout, generic MCP compatibility, customer-visible receipt wording, draft-consent leakage into child verification.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Boundary-change enumeration

Closure declaration: CLOSED.  Customer classification remains the existing database-enforced `residential|commercial|unknown` set in migration 366 / `EOM_CUSTOMER_TYPES`; this PR only recognizes the exact `residential` member for the new receipt path and safely treats every other admitted member as no residential receipt.  Receipt-delivery status is a new database-enforced closed set: `pending|sent|failed|skipped`; this slice writes only `pending` and `skipped`.

- Boundary path/seam: deployed full and slim `POST /api/v1/receivables/payments` gain a canonical CRM preflight and pass an internal receipt context to the ledger; `DatabaseCRMProvider` gains a narrow active-customer payment projection.
- Replaced-path behaviors: none.  The request body and payment fingerprint remain unchanged; full Atlas/MCP callers do not supply a receipt context and keep their existing route behavior.
- Guard-relevant fields: canonical `id`, `business_context_id`, `contact_type`, `status`, `customer_type`, `full_name`, and normalized `email`; ledger `payment_id` and receipt `delivery_status`.
- Caller x input shape: existing authenticated full and slim callers x every valid payment body retain the same body contract.  The canonical read must return an active `effingham_maids` customer; a nonmatching/missing/unavailable canonical contact fails before the payment transaction.  Only `customer_type == residential` receives an outbox row; blank/malformed email becomes `skipped/no_email` rather than an email attempt.

### Deployed-config probing

- Deployed/default config values: the deployed full application reads canonical customer identity through its initialized primary CRM provider; the slim profile continues to use the existing `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` path guarded by `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED`.  Ledger schema setup uses the full migration runner and the existing EOM migration startup switch.
- Explicit value probe: route tests provide a configured recording canonical CRM provider and prove its normalized residential/no-email outputs control the receipt row.
- Absent value probe: route tests prove an unavailable canonical CRM dependency answers a controlled error before `create_payment` is invoked; the receipt-enabled ledger path rejects a missing outbox relation before writing.
- Default-session/default-context probe: generic service/MCP tests preserve a `receipt_recipient=None` payment path and do not require canonical CRM configuration; both receipt-aware EOM HTTP routes require canonical context only for a new payment and still recover a same-key replay.
- Side-effect ordering: canonical snapshot read first; ledger payment/event/outbox row atomically commit second; transport send is intentionally absent and therefore cannot precede or roll back financial state.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/receivables.py`
- `atlas_brain/storage/migrations/369_receivables_payment_receipt_outbox.sql`
- `atlas_brain/templates/email/payment_receipt.py`
- `plans/PR-EOM-Residential-Receipt-Outbox.md`
- `scripts/open_pr.sh`
- `tests/test_eom_billing_recipients.py`
- `tests/test_eom_payment_receipts.py`
- `tests/test_eom_render_profile.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_receivables.py`

## Mechanism

The deployed full route reads the narrow active-customer projection through its primary `DatabaseCRMProvider`; the slim route reads the same projection through its separately configured canonical CRM pool.  Each passes a typed internal snapshot to `ReceivablesService`; no client body can supply classification or recipient data.  The service creates the payment exactly as it does today, then—in the same transaction and only on the first successful payment insert—creates one `payment_receipt_deliveries` row keyed uniquely by `payment_id`.  The random payment UUID anchors a deterministic `EOM-RCP-<payment-id>` receipt number; the stored subject/body are rendered once from that snapshot and the immutable payment facts.

The row is `pending` only when the canonical email normalizes successfully.  Otherwise it is `skipped` with `no_email`; no email transport is invoked.  Payment idempotency is checked and locked before any new payment/outbox insert, so same-key retries return the prior financial payment and delivery projection.  The outbox is deliberately separate from Gmail: a later slice will claim delivery, send outside the transaction with a transport idempotency/reconciliation strategy, then transition to `sent` or `failed` without creating a new payment.

The receipt table is additive and is not placed in the canonical CRM database: it is financial lifecycle evidence owned by the receivables ledger.  Canonical identity is a read-before-write preflight: the full route uses the primary CRM provider and the slim profile uses its separate canonical pool.  Failure to obtain that snapshot fails closed before a new ledger transaction while an unchanged retry can still recover its already committed payment.  The stored snapshot makes the committed receipt deterministic even if the contact later changes.

`open_pr.sh` validates draft consent before any network or Git side effect, then invokes its final local review with that one-shot consent removed.  The consent remains required to reach the draft create mutation, while a child test process cannot inherit approval for its own independent admission checks.

## Intentional

- No Gmail draft/send, Resend send, background worker, automatic retry, or browser UI is added.  The current sender API cannot prove an uncertain send was unique, and a receipt delivery failure must never roll back a payment.
- `commercial` and `unknown` customer types produce no residential receipt delivery rather than a guessed recipient or a fake invoice.
- The deployed full EOM HTTP route intentionally becomes receipt-aware because it has the authoritative primary CRM dependency; generic service and MCP callers receive no new implicit receipt behavior because they do not pass receipt context.  This preserves existing MCP callers while the website/tracker path gains the durable provider capability.
- Receipt number uses the full persisted payment UUID rather than a mutable/sequence-based display counter, making retries deterministic without adding a counter race or a destructive reset procedure.
- The small workflow correction is limited to draft-consent environment hygiene because the user-required draft path could not otherwise complete its mandatory local test gate; it does not relax consent, alter GitHub mutation targets, or change financial behavior.

## Deferred

- Slice 5 follow-up: tracker and Website show the canonical recipient before submit and display receipt status/retry controls.
- Slice 5/9 follow-up: an explicit, idempotent send/reconciliation worker transitions pending/failed rows without duplicate email and records delivery evidence.
- Customer-ledger slice: paginate/search/filter receipt delivery status alongside payments.

Parking predicate: this PR parks unrelated billing-run, invoice/PDF/Gmail-draft, Square, generic email-provider, and presentation hardening unless it directly violates the atomic receipt-outbox contract above.  Parked hardening: none.

## Verification

- Passed locally: `ruff check --ignore E402` and `python -m compileall` on every
  changed Python file; `git diff --check`; guard-class closure with a documented
  closed-schema heuristic waiver; plan
  audit and sync/check.  `main_eom.py` retains its pre-existing E402 bootstrap
  import ordering, so the changed-file lint uses the repository's established
  E402 exclusion rather than changing unrelated startup structure.
- Passed locally: receipt-focused service/route coverage, plus isolated
  PostgreSQL migration/readiness/rollback/recovery/mixed-rollout coverage.
- Passed locally: the exact invoicing CI selections -- 2
  monthly approval-blocker tests; 213 EOM/receivables/billing-recipient/
  receipt/repository tests against a disposable local PostgreSQL 16 container;
  and 43 MCP/OAuth regression tests.  No production payment, invoice, Gmail
  draft, or customer email was created.
- Passed locally: `tests/test_open_pr_wrapper.py` including the draft-consent
  child-environment regression.
- Pending before push: the repository local PR-contract bundle run by
  `scripts/push_pr.sh`; hosted GitHub checks will not be manually dispatched
  or rerun because local verification is the requested acceptance gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/api/invoicing/receivables.py` | 91 |
| `atlas_brain/eom_api/receivables.py` | 99 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/crm_provider.py` | 47 |
| `atlas_brain/services/receivables.py` | 314 |
| `atlas_brain/storage/migrations/369_receivables_payment_receipt_outbox.sql` | 68 |
| `atlas_brain/templates/email/payment_receipt.py` | 67 |
| `plans/PR-EOM-Residential-Receipt-Outbox.md` | 145 |
| `scripts/open_pr.sh` | 6 |
| `tests/test_eom_billing_recipients.py` | 6 |
| `tests/test_eom_payment_receipts.py` | 395 |
| `tests/test_eom_render_profile.py` | 10 |
| `tests/test_open_pr_wrapper.py` | 24 |
| `tests/test_receivables.py` | 682 |
| **Total** | **1962** |
