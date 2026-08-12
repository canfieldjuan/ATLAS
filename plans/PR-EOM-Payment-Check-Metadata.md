# PR-EOM-Payment-Check-Metadata

## Why this slice exists

The Billing & Payments workspace requires an optional check date and a
received-through value (for example mail, employee handoff, or customer
delivery). Slice 3 proved that the current operator entry flow can safely
record an unapplied residential check, but its deployed provider request shape
contains only received date, reference, notes, and allocations. The required
metadata cannot be invented or persisted in the browser: ATLAS is the
financial source of truth and receipt delivery must later reproduce the
recorded payment faithfully.

This is the provider-first prerequisite recorded in #2362 and #2363 after
Website PR #192. It lands only the durable, backward-compatible ATLAS contract
needed before tracker and Website consumers can add the fields.

Diff budget: this cohesive provider migration is slightly above the 400-LOC
soft cap because it needs paired full/slim contract coverage, real-schema
readiness proof, and the existing repository fixture update. Splitting those
would permit a provider deploy with an untested schema path.

### Problem-derived contract

- Root cause: `customer_payments` and both active provider request models have
  no dedicated check-date or received-through fields; the create service cannot
  include either value in its idempotency fingerprint or payment view. A
  browser-only field would be lost, and a later receipt/outbox could not
  deterministically reconstruct the recorded check.
- Correct fix must touch/change: add one additive migration with nullable
  `check_date DATE` and `received_through VARCHAR(128)` columns; enroll it in
  the EOM readiness migration closure and invoicing workflow; include both
  nullable fields in readiness, both full/slim EOM `CreatePaymentRequest`
  models, the service fingerprint/insert/payment view, and direct
  service/HTTP/migration regression tests.
- Must not change: existing required request fields, check `received` versus
  deposit/clearing lifecycle, allocation behavior, legacy/MCP payment routes,
  customer identity admission, money representation, or any receipt-email,
  tracker, Website, Gmail, and billing-recipient behavior. The new fields stay
  optional and do not decide receipt-email copy or delivery policy.

## Scope (this PR)

Ownership lane: eom/billing-payments
Slice phase: Vertical slice

1. Accept optional `check_date` and `received_through` only on the existing
   EOM payment-create provider request, retaining their normalized values in
   the canonical `customer_payments` record and response.
2. Preserve a legacy caller that omits both fields, and preserve idempotent
   replay only when the full payment intent, including either supplied
   metadata field, matches.
3. Make the additive migration part of the full and slim EOM readiness/deploy
   path, and prove the real authenticated full `/api/v1/receivables/payments`
   entrypoint persists an optional check date and received-through value in an
   isolated test schema.

### Review Contract

- Acceptance criteria:
  - [ ] Both `atlas_brain.api.invoicing.receivables.CreatePaymentRequest` and
    `atlas_brain.eom_api.receivables.CreatePaymentRequest` accept omitted,
    valid, and boundary-valid `check_date`/`received_through` values and reject
    an over-128-character received-through string; settled by
    `tests/test_receivables.py::test_payment_http_models_preserve_optional_check_metadata`.
  - [ ] `ReceivablesService.create_payment_with_outcome` persists normalised
    optional metadata, returns it in the payment view, and treats a same
    idempotency key with changed optional metadata as a request conflict;
    settled by
    `tests/test_receivables.py::test_create_payment_persists_optional_check_metadata_in_payment_intent`.
  - [ ] The existing no-metadata payment create/replay behavior remains
    unchanged; settled by the existing zero-allocation replay test plus the
    full HTTP regression test with the legacy body unchanged.
  - [ ] The real authenticated full `/api/v1/receivables/payments` entrypoint
    persists both values to `customer_payments` in an isolated Postgres schema;
    settled by `tests/test_receivables.py::test_receivables_http_and_mcp_contracts`.
  - [ ] EOM readiness refuses a schema missing either new column and the new
    migration is included in the closed EOM migration set and invoicing CI path
    filters; settled by
    `tests/test_receivables.py::test_eom_receivables_readiness_migration_set_builds_ready_schema`
    and the migration-enrollment test.
- Reachability proof: full `main.app` `POST /api/v1/receivables/payments` with
  service authentication, isolated Postgres schema, and an observable
  `customer_payments.check_date`/`received_through` row. No production payment
  or email is used.
- Affected surfaces: EOM provider API request contract; canonical receivables
  ledger persistence/readiness; EOM startup migration closure; invoicing CI
  enrollment; tests.
- Risk areas: money record truthfulness, idempotency/retry safety,
  backward-compatible API evolution, safe additive migration and deployment
  order.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R12, R14.

### Boundary-change enumeration

N/A - no permission, identity, or admission decision changes. The existing
Pydantic request models gain optional bounded data fields, and the service
fingerprint treats their normalized values as payment intent.

### Deployed-config probing

N/A - no guard or configuration boundary changes. The fields are request data,
not environment configuration.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/receivables.py`
- `atlas_brain/storage/migrations/368_receivables_payment_check_metadata.sql`
- `plans/PR-EOM-Payment-Check-Metadata.md`
- `tests/test_eom_render_profile.py`
- `tests/test_invoice_repository.py`
- `tests/test_receivables.py`

## Mechanism

The new migration uses nullable additions only, so already-recorded payments
remain valid and rollback is a code rollback (the unused nullable columns can
remain safely). Both provider models accept `check_date` as an ISO date and a
bounded `received_through` string. The service normalizes `received_through`
once, places both fields in its fingerprint payload, inserts them with the
payment, and returns the database row through the existing payment view. A retry
with identical metadata returns the original payment; a retry
that changes either field conflicts before a second receipt or allocation can
be created.

Migration `368` is appended to the closed EOM readiness set after its
receivables prerequisites. Readiness itself checks the new columns, preventing
the provider from reporting ready if a deploy has code but not the migration.

## Intentional

- `received_through` is stored as operator-entered text rather than a closed
  enum because the approved requirements give examples, not a canonical list.
  The 128-character bound limits unstructured payload size while avoiding a
  speculative product taxonomy.
- The fields are accepted for every current EOM payment method for a
  backward-compatible additive transport shape. The Website will render them
  only for checks in its downstream consumer slice; enforcing a method-specific
  rule now would reject an old/new mixed deploy without a user-facing benefit.
- No event metadata duplication: `customer_payments` is the financial record;
  the existing immutable `payment_recorded` event retains lifecycle evidence
  without turning event JSON into a second source of truth.

## Deferred

 - Tracker then Website must consume the deployed optional fields in the same
   provider-first order before receipt delivery is built (#2362; #2363).
 - Receipt numbering, deterministic receipt composition, outbox state, retries,
   and no-email behavior remain the dedicated receipt-outbox slice.
 - The duplicate full/slim route model debt remains H-06 in #2363; this slice
   deliberately maintains parity rather than extracting a shared abstraction.

Parking predicate: new receipt delivery, email content, customer-ledger search,
and taxonomy/polish findings that do not prevent durable storage or safe replay
of these two fields are parked in #2363.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_monthly_invoice_generation.py -k "update_invoice_clears_needs_hours_when_line_items_are_billable or line_items_are_billable_requires_all_positive_quantities" -q` — 2 passed.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=<isolated local PostgreSQL> python -m pytest tests/test_receivables.py -q` — 77 passed.
- `ATLAS_VOICE__ENABLED=false ATLAS_VOICE__AUTO_START_ASR=false python -m pytest tests/test_eom_render_profile.py -q` — 61 passed; the test-only setting matches GitHub's no-GPU runner and prevents local ASR start-up.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=<isolated local PostgreSQL> python -m pytest tests/test_invoice_repository.py -q` — 1 passed.
- `python -m pytest tests/test_invoicing_readonly_mcp.py tests/test_invoicing_readonly_oauth.py tests/test_invoicing_draft_writer_mcp.py tests/test_invoicing_draft_writer_oauth.py -q` — 43 passed.
- Targeted `ruff check ... --ignore E402` and `git diff --check` — passed; `main_eom.py`'s pre-import local-environment load is the established E402 exception.
- The workstation's Python 3.11 venv could not install the pinned `webrtcvad` package because Python 3.11 development headers are absent. The same workflow commands above ran locally under the provisioned Python 3.13 environment; no code or deployed configuration was changed to compensate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/api/invoicing/receivables.py` | 4 |
| `atlas_brain/eom_api/receivables.py` | 4 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/receivables.py` | 24 |
| `atlas_brain/storage/migrations/368_receivables_payment_check_metadata.sql` | 5 |
| `plans/PR-EOM-Payment-Check-Metadata.md` | 187 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_invoice_repository.py` | 1 |
| `tests/test_receivables.py` | 201 |
| **Total** | **430** |
