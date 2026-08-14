# PR-EOM-Payment-Check-Metadata-V2

Issue: #2362

## Why this slice exists

The Billing & Payments workspace needs ATLAS to retain an optional check date
and received-through detail (for example mail, employee handoff, or customer
delivery) with the canonical payment. The deployed provider currently accepts
only a received date, reference, notes, and allocations. Browser or tracker
state cannot be the financial source of truth for these details, and later
receipt delivery must be able to reconstruct the record faithfully.

The closed precursor PR #2368 cannot be reused: it was based on an older main
and bundled the MCP response projection now separately merged as #2370. Current
main also proves an additional rollout risk: `atlas_brain.main` logs a full
migration failure and keeps serving, while the payment service would otherwise
issue a new-column insert. This rebuilt provider-first slice closes that actual
root cause before a tracker or Website consumer can send the new fields.

This cohesive migration/contract slice is expected to exceed the 400-LOC soft
target because the money-safety boundary needs both full and slim ASGI proofs,
an isolated real-Postgres migration proof, compatibility/retry coverage, and
the paired EOM startup inventory assertion. Splitting those artifacts would
permit an unproven partial provider rollout.

### Problem-derived contract

- Root cause: `customer_payments`, the two active EOM payment request models,
  and `ReceivablesService` do not own the check-date/received-through data.
  Moreover, a provider binary can remain online after a full-app migration
  failure, so an unconditional new-column insert would turn a valid legacy
  payment into a schema error during a mixed rollout.
- Correct fix must touch/change: add nullable `check_date DATE` and
  `received_through VARCHAR(128)` columns in one additive migration; append it
  to the slim EOM readiness migration closure; add the two optional request
  fields to both provider models and forward them through both routes; make the
  service normalize and fingerprint supplied metadata, preserve the legacy
  payload/insert when neither field is materially supplied, and fail a
  metadata-bearing request before its transaction when canonical readiness is
  false; prove that behavior through full and slim ASGI entrypoints plus an
  isolated real-Postgres schema.
- Must not change: required payment fields; check received/deposited/cleared,
  return, void, allocation, customer-identity, money, actor, or retry
  semantics; existing MCP request/response behavior (including #2370's
  projection); tracker/Website behavior; receipt-email delivery; billing
  recipients; and any production records or configuration.

## Scope (this PR)

Ownership lane: eom/billing-payments
Slice phase: Vertical slice
Max files: 8

1. Accept and persist optional check-only `check_date` and `received_through`
   through the current full and slim EOM payment-create APIs; reject those
   fields for ACH and Square before a transaction.
2. Keep a no-metadata request byte-for-byte equivalent at the service intent
   level: its legacy fingerprint and legacy-column insert remain usable before
   migration 368, while any material metadata request is rejected with a 503
   before financial writes if readiness is false.
3. Add and enroll the nullable migration in the EOM readiness closure, then
   prove migration idempotency, ready/not-ready detection, idempotency replay,
   and both active ASGI entrypoints locally.

### Review Contract

- Acceptance criteria:
  - [ ] Both `atlas_brain.api.invoicing.receivables.CreatePaymentRequest` and
    `atlas_brain.eom_api.receivables.CreatePaymentRequest` accept omitted,
    valid, and 128-character `received_through` values, and reject malformed
    dates and 129-character values; settled by
    `tests/test_receivables.py::test_payment_http_models_preserve_optional_check_metadata`.
  - [ ] `ReceivablesService.create_payment_with_outcome` normalizes and
    persists supplied metadata, replays an unchanged request, and conflicts on
    a changed material metadata value without a second payment event; check
    metadata on ACH or Square is rejected before a transaction; settled by
    `tests/test_receivables.py::test_create_payment_persists_optional_check_metadata_in_payment_intent`
    and `tests/test_receivables.py::test_create_payment_rejects_check_metadata_for_non_check_methods`.
  - [ ] A legacy request that omits both values retains its old fingerprint and
    old insert shape, and can commit in a pre-368 isolated schema; settled by
    `tests/test_receivables.py::test_create_payment_without_check_metadata_preserves_legacy_intent`
    and `tests/test_receivables.py::test_check_metadata_schema_gate_preserves_legacy_payment_creation`.
  - [ ] A supplied check date or nonblank received-through value observes the
    canonical readiness gate before a transaction and returns a 503/no-write
    result when migration 368 is absent; settled by
    `tests/test_receivables.py::test_full_and_slim_payment_entrypoints_reject_check_metadata_when_schema_is_unready`.
  - [ ] A full authenticated and a slim authenticated
    `POST /api/v1/receivables/payments` request each forward approved metadata
    to the canonical service; settled by
    `tests/test_receivables.py::test_real_postgres_http_and_mcp_entrypoints_use_supported_dependencies`.
  - [ ] Applying the closed slim migration set, including migration 368, makes
    `ReceivablesService.is_ready()` true; removing either new column makes it
    false; settled by
    `tests/test_receivables.py::test_eom_receivables_readiness_migration_set_builds_ready_schema`
    and `tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_receivables_readiness`.
- Reachability proof: authenticated ASGI requests through `main.app` and
  `main_eom.app` observe a canonical service result or a 503 before a
  transaction. The successful full path additionally persists normalized data
  in an isolated local Postgres schema. No production payment, invoice, or
  email is created.
- Affected surfaces: full and slim EOM payment API contracts; canonical
  receivables persistence/retry/readiness; slim migration startup inventory;
  isolated database and ASGI regression tests.
- Risk areas: financial-record truthfulness, idempotency/retry safety,
  migration compatibility, full-versus-slim route parity, and incremental
  provider deployment.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: full `atlas_brain.api.invoicing.receivables` and slim
  `atlas_brain.eom_api.receivables` `POST /receivables/payments` request
  bodies, then `ReceivablesService.create_payment_with_outcome`.
- Replaced-path behaviors: an omitted or blank-normalized metadata pair keeps
  the legacy request fingerprint and legacy insert; a material `check_date` or
  `received_through` joins the payment intent and new-column insert only after
  the canonical readiness decision succeeds.
- Guard-relevant fields: `check_date` and `received_through`; all existing
  payment fields remain inputs to the unchanged validation and fingerprint.
- Caller x input shape:
  - full and slim HTTP callers omitting both fields preserve legacy behavior;
  - full and slim HTTP check callers supplying a date, nonblank route text, or
    both receive the canonical payment result only when schema-ready;
  - ACH and Square callers supplying either material check field receive a
    validation error before a transaction; blank route text normalizes to
    absent metadata and preserves the legacy path;
  - blank route text normalizes to absent metadata and follows the legacy path;
  - direct legacy MCP/service callers omit these parameters and retain their
    current behavior; no MCP input contract is added here.
- Closure declaration for the metadata-field inventory: the field set is
  **CLOSED** and **ENUMERATED** by migration
  `368_receivables_payment_check_metadata.sql` and the two paired HTTP models.
  Unlisted request fields neither affect the fingerprint nor reach the insert;
  omitted/blank values take the backward-compatible legacy path, while the two
  recognized material values fail closed on unavailable schema before any
  financial write. This direction is safer because it preserves valid legacy
  payments but never pretends a new detail was recorded when its columns are
  missing.
- Closure declaration for `EOM_RECEIVABLES_READINESS_MIGRATIONS`: the curated
  migration set is **CLOSED** and **ENUMERATED** from the required receivables
  tables, columns, indexes, and SQL prerequisites used by
  `ReceivablesService.is_ready()`. Migrations outside it intentionally do not
  run in the slim profile; a future receivables schema dependency must update
  this tuple and its ready-schema test in the same slice. Incompleteness fails
  the metadata mutation rather than creating a partial record.

### Deployed-config probing

N/A - this changes no environment/config fallback. The deployment-sensitive
decision is database schema readiness and is probed with a ready schema,
pre-368 schema, and blank/absent metadata inputs before any transaction.

### Files touched

- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/receivables.py`
- `atlas_brain/storage/migrations/368_receivables_payment_check_metadata.sql`
- `plans/PR-EOM-Payment-Check-Metadata-V2.md`
- `tests/test_eom_render_profile.py`
- `tests/test_receivables.py`

## Mechanism

Migration 368 only adds nullable fields, so existing rows and old binaries are
valid. The service normalizes `received_through` once. If neither new value is
material, it constructs the exact legacy intent and executes the legacy insert.
Material check metadata is valid only for check payments; ACH and Square fail
validation before a transaction. For a material check request, the service
first calls the established complete `is_ready()` check; a false or unavailable
answer becomes a typed schema unavailable response before a payment transaction
begins. A ready request adds the normalized values to its fingerprint and uses
the extended insert. Thus an unchanged retry returns the original payment and a
changed check detail is an idempotency conflict rather than a second
receipt/allocation.

The full app may log a failed best-effort migration and continue; the
metadata-specific service gate makes that state observable and safe. The slim
profile's curated migration tuple gains 368, so its normal startup path applies
the additive prerequisite. A code rollback is safe because older code ignores
the nullable columns; no rollback drops customer-payment history or columns.

## Intentional

- `received_through` remains bounded operator-entered text, not a speculative
  enum: the approved requirements give examples but no canonical taxonomy.
- These fields are check-specific. A material value is rejected for ACH and
  Square before a transaction so ATLAS never records a false check detail;
  blank route text remains backward-compatible absent metadata.
- The metadata schema gate is in the canonical service, not duplicated as
  route-specific business logic, so both full and slim HTTP routes and any
  future direct caller receive the same fail-closed rule.
- The already-merged MCP response projection remains untouched. MCP callers do
  not receive a new request field in this slice.

## Deferred

- eom-timetracker and Website consumption of these fields follow only after the
  provider migration is deployed (#2362).
- Receipt numbers, deterministic receipt composition, outbox state, retry, and
  no-email recovery remain the dedicated residential receipt-outbox slices
  tracked in #2362 and #2363.
- Customer ledger search, commercial billing candidates, Gmail draft recovery,
  sent-mail reconciliation, and manual Square queue behavior remain later
  Billing & Payments slices (#2362).

Parking predicate: customer-facing receipt composition/delivery, payment-search
ergonomics, billing-run behavior, and metadata taxonomy/polish that do not
block durable provider storage, legacy replay, or fail-closed metadata writes
are parked in #2363.

Parked hardening: none.

## Verification

- Passed locally: `ruff check` on all changed Python files; `compileall` on
  all changed Python files; `git diff --check`; plan sync/check; and strict
  `check_guard_class_closure.py`.
- Passed locally: focused model/service/ASGI coverage (10 passed) and isolated
  PostgreSQL migration/readiness/entrypoint coverage (4 passed).
- Passed locally: the invoicing workflow selection -- 2 monthly-invoice tests,
  194 EOM/receivables/billing-recipient/repository tests, and 43 MCP
  regression tests. No production payment, invoice, or email was created.
- Pending before push: the repository local PR gate. Hosted GitHub checks are
  not manually dispatched or rerun; the operator requested equivalent checks
  be run locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/receivables.py` | 9 |
| `atlas_brain/eom_api/receivables.py` | 9 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/receivables.py` | 142 |
| `atlas_brain/storage/migrations/368_receivables_payment_check_metadata.sql` | 5 |
| `plans/PR-EOM-Payment-Check-Metadata-V2.md` | 242 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_receivables.py` | 455 |
| **Total** | **864** |
