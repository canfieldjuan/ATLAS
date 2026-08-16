# PR-EOM-Receipt-Delivery-Migration-Recovery

## Why this slice exists

The deployed EOM receipt-delivery schema has a recorded
`378_receivables_payment_receipt_delivery` migration but is missing the
reconciliation-events table required by the current receipt-delivery readiness
contract. That makes `/api/v1/receivables/ready` fail closed even though the
receipt routes are mounted.

This production-hardening slice unblocks the already merged residential
receipt-delivery vertical: the tracker and Website consumers are deployed,
while the Atlas provider cannot safely serve the capability until the closed
receipt-delivery schema contract is complete.

The expected diff slightly exceeds the 400-LOC soft cap because the regression
test deliberately builds the real EOM migration set, records 378, preserves
real financial/receipt/operation rows, removes only the later DDL, and reruns
the production migration runner. That proof is indivisible from the recovery.

### Problem-derived contract

- Root cause: the migration runner records an applied migration by its filename
  in `schema_migrations` and determines pending work by that filename
  (`atlas_brain/storage/migrations/__init__.py`). Migration 378 was recorded
  before a later revision added its result fields and reconciliation-events
  DDL, so current code correctly skips the already-recorded filename and the
  database stays structurally partial.
- Correct fix must touch/change: add one new, forward-only recovery migration
  that restores only the receipt-delivery schema objects introduced after the
  originally recorded 378; enroll that migration in the closed EOM startup set
  and invoice-check workflow; and add an isolated-database replay test that
  records 378, removes those later objects, runs the real migration runner,
  and proves receipt-delivery readiness returns without modifying existing
  payment or receipt facts.
- Must not change: do not edit migration 378 or mutate its ledger row; do not
  change payment, allocation, check, return, void, finalization, or audit
  behavior; do not change Gmail dispatch/reconciliation semantics, routes,
  authorization, tracker, Website, dependencies, or the generic migration
  runner. This repair treats the deployed partial schema safely; a generic
  policy for immutable applied-migration contents is deferred.

## Scope (this PR)

Ownership lane: eom/billing-payments-receipt-delivery
Slice phase: Production hardening

1. Add the additive receipt-delivery recovery DDL as migration 379 and include
   it in the EOM readiness migration tuple that the slim EOM entrypoint applies
   on startup.
2. Enroll the new migration file in both invoice-check workflow path filters.
3. Prove the exact recorded-378/partial-schema replay path, readiness result,
   idempotent rerun, and preservation of pre-existing financial and receipt
   rows in an isolated PostgreSQL schema.

### Review Contract

- Acceptance criteria:
  1. Given a ledger-recorded 378 schema missing its later result fields and
     reconciliation-events table, migration 379 creates the required columns,
     table, and index and `ReceivablesService.is_receipt_delivery_ready()`
     returns true; settled by the new isolated-schema regression test in
     `tests/test_receivables.py`.
  2. The recovery SQL contains only additive schema operations and no DML over
     payment, receipt, allocation, or audit facts; settled by the migration
     file and the regression test's before/after row snapshots.
  3. A rerun after migration 379 is recorded makes no further schema or fact
     changes; settled by the same regression test's repeated
     `run_migrations(..., only=EOM_RECEIVABLES_READINESS_MIGRATIONS)` call.
  4. The real slim EOM startup path can request migration 379, and a changed
     migration file triggers invoice checks on pull requests and pushes;
     settled by `atlas_brain/main_eom.py`,
     `.github/workflows/atlas_invoicing_checks.yml`, and their enrollment/
     EOM-profile tests in `tests/test_receivables.py` and
     `tests/test_eom_render_profile.py`.
- Reachability proof: `atlas_brain/main_eom.py::_run_startup_migrations` calls
  `_apply_eom_receivables_migrations`, which passes the closed tuple to the
  production runner. The isolated replay test exercises that same tuple and
  observes receipt-delivery readiness as the observable gate. After merge,
  deployment verification will call the authenticated local
  `/api/v1/receivables/ready` endpoint and inspect only database catalog
  metadata.
- Affected surfaces: migration execution and ledger bookkeeping; the EOM
  startup migration tuple; invoice workflow path enrollment; the closed
  receipt-delivery schema/readiness contract; isolated PostgreSQL migration
  tests.
- Risk areas: preserving financial and receipt facts; idempotent replay;
  backward compatibility with already-complete 378 schemas; migration ordering
  and atomic ledger recording; startup reachability; and fail-closed readiness
  before the recovery runs.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

N/A - no guard, validator, router, classifier, or admission boundary changes.
The existing receipt-delivery readiness gate remains fail closed until the
recovered schema satisfies its unchanged closed object contract.

### Deployed-config probing

N/A - no guard/config boundary change. The rollout uses the existing database
and existing EOM migration-startup configuration; it adds no environment
setting or fallback.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/main_eom.py`
- `atlas_brain/storage/migrations/379_receivables_payment_receipt_delivery_recovery.sql`
- `plans/PR-EOM-Receipt-Delivery-Migration-Recovery.md`
- `tests/test_eom_render_profile.py`
- `tests/test_receivables.py`

## Mechanism

Migration 379 uses the runner's atomic-bookkeeping marker because it has no
concurrent DDL. It uses `IF NOT EXISTS` / catalog-guarded additive DDL to add
the two result columns and their `NOT VALID` shape constraint to
`payment_receipt_delivery_operations`, then creates the missing append-only
`payment_receipt_delivery_reconciliation_events` table and its receipt-order
index. It does not update, insert, or delete financial, receipt, operation, or
audit rows.

Appending 379 to `EOM_RECEIVABLES_READINESS_MIGRATIONS` makes the existing slim
startup path request it. Because 378 is already in `schema_migrations`, the
real runner skips 378 and records/runs only 379. The regression test models
that exact sequence in an isolated schema, snapshots pre-existing facts,
asserts false readiness before recovery and true readiness afterward, then
reruns the same migration set to prove idempotence.

## Intentional

- The repair is a new migration rather than a rewrite of 378: applied
  migrations cannot be safely replayed by the current filename-based ledger.
- It restores only the objects added after the initial 378 revision; it does
  not broaden receipt dispatch, resend, reconciliation, payment, or Gmail
  behavior.
- `NOT VALID` preserves the final 378 constraint posture: it enforces the
  result shape for future writes without rejecting historical rows during this
  schema repair.

## Deferred

Parking predicate: park generic migration-framework hardening, alternate
historical partial states not evidenced by the deployed catalog, and any
receipt-product behavior change unless it is required to restore this exact
fail-closed readiness gate.

- A content checksum or immutable-applied-migration policy for the generic
  runner is deferred. It needs a cross-product ledger/compatibility design and
  is not required to repair this recorded partial 378 state safely.

Parked hardening: none.

## Verification

- Passed locally:
  - targeted recovery, EOM tuple, complete-readiness, and CI-enrollment tests:
    8 passed;
  - invoice workflow's receivables ledger/repository selection: 625 passed in
    78.51s;
  - invoice approval-blocker selection: 2 passed, 41 deselected;
  - invoice MCP/OAuth surface selection: 43 passed;
  - targeted Ruff validation: passed. The E402 import-order exception is the
    existing intentional import-after-environment-bootstrap baseline in the
    EOM entrypoint; this slice adds no import;
  - Python compile validation: passed;
  - diff whitespace validation: passed;
  - plan synchronization and check validation: passed.
- Pending before push: the repository's `scripts/push_pr.sh` mechanical review
  entry point.
- After merge/deploy: verify the mounted route remains authorization-gated and
  the authenticated local readiness endpoint returns ready; inspect only
  schema catalog metadata and do not trigger payment or Gmail actions.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/storage/migrations/379_receivables_payment_receipt_delivery_recovery.sql` | 72 |
| `plans/PR-EOM-Receipt-Delivery-Migration-Recovery.md` | 186 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_receivables.py` | 194 |
| **Total** | **456** |
