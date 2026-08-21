# PR-EOM-Recurring-Invoice-Dedup-Recovery

## Why this slice exists

ATLAS [#2451](https://github.com/canfieldjuan/ATLAS/issues/2451) is the
provider prerequisite discovered after #2448 merged. A read-only production
preflight on 2026-08-20 showed `schema_migrations` records
`385_invoices_billing_period_dedup`, but the live catalog is the first
revision of that file: it has only the nullable `billing_period` column and
the partial recurring index. It lacks `billing_period_legacy_null`, the
reservation table, the fresh-write admission check, and the historical
backfill added later to the PR. The runtime's
`recurring_invoice_dedup_schema_ready` fence consequently returns false, so
deploying merged #2448 directly would correctly refuse to serve the enabled
receivables writer path.

The ledger row must remain historical evidence. Rewriting migration 385 or
its recorded digest would hide the deployment event rather than recover it.
This thin provider-hardening slice adds one forward-only migration that
converges the observed initial catalog into the final #2448 contract without
creating, deleting, sending, or reallocating any financial record.

The diff exceeds the 400-LOC soft cap because the migration, exact historical
catalog reconstruction, atomic-failure proof, and clean-schema no-op proof are
one financial safety claim: splitting any one of them would leave a production
repair without repeatable evidence for the state it is authorized to mutate.

### Problem-derived contract

- Root cause: migration 385 was recorded before later PR commits expanded its
  schema and backfill behavior; the migration runner therefore skips the final
  file while the #2448 readiness fence requires objects the recorded catalog
  does not have.
- Correct fix must touch/change: add a new atomic recovery migration, prove it
  against a real PostgreSQL reconstruction of the recorded initial catalog,
  and enroll the new migration in the existing invoicing workflow trigger
  paths.
- Must not change: the contents or ledger identity of migration 385; invoice,
  payment, allocation, receipt, delivery, Gmail, or customer data semantics;
  existing recurring-writer query contracts; tracker and Website consumers;
  or an unrelated generic migration-framework redesign.

## Scope (this PR)

Ownership lane: eom/billing-payments-recurring-dedup-recovery
Slice phase: Production hardening
Max files: 4

1. Add migration 387, marked `-- atlas: atomic-bookkeeping`, that converges
   the exact observed initial-385 state and is a data no-op on a clean final
   #2448 schema.
2. Add real-PostgreSQL proof for normal, partial, ambiguous, invalid-format,
   retry, and current-schema recovery states; keep it isolated in a disposable
   test schema.
3. Enroll migration 387 in both pull-request and `main` push path filters for
   the existing invoicing checks.
4. Retain the readiness fence: an unobserved missing/invalid recurring index
   remains fail-closed rather than widening this recovery into index-rebuild
   machinery. The observed index is valid, ready, unique, and has the required
   predicate, so migration 387 reuses it.

### Review Contract

- Acceptance criteria:
  - A reconstructed initial-385 ledger/catalog state has no legacy-null
    column, reservation table, or fresh-write check before migration 387, and
    `recurring_invoice_dedup_schema_ready` returns false.
  - Applying migration 387 under `run_migrations` records only 387, retains
    the already-recorded 385 ledger row and its historical null digest, and
    makes the readiness predicate return true.
  - A mechanically derivable, non-conflicting legacy `monthly_auto` or
    `eom_commercial_billing` invoice gains the exact `YYYY-MM` period; its
    source, number, amount, status, identity, payments, and allocations are
    unchanged.
  - A legacy candidate that conflicts with another legacy candidate or an
    already-populated recurring invoice remains `billing_period = NULL`, is
    explicitly marked and reserved, and makes the existing application
    pre-check return its `quarantined_collision` synthetic hit.
  - A malformed or semantically invalid legacy period is retained as a marked
    legacy exception, not guessed at; a fresh recurring invoice with a missing
    or `0000-01` period is rejected by the converged database admission
    constraints.
  - A void recurring row and a non-recurring MCP row remain outside this
    recovery, preserving reissue and ad-hoc-invoice behavior.
  - Re-running the migration is idempotent: the invoice/reservation snapshot
    does not change and no duplicate migration row, reservation, invoice,
    PDF, draft, email, payment, or ledger mutation is created.
  - A clean final-385 schema accepts 387 as a schema/data no-op apart from its
    own ledger record; it does not downgrade constraints or alter rows.
  - The new migration path is included in both invoicing workflow trigger
    lists, so a future recovery edit cannot bypass the provider proof.
- Reachability proof: `atlas_brain/main.py` runs generic migrations before it
  invokes `recurring_invoice_dedup_schema_ready` for enabled receivables or
  auto-invoice writers. The real-PostgreSQL test calls the same migration
  runner and readiness function in a fresh schema.
- Affected surfaces: the migration runner's 387 file discovery, the existing
  recurring repository readiness/pre-check contract, the invoice test module,
  and the invoicing workflow trigger lists.
- Risk areas: a corrupted historical ledger, duplicate cross-pipeline invoice,
  rewriting financial history, an incomplete schema that starts serving,
  atomic rollback, and a future stale recorded migration.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: initial recorded migration-385 catalog -> generic
  migration runner -> migration 387 ->
  `recurring_invoice_dedup_schema_ready` -> recurring invoice writers.
- Replaced-path behaviors: the initial catalog previously fails closed at
  startup; it converges to the already-merged #2448 writer contract. Missing
  or invalid index states continue to fail closed at readiness rather than
  being silently repaired by this slice.
- Guard-relevant fields: `invoices.billing_period`,
  `invoices.billing_period_legacy_null`, the two named constraints, the
  reservation table, active status, and the two recurring source values.
- Caller x input shape: legacy NULL periods are parsed only from the exact
  source-owned formats; derivable unique values populate; ambiguous and
  unparseable values remain explicit exceptions; all new recurring writes use
  the existing database and application checks.

### Deployed-config probing

- Deployed/default config values: read-only live inspection found the active
  `atlas-api.service` runs pre-#2448 code but enables the receivables API; no
  environment value, credential, service unit, or financial record is changed
  by this PR.
- Explicit value probe: the real-PostgreSQL test invokes the production
  migration runner and repository readiness function against the reconstructed
  catalog.
- Absent value probe: schema objects absent from the initial-385 reconstruction
  produce a false readiness result before the recovery migration runs.
- Default-session/default-context probe: migration 387 is atomic-bookkeeping;
  the existing runner test proves marked migrations record DDL and their ledger
  entry in one transaction, while this PR's regression exercises the actual
  migration on isolated data.
- Side-effect ordering: migration 387 executes before API readiness; it has no
  invoice/PDF/Gmail/email writer and does not restart or deploy the service.

### Closure declaration

The recovery state set is **CLOSED** and **DERIVED** from the observed
initial-385 database catalog and the #2448 readiness function. Every active
legacy recurring invoice is either mechanically backfilled, explicitly
quarantined, or left as a marked unparseable exception; void and non-recurring
rows are deliberately outside the writer-dedup boundary.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/storage/migrations/387_eom_recurring_invoice_dedup_recovery.sql`
- `plans/PR-EOM-Recurring-Invoice-Dedup-Recovery.md`
- `tests/test_invoice_repository.py`

## Mechanism

Migration 387 uses the runner's advisory lock and atomic-bookkeeping
transaction. It adds the missing legacy-null flag and reservation table,
replaces only the recorded initial version of the period check with the
nonzero-year contract required by #2448, and installs the required-write
check. Before changing the check it fails atomically if a non-null stored
period cannot satisfy the final grammar.

It derives a candidate period only from the existing writer-owned formats:
`monthly_auto.source_ref` ending in `_YYYY-MM`, and
`eom_commercial_billing.invoice_number` in `INV-YYYY-Mon-sequence` form. It
checks each candidate against other candidate rows and non-void pre-populated
recurring invoice periods. A unique candidate becomes a real period. Any
collision stays NULL, receives immutable-looking explanatory metadata and a
reservation for the existing pre-check. A non-derivable legacy row stays NULL
with an explicit legacy marker. This avoids selecting a historical winner or
inventing an invoice.

The migration does not recreate the index: the exact observed state already
has the correct valid index, and #2448's readiness guard remains the safe stop
for any unobserved index corruption. A clean final-385 schema meets every
catalog condition, so 387 only adds its own migration ledger entry there.

## Intentional

- Preserve the historical `schema_migrations` row for 385 rather than amend
  its digest or source identity.
- Prefer an atomic, short recovery transaction over `CONCURRENTLY` index DDL:
  the live preflight found 59 candidate rows and a valid existing index, so
  there is no reason to introduce the original migration's non-atomic window.
- Do not create an invoice, choose a collision winner, alter an amount/status,
  mark a service invoiced, create a PDF/Gmail draft, or deliver customer mail.
- Keep the index-rebuild problem out of this vertical slice. A missing or
  invalid index remains a visible startup/readiness failure until its own
  evidence-backed repair is justified.
- Deployment remains provider-first: merge and deploy this ATLAS repair before
  moving the #2448 runtime forward, then verify health and readiness before
  resuming tracker or Website work.

## Deferred

- H-18 phase 2 migration-forensics / policy work remains in [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363);
  this slice heals the proven 385 state without redesigning historical digest
  policy.
- An invalid or missing recurring dedup index has no observed production
  evidence. The readiness fence protects it; a live finding belongs in the
  Billing & Payments Hardening & Deferred issue before a separate repair.
- Durable commercial billing recipient override #2433 remains product work
  gated on repeated `source_correction_pending` evidence, not a consequence of
  this catalog recovery.

Parking predicate: any change to invoice/product delivery, customer-facing
copy, recipient selection, tracker proxy, Website UI, migration framework, or
index-rebuild behavior is parked unless it is strictly required to prove this
observed recovery state.

Parked hardening: H-18 phase 2 and unobserved index repair, tracked above.

## Verification

- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://…:55487/… python -m
  pytest tests/test_invoice_repository.py -k 'recorded_initial_385_recovery
  or 387_is_data_noop' -q` -- 3 passed, 9 deselected against a disposable
  Docker PostgreSQL 16 database; no production database, customer, PDF, Gmail,
  or email surface was used.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://…:55487/… python -m
  pytest tests/test_invoice_repository.py -q` -- 12 passed.
- `python -m pytest tests/test_migrations_runner.py -k 'marked_migration_records_its_ledger_entry_in_one_transaction or marked_migration_rejects_concurrently_ddl_before_a_transaction' -q`
  -- 2 passed, 49 deselected.
- The exact `atlas-invoicing-checks` receivables-ledger job command completed
  locally against the disposable PostgreSQL target; its collection confirmed
  751 tests across the 14 workflow files. The adjacent approval blocker,
  legacy opt-in, and MCP/OAuth workflow commands also passed: 2 passed / 41
  deselected, 11 passed, and 47 passed respectively.
- `python -m ruff check tests/test_invoice_repository.py` and `python -m
  py_compile tests/test_invoice_repository.py` -- passed.
- `git diff --check` -- passed.
- `python scripts/sync_pr_plan.py
  plans/PR-EOM-Recurring-Invoice-Dedup-Recovery.md --check` and
  `python scripts/audit_plan_doc.py
  plans/PR-EOM-Recurring-Invoice-Dedup-Recovery.md` -- passed.
- Skipped locally: the unrelated legacy-monthly writer-harness workflow uses
  an exact loopback `:5432` contract, but that port is owned by a non-owned
  existing disposable test container. This PR does not modify its task,
  harness, or migration set; hosted CI remains the independent run.
- Before deploy after merge: repeat the redacted live catalog preflight,
  restart only the ATLAS provider, prove active runtime SHA plus `/health` and
  receivables readiness, and retain the current #2441 runtime as rollback.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/storage/migrations/387_eom_recurring_invoice_dedup_recovery.sql` | 273 |
| `plans/PR-EOM-Recurring-Invoice-Dedup-Recovery.md` | 252 |
| `tests/test_invoice_repository.py` | 673 |
| **Total** | **1200** |
