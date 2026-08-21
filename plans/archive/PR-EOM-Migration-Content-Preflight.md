# PR-EOM-Migration-Content-Preflight

## Why this slice exists

H-18 in [ATLAS #2363](https://github.com/canfieldjuan/ATLAS/issues/2363)
records the financial-provider deployment failure mode where a filename-recorded
migration can conceal changed source bytes. Phase one ([#2447](https://github.com/canfieldjuan/ATLAS/pull/2447)) correctly stores identities for newly applied migrations and classifies legacy, mismatched, and missing-source rows, but its classifier is reachable only through the write-capable migration runner.

The production checkpoint for [#2452](https://github.com/canfieldjuan/ATLAS/pull/2452)
then established the first real H-18 evidence case: migration 387 has a
historical digest/timestamp discrepancy, while the final catalog/readiness
predicate proves its required schema is currently present. An operator needs a
safe way to inspect that discrepancy before a future migration-bearing cutover.
Calling `run_migrations()` as an inspection tool is not safe: it evolves the
ledger bootstrap and can apply pending SQL.

This is Slice 1 of [#2461](https://github.com/canfieldjuan/ATLAS/issues/2461):
read-only provenance preflight. It is production hardening justified by a real
financial-provider migration incident, not a new billing behavior or a generic
CI/process slice.

**Diff-budget exception:** the expected 700-plus-line diff includes the
indivisible plan/contract, one small no-write CLI, failure-branch fixtures that
prove its nonzero, redaction, read-only, and target-confirmation behavior, and
CI enrollment. Splitting the checker from the tests or enrollment would violate
the repository's checker-fixture/CI requirements; no unrelated refactor is
included.

### Problem-derived contract

- Root cause: the only migration-content classifier is invoked inside
  `run_migrations()`. That entrypoint first performs DDL bootstrap and may run
  pending migration SQL, so it cannot be used as a no-mutation production
  provenance check. The current runtime therefore has no canonical
  operator-facing way to distinguish verified, legacy-unverified, mismatched,
  and missing-source ledger evidence before a cutover.
- Correct fix must touch/change: promote the existing classifier to one
  canonical reusable helper without changing its classifications; add a
  standalone `scripts/check_migration_content_integrity.py` entrypoint that
  connects through the existing typed Atlas database settings, wraps the query
  in a read-only transaction, emits deterministic redacted JSON, and returns a
  nonzero status only for unresolved mismatch/missing-source evidence or an
  unavailable check. Add failure-branch tests and enroll them in the existing
  migration-runner workflow.
- Must not change: historical migration SQL or `schema_migrations` rows;
  pending migration selection, runner startup behavior, source-hash recording,
  runtime service units, financial records, invoices, payments, Gmail, Square,
  #2034's generic collision investigation, or #2035's branch-protection/CI
  policy.

### Contract revision

- New evidence: the existing typed database configuration defaults to
  `localhost:5433/atlas` when no deployed `ATLAS_DB_*` value is present
  (`atlas_brain/storage/config.py:25-52,77-131`). A no-argument preflight
  could therefore return a truthful report for the wrong local database and be
  misread as a production cutover receipt.
- Revised root cause: besides lacking a standalone no-write classifier path,
  the first draft lacked an explicit operator-to-config target binding.
- Revised required change surface: the CLI must expose the log-safe current
  `db_settings.target_label`, require an exact `--expected-target` before it
  opens a connection, include that label in the emitted receipt, and return the
  existing unavailable exit code before any database query when the expected
  target is absent or different. Tests must prove matching, absent, and
  mismatched target paths.
- Revised explicit non-scope: connection target confirmation is limited to the
  existing log-safe typed configuration label. It does not add a DSN argument,
  invent a second connection configuration source, prove cluster identity, or
  change application/runtime configuration.
- Revised assumptions/blockers: the label is an operator confirmation guard,
  not cryptographic database identity. The next H-18 reconciliation/enforcement
  slices remain responsible for durable policy after the preflight reports.
- Revised verification plan: add focused CLI tests showing no connection is
  attempted for an absent or mismatched target, and that only a matching label
  reaches the async preflight.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 5

1. Expose the phase-one content-integrity classifier as the migration module's
   canonical reusable read function while retaining the existing internal name
   as a compatibility alias; the runner retains its current diagnostic
   behavior.
2. Add a read-only operator preflight command. It uses the existing typed
   `ATLAS_DB_*` configuration, creates no schema object, executes no migration
   SQL, and runs its catalog query inside an asyncpg read-only transaction. The
   command exposes a log-safe configured-target label and requires an exact
   `--expected-target` confirmation before connecting.
3. Emit only classification names/counts and generic error class information:
   no connection string, credentials, SQL text, invoice data, or customer data.
   A mismatch or missing source returns exit code 2; an unavailable preflight
   returns exit code 3; legacy-unverified evidence remains visible but retains
   phase one's observe-only exit behavior.
4. Add focused behavior tests and enroll the new test file in
   `.github/workflows/atlas_migrations_runner_checks.yml`.

### Review Contract

- Acceptance criteria:
  1. `scripts/check_migration_content_integrity.py` calls the exact migration
     content classifier through a read-only transaction and does not invoke
     `execute`, `_ensure_migrations_table`, or `run_migrations`; settled by
     `tests/test_migration_content_integrity_preflight.py` fake connection
     state and source inspection.
  2. A verified row, a readable null-digest legacy row, a mismatched row, and
     a missing-source row appear under their exact stable JSON keys; a
     mismatch/missing-source result exits 2; settled by
     `tests/test_migration_content_integrity_preflight.py::test_preflight_reports_unresolved_drift_without_writes`.
  3. A legacy-only report remains explicit but exits 0 under phase one's
     observe-only policy; settled by
     `tests/test_migration_content_integrity_preflight.py::test_preflight_keeps_legacy_evidence_visible_without_treating_it_as_drift`.
  4. A connection/classification failure exits 3 with a generic error class
     rather than an exception message or DSN; settled by
     `tests/test_migration_content_integrity_preflight.py::test_main_redacts_database_failure_details`.
  5. The migration workflow runs the new test file on changes to the script or
     test; settled by `.github/workflows/atlas_migrations_runner_checks.yml`
     and its current-head GitHub job.
  6. The real CLI does not open a connection for an absent or mismatched
     expected target, and a matching log-safe label is included in the receipt;
     settled by target-confirmation tests in
     `tests/test_migration_content_integrity_preflight.py`.
- Reachability proof: executing the script's real async entrypoint against an
  isolated fake external database connection returns JSON plus its documented
  process status; no HTTP or customer-facing surface is added.
- Affected surfaces: migration-content classifier utility, one operator CLI,
  its focused tests, and the migration-runner CI workflow.
- Risk areas: migration source truthfulness, read-only database safety,
  historical ledger availability, secret redaction, CI enrollment, and
  backward-compatible runner diagnostics.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R10, R12, and R14.

### Boundary-change enumeration

- Boundary path/seam: the new CLI binds an operator-provided
  `--expected-target` to the existing `db_settings.target_label` before it
  opens a database connection; after that admission, the existing closed-set
  content classifier maps its four categories to the documented receipt and
  exit status.
- Replaced-path behaviors: no production caller is replaced. `run_migrations()`
  retains its diagnostic behavior; the new operator path replaces unsafe use of
  that write-capable runner as an inspection command.
- Guard-relevant fields: the non-secret configured target label, expected
  target label, connection availability, and existing `verified`,
  `legacy_unverified`, `mismatched`, and `missing_source` category values.
  Absent/mismatched target and unavailable connection default to exit 3 before
  the catalog query; mismatch/missing source default to exit 2; legacy evidence
  remains explicit but observe-only under phase one's policy.
- Caller x input shape: `--show-target` returns the log-safe label without a
  connection; absent/mismatched `--expected-target` is rejected before any
  connection; an exact label executes one read-only catalog query against the
  packaged SQL catalog.

### Guard class-closure declaration

- The classifier input set is closed per invocation: migration source members
  come only from `sorted(MIGRATIONS_DIR.glob("*.sql"))`, and ledger members come
  only from the one `schema_migrations` query. The command accepts no caller
  supplied migration names, digest values, or SQL.
- The target admission input is a single exact equality comparison against the
  repository's existing log-safe typed configuration label. Any absent,
  unmatched, or unavailable value fails before a query. The tests exercise both
  error directions: a matching label reaches the preflight and a mismatched or
  absent label does not.

### Deployed-config probing

The command introduces no new setting or fallback. It reuses the existing typed
`atlas_brain.storage.config.db_settings` and requires the operator to bind the
non-secret `db_settings.target_label` explicitly before any query.

- Deployed/default config values: `DatabaseConfig` is the existing
  `ATLAS_DB_*` authority; its effective deployed target is deliberately exposed
  only as `target_label`, never as a DSN or credential.
- Explicit value probe: an exact `--expected-target` reaches the read-only
  preflight and returns that label in JSON.
- Absent value probe: argparse rejects a preflight without
  `--expected-target` (except the no-query `--show-target` command).
- Default-session/default-context probe: the current typed default label is
  treated like any other label; only an exact explicit confirmation admits a
  query.
- Side-effect ordering: target admission completes before `_connect_read_only`;
  the catalog query itself is inside `readonly=True` transaction.

### Files touched

- `.github/workflows/atlas_migrations_runner_checks.yml`
- `atlas_brain/storage/migrations/__init__.py`
- `plans/PR-EOM-Migration-Content-Preflight.md`
- `scripts/check_migration_content_integrity.py`
- `tests/test_migration_content_integrity_preflight.py`

## Mechanism

`MigrationContentIntegrityReport` remains the source of classification truth.
The module's canonical helper receives the actual migration catalog plus a
database executor and only reads `schema_migrations`; the former private helper
continues delegating to it so the existing runner behavior and direct tests stay
compatible.

The script obtains the already configured database connection target through
`db_settings`. `--show-target` displays its log-safe label without connecting;
the query path requires a matching `--expected-target`, then establishes a
dedicated connection and opens
`connection.transaction(readonly=True)` before invoking that canonical helper.
It never imports or calls `run_migrations()`. Its payload contains sorted names
in the existing four categories and a status. `mismatched` or
`missing_source` is `unresolved_drift` / exit 2; a connection or catalog error
is `could_not_determine` / exit 3; a legacy-only result stays visibly
`legacy_unverified` but exits 0 because this phase deliberately does not yet
block a deployment.

## Intentional

- This slice does not fail application startup or alter pending migration
  selection. The current production 387 mismatch is real; applying a global
  fail-closed rule before an explicit reconciliation path exists would make
  ordinary migration-bearing deployments unavailable.
- The preflight does not offer a `--database-url` argument, avoiding DSNs on
  process listings and keeping it on the existing typed Atlas configuration
  boundary. It also does not load a `.env` itself.
- It requires a non-secret target confirmation instead of silently trusting
  fallback database defaults. This reduces wrong-environment receipts without
  pretending that host/database naming is a cryptographic cluster identity.
- Legacy null digests remain an explicit observation rather than a failure
  because phase one deliberately preserved those historical rows and did not
  prove their source bytes. This script must not relabel them as verified.
- The command reuses the one phase-one classifier rather than copying its
  categories into a second reporting implementation.

## Deferred

Parking predicate: historical-source reconstruction, any mutation of a
recorded ledger row, policy that blocks a new migration, rollback automation,
and parser expansion for dynamic self-recording SQL are parked unless they are
required to make this read-only preflight truthful.

- #2461 Slice 2: a separately reviewed, additive reconciliation-evidence model
  for the known 387 discrepancy. It must preserve the source mismatch and
  attach catalog/readiness proof rather than claim historical-byte verification.
- #2461 Slice 3: pending-migration admission policy plus documented rollout,
  rollback, and recovery after an explicit reconciliation record exists.
- #2461 Slice 4: dynamic self-recording migration authoring policy; retain the
  existing `-- atlas: atomic-bookkeeping` marker until then.
- #2034 remains a non-overlapping investigation-only issue; this PR does not
  change its semantic reservation/drift audit. #2035 remains the owner of
  branch-protection policy; this PR only enrolls its own focused test.

Parked hardening: none against the stated predicate.

## Verification

- Fast local checks planned: `python -m py_compile
  atlas_brain/storage/migrations/__init__.py
  scripts/check_migration_content_integrity.py`, then the focused pytest file.
- GitHub: `Atlas Migrations Runner Checks` executes the new focused test on the
  exact PR head. No full test suite is duplicated locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_migrations_runner_checks.yml` | 5 |
| `atlas_brain/storage/migrations/__init__.py` | 12 |
| `plans/PR-EOM-Migration-Content-Preflight.md` | 270 |
| `scripts/check_migration_content_integrity.py` | 198 |
| `tests/test_migration_content_integrity_preflight.py` | 248 |
| **Total** | **733** |
