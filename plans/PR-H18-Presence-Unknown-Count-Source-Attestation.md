# PR-H18-Presence-Unknown-Count-Source-Attestation

## Why this slice exists

The target-confirmed, read-only H-18 preflight for [#2476](https://github.com/canfieldjuan/ATLAS/issues/2476) still refuses pending migration 389 because its immutable source-identity report contains six missing packaged-source names. One is `022b_presence_unknown_count`: its one NULL-digest ledger row was applied at `2026-02-17T23:34:17.949845Z`, but the exact one-line source was later retained only through the source-controlled `022 -> 022b -> 027` rename chain. The current target has the exact intended `presence_events.unknown_count` catalog signature.

The generic report must keep showing this historical filename gap. Unlike migration 382, its byte identity is retained under the current `027_presence_unknown_count` package name, but the legacy ledger still has no content digest. This slice creates a named, fail-closed receipt for only that record. It unblocks no other discrepancy by construction and remains a provider prerequisite for the already-merged missed-call recovery migration.

### Problem-derived contract

- Root cause: `migration_content_integrity_report()` correctly classifies the historical `022b` ledger name as `missing_source` after its source file was renamed to `027`; the runner has no source-controlled, target-proven receipt that can decide whether this one name is safely admissible, so it must block every pending migration.
- Correct fix must touch/change: add one immutable `022b` reconciliation record and its catalog-only attestation to `atlas_brain/storage/migrations/reconciliation.py`; register only that record with the existing report-derived attestation dispatcher; add focused preflight, runner-admission, and disposable-PostgreSQL regression proofs; enroll that disposable proof in the existing PostgreSQL-backed migration CI workflow; archive this session's merged #2478 plan and refresh the plan index.
- Must not change: no packaged SQL migration; no `schema_migrations` rewrite, historical replay, source-file rename, database/table/lead/customer/email/financial data mutation, or deployment; no generic rename/allowlist mechanism; no change to 386, 387, 382, or any other missing-source record; no change to the missed-call recovery provider, public intake, CRM UI, product configuration, or customer-facing behavior.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 8

1. Add the one named `022b_presence_unknown_count` source-absence receipt. It must require exactly one NULL-digest ledger row at the observed timestamp, the exact retained package digest at current `027_presence_unknown_count`, and a catalog-only `presence_events.unknown_count` signature of nullable `integer DEFAULT 0` with no column constraint.
2. Permit the existing runner seam to subtract only this reported name when that receipt is `attested`; preserve the generic report's `missing_source` entry and block pending SQL for any non-attested or other discrepancy.
3. Add normal, malformed-evidence, no-row, duplicate-row, changed-package, changed-column/default/nullability/constraint, retry, and unrelated-missing-source proofs without querying business rows.
4. Archive the merged #2478 plan as the required same-session housekeeping; do not archive any other in-flight plan.
5. Run the new disposable-PostgreSQL proof in the existing migration-runner CI job, including path filters, without changing its runner service, permissions, dependencies, or any product behavior.

### Review Contract

- Acceptance criteria:
  - [ ] `attest_known_historical_migration_reconciliations(..., candidate_names={"022b_presence_unknown_count"})` returns `attested` only when the exact one-row NULL-digest ledger evidence, recorded timestamp, retained `027` SHA-256, and catalog-only column/constraint predicate all match; `tests/test_migration_content_integrity_preflight.py` and `tests/test_presence_unknown_count_migration_repair.py` settle both positive and negative cases.
  - [ ] The real `run_migrations()` admission path executes no pending SQL or ledger writes when the named evidence is absent/non-attested, and performs exactly one ordinary retry only after the same named evidence becomes attested; `tests/test_migrations_runner.py` settles this.
  - [ ] An attested 022b receipt never clears a different missing-source record; the generic preflight remains `unresolved_drift` / exit 2 while independent H-18 entries remain; `tests/test_migrations_runner.py` and the target-confirmed `scripts/check_migration_content_integrity.py --attest-known-reconciliations` receipt settle this.
  - [ ] The receipt reads only `schema_migrations` metadata and PostgreSQL catalog metadata, never `presence_events` rows or customer/lead/financial data; focused fake-query assertions and the disposable PostgreSQL proof settle this.
  - [ ] Existing no-candidate/default helper behavior and the existing 387/382 named receipts remain unchanged; focused regression tests settle this.
  - [ ] The existing PostgreSQL-backed migration-runner workflow runs the new disposable 022b proof whenever its file or migration-integrity implementation changes; `.github/workflows/atlas_migrations_runner_checks.yml` settles that CI enrollment without relying on the unit gate's intentionally absent test-database configuration.
- Reachability proof: `run_migrations()` is the actual migration admission entrypoint. Its observable effect is either `PendingMigrationContentIntegrityError` before any pending SQL or one ordinary application after the named receipt passes. The separate read-only preflight exposes the same receipt without applying a migration.
- Affected surfaces: H-18 migration integrity evidence, migration-runner admission, read-only integrity preflight, test-only disposable PostgreSQL schema, the existing migration-runner CI invocation, and plan archive/index.
- Risk areas: migration/ledger safety, false admission, false block, backward compatibility of the attestation helper, retry safety, and release deployment ordering.
- Reviewer rules triggered: R1, R2, R4, R5, R8, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: the report-derived `missing_source` admission decision in `_unresolved_pending_migration_content_evidence()`.
- Replaced-path behaviors: reported `022b_presence_unknown_count` remains globally visible as missing source but is subtracted only after its exact named receipt is `attested`; every unregistered, malformed, absent, duplicate, wrong-digest, wrong-timestamp, or wrong-catalog case stays a pending-SQL block.
- Guard-relevant fields: migration name, ledger-row cardinality, NULL ledger digest, UTC applied timestamp, current `027` packaged SHA-256, `unknown_count` data type/nullability/default, and the count of constraints involving that column.
- Caller x input shape:
  - `run_migrations()` x report containing only non-attested 022b: block before pending SQL and ledger writes.
  - `run_migrations()` x exact attested 022b plus no other discrepancy: admit ordinary pending SQL once.
  - `run_migrations()` x attested 022b plus another missing-source name: keep that other name blocking.
  - read-only preflight x report containing 022b: emit named evidence but retain its generic forensic entry and unresolved exit while unrelated drift remains.
  - direct helper without explicit candidates: preserve existing 387-only compatibility behavior.

### Deployed-config probing

- Deployed/default config values: Atlas target label is `host=localhost, port=5433, db=atlas`; the database target itself is not changed.
- Explicit value probe: `scripts/check_migration_content_integrity.py --expected-target 'host=localhost, port=5433, db=atlas' --attest-known-reconciliations` runs inside a read-only transaction.
- Absent value probe: the existing script rejects a missing `--expected-target` before opening a database connection; its behavior is unchanged and covered by existing preflight tests.
- Default-session/default-context probe: the exact target is displayed first with `--show-target`; the manual catalog query uses `default_transaction_read_only=on` plus `transaction(readonly=True)`.
- Side-effect ordering: the named receipt is evaluated under the runner's existing advisory lock before the first pending SQL or `schema_migrations` write; non-attestation raises before either action.

### Files touched

- `.github/workflows/atlas_migrations_runner_checks.yml`
- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/INDEX.md`
- `plans/PR-H18-Presence-Unknown-Count-Source-Attestation.md`
- `plans/archive/PR-H18-EOM-Public-Onboarding-Source-Attestation.md`
- `tests/test_migration_content_integrity_preflight.py`
- `tests/test_migrations_runner.py`
- `tests/test_presence_unknown_count_migration_repair.py`

## Mechanism

`HistoricalRenamedMissingSourceReconciliation` records the exact old ledger name, the current retained package name, the SHA-256 of the unchanged source bytes, and the target-observed timestamp. Its attestation reads the named ledger row and PostgreSQL catalog metadata only. It proves (a) exactly one legacy NULL-digest row, (b) exact timestamp, (c) current packaged `027` bytes still equal the source-controlled digest, and (d) the sole field this migration introduced still has its final shape.

The source-verification result remains explicitly weaker than historical ledger verification: a NULL historical digest cannot prove which bytes executed. The attestation is therefore immutable target evidence for one admission decision, not a backfilled checksum and not a generic filename-rename rule.

The existing runner derives candidates only from the live report, asks only registered named records to attest, and subtracts only `attested` names from the pending-SQL blocker. The preflight reports the same named evidence but deliberately leaves the generic report unchanged. Migration 389 remains blocked until each other H-18 record has its own passing receipt or forward recovery.

The existing PostgreSQL-backed migration-runner workflow now invokes the new
disposable proof and includes its path in both PR and `main` filters. The test
remains isolated to `ATLAS_MIGRATION_TEST_DATABASE_URL`; the unit gate retains
its normal no-database behavior instead of silently skipping the only real
database proof.

## Intentional

- The historical ledger digest remains NULL. The receipt does not rewrite it, claim that the original execution bytes are cryptographically verified, or turn the generic report green.
- The retained `022 -> 022b -> 027` history is one source-controlled fact embedded in one named record, not a reusable rename map or a filename-compatibility fallback.
- The catalog probe does not read `presence_events` data. It verifies the structural result of the one-column migration only.
- This PR contains no SQL file and does not deploy or attempt migration 389. Runtime deployment continues to wait for all H-18 blockers.

## Deferred

- #2476 remains responsible for 386's separate forward recovery and the five remaining missing-source records (`067`, `272`, `297`, `379`, and the generic 382 entry). This slice does not make any of them admissible.
- The missed-call recovery provider #2475 remains deployed only after H-18 is fully resolved, migration 389 has applied through the normal runner, and recovery delivery remains disabled during schema rollout.

Parking predicate: adjacent historical migration evidence, source-rename automation, and any catalog/data repair not required to attest only 022b are parked in #2476 rather than expanded into this slice.

Parked hardening: none.

## Verification

- Before implementation: target-confirmed read-only preflight remained `unresolved_drift` / exit 2 with 2 mismatches and 6 missing-source rows; 387 and 382 were individually attested. A separate read-only catalog probe observed exactly one NULL-digest `022b` ledger row at `2026-02-17T23:34:17.949845Z`, nullable `integer DEFAULT 0`, and zero constraints on `unknown_count`.
- Focused local regression: `pytest -q tests/test_migration_content_integrity_preflight.py tests/test_migrations_runner.py tests/test_presence_unknown_count_migration_repair.py` — `113 passed, 4 skipped in 0.71s` before disposable-DB configuration. The selected `022b` cases first passed `17 passed, 2 skipped`.
- Disposable PostgreSQL: a fresh `postgres:16-alpine` container bound only to `127.0.0.1:55437`, with `ATLAS_MIGRATION_TEST_DATABASE_URL` directed to that container, ran `pytest -q tests/test_presence_unknown_count_migration_repair.py` — `3 passed in 0.31s`; the container was removed by the shell trap. It exercised the exact no-synthetic-022b catalog, a real `unknown_count` constraint rejection, and the actual `run_migrations()` admission path.
- CI enrollment repair: the existing `Atlas Migrations Runner Checks` workflow now includes `tests/test_presence_unknown_count_migration_repair.py` in both path filters and its PostgreSQL-backed pytest invocation. A fresh auto-assigned-loopback `postgres:16-alpine` rerun passed `3 passed in 0.30s`; YAML parsing and `git diff --check` passed. GitHub remains the authority for the complete migration-runner workflow and unit gate.
- Fast mechanical checks: `ruff check` and `python -m py_compile` over all four changed Python files passed; `git diff --check` passed.
- Target-confirmed read-only receipt: `python scripts/check_migration_content_integrity.py --expected-target 'host=localhost, port=5433, db=atlas' --attest-known-reconciliations` returned exit `2` only because the generic report still contains 2 mismatches and 6 missing-source names. The new `022b` evidence was `attested` with every predicate true; the report remained deliberately unchanged and migration 389 was not attempted.
- Planned remote checks: GitHub retains the full unit gate, EOM lead-pipeline, migration-runner, pre-push, and live-reconciliation suites. Do not duplicate the broad unit gate locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_migrations_runner_checks.yml` | 3 |
| `atlas_brain/storage/migrations/reconciliation.py` | 217 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-Presence-Unknown-Count-Source-Attestation.md` | 124 |
| `plans/archive/PR-H18-EOM-Public-Onboarding-Source-Attestation.md` | 0 |
| `tests/test_migration_content_integrity_preflight.py` | 234 |
| `tests/test_migrations_runner.py` | 184 |
| `tests/test_presence_unknown_count_migration_repair.py` | 200 |
| **Total** | **965** |
