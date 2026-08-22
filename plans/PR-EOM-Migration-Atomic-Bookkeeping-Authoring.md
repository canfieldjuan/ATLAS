# PR-EOM-Migration-Atomic-Bookkeeping-Authoring

## Why this slice exists

H-18 in [ATLAS #2461](https://github.com/canfieldjuan/ATLAS/issues/2461)
is now through its runtime admission-policy slice (#2469). The migration runner
correctly detects direct executable `INSERT INTO schema_migrations` statements,
but it deliberately masks quoted and dollar-quoted SQL before that recognizer.
Consequently a migration that writes its own ledger row through `EXECUTE`,
`format`, or a procedure body is opaque to automatic detection. The existing
`-- atlas: atomic-bookkeeping` first-line marker is the designed escape hatch,
but the repository has no durable authoring guide or exact dynamic-SQL proof
that explains why it is required.

This slice completes that bounded H-18 follow-up. It documents the authoring
contract and proves the real runner still selects the atomic path for a marked,
otherwise-opaque dynamic self-recorder. It does not widen the SQL recognizer,
invent a heuristic for string literals, or alter any packaged migration.

### Problem-derived contract

- Root cause: the static self-recording recognizer must mask SQL literals and
  PL/pgSQL bodies to avoid false positives. A dynamic ledger write is therefore
  intentionally not discoverable from the executable-token grammar, and an
  author who omits the explicit marker could reintroduce the SQL-to-ledger crash
  window that atomic bookkeeping closes.
- Correct fix must touch/change: add one durable migration-authoring reference
  that names the exact first-nonempty-line marker, the direct-vs-dynamic
  boundary, `CONCURRENTLY` incompatibility, and required proof. Add focused
  `run_migrations()` tests that use dynamic `EXECUTE` source to demonstrate the
  opaque boundary and a disposable-PostgreSQL source-row/digest rollback and
  retry proof for the marked atomic entrypoint.
- Must not change: `_SELF_RECORDING_INSERT_RE`, `_executable_sql`, marker
  spelling/placement semantics, `run_migrations()` execution behavior,
  packaged migration SQL, `schema_migrations` history, migration admission,
  database configuration, financial/customer data, or any Gmail/Square/payment
  behavior. This is authoring guidance plus regression proof, not a dynamic SQL
  parser or a new runtime rejection policy.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 3

1. Add a migration-authoring guide for dynamic self-recording SQL: its first
   non-empty source line must be exactly `-- atlas: atomic-bookkeeping`; direct
   static inserts remain automatically covered; marked migrations cannot use
   executable `CONCURRENTLY`; and authors must prove the exact source through
   `run_migrations()`.
2. Add focused runner regression coverage proving an `EXECUTE`-hosted ledger
   insert is intentionally opaque to the direct recognizer, a late marker does
   not select the atomic path, and the exact first-line marker makes the real
   runner execute the same dynamic source in its atomic bookkeeping path.
3. Keep the dynamic parser boundary explicit and forward compatible: any future
   automatic detection requires a separate, evidence-backed H-18 slice rather
   than a literal/regex heuristic in this PR.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/MIGRATION_AUTHORING.md` names the exact first-nonempty marker,
    distinguishes direct static self-recording from dynamic SQL, explains the
    transaction/`CONCURRENTLY` constraint, and requires a real-runner proof;
    settled by the checked-in guide.
  - [ ] A dynamic `EXECUTE` source containing a ledger insert remains outside
    `_contains_executable_self_recording_insert`, and moving the marker below
    another non-empty line does not select atomic bookkeeping; settled by the
    new focused test in `tests/test_migrations_runner.py`.
  - [ ] The same dynamic source with the exact marker as its first non-empty
    line reaches `run_migrations()` against PostgreSQL; a forced digest-update
    failure rolls back its table and self-recorded ledger row, and an unchanged
    retry persists the expected version and source digest; settled by the
    focused real-PostgreSQL runner test.
  - [ ] Existing direct self-recording, marker, concurrent-DDL, retry, and
    no-pending behavior remain covered by the focused migration runner suite;
    settled by `tests/test_migrations_runner.py` and GitHub's `Atlas Migrations
    Runner Checks`.
  - [ ] No production migration source, runtime parser, ledger row, target
    configuration, or financial/customer state changes; settled by cold diff
    reconstruction and changed-file review.
- Reachability proof: the test writes a dynamic migration into a temporary
  catalog and calls the real `run_migrations()` entrypoint against a disposable
  PostgreSQL schema; observable results are the rolled-back table/ledger row
  on failure and the persisted version/source digest after retry.
- Affected surfaces: `docs/MIGRATION_AUTHORING.md`, the runner's existing
  marker/recognizer contract, `tests/test_migrations_runner.py`, and the
  already-enrolled `Atlas Migrations Runner Checks` workflow.
- Risk areas: false-positive parser expansion, accidental autocommit of dynamic
  ledger bookkeeping, unsafe `CONCURRENTLY` guidance, rewrites of historic SQL,
  and a test that proves only a helper rather than the runner entrypoint.
- Reviewer rules triggered: R1, R2, R4, R5, R10, R12, R13, R14.

### Boundary-change enumeration

N/A — the runner's parser, marker predicate, and admission behavior do not
change. This slice documents and regression-tests the existing boundary.

### Deployed-config probing

N/A — no configuration, environment fallback, deployed target, or runtime
connection behavior changes.

### Files touched

- `docs/MIGRATION_AUTHORING.md`
- `plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md`
- `tests/test_migrations_runner.py`

## Mechanism

The guide makes the existing boundary operational rather than implicit. Static
executable `INSERT INTO schema_migrations` statements keep their automatic
atomic selection. Dynamic SQL is intentionally opaque because the recognizer
masks literals and dollar-quoted bodies; its author must put the exact existing
marker first, before any prose comment or SQL. The runner then uses the same
atomic transaction it already uses for marked migrations, where migration SQL,
ledger row, and digest verification either commit together or roll back.

The regression fixture contains a valid-looking PL/pgSQL `EXECUTE` string. It
first proves the direct recognizer cannot see that literal and that a late
marker is not an opt-in. The real PostgreSQL runner fixture then injects a
digest-update failure after the dynamic ledger insert: both the table and ledger
row must roll back. Removing that fault and retrying the unchanged source must
persist the expected version and digest. The test intentionally does not claim
to interpret arbitrary dynamic SQL; that omission is the safety boundary being
preserved.

## Intentional

- Preserve automatic direct-insert recognition and the exact marker spelling;
  existing/historical migration source remains byte-stable.
- Do not scan string literals, `EXECUTE`, `format`, procedures, or arbitrary
  dynamic constructs for hidden ledger operations. Such a parser would have
  false-positive and false-negative semantics that require separate evidence.
- Do not make startup reject every dynamic SQL migration. The authoring policy
  applies only when the migration itself owns `schema_migrations` bookkeeping.
- Do not add an operator rollout, database query, external API, or financial
  side effect; the guide is source-authoring guidance.

## Deferred

- A broader dynamic-SQL classifier or runtime enforcement requires a separately
  reviewed H-18 issue with a concrete observed failure mode and a bounded
  grammar; it is explicitly outside this slice.
- #2363 remains the target-confirmed forensic decision for the unrelated 382
  missing-source record; this authoring policy creates no reconciliation or
  admission exception.
- Migration SQL/data repair, historical source rewriting, and generic migration
  framework replacement remain outside this thin authoring/proof slice.

Parking predicate: Keep findings in this slice only when they are necessary to
make the existing marker's authoring boundary or its disposable-PostgreSQL
rollback/retry proof materially correct. Park parser expansion, generic dynamic
SQL enforcement, historical migration/data repair, framework replacement, and
deployment orchestration unless a verified defect blocks that predicate.

Parked hardening: none against this predicate.

## Verification

- Executed locally: Command: python -m pytest -q tests/test_migrations_runner.py
  tests/test_migration_content_integrity_preflight.py. Result: 77 passed, 1
  skipped; the skipped real-PostgreSQL test had no configured local disposable
  target.
- Executed locally: Command: ruff check tests/test_migrations_runner.py.
  Result: passed. Command: python -m py_compile
  tests/test_migrations_runner.py. Result: passed. Command: git diff --check.
  Result: passed.
- Executed locally: Command: python scripts/audit_plan_code_consistency.py
  --base-ref origin/main plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md.
  Result: passed. Command: python scripts/sync_pr_plan.py --check
  plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md origin/main. Result:
  passed.
- Executed locally: Command: python scripts/audit_plan_doc.py
  plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md. Result: passed.
  Command: python scripts/audit_plan_doc_files_touched.py
  plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md origin/main. Result:
  passed. Command: python scripts/audit_plan_doc_diff_size.py
  plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md origin/main. Result:
  passed.
- Executed locally: Command: python scripts/audit_review_rules_triggered.py
  --plan plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md origin/main.
  Result: passed. Command: python scripts/check_boundary_change_enumeration.py
  --base origin/main --strict. Result: passed. Command: python
  scripts/check_guard_class_closure.py --base origin/main --strict. Result:
  passed.
- Skipped locally: the real PostgreSQL dynamic self-recording proof because
  ATLAS_MIGRATION_TEST_DATABASE_URL is not configured locally; GitHub's
  already-enrolled Atlas Migrations Runner Checks provides the disposable
  PostgreSQL service.
- Skipped locally: repository unit gate and broad CI mirror. Per operator
  direction, GitHub runs the full unit, security, integration, and required
  checks on the published head.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/MIGRATION_AUTHORING.md` | 92 |
| `plans/PR-EOM-Migration-Atomic-Bookkeeping-Authoring.md` | 203 |
| `tests/test_migrations_runner.py` | 77 |
| **Total** | **372** |
