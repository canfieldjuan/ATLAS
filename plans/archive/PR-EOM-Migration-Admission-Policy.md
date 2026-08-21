# PR-EOM-Migration-Admission-Policy

## Why this slice exists

H-18 phase two in [ATLAS #2461](https://github.com/canfieldjuan/ATLAS/issues/2461)
now has both prerequisites: #2462 provides a target-confirmed, read-only
provenance preflight and #2467 supplies immutable evidence for the real
migration-387 historical source gap. The production migration runner still only
logs mismatched or missing-source evidence, then applies any pending SQL. That
would allow a newly packaged migration to change a financial provider's schema
while the same deployment has unresolved provenance drift.

This production-hardening slice closes that precise admission gap. It is
justified by the recorded financial migration incident, not a new billing
workflow: it permits a pending migration only when every mismatch is absent or
represented by a current, attested, source-controlled reconciliation record and
no missing-source row exists. It leaves a no-pending startup available, and
documents the operator rollout, rollback, and recovery path.

### Contract revision — existing repair proof exposed by GitHub CI

GitHub's required migration-runner job proved that the real public-onboarding
repair test deliberately recreates an observed negative ledger record named
`382_eom_public_onboarding_tokens`, for which no same-named packaged source
exists. The new policy correctly rejects its pending 384 repair before SQL.
That is not evidence to weaken missing-source admission: phase one's archived
contract explicitly defines an unreadable/absent source as `missing_source`,
and the 384 repair plan records the historical collision as a real production
condition.

The narrow correction is test isolation, not a runtime exception. Preserve the
real-package refusal as a disposable-PostgreSQL regression; let the older 384
DDL tests use a test-owned readable legacy catalog solely to exercise the DDL
contract. Do not add an unproven reconciliation record for 382. Record its
target-confirmed evidence/reconciliation decision as a dedicated H-18 follow-up
in #2363 before a production migration-bearing rollout relies on this policy.

**Diff-budget exception:** the expected 600-plus-line change consists of the
required plan/Review Contract, one durable operator runbook, a small admission
decision, and an indivisible real-runner fixture matrix that proves refusal,
known attestation, retry recovery, and no-pending availability. Splitting the
runbook or the recovery tests would either leave operators without a safe
response to a financial migration refusal or make the new fail-closed boundary
unproven; no unrelated refactor is included.

### Problem-derived contract

- Root cause: `run_migrations()` computes and logs the generic
  `MigrationContentIntegrityReport`, but its pending-file loop does not use
  that report for admission. A known 387 reconciliation record exists but is
  deliberately not source verification, so the runner needs a separate policy
  decision that consumes only its current attestation rather than relabeling
  history as verified.
- Correct fix must touch/change: make the existing runner calculate pending
  files under its advisory lock, resolve only the candidate names exposed by
  the source-controlled reconciliation module, and refuse all pending SQL when
  any mismatch/missing-source evidence remains unresolved. Add behavior tests
  for unknown drift, attested 387, returned/transport-failed 387
  attestation/retry recovery, mixed known-plus-unresolved evidence, and
  no-pending availability. Add a stable operator runbook with exact rollout,
  rollback, and recovery constraints.
- Must not change: migration SQL, `schema_migrations` historical rows, source
  hashes, the generic preflight categories/exit code, invoice/payment/Gmail/
  Square behavior, target/database configuration, current no-pending startup,
  or any unrelated migration, CI, deployment, or billing lane. A historical
  record must never become `verified` merely because it is admission-attested.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 6

1. Add an advisory-lock-held pending-migration admission decision to the
   canonical runner. It blocks before executing a pending migration when
   unresolved generic mismatch/missing-source evidence remains.
2. Reuse the immutable reconciliation module as the sole source for names that
   can be attested. An attested record can satisfy admission for its own current
   mismatch only; unknown, missing-source, or non-attested evidence remains a
   blocker.
3. Preserve the pre-existing no-pending behavior: diagnostics remain visible,
   but no reconciliation catalog query or startup failure is introduced solely
   by historical drift when this run would apply no SQL.
4. Add focused runner tests and a durable runbook. The runbook documents the
   target-confirmed read-only preflight, deployment sequence, safe recovery,
   and the fact that rollback is never a way to bypass an admission refusal.
5. Keep the existing public-onboarding repair DDL proof meaningful without
   creating a production bypass: its test-owned readable legacy catalog proves
   migration 384's schema behavior, while a separate real packaged-catalog
   case proves the known missing 382 source now refuses the repair before SQL.
6. Scope a known reconciliation's non-attested result to an actively reported
   mismatch, and prove both a thrown attestation transport error and an
   attested 387 plus another unresolved row remain fail-closed before SQL.

### Review Contract

- Acceptance criteria:
  - [ ] With a pending temporary migration and either an unknown digest mismatch
    or a missing packaged source, `run_migrations()` raises the named admission
    error before the pending SQL or ledger record is executed; settled by the
    parametrized failure test in `tests/test_migrations_runner.py`.
  - [ ] With the existing 387 mismatch plus a current full attestation,
    `run_migrations()` accepts a later pending migration and records its exact
    digest; settled by the attested-runner test using the real reconciliation
    predicate fixture in `tests/test_migrations_runner.py`.
  - [ ] A failed known attestation applies no pending SQL; after the controlled
    catalog evidence is restored, retry applies the same pending file exactly
    once; settled by the recovery test in `tests/test_migrations_runner.py`.
  - [ ] A thrown known-attestation transport error raises the named refusal,
    releases the advisory lock and connection, writes no pending SQL or ledger
    row, and applies the unchanged pending file once after retry; settled by
    the transport-failure runner test in `tests/test_migrations_runner.py`.
  - [ ] A current attestation for 387 cannot clear an additional unknown
    mismatch or missing source: each mixed fixture still refuses before pending
    SQL and ledger writes; settled by the parametrized mixed-evidence runner
    test in `tests/test_migrations_runner.py`.
  - [ ] With no pending migration, a recorded mismatch remains a logged
    diagnostic and the runner returns without calling the reconciliation
    catalog probe; settled by the no-pending runner test and its fake executor
    call surface.
  - [ ] The real package's observed 382 missing-source ledger shape rejects
    pending 384 before its SQL or ledger record; the schema-repair test keeps
    its old DDL/failure proof only with an explicitly test-owned readable
    legacy catalog, settled by
    `tests/test_eom_public_onboarding_migration_repair.py` on disposable
    PostgreSQL.
  - [ ] `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` directs operators to the
    existing target-confirmed read-only preflight, refuses ledger edits/replay
    as recovery, and names safe rollout/rollback/reconciliation decisions;
    settled by cold diff review and the checked-in document.
- Reachability proof: the real `run_migrations()` entrypoint runs against its
  existing acquired-connection fake, and observable pending SQL/ledger state
  proves the admission decision occurs before migration execution. The operator
  runbook points to the existing real preflight entrypoint; no new public API is
  introduced.
- Affected surfaces: migration runner admission, immutable reconciliation
  evidence lookup, focused runner tests, and the Atlas operator migration
  runbook.
- Risk areas: financial-schema deployment safety, source provenance, startup
  availability, retry/idempotency, no-write ordering, backward compatibility,
  and operator recovery.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `run_migrations()` now uses the pre-existing content
  report at the single point after it knows a run has pending files and before
  any pending SQL is executed. The policy consumes generic
  `mismatched`/`missing_source` names plus named reconciliation attestations.
  Only a reported mismatch can be cleared by a matching current attestation;
  every missing source remains admission-blocking.
- Replaced-path behaviors: phase one's log-only handling is replaced only for a
  run that would apply at least one pending file. A no-pending run remains
  diagnostic-only; legacy-null evidence remains observe-only.
- Guard-relevant fields: pending migration stems, `mismatched` names,
  `missing_source` names, the closed derived set of reconciliation candidates,
  and each candidate's `attested`/`not_attested` status. The safe default is
  refusal: unrecognized, missing, unavailable, or non-attested evidence cannot
  satisfy admission, because executing new financial-schema SQL is the more
  expensive error direction.
- Caller x input shape: full application/MCP runner with pending SQL; `only=`
  runner with a targeted pending prerequisite; no pending files; known 387
  evidence attested; known evidence not attested; and unknown mismatch/missing
  source. The runner never accepts a caller-supplied digest or reconciliation
  identity.

### Closure declaration

1. **Closed or open:** the packaged migration and applied-ledger members are
   **CLOSED per run**: they come from the migration directory and the one
   `schema_migrations` snapshot. Reconciliation candidates are also **CLOSED**
   at each source revision: the reconciliation module exposes the reviewed
   named records. The caller's `only` collection is open, but its valid members
   are derived from the actual packaged directory and unknown names already
   raise before any migration SQL.
2. **Membership source:** all three decision sets are **DERIVED**: packaged
   files use `directory.glob("*.sql")`, applied names use the locked ledger
   snapshot, and known reconciliation names use the source-controlled
   reconciliation module. No copied allowlist exists in the runner or runbook.
3. **Outside-set behavior:** any mismatch not returned as currently attested,
   and every missing-source name, remains unresolved and rejects pending
   application. An absent/unreadable/non-attested known record has the same
   result. This is the safer default because a false refusal is recoverable by
   evidence/reconciliation, while applying schema SQL alongside unexplained
   provenance drift can create an unreconstructible financial deployment state.

### Deployed-config probing

- Deployed/default config values: no new setting, environment variable, or
  fallback is introduced. The normal runner retains its existing packaged
  `MIGRATIONS_DIR` default and existing typed database configuration.
- Explicit value probe: a caller may use the existing `only=` collection; a
  targeted pending migration is still subject to the same admission decision.
- Absent value probe: omitted `only` retains the full packaged-run behavior;
  no pending file returns before reconciliation catalog attestation.
- Default-session/default-context probe: the configured production runner and
  standalone MCP callers all reach the same `run_migrations()` implementation,
  so no caller can opt out through a new setting.
- Side-effect ordering: the runner obtains its existing lock/bootstrap and
  locked ledger snapshot, computes pending files, then resolves evidence before
  its first pending migration `execute()` or `_record_migration()` call.

### Files touched

- `atlas_brain/storage/migrations/__init__.py`
- `atlas_brain/storage/migrations/reconciliation.py`
- `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md`
- `plans/PR-EOM-Migration-Admission-Policy.md`
- `tests/test_eom_public_onboarding_migration_repair.py`
- `tests/test_migrations_runner.py`

## Mechanism

The runner keeps its existing advisory-lock and diagnostic report. Only if its
selected migration set contains a pending file does it ask the reconciliation
module which named historical records are eligible for attestation. It evaluates
those records only when the generic report names one as mismatched; every other
report name remains unresolved without extra catalog work. An attested
reconciliation removes only its exact migration name from the admission-blocking
mismatch set. Missing source evidence is never removed from that blocking set.

The runner raises one log-safe error containing only migration names/categories
before entering the pending-file loop. It makes no migration SQL or ledger
record write on that refusal. Existing retries remain safe because a corrected
attestation re-evaluates the unchanged pending set under the same advisory lock
and executes/records each file through the established idempotent mechanism.

The runbook makes the preflight an evidence receipt, not an alternate migration
runner: operators confirm the target, inspect the JSON, deploy the policy before
future migration-bearing releases, and use a reviewed forward reconciliation or
repair when admission blocks. It expressly prohibits editing historical ledger
rows, replaying migration 387, or rolling back to evade the policy.

## Intentional

- Generic preflight output remains `unresolved_drift` / exit 2 even for a
  currently attested historical record. Admission attestation is not historical
  source verification and must not obscure the forensic truth.
- The policy does not add a new migration, feature flag, CLI mode, schema table,
  or automated recovery. The existing runner is the canonical admission point;
  a parallel deployment gate would drift from actual application behavior.
- No-pending startup does not query current catalog reconciliation evidence.
  This preserves availability for services that merely inspect an already
  migrated database while still blocking any new migration application.
- Existing legacy-null hashes with a readable packaged source remain explicit
  but non-blocking. A legacy row with no readable same-named source is still
  `missing_source`, so it stays fail-closed until separately reconciled.
- A test-only readable 382 fixture is not a provenance reconciliation and does
  not add, replace, or bless a packaged migration source. It merely keeps the
  pre-existing 384 DDL regression test focused on its schema contract; the
  real packaged catalog remains fail-closed in a neighboring assertion.

## Deferred

Parking predicate: park any new historical reconciliation record, source-byte
reconstruction, migration SQL/data repair, startup/profile behavior change,
general deployment orchestrator, or dynamic-SQL parser expansion unless it is
required to enforce the single runner admission decision above.

- #2461 Slice 4: document/validate dynamic self-recording migration authoring
  around the existing atomic-bookkeeping marker without broadening the SQL
  parser speculatively.
- H-18 382 public-onboarding missing-source evidence: GitHub CI proved the
  existing repair fixture models a historical, same-named source absence. Do
  not create an admission exception from test evidence; #2363 must receive a
  target-confirmed forensic/preflight follow-up before a production migration
  run is unblocked on that name.
- H-18 known-reconciliation expansion remains parked: this loop documents the
  inactive-record condition and proves the current 387 boundary; #2363 remains
  the only target-confirmed forensic/evidence path for any additional name.
- A future source discrepancy requires its own reviewed evidence record and
  named catalog predicate; it must not be added as an ad hoc runner exception.
- #2034 generic cross-session reservation/semantic-drift investigation and
  #2035 branch-protection/CI policy remain outside this ownership lane.

Parked hardening: none against the stated predicate.

## Verification

- Fast local checks executed:
  `python -m pytest -q tests/test_migrations_runner.py
  tests/test_migration_content_integrity_preflight.py` — 73 passed, 1 skipped.
  This is the focused migration runner/preflight selection, not the repository
  unit gate.
- Fast local checks executed:

      ruff check atlas_brain/storage/migrations/__init__.py atlas_brain/storage/migrations/reconciliation.py tests/test_migrations_runner.py
      python -m py_compile atlas_brain/storage/migrations/__init__.py atlas_brain/storage/migrations/reconciliation.py tests/test_migrations_runner.py
      git diff --check

  All passed.
- Fast local collection check executed with the migration-test database URL
  explicitly unset: `tests/test_eom_public_onboarding_migration_repair.py` —
  4 skipped. The disposable-PostgreSQL proof is intentionally GitHub-only
  because no local target was independently verified as non-production.
- Fast local checks executed: `python scripts/sync_pr_plan.py --check
  plans/PR-EOM-Migration-Admission-Policy.md`, the reviewer-rule audit, and
  `python scripts/check_guard_class_closure.py --base origin/main --strict` —
  passed after the planned files are visible to Git's diff.
- GitHub-only: the complete `Atlas Migrations Runner Checks`, repository unit
  gate, security scans, and all required PR checks. Per operator direction, no
  broad unit gate will be run locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/__init__.py` | 81 |
| `atlas_brain/storage/migrations/reconciliation.py` | 11 |
| `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` | 107 |
| `plans/PR-EOM-Migration-Admission-Policy.md` | 314 |
| `tests/test_eom_public_onboarding_migration_repair.py` | 81 |
| `tests/test_migrations_runner.py` | 277 |
| **Total** | **871** |
