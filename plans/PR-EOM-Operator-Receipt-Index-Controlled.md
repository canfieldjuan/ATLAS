# PR-EOM-Operator-Receipt-Index-Controlled

## Why this slice exists

Fixes a defect introduced by ATLAS #2548 (merge 5f6e74d04), slice A of the
eom-timetracker customer create match contract. Migration
`401_eom_operator_contact_matched_receipt_index` was placed in the automatic
startup chain. When the office `atlas-api` started on 5f6e74d04 it failed with
`must be owner of index idx_eom_lead_lifecycle_operator_contact_operation_key`:
migration 394 gave `eom_lead_lifecycle_events` to the no-login guard owner
`atlas_eom_handoff_owner`, and the runtime `atlas` role is not a member of that
role, so it can neither drop nor create an index on that table.

Consequences on the running service:

- Correctness is unaffected: matched receipts are written and replayed.
- The receipt lookup is unindexed.
- Every startup retries 401 and logs the failure.
- The generic runner stops at the first failure, so any later migration
  merged to main would also stay unapplied.

### Problem-derived contract

- Root cause: 401 needs guard-owner authority but was shipped as an ordinary
  startup migration. Atlas already has the mechanism for exactly this class:
  `CONTROLLED_DBA_MIGRATION_NAMES` keeps a migration out of startup, and the
  shared controlled runner applies it with a protected superuser DSN, its SQL
  and ledger row in one transaction (393-399 use it).
- Correct fix must touch/change: 401's SQL (no CONCURRENTLY, which the atomic
  controlled runner cannot run; a DBA-only guard like 399's), the startup
  exclusion set, the runner allowlist and predecessors (364 built the index,
  394 moved its table), a fixed `./ops db controlled` operation with its
  wrapper, and the capability map and runbook entries every controlled
  migration has.
- Must not change: what 401 builds (same index name, columns, and predicate),
  any runtime code path, every other migration, and the controlled runner's
  guards.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 12

1. Rewrite migration 401 as an atomic-bookkeeping, DBA-only migration: a
   superuser guard, `DROP INDEX IF EXISTS`, then `CREATE INDEX` with the same
   definition.
2. Add 401 to `CONTROLLED_DBA_MIGRATION_NAMES`, so startup skips it.
3. Allowlist 401 in the shared controlled runner, with predecessors 364 and
   394.
4. Add `scripts/apply_eom_operator_contact_receipt_index_schema.py` and the
   `./ops db controlled eom-operator-contact-receipt-index {preflight|apply}`
   operation.
5. Document the operation in `.agent/capabilities.yaml` and
   `.agent/runbooks/database.md`.

### Review Contract

- Acceptance criteria:
  - Startup skips 401: the controlled set equals the seven named migrations,
    and a generic run applies none of them --
    `tests/test_migrations_runner.py::test_generic_run_skips_controlled_dba_migration_until_explicitly_selected`.
  - 401 is atomic bookkeeping with no CONCURRENTLY. Its statements are the
    DBA guard, the drop, and the create, and the index predicate lists exactly
    `EOM_OPERATOR_CONTACT_EVENT_TYPES` --
    `tests/test_migrations_runner.py::test_operator_contact_receipt_index_lists_exactly_the_receipt_types`.
  - The controlled runner applies 401 alone, in the pinned transaction --
    `tests/test_eom_first_clean_completion_dba_runner.py::test_runner_applies_only_its_named_migration_in_pinned_transaction[401_eom_operator_contact_matched_receipt_index]`.
  - It refuses 401 when 364 or 394 is unrecorded, and applies nothing --
    `...::test_operator_contact_receipt_index_refuses_any_unrecorded_predecessor`.
  - The wrapper pins migration 401 and rejects any other migration --
    `...::test_operator_contact_receipt_index_entrypoint_pins_migration_401`.
  - `./ops` routes both modes to the wrapper with `--json`; the capabilities
    entry and runbook carry the same commands --
    `tests/test_agent_operations_contract.py`.
  - Real PostgreSQL, on a disposable schema where 394 has moved the
    lifecycle table to the guard owner and 364's index exists --
    `tests/test_eom_first_clean_completion.py::test_receipt_index_migration_401_needs_the_dba_and_commits_with_its_record`:
    - A non-superuser executor is refused before any change.
    - The controlled runner's pinned transaction rolls the rebuild back when its
      ledger write fails, recording nothing.
    - It then commits the widened, valid, guard-owned index together with a
      ledger row carrying the file's digest.
- Reachability proof: the ops dispatcher, the wrapper, and the shared runner
  are exercised end to end with a fake DBA pool; startup exclusion is
  exercised through `run_migrations`.
- Affected surfaces: migration 401, `atlas_brain/storage/migrations/__init__.py`,
  `scripts/apply_eom_first_clean_completion_schema.py`, the new wrapper, `ops`,
  `.agent/capabilities.yaml`, `.agent/runbooks/database.md`, and tests.
- Risk areas: a startup run still attempting 401; the runner admitting 401
  without its predecessors; a predicate that drifts from the receipt types.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R12, R13, R14.

### Guard-class closure declaration

- Member set: `CONTROLLED_DBA_MIGRATION_NAMES` and the runner's
  `CONTROLLED_MIGRATION_NAMES` each gain 401; `CONTROLLED_DATABASE_OPERATIONS`
  gains `eom-operator-contact-receipt-index`.
- Membership is **CLOSED** and **ENUMERATED**: each set is a literal whose
  tests compare it with the full expected set.
- Out-of-set default is **REJECT**: the runner refuses a non-allowlisted
  migration, and ops refuses an unknown operation.

### Boundary-change enumeration

- Boundary path/seam: startup `run_migrations` and the controlled
  `./ops db controlled` path.
- Replaced-path behaviors: startup no longer attempts 401; only the controlled
  operation does.
- Guard-relevant fields: migration name and recorded predecessors.
- Caller x input shape: startup x 401 (skipped); controlled runner x 401 with
  364 and 394 recorded (applied); controlled runner x 401 with either missing
  (refused); a non-superuser running 401 (refused by its DO block).

### Deployed-config probing

The apply needs the protected superuser DSN
(`ATLAS_EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL`) plus the schema, exactly
as for 394-399. No runtime configuration changes.

### Files touched

- `.agent/capabilities.yaml`
- `.agent/runbooks/database.md`
- `atlas_brain/storage/migrations/401_eom_operator_contact_matched_receipt_index.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `ops`
- `plans/PR-EOM-Operator-Receipt-Index-Controlled.md`
- `scripts/apply_eom_first_clean_completion_schema.py`
- `scripts/apply_eom_operator_contact_receipt_index_schema.py`
- `tests/test_agent_operations_contract.py`
- `tests/test_eom_first_clean_completion.py`
- `tests/test_eom_first_clean_completion_dba_runner.py`
- `tests/test_migrations_runner.py`

## Mechanism

401 now begins with `-- atlas: atomic-bookkeeping`, so the shared runner
applies its statements and the ledger row in one transaction. It refuses
CONCURRENTLY there, so the drop and create are plain. A DO block refuses a
non-superuser executor with the same wording the other controlled migrations
use. `CONTROLLED_DBA_MIGRATION_NAMES` makes generic startup runs skip the file.
The controlled runner admits it by name, requires 364 and 394 to be recorded,
and pins the run to one connection under the migration lock. The new `./ops`
operation is a fixed entry like 396-399's.

401 was never recorded in any persistent database: it failed on the office
database and CI databases are disposable. So it is rewritten in place rather
than replaced by a new number.

## Intentional

- Same index name and predicate as the merged 401. Only how it is applied
  changes, so the receipt lookup and its test are untouched.
- No readiness attestation like 398/399's. The index is a performance aid, and
  the runtime is correct without it.

## Deferred

- Running the apply on the office database. It needs the protected superuser
  DSN, which only the operator holds; the runbook section gives the commands.
- The pre-existing startup log `Migration content integrity mismatch` for 386
  and 387. It was present before this arc and is unrelated.

Parking predicate: anything that does not change how migration 401 reaches the
database stays parked.

Parked hardening: none.

## Verification

- Passed locally, with every database setting pointed at a closed port:
  `tests/test_eom_first_clean_completion_dba_runner.py`,
  `tests/test_agent_operations_contract.py`, and
  `tests/test_migrations_runner.py` (266 passed; the one skip needs
  `ATLAS_MIGRATION_TEST_DATABASE_URL`).
- Skipped locally, proven in CI: the real-PostgreSQL 401 test in
  `tests/test_eom_first_clean_completion.py`. It needs the disposable runtime
  and DBA URLs that the eom-lead-pipeline workflow provides.
- Passed locally: `ruff check` on the changed Python files and `ops`; no
  non-ASCII added; `python scripts/check_guard_class_closure.py --base origin/main --strict`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 11 |
| `.agent/runbooks/database.md` | 23 |
| `atlas_brain/storage/migrations/401_eom_operator_contact_matched_receipt_index.sql` | 42 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 7 |
| `plans/PR-EOM-Operator-Receipt-Index-Controlled.md` | 199 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 10 |
| `scripts/apply_eom_operator_contact_receipt_index_schema.py` | 42 |
| `tests/test_agent_operations_contract.py` | 38 |
| `tests/test_eom_first_clean_completion.py` | 150 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 108 |
| `tests/test_migrations_runner.py` | 42 |
| **Total** | **673** |
