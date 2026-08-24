# PR-EOM-Missed-Call-Recovery-Runtime-Privileges

## Why this slice exists

The first production application of the EOM missed-call recovery schema reached
the new additive migration through a DBA account. PostgreSQL therefore made the
new recovery tables and functions owned by `postgres`; the normal `atlas`
runtime role has no table privileges. The recovery service nevertheless reads,
locks, inserts, and updates those tables. Its startup probe currently checks
only that relations, one index/column, triggers, and one function exist, so it
can report a structurally present but unusable schema as ready.

### Problem-derived contract

- Root cause: a privileged migration executor becomes the owner of newly created
  PostgreSQL objects unless the migration assigns a protected no-login owner and
  materializes the runtime ACL after ownership transfer. Migration 389 creates
  the missed-call tables/functions without either step; the service readiness
  predicate has no privilege check.
- Correct fix must touch/change: add one forward-only, DBA-only migration after
  389 that assigns recovery-object ownership to `atlas_eom_handoff_owner`,
  provides the precise `atlas` privileges needed by the recovery service
  (including the `UPDATE` capability required for its `FOR UPDATE` row locks),
  preserves NocoDB's direct denial across table, column, and function ACLs,
  and makes the CRM-trigger bridge run with the isolated owner and a fixed
  search path. Extend `missed_call_recovery_schema_ready` to prove that the
  configured runtime role can actually use only the intended schema surface,
  the guard remains membership-isolated, and immutable-evidence triggers are
  intact. Keep migration 393 out of ordinary EOM startup migration selection;
  provide a controlled DBA-only runner and runbook step so runtime startup sees
  the recorded repair. Add disposable-PostgreSQL proof for both roles, trigger
  fencing, and the controlled runner.
- Must not change: do not rewrite migration 389 or its recorded receipt; do not
  modify existing recovery evidence, provider defaults, booking/email behavior,
  authenticated routes, CRM UI, role memberships, billing logic, or historical
  migration contents or the generic full migration runner. Do not repair
  production through manual grants or restart services before the forward
  migration is source-reviewed and applied.

## Scope (this PR)

Ownership lane: eom/missed-call-recovery-runtime-privileges
Slice phase: Production hardening

1. Add the additive recovery-ACL migration and security-definer trigger bridge.
2. Make readiness fail closed when the runtime role cannot execute the service's
   actual recovery-table operations.
3. Keep DBA-only/historical recovery SQL out of the slim-profile's ordinary
   migration set and provide a controlled runner for migration 393.
4. Prove the runtime/NocoDB boundary in a disposable Postgres schema.
5. Document the redacted-DSN DBA cutover before normal runtime startup.

### Review Contract

- Acceptance criteria:
  1. The new migration is forward-only and DBA-gated; it keeps the existing
     no-login, membership-isolated guard model, transfers all six recovery
     tables plus the privileged CRM-trigger functions to that guard, clears
     stale table/column/function ACLs, and refuses to grant mutable evidence
     access unless the exact append-only trigger bindings are intact.
  2. After applying the migration in a disposable schema, the runtime role has
     the exact required table privileges: read/insert/lock for receipts and
     attempts, read/insert for suppressions/events, and read/insert/update for
     sequences/steps. `tests/test_eom_missed_call_recovery.py` proves the real
     role can perform the service's locked read and mutations.
  3. `atlas_nocodb` cannot directly select, insert, update, or delete recovery
     tables, but an allowed CRM-table mutation still reaches the guarded trigger
     and terminalizes affected recovery work. The same disposable-Postgres test
     proves both sides.
  4. `missed_call_recovery_schema_ready` returns false for a structurally
     complete schema with missing or extra runtime ACLs, runtime guard
     membership, a disabled/rebound immutable-evidence trigger, or an untrusted
     NocoDB direct path; it returns true only after the migration establishes
     the expected privilege/guard properties. The integration test settles the
     effect through the real query.
  5. `EOM_MISSED_CALL_RECOVERY_READINESS_MIGRATIONS` excludes DBA-only and
     historical recovery SQL. The dedicated DBA runner defaults to a redacted,
     read-only preflight, applies only migration 393 after `--apply`, and
     verifies the migration ledger before a full generic Atlas startup or
     enabling recovery. A fresh target first receives migration 389 from the
     slim EOM bootstrap path. The corresponding source-level tests settle both
     boundaries.
  6. No route, provider, delivery configuration, email copy, booking state, or
     recorded recovery evidence changes. Settled by the focused existing
     recovery test suite and a cold diff audit.
- Reachability proof: the service calls `missed_call_recovery_schema_ready`
  before starting the worker; the disposable Postgres proof applies the actual
  migration, connects as each real role, executes the schema probe, and observes
  the corresponding allowed/denied database effects. The controlled DBA runner
  uses the existing migration ledger after the slim bootstrap records 389 and
  before a full generic Atlas process starts or recovery is enabled.
- Affected surfaces: `atlas_brain/storage/migrations/`,
  `atlas_brain/main_eom.py`, `atlas_brain/services/eom_missed_call_recovery.py`,
  the controlled migration runner/runbook, and their disposable Postgres/unit
  tests.
- Risk areas: PostgreSQL ownership/ACL semantics, `FOR UPDATE` lock privileges,
  security-definer search paths and direct function execution, NocoDB direct
  data exposure, migration retry/atomicity, immutable-evidence fences, and
  worker startup gating.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R13.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `atlas` service role -> recovery tables; `atlas_nocodb`
  CRM-table write -> guarded recovery trigger functions; startup probe -> worker
  admission; protected DBA DSN -> recorded migration 393 -> normal runtime
  startup.
- Replaced-path behaviors: previously a table's mere existence admitted the
  worker; after this change only structural presence plus usable runtime ACLs
  and guarded trigger properties admit it.
- Guard-relevant fields: table/column ACL, table owner, function ACL/owner,
  `prosecdef`, function `search_path`, trigger identity/enabled state, role
  login/inheritance/membership properties, and the six recovery table names.
- Caller x input shape: `atlas` executes its existing service SQL; NocoDB only
  performs its documented contacts/contact-interactions mutations. Neither
  caller receives a new public route or payload shape.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: recovery delivery remains disabled by current
  configuration; this slice neither enables it nor changes its defaults.
- Explicit value probe: with a structurally complete schema and correct ACLs,
  the real schema probe returns true.
- Absent value probe: removing a required runtime privilege makes the same probe
  return false before the worker can be started.
- Default-session/default-context probe: the query uses `current_user`, so the
  integration proof connects as the runtime role rather than spoofing a role
  name in a superuser session.
- Side-effect ordering: readiness is a catalog/ACL read before worker admission;
  role-bound runtime mutations are proved only after the migration establishes
  grants in the disposable schema.

### Files touched

- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_missed_call_recovery.py`
- `atlas_brain/storage/migrations/393_eom_missed_call_recovery_runtime_privileges.sql`
- `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md`
- `plans/PR-EOM-Missed-Call-Recovery-Runtime-Privileges.md`
- `scripts/apply_eom_missed_call_recovery_runtime_privileges.py`
- `tests/test_eom_missed_call_privilege_runner.py`
- `tests/test_eom_missed_call_recovery.py`
- `tests/test_eom_render_profile.py`

## Mechanism

The migration validates that a DBA is applying it and that
`atlas_eom_handoff_owner` remains a no-login, membership-isolated guard. It
moves recovery tables and the CRM-trigger execution chain under that guard,
sets privileged trigger functions to `SECURITY DEFINER` with a fixed schema
search path, clears public/NocoDB/runtime table and column ACLs plus explicit
NocoDB function execution, and then grants the runtime only its proven SQL
surface. It refuses to proceed unless the receipt/attempt append-only triggers
are still enabled and bound to their rejecting functions.

The service's existing structural probe becomes an admission predicate for that
same privilege/guard contract, so a partial, over-privileged, trigger-drifted,
or DBA-owned deployment fails closed before any provider worker can run. The
test applies the source migration to an empty disposable schema, connects as
each restricted role, and exercises both allowed runtime locking/writes and
denied direct NocoDB data access plus the guarded CRM trigger path. A fresh
target's slim EOM bootstrap records migration 389; the separate DBA runner then
records migration 393 before a full generic Atlas runtime starts or recovery is
enabled. The slim readiness tuple intentionally carries only ordinary schema
prerequisites.

## Intentional

- Preserve `FOR UPDATE` concurrency semantics rather than weakening locks to
  avoid an `UPDATE` privilege grant; immutable-row triggers remain the actual
  mutation fence.
- Reuse the existing no-login handoff guard rather than create a second EOM
  ownership role.
- Do not give NocoDB a direct recovery-table grant. Its CRM writes use guarded
  trigger functions instead.
- Do not alter historical migration 389; a new, recorded migration is the
  auditable recovery path.

## Deferred

- Production cutover happens only after this source slice is reviewed, merged,
  migration 389 is present, and its forward repair is applied through the
  controlled DBA runner. The required post-apply role probes and service restart
  are operational proof, not a manual ACL workaround.

Parking predicate: speculative privilege hardening unrelated to the six
recovery tables or the existing CRM trigger bridge is parked; only a proven
runtime-access or direct-data-exposure gap belongs in this slice.

Parked hardening: none.

## Verification

- `python -m compileall -q atlas_brain/main_eom.py
  atlas_brain/services/eom_missed_call_recovery.py
  scripts/apply_eom_missed_call_recovery_runtime_privileges.py
  tests/test_eom_missed_call_recovery.py
  tests/test_eom_missed_call_privilege_runner.py
  tests/test_eom_render_profile.py` — passed locally.
- `python -m pytest -q tests/test_eom_missed_call_recovery.py` — `13 passed,
  42 skipped` locally. The disposable-PostgreSQL role test is skipped because
  `ATLAS_MIGRATION_TEST_DATABASE_URL` is absent in this workspace; hosted CI
  remains the required real-role proof.
- `python -m pytest -q tests/test_eom_missed_call_privilege_runner.py
  tests/test_eom_render_profile.py` — `67 passed` locally.
- `git diff --check` and the repository mechanical review wrapper remain
  required before publishing. No production database, role, migration, or
  service restart was touched by this source slice.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/main_eom.py` | 11 |
| `atlas_brain/services/eom_missed_call_recovery.py` | 191 |
| `atlas_brain/storage/migrations/393_eom_missed_call_recovery_runtime_privileges.sql` | 414 |
| `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md` | 44 |
| `plans/PR-EOM-Missed-Call-Recovery-Runtime-Privileges.md` | 238 |
| `scripts/apply_eom_missed_call_recovery_runtime_privileges.py` | 174 |
| `tests/test_eom_missed_call_privilege_runner.py` | 158 |
| `tests/test_eom_missed_call_recovery.py` | 563 |
| `tests/test_eom_render_profile.py` | 3 |
| **Total** | **1796** |

## Diff size rationale

The source/test diff exceeds the 400-LOC soft budget because the indivisible
security repair requires one forward migration, an executable readiness
predicate, a controlled DBA-only runner, and a real-role disposable-Postgres
proof. No provider, route, configuration, UI, or historical-evidence behavior
is included.
