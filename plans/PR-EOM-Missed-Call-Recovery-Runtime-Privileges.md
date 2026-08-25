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
  provides the precise configured EOM funnel runtime privileges needed by the recovery service
  (including the `UPDATE` capability required for its `FOR UPDATE` row locks),
  preserves NocoDB's direct denial across table, column, and function ACLs,
  and verifies each CRM-trigger bridge body against migration 389 before making
  it security-definer under the isolated owner and a fixed search path. Extend
  `missed_call_recovery_schema_ready` to prove that the
  configured runtime role can actually use only the intended schema surface,
  the guard remains membership-isolated, and immutable-evidence triggers are
  intact. Keep migration 393 out of ordinary EOM startup migration selection;
  provide a controlled DBA-only runner and runbook step so runtime startup sees
  the recorded repair only after the ordinary migration-389 receipt is present.
  Add disposable-PostgreSQL proof for both roles, trigger fencing, and the
  controlled runner.
- Must not change: do not rewrite migration 389 or its recorded receipt; do not
  modify existing recovery evidence, provider defaults, booking/email behavior,
  authenticated routes, CRM UI, role memberships, billing logic, or historical
  migration contents or the generic full migration runner. The controlled
  runner must not apply or recreate migration 389; the slim bootstrap remains
  its only application path. Do not repair production through manual grants or
  restart services before the forward migration is source-reviewed and applied.

## Scope (this PR)

Ownership lane: eom/missed-call-recovery-runtime-privileges
Slice phase: Production hardening
Max files: 10

1. Add the additive recovery-ACL migration and security-definer trigger bridge.
2. Make readiness fail closed when the runtime role cannot execute the service's
   actual recovery-table operations.
3. Keep DBA-only/historical recovery SQL out of the slim-profile's ordinary
   migration set and provide a controlled DBA runner that admits only the
   exact 390-392 historical prelude when the generic migration guard requires
   it, then refuses to apply 393 before migration 389 has a ledger receipt.
4. Prove the runtime/NocoDB boundary in a disposable Postgres schema.
5. Document the redacted-DSN DBA cutover before normal runtime startup.
6. Enroll the controlled DBA-runner migration/test surface in the explicit EOM
   workflow so a runner-only or migration-393 change cannot bypass its required
   PostgreSQL proof.

### Fix-loop repair contract (current review round)

- Root cause: `run_migrations(..., only=...)` still computes global unresolved
  historical migration evidence before it applies the selected pending files.
  For an exact 379/386 recovery state it will select 391/392/390 only when that
  name is in the caller's pending set. The slim migration tuple deliberately
  omits those DBA/historical names, so a legacy target can fail before its
  ordinary migration-389 bootstrap. Separately, migration 393 changes six
  existing migration-389 bridge functions to security-definer ownership without
  proving their stored `pg_proc.prosrc` bodies are the trusted source bodies.
  The trusted-body comparison itself depends on an installed `pgcrypto`
  extension, but neither migration 393 nor the fresh disposable-schema path
  creates it. The explicit EOM workflow also uses path allowlists and a named
  pytest list which omit migration 393, the controlled runner, and its focused
  unit test, so a change to either migration gate can receive no EOM CI proof.
  The generic migration runner deliberately commits one selected historical
  recovery and then raises `PendingMigrationContentIntegrityError` while more
  historical evidence remains; the controlled DBA loop does not distinguish
  that committed-progress stop from a no-progress integrity failure. Finally,
  the receipt/attempt fence checks attest only each trigger's OID binding.
  PostgreSQL preserves that OID under `CREATE OR REPLACE FUNCTION`, so a
  replaced append-only function body can admit a table-wide runtime `UPDATE`.
  Finally, migration 393 clears direct definer execution for `atlas_nocodb`
  but not `atlas`; an explicit pre-existing or later `atlas` `EXECUTE` grant
  can bypass the intended CRM-trigger-only path and terminalize arbitrary
  active sequences. The resulting repair still treats the literal role `atlas`
  as the runtime authority even though missed-call recovery actually runs from
  the configured EOM funnel connection, whose login can differ. It also leaves
  the two append-only fence functions and the helper called inside the definer
  chain owned by migration 389's runtime executor after attesting them; that
  executor can replace a fence body after the repair and before using the
  existing mutable-row grants. The two scope-validation trigger functions also
  remain `SECURITY INVOKER` after their ownership transfer: the restricted
  runtime must invoke them for recovery writes, but is intentionally not
  granted direct `SELECT` on `contacts`, which those validators read. Finally,
  391 and 392 require `pgcrypto`, but the controlled runner selects those
  historical preludes before migration 393 creates the extension.
- Required change surface: update only the controlled DBA runner, migration
  393, their existing focused tests, the EOM workflow enrollment, the rollout
  runbook, and this plan. On
  `--apply`, an existing migration ledger lets the runner first give the
  existing migration runner only the three explicitly enumerated 390-392
  historical recovery names, allowing it to apply at most one selected prelude
  per invocation; an absent ledger must reject without invoking that generic
  runner. It must never apply 389 itself, and must still reject a missing 389
  receipt before applying 393. Its read-only result must report all prelude
  receipts. After it validates the protected DBA/role boundary and before it
  calculates any trusted bridge-function digest, migration 393 must provision
  and locate `pgcrypto`; the disposable-schema test must observe that the
  applied migration established that extension rather than pre-seeding it in a
  fixture. Migration 393 must verify
  the exact trusted body digest for all six bridge functions before any
  `SECURITY DEFINER` or ownership change. Tests must prove the DBA prelude is
  ordered before 393, an absent ledger and a missing 389 never reach 393, and
  each bridge function rejects a tampered body without gaining definer
  authority. The EOM workflow must trigger on and execute the controlled
  runner/migration-393 proof on both pull requests and post-merge pushes.
  The DBA loop must catch only the expected post-recovery
  `PendingMigrationContentIntegrityError`, re-read the ledger, and continue
  only when an explicitly authorized 390-392 receipt advanced; an unchanged
  ledger must re-raise. Migration 393 must attest both receipt/attempt fence
  function bodies before it grants `UPDATE`, and runtime readiness must reject
  either body outside the exact migration-389 source body. Migration 393 must
  revoke direct `EXECUTE` from `atlas` on all six definer helpers, and readiness
  must reject an admitted runtime role that regains direct execution on any of
  them; tests must prove the migration clears all six and the probe rejects then
  re-admits one runtime-role grant/revoke boundary. The controlled runner must
  derive the EOM runtime login from the configured EOM funnel DSN without
  exposing its credentials, install `pgcrypto` through the already-admitted DBA
  connection before an authorized 390-392 prelude can run, and bind that exact
  login into the migration session. Migration 393 must validate the configured
  login as an unprivileged, guard-isolated runtime role, revoke stale recovery
  table ACLs from it (and from a present legacy `atlas` role), remove every
  non-guard direct `EXECUTE` grant from the definer helpers, and grant only that
  configured login. It must attest and transfer every guard-critical recovery
  function--the CRM bridge, all append-only fences, scope validators, and the
  nested inbound-SMS helper--to the no-login guard before granting runtime
  access. The two scope validators must run as guard-owned `SECURITY DEFINER`
  functions with the fixed schema search path, so the restricted runtime can
  invoke its existing recovery-table triggers without a direct `contacts`
  grant. Readiness must reject a missing/incorrect owner or validator definer
  configuration on that exact guard-function set. Tests must prove a non-default
  configured runtime receives the allowed surface, owner/validator-authority
  drift rejects readiness, and `pgcrypto` setup precedes every selected prelude.
- Explicit non-scope: do not alter migration 389, the generic migration runner,
  the slim startup tuple, application connection-pool configuration, database
  schema, role memberships, service routes, delivery behavior, PostgreSQL CI
  image, or production state. Do not reapply 389 from the DBA runner and do not
  let the normal runtime execute 390-393.
- Assumption/blocker: migration 393 remains unrecorded until this source repair
  is merged and a DBA follows the runbook. The existing migration ledger and
  reconciliation selector remain authoritative for whether an exact historical
  prelude is permitted. The local workspace lacks the disposable PostgreSQL
  URL, so hosted CI remains the real-role proof for the SQL migration behavior.
- Verification plan: run the runner unit tests for no-op, prelude, and missing-
  prerequisite branches; run the focused recovery test file (with the
  disposable role test skipped locally when no test DSN is configured); compile
  changed Python files; run the plan sync and whitespace checks; then let hosted
  `eom-lead-pipeline` exercise the real PostgreSQL extension provisioning,
  bridge/fence tamper rejection, and controlled-runner test enrollment.

### Review Contract

- Acceptance criteria:
  1. The new migration is forward-only and DBA-gated; it keeps the existing
     no-login, membership-isolated guard model, transfers all six recovery
     tables plus every guard-critical bridge, fence, validator, and nested
     definer-chain helper to that guard, makes the two CRM-reading scope
     validators guard-owned `SECURITY DEFINER` functions with a fixed search
     path, clears stale table/column/function
     ACLs, and refuses to grant mutable evidence access unless the exact
     append-only trigger bindings and trusted migration-389 function bodies are
     intact. It also leaves neither NocoDB nor the configured runtime role with
     direct `EXECUTE` on the six definer helpers.
  2. The controlled DBA runner derives the EOM runtime login from the configured
     funnel DSN without logging its credentials, installs `pgcrypto` before an
     allowed historical prelude, and binds that login to migration 393. The
     migration fails closed if that role is absent, elevated, or has guard
     membership; it does not assume a literal `atlas` login exists.
  3. After applying the migration in a disposable schema, the configured runtime
     role has
     the exact required table privileges: read/insert/lock for receipts and
     attempts, read/insert for suppressions/events, and read/insert/update for
     sequences/steps. `tests/test_eom_missed_call_recovery.py` proves the real
     role can perform the service's locked read and mutations.
  4. `atlas_nocodb` cannot directly select, insert, update, or delete recovery
     tables, but an allowed CRM-table mutation still reaches the guarded trigger
     and terminalizes affected recovery work. The same disposable-Postgres test
     proves both sides.
  5. `missed_call_recovery_schema_ready` returns false for a structurally
     complete schema with missing or extra runtime ACLs, runtime guard
     membership, a disabled/rebound immutable-evidence trigger, or an untrusted
     NocoDB direct path; it returns true only after the migration establishes
     the expected privilege/guard properties, rejects any guard-function owner
     or scope-validator authority drift, and rejects either replaced
     receipt/attempt fence body. The
     integration test settles the effect through the real query.
  6. `EOM_MISSED_CALL_RECOVERY_READINESS_MIGRATIONS` excludes DBA-only and
     historical recovery SQL. The dedicated DBA runner defaults to a redacted,
     read-only preflight that reports the three historical-prelude receipts plus
     migration 389/393. With `--apply`, it first gives the existing selector
     only 390-392, then refuses to apply 393 until 389 is recorded; it never
     applies 389 and the normal runtime never executes the prelude. A fresh
     target first receives 389 from the slim EOM bootstrap path. Migration 393
     provisions `pgcrypto` only after DBA admission, rejects every altered
     bridge or mutable-evidence fence body before granting definer authority or
     runtime `UPDATE`, retries only a ledger-advancing historical-progress stop,
     and is enrolled with the controlled-runner proof in the EOM workflow. The
     corresponding source and disposable-Postgres tests settle both boundaries.
  7. No route, provider, delivery configuration, email copy, booking state, or
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

- Boundary path/seam: configured EOM funnel service role -> recovery tables;
  configured-runtime/NocoDB direct function execution -> guarded CRM-trigger
  functions; runtime-owned guard function -> immutable/scope/definer chain;
  startup probe -> worker admission; protected DBA DSN -> `pgcrypto` -> exact
  390-392 historical prelude -> recorded migration 389 -> recorded migration
  393 -> normal runtime startup.
- Replaced-path behaviors: previously a table's mere existence admitted the
  worker; after this change only structural presence plus usable runtime ACLs
  and guarded trigger properties admit it. Previously the DBA runner could not
  clear an exact required historical prelude before a missing 389 receipt, and
  393 trusted pre-existing bridge code; it now gives only the proven prelude
  set to the existing selector, requires the 389 receipt before 393, and
  rejects bridge and mutable-evidence fence bodies outside the migration-389
  allowlist before elevation or runtime `UPDATE`, and removes/rejects direct
  runtime execution of the definer helpers. The configured EOM funnel role now
  receives the exact ACL, every guard-critical function is guard-owned, and the
  DBA runner establishes `pgcrypto` before it can select a historical prelude.
  A committed historical prelude's expected integrity stop now continues only
  when the ledger proves that exact prelude advanced.
- Guard-relevant fields: table/column/function ACL, table owner, function owner,
  `prosecdef`, function `search_path`, trusted bridge/fence/validator/helper
  function-body SHA-256 or source body, trigger identity/enabled state, role
  login/inheritance/membership properties, the configured EOM funnel DSN user,
  the six recovery table names, installed `pgcrypto` extension, and the 390-393
  ledger receipts.
- Caller x input shape: the configured EOM funnel runtime executes its existing
  service SQL; NocoDB only performs its documented contacts/contact-interactions
  mutations. Neither caller receives a new public route or payload shape.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: recovery delivery remains disabled by current
  configuration; this slice neither enables it nor changes its defaults.
- Explicit value probe: with a structurally complete schema and correct ACLs,
  the real schema probe returns true.
- Absent value probe: removing a required runtime privilege makes the same probe
  return false before the worker can be started.
- Explicit migration-ledger probe: an exact historical state permits only its
  selected 390-392 prelude through the DBA runner; a recorded migration 389 and
  absent 393 receipt then lets the runner select 393.
- Absent migration-ledger probe: an absent ledger table rejects before the DBA
  runner can invoke the generic migration runner or create a ledger. An empty
  ledger, unrelated receipt, 393-only receipt, and mixed unrelated/393 receipt
  all reject before 393 can run.
- Default-session/default-context probe: the query uses `current_user`, so the
  integration proof connects as the runtime role rather than spoofing a role
  name in a superuser session; the DBA runner reads the target's ledger rather
  than trusting an environment flag for migration-389 completion.
- Side-effect ordering: readiness is a catalog/ACL read before worker admission;
  role-bound runtime mutations are proved only after the migration establishes
  grants in the disposable schema.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
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

The controlled runner reads the configured EOM funnel DSN only to derive its
username, never logs that DSN, and binds the role into each selected migration
connection. After the DBA admission and before any selected historical prelude,
it establishes `pgcrypto`. Migration 393 validates that the configured login is
an unprivileged, guard-isolated runtime and that `atlas_eom_handoff_owner`
remains a no-login, membership-isolated guard. Before it changes any bridge
function's authority or grants runtime `UPDATE`, it compares every guard-critical
bridge, fence, validator, and nested helper body against trusted migration-389
source. It then moves recovery tables and every such function under that guard,
  sets the six CRM-trigger bridge functions and two CRM-reading scope validators
  to `SECURITY DEFINER` with a fixed schema search path, clears public/NocoDB/runtime table and column ACLs plus
explicit NocoDB/runtime definer execution, and grants only the configured runtime
its proven SQL surface. It refuses to proceed unless the receipt/attempt
append-only triggers are still enabled, bound to their rejecting functions, and
retain their trusted bodies.

The service's existing structural probe becomes an admission predicate for that
same privilege/guard contract, so a partial, over-privileged, trigger-drifted,
or DBA-owned deployment fails closed before any provider worker can run. The
test applies the source migration to an empty disposable schema, connects as
each restricted role, and exercises both allowed runtime locking/writes and
denied direct NocoDB data access plus the guarded CRM trigger path. A fresh
target's slim EOM bootstrap records migration 389. Before the runner requires
that receipt, its `--apply` path gives only 390-392 to the existing attested
historical selector, so an exact legacy recovery can be completed under the DBA
connection without handing that authority to normal startup. When the generic
runner raises its expected post-commit integrity stop, the DBA command
continues only if its new ledger snapshot advanced an authorized prelude. It
never applies 389, then records 393 before a full generic Atlas runtime starts
or recovery is enabled. The slim readiness tuple intentionally carries only
ordinary schema prerequisites. The EOM workflow lists the migration,
controlled runner, and runner test in both trigger filters and runs the focused
runner test explicitly.

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
- Provision `pgcrypto` inside the DBA-gated forward migration rather than in
  the test fixture or normal runtime configuration, so a stock PostgreSQL
  target proves the same deployment prerequisite.

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

- `python -m pytest -q tests/test_eom_missed_call_privilege_runner.py` — `11
  passed` locally, including committed-prelude retry and non-prelude-progress
  rejection boundaries.
- `python -m pytest -q tests/test_eom_missed_call_recovery.py` — `14 passed,
  50 skipped` locally. The disposable-PostgreSQL role test is skipped because
  `ATLAS_MIGRATION_TEST_DATABASE_URL` is absent in this workspace; hosted CI
  remains the required real-role proof.
- `python -m pytest -q tests/test_eom_render_profile.py` — `64 passed` locally.
- `python -m compileall -q scripts/apply_eom_missed_call_recovery_runtime_privileges.py
  atlas_brain/services/eom_missed_call_recovery.py
  tests/test_eom_missed_call_privilege_runner.py
  tests/test_eom_missed_call_recovery.py` — passed locally.
- `git diff --check` and `python scripts/check_guard_class_closure.py --base
  origin/main --strict` — passed locally. The repository mechanical review
  wrapper remains required before publishing. No production database, role,
  migration, or service restart was touched by this source slice.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 7 |
| `atlas_brain/main_eom.py` | 11 |
| `atlas_brain/services/eom_missed_call_recovery.py` | 253 |
| `atlas_brain/storage/migrations/393_eom_missed_call_recovery_runtime_privileges.sql` | 753 |
| `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md` | 76 |
| `plans/PR-EOM-Missed-Call-Recovery-Runtime-Privileges.md` | 409 |
| `scripts/apply_eom_missed_call_recovery_runtime_privileges.py` | 356 |
| `tests/test_eom_missed_call_privilege_runner.py` | 492 |
| `tests/test_eom_missed_call_recovery.py` | 1187 |
| `tests/test_eom_render_profile.py` | 7 |
| **Total** | **3551** |

## Diff size rationale

The source/test diff exceeds the 400-LOC soft budget because the indivisible
security repair requires one forward migration, an executable readiness
predicate, a controlled DBA-only runner, and a real-role disposable-Postgres
proof. No provider, route, configuration, UI, or historical-evidence behavior
is included.
