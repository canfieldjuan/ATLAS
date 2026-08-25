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
Max files: 13

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

### Follow-up repair contract (current review round)

- Root cause: the controlled runner checks only that its executor is a
  superuser before calling `_ensure_pgcrypto()`. The authoritative role
  admissions live later in migration 393, so an invalid guard, runtime, or
  NocoDB role can cause a permanent extension write before the migration
  rejects. Separately, the body-tamper tests configure `atlas` as the runtime
  role even though the disposable CI service makes `atlas` a PostgreSQL
  superuser; migration 393 therefore stops at role admission before either
  trusted-body assertion is exercised.
- Correct fix must touch/change: add a read-only runner preflight that mirrors
  migration 393's guard/runtime/NocoDB role admissions before `pgcrypto` or an
  authorized historical prelude can run, with unit proof for each rejected
  admission and the valid ordering. Change only the disposable tamper-test
  setup to bind a unique `LOGIN NOINHERIT` probe role and clean it up, so each
  case reaches its expected bridge/fence body rejection.
- Must not change: migration 393 remains the authoritative, in-transaction
  role-enforcement point; do not change its SQL, the PostgreSQL CI service's
  `atlas` superuser identity, role memberships, normal startup migration tuple,
  historical selector, runtime connection configuration, routes, providers,
  delivery behavior, billing, or production state.
- Acceptance criteria: an invalid guard, runtime, or NocoDB admission rejects
  before extension creation and before any migration runner call; the valid
  path still establishes `pgcrypto` before an allowed historical prelude; and
  every bridge/fence tamper case fails for its trusted-body message rather than
  for a superuser-runtime admission failure.

### Hosted-PostgreSQL repair contract (current review round)

- Root cause: migration 393 revokes table and column privileges from the
  legacy `atlas` login before it transfers a recovery relation to the
  no-login guard. On a fresh disposable database, `atlas` owns migration-389
  objects. PostgreSQL represents that owner-targeted revoke as an explicit
  empty ACL item; `ALTER TABLE ... OWNER TO atlas_eom_handoff_owner` carries
  it to the new owner. The runtime still receives its cataloged direct ACL,
  but the guard-owned `SECURITY DEFINER`
  `validate_eom_missed_call_sequence_scope()` cannot read
  `eom_missed_call_attempts`, so an otherwise allowed sequence insert fails.
  Separately, the tamper helpers use `CREATE OR REPLACE FUNCTION` definitions
  that omit existing argument names. PostgreSQL treats that as an attempted
  parameter rename and rejects before migration 393 can prove its trusted-body
  boundary.
- Correct fix must touch/change: in migration 393, transfer each recovery
  table to the guard before revoking direct table or column ACLs from `PUBLIC`,
  NocoDB, legacy `atlas`, or the configured runtime. Keep those revokes and
  the exact runtime allowlist in the same transaction, after ownership has
  changed. Update the disposable PostgreSQL proof to assert that the guard has
  effective access to the recovery relation it reads and to exercise the real
  runtime insert path. Update only the tamper replacement definitions so every
  existing named input parameter is preserved while its body is changed.
- Must not change: do not alter migration 389, role admissions, guard-role
  flags or memberships, the configured-runtime allowlist, the normal startup
  tuple, generic migration runner, service/readiness implementation, routes,
  providers, billing, production state, or the PostgreSQL CI identity. Do not
  add broad runtime CRM or guard membership grants.
- Acceptance criteria: applying 393 where `atlas` owns the migration-389
  relations leaves the guard able to execute its trusted validator and the
  configured unprivileged runtime able to create an allowed sequence; the
  runtime and NocoDB direct-denial checks remain intact. Each bridge/support
  tamper fixture reaches migration 393's trusted-body failure rather than a
  PostgreSQL argument-name error.

### Hosted-PostgreSQL test-fixture correction contract (current review round)

- Root cause: the NocoDB integration proof creates an estimate-request
  interaction whose `submitted_email` is intentionally the effective recovery
  recipient. It then changes only `contacts.email` and expects a
  `recipient_changed` cancellation. Migration 389 deliberately preserves that
  submitted recipient across an ordinary canonical-email edit, so the update
  correctly leaves the sequence active and the test asserts behavior the
  product does not implement.
- Correct fix must touch/change: change only that disposable NocoDB fixture to
  grant and update `lead_stage`, then assert its existing `lead_advanced`
  cancellation branch. This still proves that an allowed NocoDB CRM mutation
  reaches the guard-owned trigger while NocoDB retains no direct recovery-table
  access.
- Must not change: do not change migration 389's effective-recipient
  precedence, recipient-change behavior, CRM trigger definitions, runtime or
  NocoDB production privileges, service logic, routes, provider behavior, or
  data.
- Acceptance criteria: the disposable real-role test proves a permitted NocoDB
  contact update terminalizes an active sequence with `lead_advanced`; its
  direct recovery-table reads remain denied, and the separate interaction path
  remains covered.

### Recovery ACL allowlist repair contract (current review round)

- Root cause: migration 393 clears direct recovery-table and column grants
  only from `PUBLIC`, NocoDB, the literal legacy `atlas` role, and the current
  runtime role. PostgreSQL preserves grants to every other login or group role
  across ownership transfer. Readiness separately checks only the current
  runtime and NocoDB, so an inherited group grant or an old runtime's direct
  grant can keep reading or mutating recovery evidence while the worker still
  reports the schema ready.
- Correct fix must touch/change: rebuild each recovery table's direct ACL from
  the catalog after ownership transfer: make the no-login guard's effective
  table authority explicit, revoke every other direct table grantee and every
  direct column grantee, then re-grant only the existing configured runtime
  allowlist. Extend readiness to fail closed on any direct table ACL grantee
  other than the guard/current runtime or on any direct column ACL.
  Extend the disposable PostgreSQL proof with a unique stale login and inherited
  group role: it must prove migration cleanup removes pre-existing table and
  column grants, readiness rejects post-migration grants, and revocation
  re-admits the schema.
- Must not change: do not change the guard or runtime role flags/memberships,
  the existing runtime privilege matrix, NocoDB's CRM permissions, definer
  functions, migration 389, generic migration runner, normal startup tuple,
  routes, providers, billing, production grants, or PostgreSQL superuser
  semantics. Superuser access remains PostgreSQL's DBA boundary, not an ACL
  allowlist exception to be silently widened.
- Acceptance criteria: a pre-existing stale login/group table or column grant
  cannot survive migration 393; a newly granted stale login/group table or
  column privilege makes `missed_call_recovery_schema_ready` return false;
  revoking it restores readiness; and the existing exact runtime/NocoDB
  allowed/denied real-role paths remain green.

### Definer function-resolution repair contract (current review round)

- Root cause: migration 393 deliberately retains the application schema in its
  guard-owned `SECURITY DEFINER` functions' search paths so the existing CRM and
  recovery relations remain reachable. Migration 354 and its supported
  deployment shape can leave that schema owned or `CREATE`-writable by the
  normal runtime. Three guarded callbacks then call user-defined helpers with
  non-exact `VARCHAR` or unknown-literal arguments, so a runtime-created
  overload can win PostgreSQL function resolution and execute in the guard's
  definer context.
- Correct fix must touch/change: after attesting the trusted migration-389
  functions and before marking them `SECURITY DEFINER`, replace only the three
  callbacks that make user-defined calls. Every such call must be schema
  qualified and its arguments explicitly cast to the trusted signature. Extend
  the disposable real-role proof by granting the configured runtime schema
  `CREATE`, installing competing overloads before migration 393, and proving a
  permitted NocoDB contact update neither invokes them nor loses its normal
  guarded terminalization behavior.
- Must not change: do not transfer the application schema's ownership, revoke
  the runtime's general schema `CREATE` capability, change migration 354,
  change table/function role membership or the direct runtime ACL matrix,
  introduce a new schema/role/table, alter migration 389, alter generic startup
  or migration-runner behavior, or change CRM/NocoDB APIs, routes, providers,
  billing, or production deployment data.
- Acceptance criteria: a malicious `UUID, VARCHAR` effective-recipient
  overload and a `UUID, TEXT, TEXT` cancellation overload remain present and
  callable by the runtime, but cannot run during the NocoDB-triggered guarded
  contact-update path; the legitimate sequence still terminalizes as
  `recipient_changed` and the existing runtime/NocoDB proof remains green.

### Current-head integration contract (current review round)

- Root cause: this branch and current `origin/main` independently extend the
  manually enumerated EOM lead-pipeline workflow. The missed-call slice adds
  migration 393 and its controlled-runner test; the merged first-clean slice
  adds its service, migration, database service, and tests. The independently
  correct list edits overlap, so Git cannot select a complete workflow without
  an explicit union.
- Correct fix must touch/change: retain both slices' workflow paths and test
  commands in each `pull_request` and `push` registration, while retaining the
  first-clean PostgreSQL service/configuration. Preserve the missed-call
  startup tuple that excludes historical/DBA-only migrations and the upstream
  first-clean app-state pool hook; neither behavior is a conflict to rewrite.
- Must not change: do not change recovery privilege policy, migration 393,
  first-clean implementation, service routes, database credentials, tests,
  Docker configuration, or any production data. The integration may edit only
  the already-scoped workflow and plan; `atlas_brain/main_eom.py` must merge
  its independent changes without a manual behavioral edit.
- Acceptance criteria: Git can merge the branch with current `origin/main`;
  the resulting workflow contains both the missed-call migration/runner paths
  and first-clean paths/tests/services, while `main_eom` retains both the
  historical/DBA-only recovery exclusion and the first-clean pool hook.

### Delegated-login admission repair contract (current review round)

- Root cause: the intended exact runtime/NocoDB identity boundary is an
  authorization-graph invariant, but migration 393 and the controlled DBA
  runner inspect only memberships granted *to* the configured runtime or
  `atlas_nocodb`. They do not reject a separate non-superuser login that can
  assume either admitted login through a direct or transitive membership.
  PostgreSQL's `pg_has_role(actor, target, 'MEMBER')` reports that ability, and
  the admitted actor can then `SET ROLE` to the target. The service readiness
  predicate checks table and guard state but does not attest either admitted
  login against that incoming delegation path.
- Correct fix must touch/change: before migration 393 creates `pgcrypto` or
  changes an ACL, and before the runner creates `pgcrypto` or runs a historical
  prelude, reject any non-superuser `LOGIN` other than the target role for which
  `pg_has_role(login, target, 'MEMBER')` is true. Apply the same admission
  boundary to the runtime worker's schema-readiness predicate for both the
  current runtime identity and `atlas_nocodb`. Extend disposable-PostgreSQL
  proof to cover direct and transitive delegated-login paths for both admitted
  identities, including failed migration/readiness admission followed by
  revocation and successful re-admission. Exercise the controlled runner's
  read-only admission query against the same real role graph.
- Must not change: do not grant, revoke, alter, or otherwise repair production
  role memberships; the code must only refuse an unsafe graph. Do not change
  the existing direct table/function ACL matrix, guard role flags, migration
  389, generic migration runner, historical selector, normal startup tuple,
  CRM routes/APIs, providers, billing, first-clean behavior, database schema,
  or production data. Preserve the existing superuser DBA boundary: the new
  delegated-login scan excludes superusers rather than treating DBA authority as
  an ordinary runtime admission path.
- Assumptions: a direct or transitive non-superuser `LOGIN` membership that
  PostgreSQL reports through `pg_has_role(..., 'MEMBER')` is an assumable target
  identity; the disposable database probes establish that behavior. The target
  role itself is excluded because PostgreSQL reports a role as a member of
  itself.
- Acceptance criteria: migration 393 and the DBA runner reject direct and
  transitive non-superuser login delegation to either admitted identity before
  mutable work; worker readiness returns false for the same post-migration
  graph and true again after revocation. The configured runtime and NocoDB
  retain their prior allowed/denied operations when no unsafe delegation
  exists.

### Controlled-runner config and target-attestation contract (current review round)

- Root cause: the controlled runner accepts independent protected-DBA and EOM
  runtime DSNs, but it reads them directly from `os.environ` and uses the
  runtime DSN only to parse a username. It consequently bypasses the supported
  `.env`/`.env.local` settings boundary and has no evidence that the DBA pool
  mutates the runtime database, cluster, or schema. A valid runtime DSN can
  therefore point at a different target from the DBA DSN while `--apply`
  reports the wrong target repaired.
- Correct fix must touch/change: define a dedicated typed `SecretStr` DBA
  setting under the canonical `ATLAS_EOM_MISSED_CALL_RECOVERY_*` namespace and
  reuse `EOMFunnelConfig` for the runtime DSN. Before any extension or
  migration mutation, connect to the runtime target read-only, record its
  schema/database/session identity, bind the DBA pool to that schema, and
  require both pools to contend on one transaction-scoped random advisory lock.
  Re-attest the DBA pool's schema/database identity before and after controlled
  migration work. Extend the DBA-runner tests to prove valid same-target
  execution, typed `.env` loading, mismatched runtime/DBA database or schema
  rejection, and same-name clone rejection before any `pgcrypto` or migration
  call.
- Must not change: do not change migration 393 SQL, recovery-table/function
  ACL policy, the ordinary EOM runtime tuple, generic migration runner,
  first-clean runner, route/provider/billing behavior, existing production
  environment values, database contents, or role grants. Do not make the normal
  application runtime load or require the protected DBA secret. Preserve the
  runner's `--apply`/`--json` behavior and redacted target result; only remove
  the newly introduced arbitrary environment-variable indirection in favor of
  the canonical typed configuration.
- Assumptions: the controlled DBA target must be the runtime target's exact
  current schema and database identity, and an advisory-lock contention probe
  distinguishes a live shared PostgreSQL cluster from a same-named clone. The
  runtime DSN must establish its direct configured login; the existing funnel
  setting is typed and `.env` aware but intentionally remains a plain string.
- Acceptance criteria: deployment values supplied through the supported typed
  settings files are accepted without exposing a DSN; a wrong database, schema,
  cluster, or runtime session identity is rejected before any mutable call;
  matching runtime/DBA targets retain the existing preflight/apply path and
  redacted output.

### Contract revision: transaction-pooled target attestation

- New evidence: the adjacent controlled first-clean DBA runner disables
  asyncpg's prepared-statement cache on both pools because acquire/release
  boundaries can cross PostgreSQL backends under transaction pooling
  (`scripts/apply_eom_first_clean_completion_schema.py`). The missed-call
  runner now holds a transaction-scoped advisory lock across two controlled
  pools, so the same backend-switching risk applies to its new attestation
  queries.
- Revised required change surface: disable asyncpg's statement cache in the
  controlled missed-call runtime and DBA pools while preserving their one-slot
  pool bounds; bind only the DBA pool's validated runtime schema through startup
  settings.
- Explicit non-scope remains unchanged: this does not change the normal Atlas
  application pool or any generic migration-pool configuration.

### Contract revision: executable-runner test import

- New evidence: `scripts/` is not an importable Python package in this
  repository, while the disposable-role test needs the runner's admission
  helper as a direct unit under test.
- Revised required test surface: load that executable file through an explicit
  `importlib` file-spec fixture in the test instead of relying on package
  import discovery.
- Explicit non-scope remains unchanged: do not turn `scripts/` into an import
  package, alter Python packaging, or change the command's CLI behavior merely
  to make its internal helper importable.

### Controlled-migration reservation and backend-pinning contract (current review round)

- Root cause: migration 393 is a DBA-only migration, but the generic catalog
  reserves only migration 394. An unrestricted `run_migrations(...)` therefore
  selects 393 and reaches its superuser guard during ordinary startup instead
  of leaving it for the controlled DBA command. Separately, the controlled
  command writes the runtime-role setting through a pool wrapper, then lets the
  generic migration runner acquire from that pool again. Under transaction
  pooling those two acquisitions can use different PostgreSQL backends, so the
  migration can lose the session setting it requires even though the command
  preflighted the correct target.
- Correct fix must touch/change: add migration 393 to the existing controlled
  DBA catalog reservation, without changing the generic runner's selection or
  locking algorithm. Extend its existing runner test so an unrestricted run
  skips both controlled migrations and explicit selection reaches each one.
  In the missed-call controlled command, run only migration 393 through one
  explicit transaction-pinned connection, acquire and release the canonical
  migration serialization lock around that invocation, and bind the runtime
  role setting through that same pinned connection. Extend the focused command
  tests to prove a contention retry waits outside the transaction and that the
  setting, generic-runner invocation, lock, migration, and bookkeeping share
  the one pinned connection.
- Must not change: do not edit migration 393 SQL, historical migrations
  390--392, the normal generic migration algorithm, the first-clean controlled
  runner, the slim startup tuple, role/ACL policy, API or UI behavior, workflow
  configuration, production data, or the command's public CLI/result shape.
  Historical prelude selection remains limited to 390--392; only 393 receives
  the new transaction-pinned execution boundary.
- Acceptance criteria: normal generic startup never selects an unrecorded 393
  or 394; each remains selectable only through an explicit `only` request. A
  controlled 393 run retries a busy canonical lock without an open transaction,
  then runs the real migration adapter inside one transaction on the exact
  acquired DBA connection with the runtime-role setting installed there.

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
     membership, or if a separate non-superuser login can assume either
     admitted runtime/NocoDB identity; it does not assume a literal `atlas`
     login exists.
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
     membership, incoming delegation to either admitted login, a
     disabled/rebound immutable-evidence trigger, or an untrusted NocoDB direct
     path; it returns true only after the migration establishes the expected
     privilege/guard properties, rejects any guard-function owner or
     scope-validator authority drift, and rejects either replaced
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
- `atlas_brain/config.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_missed_call_recovery.py`
- `atlas_brain/storage/migrations/__init__.py`
- `atlas_brain/storage/migrations/393_eom_missed_call_recovery_runtime_privileges.sql`
- `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md`
- `plans/PR-EOM-Missed-Call-Recovery-Runtime-Privileges.md`
- `scripts/apply_eom_missed_call_recovery_runtime_privileges.py`
- `tests/test_eom_missed_call_privilege_runner.py`
- `tests/test_eom_missed_call_recovery.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migrations_runner.py`

## Mechanism

The generic catalog reserves migrations 393 and 394 from ordinary startup; a
dedicated command must select either one explicitly. The controlled runner reads
the configured EOM funnel DSN only to derive its username, never logs that DSN,
and binds the role into each selected migration connection. For atomic migration
393, it holds the canonical migration lock and pins the generic runner plus that
role setting to one explicit transaction connection, so a transaction-pooling
proxy cannot split the migration from its required session setting. After the
DBA admission and before any selected historical prelude, it establishes
`pgcrypto`. Migration 393 validates that the configured login is
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

- `pytest tests/test_eom_missed_call_privilege_runner.py -q` — `28 passed`
  locally, including typed `.env`/`.env.local` loading, exact target/schema
  binding, direct-runtime-session rejection, database/OID mismatch rejection,
  and same-name clone rejection before any mutable call.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=... python -m pytest -q
  tests/test_eom_missed_call_recovery.py` — `72 passed` locally against the
  disposable PostgreSQL database. The new proof creates a non-superuser
  `LOGIN NOINHERIT` actor and a `NOLOGIN NOINHERIT` intermediary, confirms
  direct and transitive `pg_has_role(..., 'MEMBER')` plus actual `SET ROLE`,
  then proves migration/runner rejection before privilege mutation and
  readiness rejection/re-admission for both admitted identities.
- `pytest tests/test_eom_render_profile.py -q` — `64 passed, 1 warning`
  locally.
- `python -m compileall -q scripts/apply_eom_missed_call_recovery_runtime_privileges.py
  atlas_brain/config.py
  tests/test_eom_missed_call_privilege_runner.py
  tests/test_eom_missed_call_recovery.py` — passed locally.
- `ruff check` for the changed Python sources and `ruff format --check` for
  the runner/test sources — passed locally. `atlas_brain/config.py` retains
  its existing repository formatting: a whole-file format check would rewrite
  unrelated baseline lines, so that churn was deliberately not applied.
- `git diff --check` and `python scripts/check_guard_class_closure.py --base
  origin/main --strict`, plus `python scripts/audit_plan_doc.py` — passed
  locally. The repository mechanical review wrapper remains required before
  publishing. No production database, role, migration, or service restart was
  touched by this source slice.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 7 |
| `atlas_brain/config.py` | 23 |
| `atlas_brain/main_eom.py` | 11 |
| `atlas_brain/services/eom_missed_call_recovery.py` | 344 |
| `atlas_brain/storage/migrations/__init__.py` | 5 |
| `atlas_brain/storage/migrations/393_eom_missed_call_recovery_runtime_privileges.sql` | 988 |
| `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md` | 81 |
| `plans/PR-EOM-Missed-Call-Recovery-Runtime-Privileges.md` | 744 |
| `scripts/apply_eom_missed_call_recovery_runtime_privileges.py` | 708 |
| `tests/test_eom_missed_call_privilege_runner.py` | 1024 |
| `tests/test_eom_missed_call_recovery.py` | 1655 |
| `tests/test_eom_render_profile.py` | 7 |
| `tests/test_migrations_runner.py` | 41 |
| **Total** | **5638** |

## Diff size rationale

The source/test diff exceeds the 400-LOC soft budget because the indivisible
security repair requires one forward migration, an executable readiness
predicate, a controlled DBA-only runner, and a real-role disposable-Postgres
proof. No provider, route, configuration, UI, or historical-evidence behavior
is included.
