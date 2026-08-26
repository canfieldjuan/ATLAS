# PR-Postgres-Role-Topology-Preflight

## Why this slice exists

The operator asked to begin the deferred least-privilege database-role work
from `PR-Postgres-Loopback-Scram-Hardening`. That prior slice deliberately did
not alter role topology. Current source still routes the application pool,
generic migration runner, and fixed operations inspection through the same
`DatabaseConfig` connection path. A direct ownership/privilege cutover without
an independently target-attested catalog receipt could alter the wrong database
or break the EOM guard-owned namespace.

This is a production-hardening prerequisite, not the role cutover itself. It
adds a controlled, read-only evidence command that compares the configured
runtime target with a separately configured DBA connection and emits only the
catalog facts a later role-design slice needs. The currently deployed
`atlas-api.service` is behind current `origin/main`, so this slice must not
change production roles, ownership, credentials, or startup behavior.

This slice intentionally exceeds the 400-LOC soft cap. Its typed privileged
configuration, two-target admission checks, fixed catalog projection,
fail-closed boundary fixtures, disposable PostgreSQL 16 integration proof, and
operator procedure are one atomic safety surface: splitting them would either
publish an unverified privileged reader or leave the operator without a usable,
documented receipt. It does not carry the later role/grant/ownership cutover,
which remains a separate slice.

### Problem-derived contract

- Root cause: Atlas has no independent, target-attested role-topology
  inspection path. `DatabasePool` initializes with
  `DatabaseConfig.connection_kwargs()`, the application lifespan passes that
  pool to the generic migration runner, and `./ops db inspect` opens a
  read-only connection from the same `DatabaseConfig`. As a result, source
  alone cannot establish the live role graph required before separating runtime
  DML, migration DDL, and inspection authority. A role/ownership change made
  from that ambiguity could invalidate the existing EOM no-login guard boundary.
- Correct fix must touch/change:
  1. Add a typed, secret-redacted DBA-only configuration boundary that is
     instantiated only by a new preflight command, never by normal Atlas
     startup.
  2. Add a fixed-query, read-only preflight that opens the normal runtime
     target and the DBA target, proves that both reach the same PostgreSQL
     database/cluster, rejects an insufficient DBA identity, and reports the
     role, membership, database-owner, schema-owner, and per-object ownership
     and ACL facts needed for a later cutover without customer rows or
     credentials.
  3. Add focused unit coverage for successful target attestation and each
     fail-closed admission path; prove every connection enters a read-only
     transaction and no apply/mutation path exists.
  4. Document the command as evidence-only in the database runbook, including
     the prerequisite that the live runtime revision must be converged before
     any later role cutover.
- Must not change:
  1. Do not create, alter, grant, revoke, drop, or assume PostgreSQL roles;
     do not change database/schema/object ownership or ACLs.
  2. Do not change `DatabaseConfig`, `DatabasePool`, generic migration startup,
     EOM funnel connection behavior, application role name `atlas`, the
     existing `atlas_eom_handoff_owner` guard, or NocoDB privileges.
  3. Do not add a migration, service restart, deployment action, runtime
     credential, user-facing behavior, API, schema, dependency, or historical
     maintenance-script refactor.
  4. Do not expose a generic SQL runner, print DSNs/passwords, or add a
     production role change disguised as a preflight.

### Contract revision — current-head review findings

New evidence:

- The fixed membership projection records only `admin_option`, while PostgreSQL
  16 stores distinct `inherit_option` and `set_option` authority on each
  membership edge.
- Runtime identity, DBA identity, lock attestation, and the catalog projection
  currently use separate acquired transactions. A receipt needs one pinned
  runtime connection and one pinned DBA connection, with the DBA catalog facts
  taken from the same stable snapshot that passed attestation.
- Relation/function owner and ACL records, column grants, and RLS-policy roles
  are aggregated into counts, losing the object identity required to design a
  least-privilege cutover.
- The focused unit double checks query routing but does not execute the catalog
  SQL through PostgreSQL 16.

Revised required change surface:

1. Record each PostgreSQL 16 membership row's grantor and `admin_option`,
   `inherit_option`, and `set_option` fields.
2. Pin target identity, shared-lock attestation, and all catalog reads to the
   same acquired runtime/DBA connections, in read-only repeatable-read
   transactions, before rendering a receipt.
3. Replace lossy object-level counts with stable relation, function, column,
   and policy identities alongside their owner, ACL, and RLS metadata.
4. Add a disposable PostgreSQL 16 integration test that seeds only test roles,
   objects, grants, and a policy; invokes the real command through asyncpg; and
   is enrolled in its PostgreSQL-backed GitHub workflow.

Revised non-scope:

- The command remains evidence-only. No production database action is added.
  The integration fixture may create and tear down only uniquely named roles,
  schema objects, grants, and policies inside the disposable GitHub PostgreSQL
  16 service; it must never receive a production or operator DSN.
- Normal startup, `DatabaseConfig`, `DatabasePool`, migrations, EOM funnel
  behavior, APIs, and role topology remain outside this revision.

### Contract revision — Python 3.11 test-collection compatibility

New evidence:

- The disposable-test identifier helper embeds escaped quote literals in an
  f-string expression. GitHub's pinned Python 3.11 parser rejects that form
  during test collection, before the PostgreSQL integration test can run.

Root cause:

- The test-only identifier helper relies on newer f-string grammar rather than
  the Python 3.11 grammar the enrolled workflow runs.

Required change surface:

1. Replace only the helper's f-string expression with Python-3.11-compatible
   string concatenation while retaining doubled-double-quote identifier
   escaping.
2. Add a direct focused assertion for ordinary and embedded-double-quote
   identifiers, and run the focused file through the available Python 3.11
   interpreter before push.

Must not change:

- Do not change the standalone preflight command, its fixed catalog SQL, the
  disposable database fixture's role/ACL/policy topology, workflow service
  topology, runtime configuration, or any production behavior.

### Contract revision — ACL grantor provenance and disposable cleanup

New evidence:

- Every ACL projection emits its grantee and privilege but omits
  `aclexplode(...).grantor`, so the receipt cannot distinguish a direct owner
  grant from a delegated grant that disappears when an intermediate grantor is
  revoked.
- The disposable integration fixture grants the temporary runtime role
  `USAGE` on `public`, but does not revoke that persistent schema ACL before
  it drops the role.

Root cause:

- The role-topology receipt preserves object identity but not the provenance of
  ACL authority, and the test fixture leaves one cross-schema dependency owned
  by its temporary role.

Required change surface:

1. Add stable grantor OID and role-name fields to database, schema, relation,
   function, column, and default-ACL records, and include those identities in
   deterministic ordering.
2. Extend the unit fixture and source-closure assertion for every ACL
   projection; seed one delegated relation grant in the disposable PostgreSQL
   16 test and assert its grantor OID/name in the real receipt.
3. Reset any switched test role and revoke the temporary runtime role's
   `public` schema `USAGE` before dropping it, so the integration fixture
   proves cleanup rather than retaining a privilege dependency.

Must not change:

- Do not make the standalone command mutate a database, alter production roles
  or privileges, add a migration, change normal runtime configuration, or
  broaden the disposable test target beyond its loopback-only guard.

### Contract revision — authoritative privilege semantics and independent DBA identity

Root cause:

- The fixed receipt records named RLS policies and direct membership grants, but
  omits the policy expressions and view-invoker state that route a privilege
  decision through an authenticated role. It also reports a direct grant to a
  predefined PostgreSQL role while dropping the predefined-role memberships
  through which that role inherits authority. The result is a topology snapshot
  that can understate the effective authority a later cutover must preserve.
- The admission boundary proves that the two connections reach one target and
  that the DBA session is a direct superuser, but it does not reject two
  sessions authenticated as the same PostgreSQL principal. That contradicts the
  command's separate DBA-identity requirement and lets a copied application
  superuser configuration produce an apparently independently attested receipt.

Required change surface:

1. Extend the fixed relation and RLS-policy projections with the catalog fields
   that control execution identity: security-invoker view state plus deparsed
   `USING` and `WITH CHECK` policy expressions. Preserve the fixed-query,
   JSON-scalar, redacted receipt boundary.
2. Replace the one-hop `pg_` membership filter with a fixed recursive closure
   from every reported non-system role through its granted roles. Emit the
   closure's referenced role records and membership edges, including reachable
   predefined roles, in deterministic order.
3. Reject a runtime and DBA session that share either authenticated or effective
   PostgreSQL identity before the shared-lock probe or any catalog read.
4. Extend the controlled unit doubles and the disposable PostgreSQL 16 test to
   prove all three conditions: a role-sensitive policy plus a
   security-invoker view appear in the real receipt, a predefined-role chain is
   retained, and same-principal admission fails before catalog access. The test
   fixture must revoke every added predefined-role membership before dropping
   its temporary runtime role.
5. Update the database runbook's command contract to state the new fail-closed
   same-principal check and the added evidence classes.

Must not change:

- Do not create, alter, grant, revoke, drop, or assume a production PostgreSQL
  role; the standalone command remains read-only and has no mutation path.
- Do not change `DatabaseConfig`, `DatabasePool`, application startup,
  migrations, EOM behavior, existing role names, production credentials, or
  deployment state.
- Do not expose policy business rows, DSNs, passwords, generic SQL input, or a
  role-cutover mechanism. The disposable PostgreSQL 16 fixture remains the
  only test code allowed to create and clean up uniquely named test topology.

### Contract revision — process-environment-only DBA credential

Root cause:

- The topology command invokes `DatabaseRoleTopologyDBAConfig()` with its
  default settings source. That settings class currently includes the shared
  `.env` and `.env.local` files, so an absent process environment variable can
  silently resolve to a stale privileged DBA DSN from a worktree file. The
  command's missing-configuration guard therefore does not actually mean that
  the operator supplied the protected credential for this invocation.

Required change surface:

1. Make `DatabaseRoleTopologyDBAConfig` read its privileged DSN from the
   process environment only; it must not read either shared dotenv file while
   preserving the existing environment-variable name, `SecretStr` redaction,
   and empty-value fail-closed behavior.
2. Add default-constructor and command-entry tests from a temporary working
   directory containing both dotenv files. With the process variable absent,
   both tests must prove the key is ignored and the command fails before it
   opens either database pool.
3. State the process-environment-only requirement in the database runbook so
   operators do not store the DBA credential in the worktree dotenv files.

Must not change:

- Do not change `ENV_FILES`, `DatabaseConfig`, any other settings class, normal
  application startup, the database URL environment-variable name, or the
  standalone command's fixed-query/read-only database behavior.
- Do not read, write, migrate, or rotate a production credential as part of
  this change. Test-only dotenv files may contain only a synthetic DSN.

## Scope (this PR)

Ownership lane: eom-crm/runtime-security
Slice phase: Production hardening
Max files: 6

1. Add a fail-closed, read-only catalog receipt that establishes the live
   PostgreSQL role/ownership topology before any authority separation is
   attempted.
2. Add a standalone `scripts/check_database_role_topology.py` command that is
   evidence-only and defaults to a redacted JSON receipt.
3. Add a typed `DatabaseRoleTopologyDBAConfig` used only by that command.
4. Add focused fixture tests and a database-runbook entry for the command.

### Review Contract

- Acceptance criteria:
  1. With an explicit valid DBA DSN and matching runtime/DBA identity, the
     command returns a redacted receipt containing only fixed catalog fields;
     settled by `tests/test_database_role_topology_preflight.py`.
  2. With the DBA DSN absent, a runtime/DBA database or cluster mismatch, a
     reused runtime/DBA authenticated or effective identity, or a non-superuser
     DBA session, the command raises before any catalog report; settled by
     focused unit tests in the same file.
  3. Both connections execute inside read-only transactions and the command has
     no `--apply`/mutation branch; settled by its controlled test doubles and
     direct inspection of the command's connection flow.
  4. Normal `DatabaseConfig` startup/migration behavior is unchanged; settled
     by unchanged `atlas_brain/storage/config.py`,
     `atlas_brain/storage/database.py`, and `atlas_brain/main.py` plus the
     focused regression suite.
- Reachability proof: `python scripts/check_database_role_topology.py` is the
  real operator entrypoint. Its observable result is a redacted, fixed-shape
  JSON evidence receipt or a fail-closed error before catalog reporting.
- Affected surfaces: typed DBA-only config, the new preflight script, focused
  test fixtures, and the database runbook. No runtime service caller is added.
- Risk areas: credential redaction, target confusion, runtime/DBA identity
  reuse, cross-cluster false attestation, insufficient inspection authority,
  read-only enforcement, current production/runtime drift, and accidental
  role/DDL mutation.
- Reviewer rules triggered: R1, R2, R3, R4, R6, R8, R10, R11, R12, R13, R14.

### Boundary-change enumeration

The preflight is an admission boundary for a privileged catalog receipt.

- Boundary path/seam: `scripts/check_database_role_topology.py` admits the
  runtime target and configured DBA target before it reads catalog facts.
- Replaced-path behaviors: no prior role-topology command exists; generic
  `./ops db inspect` remains restricted to connectivity/migration-count
  queries and is not widened.
- Guard-relevant fields: runtime database name/OID/current schema/current and
  session role; DBA database name/OID/current schema/current and session role;
  advisory-lock contention; DBA superuser status.
- Caller x input shape: absent DBA configuration -> reject before connect;
  explicit DBA DSN + matching independent runtime target -> fixed read-only
  report; explicit DBA DSN + target mismatch, reused identity, or non-superuser
  session -> reject before report.

### Deployed-config probing

Required because this slice adds a DBA-only configuration boundary.

- Deployed/default config values: the normal runtime keeps its current
  `DatabaseConfig` selection. The new DBA DSN is absent by default and the
  command must fail closed when absent.
- Explicit value probe: a test injects a valid typed DBA DSN and proves the
  command uses it only for the dedicated DBA pool.
- Absent value probe: a test proves missing/blank DBA DSN prevents pool creation.
- Default-session/default-context probe: tests distinguish a direct runtime
  session from a `SET ROLE`/mismatched session, reject a reused runtime/DBA
  authenticated or effective identity, and prove target attestation uses both
  database identity and one advisory-lock namespace.
- Side-effect ordering: all identity/config admissions precede catalog report
  construction; every catalog query runs inside `readonly=True` transactions;
  the command contains no mutation path.

### Catalog-projection closure

- The catalog row set is **OPEN**: roles, membership edges, non-system schemas,
  objects, and ACL entries can change after this PR. Their membership is
  **DERIVED** on every command invocation from fixed `pg_catalog` queries; the
  command does not maintain a copied role or object inventory.
- The projection classes are **CLOSED for this slice**: target identity, role
  attributes, each reported non-system role's reachable membership closure
  (including referenced predefined-role records), current-database
  ownership/ACL, non-system schema ownership/ACL, per-object relation and
  function ownership/effective ACL records with row-security,
  security-invoker, and security-definer flags, explicit column-ACL and
  row-security-policy role/expression records, and explicit default-ACL
  overrides. Their canonical definition is this problem-derived contract and
  the static query constants in the command.
- Catalog classes outside that closed projection, plus `pg_*` and
  `information_schema` objects other than the reachable predefined-role
  membership closure, are deliberately not interpreted as permission to
  proceed. Incomplete, malformed, mismatched, or unrecognized evidence fails
  to produce a receipt that can authorize a cutover; that is the safe side
  because this command never changes authority and a later cutover remains
  blocked on a reviewed, complete receipt.

### Files touched

- `.agent/runbooks/database.md`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/config.py`
- `plans/PR-Postgres-Role-Topology-Preflight.md`
- `scripts/check_database_role_topology.py`
- `tests/test_database_role_topology_preflight.py`

## Mechanism

The command uses the ordinary `DatabaseConfig` only to open the same runtime
target that Atlas would use. A separate typed `SecretStr` DBA configuration is
loaded only inside the command. It opens bounded, statement-cache-free pools,
pins one runtime and one DBA connection in read-only repeatable-read
transactions, attests that those sessions name the same database/schema and
share a transaction-scoped advisory-lock namespace, then verifies the DBA
session is a direct superuser with a different authenticated and effective
identity before reading a fixed catalog projection from that same DBA
transaction.

The projection contains no business rows and no DSN text. It reports only
database/current-schema ownership, relevant roles and PostgreSQL 16 membership
options through each non-system role's reachable predefined-role closure,
protected EOM guard isolation, runtime privilege flags, per-object
relation/sequence/function ownership with row-security, security-invoker, and
security-definer flags, and effective ACL records (including PostgreSQL
defaults, explicit column grants, and deparsed RLS policy bindings/expressions)
needed to design the next role-separation slice. The JSON renderer redacts
targets to host/port/database labels and never serializes a `SecretStr` value.

The command has no `--apply` option and never calls `execute` for DDL/DML. It
does not extend `./ops db inspect`, so the ordinary operator surface remains
two fixed low-privilege queries instead of becoming a privileged catalog shell.

## Intentional

- A superuser DBA connection is intentionally required for a complete role and
  ownership receipt; a normal application connection can be insufficient and
  would make a false-complete report unsafe.
- The preflight does not create the future inspection role. Creating roles,
  transferring ownership, and changing runtime migration behavior are the
  next controlled DBA slice after this receipt is collected and reviewed.
- The current role name `atlas` remains a report field, not a newly enforced
  global requirement. Existing EOM completion code has a stricter `atlas`
  readiness invariant that remains unchanged.
- Historical scripts that connect with `db_settings.dsn` remain untouched; they
  are a separately deferred maintenance-runner refactor, not a reason to widen
  this evidence command.

## Deferred

- Controlled role-topology cutover: create separate runtime, migration, and
  inspection authorities; enumerate every application and maintenance caller;
  transfer ownership only after a reviewed preflight receipt and deployed-code
  convergence.
- Runtime migration behavior: remove generic DDL from application startup and
  replace it with a controlled migration deployment path plus readiness fence.
- Historical maintenance-script refactoring into a supported operations runner.
- Cross-host PostgreSQL authentication and TLS policy.
- Deployment convergence: the running `atlas-api.service` must reach a revision
  containing the current socket/guard code before any production role change.

Parking predicate: any role, ownership, grant, service credential, migration,
or deployment mutation is parked by default. This slice owns only the
evidence-gathering command required to make those later changes safe.

Parked hardening: none.

## Verification

- Passed locally after the compatibility, cleanup, provenance,
  privilege-semantics, and dotenv-boundary corrections:
  `python3.11 -m py_compile
  tests/test_database_role_topology_preflight.py`; `python -m py_compile
  scripts/check_database_role_topology.py
  tests/test_database_role_topology_preflight.py atlas_brain/config.py`;
  `./ops test focused tests/test_database_role_topology_preflight.py -q`
  (`23 passed, 1 skipped`); `./ops test focused
  tests/test_database_role_topology_preflight.py
  tests/test_agent_operations_contract.py
  tests/test_eom_first_clean_completion_dba_runner.py -q` (`144 passed, 1
  skipped`); and `bash scripts/check_ascii_python.sh`.
- Not locally runnable: `python3.11 -m pytest
  tests/test_database_role_topology_preflight.py -q` because that isolated
  Python 3.11 interpreter has no `pytest` module. This is not bypassed; the
  enrolled GitHub Python 3.11 workflow remains the execution gate.
- The real PostgreSQL 16 command test skips locally unless the explicitly named,
  loopback-only disposable DBA test DSN is present. The EOM PostgreSQL workflow
  supplies that service and runs the test; it must pass there before merge.
- Not locally runnable: `gitleaks protect --staged --redact --verbose` because
  the executable is absent in this worktree environment. Do not bypass it; the
  required CI secret scan remains the release gate.
- Current-head plan-file/diff-size audits and the local PR-review wrapper pass
  with the synchronized PR body. GitHub remains the required full-unit,
  PostgreSQL-16, and secret-scan execution authority.
- No production database role/configuration/deployment action is part of local
  verification. A real receipt is deferred until a protected DBA DSN is
  provisioned and the deployed runtime has converged.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/runbooks/database.md` | 55 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 6 |
| `atlas_brain/config.py` | 26 |
| `plans/PR-Postgres-Role-Topology-Preflight.md` | 459 |
| `scripts/check_database_role_topology.py` | 848 |
| `tests/test_database_role_topology_preflight.py` | 1216 |
| **Total** | **2610** |
