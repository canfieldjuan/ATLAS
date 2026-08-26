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
fail-closed boundary fixtures, and operator procedure are one atomic safety
surface: splitting them would either publish an unverified privileged reader
or leave the operator without a usable, documented receipt. It does not carry
the later role/grant/ownership cutover, which remains a separate slice.

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
     role, membership, database-owner, schema-owner, relation-owner, and ACL
     facts needed for a later cutover without customer rows or credentials.
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

## Scope (this PR)

Ownership lane: eom-crm/runtime-security
Slice phase: Production hardening

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
  2. With the DBA DSN absent, a runtime/DBA database or cluster mismatch, or a
     non-superuser DBA session, the command raises before any catalog report;
     settled by focused unit tests in the same file.
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
- Risk areas: credential redaction, target confusion, cross-cluster false
  attestation, insufficient inspection authority, read-only enforcement,
  current production/runtime drift, and accidental role/DDL mutation.
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
  explicit DBA DSN + matching runtime target -> fixed read-only report;
  explicit DBA DSN + target mismatch/non-superuser session -> reject before
  report.

### Deployed-config probing

Required because this slice adds a DBA-only configuration boundary.

- Deployed/default config values: the normal runtime keeps its current
  `DatabaseConfig` selection. The new DBA DSN is absent by default and the
  command must fail closed when absent.
- Explicit value probe: a test injects a valid typed DBA DSN and proves the
  command uses it only for the dedicated DBA pool.
- Absent value probe: a test proves missing/blank DBA DSN prevents pool creation.
- Default-session/default-context probe: tests distinguish a direct runtime
  session from a `SET ROLE`/mismatched session and prove runtime/DBA target
  attestation uses both database identity and one advisory-lock namespace.
- Side-effect ordering: all identity/config admissions precede catalog report
  construction; every catalog query runs inside `readonly=True` transactions;
  the command contains no mutation path.

### Catalog-projection closure

- The catalog row set is **OPEN**: roles, membership edges, non-system schemas,
  objects, and ACL entries can change after this PR. Their membership is
  **DERIVED** on every command invocation from fixed `pg_catalog` queries; the
  command does not maintain a copied role or object inventory.
- The projection classes are **CLOSED for this slice**: target identity, role
  attributes, membership edges, current-database ownership/ACL, non-system
  schema ownership/ACL, relation and function owner/effective-ACL summaries
  with row-security and security-definer flags, explicit column-ACL and
  row-security-policy role summaries, and explicit default-ACL overrides. Their
  canonical definition is this problem-derived contract and the static query
  constants in the command.
- Catalog classes outside that closed projection, plus `pg_*` and
  `information_schema` objects, are deliberately not interpreted as permission
  to proceed. Incomplete, malformed, mismatched, or unrecognized evidence
  fails to produce a receipt that can authorize a cutover; that is the safe
  side because this command never changes authority and a later cutover remains
  blocked on a reviewed, complete receipt.

### Files touched

- `.agent/runbooks/database.md`
- `atlas_brain/config.py`
- `plans/PR-Postgres-Role-Topology-Preflight.md`
- `scripts/check_database_role_topology.py`
- `tests/test_database_role_topology_preflight.py`

## Mechanism

The command uses the ordinary `DatabaseConfig` only to open the same runtime
target that Atlas would use. A separate typed `SecretStr` DBA configuration is
loaded only inside the command. It opens bounded, statement-cache-free pools,
attests that the runtime and DBA sessions name the same database/schema and
share a transaction-scoped advisory-lock namespace, then verifies the DBA
session is a superuser before reading a fixed catalog projection.

The projection contains no business rows and no DSN text. It reports only
database/current-schema ownership, relevant roles and memberships, protected
EOM guard isolation, runtime privilege flags, relation/sequence/function owner
summaries with row-security and security-definer flags, and effective ACL
summaries (including PostgreSQL defaults, explicit column grants, and RLS policy
role bindings) needed to design the next role-separation slice. The JSON
renderer redacts targets to host/port/database labels and never serializes a
`SecretStr` value.

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

- Passed locally: `python -m py_compile scripts/check_database_role_topology.py
  tests/test_database_role_topology_preflight.py atlas_brain/config.py`;
  `./ops test focused tests/test_database_role_topology_preflight.py
  tests/test_agent_operations_contract.py
  tests/test_eom_first_clean_completion_dba_runner.py -q` (`137 passed`);
  `bash scripts/check_ascii_python.sh`; `git diff --cached --check`;
  `python scripts/sync_pr_plan.py --check
  plans/PR-Postgres-Role-Topology-Preflight.md origin/main`; and
  `python scripts/audit_plan_doc.py
  plans/PR-Postgres-Role-Topology-Preflight.md`.
- Not locally runnable: `gitleaks protect --staged --redact --verbose` because
  the executable is absent in this worktree environment. Do not bypass it; the
  required CI secret scan remains the release gate.
- Passed after the first commit: `python scripts/audit_plan_doc_files_touched.py
  plans/PR-Postgres-Role-Topology-Preflight.md origin/main` and
  `python scripts/audit_plan_doc_diff_size.py
  plans/PR-Postgres-Role-Topology-Preflight.md origin/main` (the plan's five
  files and `1757` LOC match `origin/main...HEAD`).
- Passed before push: the local PR-review wrapper completed successfully with
  its isolated session-state file and the current PR body.
- No production database role/configuration/deployment action is part of local
  verification. A real receipt is deferred until a protected DBA DSN is
  provisioned and the deployed runtime has converged.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/runbooks/database.md` | 49 |
| `atlas_brain/config.py` | 24 |
| `plans/PR-Postgres-Role-Topology-Preflight.md` | 256 |
| `scripts/check_database_role_topology.py` | 761 |
| `tests/test_database_role_topology_preflight.py` | 667 |
| **Total** | **1757** |
