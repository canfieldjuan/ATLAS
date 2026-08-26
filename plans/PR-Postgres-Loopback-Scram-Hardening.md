# PR-Postgres-Loopback-Peer-Hardening

## Why this slice exists

The authorized EOM recovery exposed two deferred operational defects.

Loopback TCP `trust` accepts unauthenticated connections as every role,
including `postgres`. Atlas defaults to `atlas` without a password, and its
live pool/fixed inspection call `connection_kwargs()`, which ignores the
existing `ATLAS_DB_SOCKET_PATH`. The deployed application therefore cannot
move to Unix-socket peer authentication by setting that variable alone.

The same recovery reported raw `missing_source` and `mismatched` evidence even
when matching, attested reconciliations admit a pending migration. The runbook
incorrectly said that admissible state could contain no `missing_source`.

### Problem-derived contract

#### Root cause

- Loopback TCP `trust` lets a local process request any database role.
- `connection_kwargs()` ignores `socket_path`, while the pool and fixed
  inspection call it rather than `DatabaseConfig.dsn`.
- A socket target label omits its configured port even though the port selects
  the socket filename and the migration receipt confirms that label exactly.
- A successful service or inspector connection is not socket proof while a
  complete DSN retains deliberate precedence over split configuration.
- The migration runbook confuses raw forensic output with attested admission.

#### Required change surface

1. Update `atlas_brain/storage/config.py` so both connection construction
   forms and its log-safe target label honour `socket_path` and the configured
   PostgreSQL port:
   - `dsn` includes the socket host and port for direct asyncpg callers.
   - `connection_kwargs()` uses the socket directory as `host`, retaining the
     configured port, so `DatabasePool` and `./ops db inspect` reach the Unix
     socket rather than loopback TCP.
   - `target_label` includes the socket port so exact-target confirmation
     distinguishes same-directory, same-database PostgreSQL clusters.
2. Update focused `DatabaseConfig` tests in
   `tests/test_eom_render_profile.py` to pin both socket-path forms, distinct
   socket ports, and the existing TCP/complete-DSN precedence behavior, then
   assert the actual pool and raw-connection callers receive the socket kwargs.
3. Replace the provisional credential/SCRAM procedure with non-secret socket
   configuration, an exact `atlas-api` OS-user → `atlas` peer map, staged
   service/CRM/backend-transport/inspection proof, loopback-SCRAM replacement,
   and a rollback that restores TCP settings before restarting the service.
4. Correct `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` to require matching,
   currently attested evidence for every raw mismatch **and** missing-source
   item while preserving the raw report and its forensic nonzero exit.
5. Exclude the unviable encrypted-credential experiment from this PR; it would
   require an unsupported plaintext-secret fallback.

#### Explicit non-scope

- Do not alter application APIs, CRM/customer behavior, email, payroll,
  billing, schemas, migrations, migration-ledger rows, historical migration
  source bytes, or database data.
- Do not change database roles, role privileges, network/non-loopback HBA
  rules, PostgreSQL ownership, or add an application password/credential.
- Do not introduce a generic SQL interface, alter `./ops db inspect`'s two
  fixed read-only statements, refactor arbitrary maintenance scripts, add
  dependencies, or make runtime-routing changes.
- Do not change the `postgres` Unix-socket peer break-glass rule.

#### Assumptions and blockers

- The service user is `juan-canfield` and the socket is `/var/run/postgresql`;
  recheck both immediately before cutover.
- Recheck the peer map and local replication activity before changing HBA.
- HBA/ident/shared configuration changes occur only after this source revision
  is deployed.

#### Verification plan

- Focused regression: socket `dsn` and `connection_kwargs()` assertions,
  distinct socket target-label assertions, existing TCP and complete-DSN
  assertions, and the pool/raw caller seam.
- Cheap local gates: focused test target, `bash scripts/check_ascii_python.sh`,
  `git diff --check`, and `python scripts/sync_pr_plan.py ... --check`.
- GitHub remains the complete unit gate.
- Post-merge proof while existing TCP trust remains: deploy source; configure
  `ATLAS_DB_SOCKET_PATH` only when no complete DSN overrides it; add/reload the
  exact identity map and specific peer HBA rule; restart `atlas-api`; prove
  health, an authenticated EOM CRM read, the application's Unix-socket backend,
  and fixed inspection.
- Only then replace every loopback TCP `trust` rule with `scram-sha-256`,
  reload PostgreSQL, repeat the proofs, prove passwordless TCP rejection, and
  prove `sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -Atc
  'SELECT current_user'` succeeds.

## Scope (this PR)

Ownership lane: eom-crm/runtime-security
Slice phase: production hardening
Max files: 5

### Review contract

- Acceptance criteria:
  1. A configured socket path reaches the configured PostgreSQL socket port in
     both `DatabaseConfig.dsn` and `connection_kwargs()`, and that port appears
     in its log-safe exact-target label.
  2. Complete DSNs retain their precedence; non-socket split settings retain
     existing TCP kwargs.
  3. `DatabasePool` and fixed `./ops db inspect` inherit the corrected path
     because both already call `connection_kwargs()`.
  4. The operational procedure rejects an overriding complete DSN, authenticates
     the specific service OS account as `atlas` over the Unix socket before
     removing loopback `trust`, retains `postgres` peer recovery, and restores
     TCP settings before a rollback restart.
  5. The migration runbook accurately distinguishes raw forensic output from
     attested admission without changing runner behavior.
- Reachability proof: `atlas_brain/storage/database.py` initializes the pool
  and raw connections from `db_settings.connection_kwargs()`; `ops` creates
  `DatabaseConfig(_env_file=None)` and passes those kwargs to its fixed asyncpg
  inspection. Direct maintenance scripts that call `db_settings.dsn` receive
  the corrected socket port as well.
- Boundary-change enumeration:
  - `socket_path=None` continues to produce TCP host/port kwargs.
  - `connection_string` continues to win over split socket/TCP settings.
  - `socket_path` replaces only the host, retaining the configured port needed
    to select PostgreSQL's socket filename.
  - The socket target label includes that port, so confirmation cannot conflate
    same-directory, same-database clusters on distinct ports.
  - Cutover order is peer proof before removal of any trust rule; rollback
    restores TCP configuration before restarting the application.

### Files touched

- `.agent/runbooks/database.md`
- `atlas_brain/storage/config.py`
- `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md`
- `plans/PR-Postgres-Loopback-Scram-Hardening.md`
- `tests/test_eom_render_profile.py`

## Mechanism

`DatabaseConfig` now carries its existing socket path and port into asyncpg.
No password is added: the post-merge identity map lets `atlas-api` authenticate
as `atlas` over peer. The map/rule are proved while TCP remains available;
only then do all loopback `trust` entries become `scram-sha-256`.

The migration-runbook change is documentation only: raw discrepancies stay
visible and retain their forensic nonzero status.

## Intentional

- Raw migration reporting/exit semantics, database role/ownership, and
  non-loopback access remain unchanged.
- `ops` already carries `ATLAS_DB_SOCKET_PATH`; CRM read is production proof,
  not a feature.

## Deferred

- Least-privilege database-role redesign (separate application ownership,
  migration DDL, and inspection authority).
- Refactoring historical maintenance scripts into a supported operations runner.
- Cross-host PostgreSQL authentication policy and TLS posture.

Parking predicate: role topology, maintenance-script ownership, or non-local
database access gets a new slice only when it blocks a future capability. None
blocks the socket-peer path.

## Verification

- Pending before push: `./ops test focused tests/test_eom_render_profile.py -q
  -k 'database_config or database_pool_uses_configured_connection_kwargs'`.
- Pending before push: `bash scripts/check_ascii_python.sh`.
- Pending before push: `git diff --check`.
- Pending before push: `python scripts/sync_pr_plan.py
  plans/PR-Postgres-Loopback-Scram-Hardening.md --check`.
- Pending before push: cold diff / contract audit and normal PR mechanical
  review helpers.
- Post-merge: follow the exact peer-socket cutover/rollback procedure in
  `.agent/runbooks/database.md`; GitHub Actions remains the complete unit gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/runbooks/database.md` | 171 |
| `atlas_brain/storage/config.py` | 11 |
| `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` | 20 |
| `plans/PR-Postgres-Loopback-Scram-Hardening.md` | 188 |
| `tests/test_eom_render_profile.py` | 60 |
| **Total** | **450** |
