# PR-Postgres-Loopback-Peer-Hardening

## Why this slice exists

The authorized EOM recovery exposed two deferred operational defects.

PostgreSQL currently accepts every TCP connection from `127.0.0.1` and `::1`
with `trust`, including connections requested as the `postgres` superuser.
Atlas defaults to the `atlas` database role without a password. Although
`DatabaseConfig` exposes `ATLAS_DB_SOCKET_PATH`, the live pool and fixed
`./ops db inspect` path call `connection_kwargs()`, which currently ignores
that setting and always selects the TCP host/port. The deployed application
cannot move from TCP `trust` to Unix-socket peer authentication merely by
setting the existing environment variable.

The recovery also emitted a migration-content-integrity warning. The runner
deliberately reports raw `missing_source` and `mismatched` historical evidence
even when matching, attested reconciliation records admit a pending migration.
The runbook previously described that admissible state as requiring no
`missing_source` records, contradicting the runner.

### Problem-derived contract

#### Root cause

- The local database boundary is too broad: loopback TCP `trust` lets a local
  process request any database role without authentication.
- The intended peer-auth replacement is ineffective because
  `DatabaseConfig.connection_kwargs()` ignores `socket_path`; the generic pool
  and fixed inspection use those kwargs rather than `DatabaseConfig.dsn`.
- The migration runbook conflates raw forensic reporting with admission. The
  runner admits known mismatches and missing sources only after current
  attestation; it does not erase or suppress their raw report.

#### Required change surface

1. Update `atlas_brain/storage/config.py` so both connection construction
   forms honour `socket_path` and the configured PostgreSQL port:
   - `dsn` includes the socket host and port for direct asyncpg callers.
   - `connection_kwargs()` uses the socket directory as `host`, retaining the
     configured port, so `DatabasePool` and `./ops db inspect` reach the Unix
     socket rather than loopback TCP.
2. Update focused `DatabaseConfig` tests in
   `tests/test_eom_render_profile.py` to pin both socket-path forms and the
   existing TCP/complete-DSN precedence behavior.
3. Replace the provisional credential/SCRAM procedure in
   `.agent/runbooks/database.md` with the peer procedure: configure the
   non-secret socket-path setting, map the actual `atlas-api` OS user to the
   existing `atlas` PostgreSQL role in `pg_ident.conf`, add a specific
   socket-peer HBA rule, prove the service/inspection/CRM path, then replace
   all loopback TCP `trust` rules with `scram-sha-256`. Preserve the existing
   Unix-socket `postgres` peer recovery path and a rollback sequence.
4. Correct `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` to require matching,
   currently attested evidence for every raw mismatch **and** missing-source
   item while preserving the raw report and its forensic nonzero exit.
5. Remove the unviable, uncommitted systemd encrypted-credential experiment
   and its tests. The running service is a non-root user service; this host
   cannot provision a protected credential to that manager without a plaintext
   secret fallback, so it is not a valid solution.

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

- `atlas-api.service` runs as local user `juan-canfield` and PostgreSQL's
  socket directory is `/var/run/postgresql`; the runbook requires a fresh
  check of both just before cutover.
- PostgreSQL accepts the `atlas_app` identity map and no live local TCP
  replication client depends on loopback `trust`; the cutover requires an
  activity/replication probe before changing those rules.
- HBA/ident/shared-configuration changes are an owned, post-merge controlled
  operation. They require deployment of the source revision containing the
  socket-kwargs correction first.

#### Verification plan

- Focused regression: socket `dsn` and `connection_kwargs()` assertions plus
  existing TCP and complete-DSN assertions.
- Cheap local gates: focused test target, `bash scripts/check_ascii_python.sh`,
  `git diff --check`, and `python scripts/sync_pr_plan.py ... --check`.
- GitHub remains the complete unit gate.
- Post-merge proof while existing TCP trust remains: deploy source; configure
  `ATLAS_DB_SOCKET_PATH`; add/reload the exact identity map and specific peer
  HBA rule; restart `atlas-api`; prove health, fixed inspection, and an
  authenticated EOM CRM read.
- Only then replace every loopback TCP `trust` rule with `scram-sha-256`,
  reload PostgreSQL, repeat the proofs, prove passwordless TCP rejection, and
  prove `sudo -u postgres psql -d atlas -Atc 'SELECT current_user'` succeeds.

## Scope (this PR)

Ownership lane: eom-crm/runtime-security
Slice phase: production hardening
Max files: 5

### Review contract

- Acceptance criteria:
  1. A configured socket path reaches the configured PostgreSQL socket port in
     both `DatabaseConfig.dsn` and `connection_kwargs()`.
  2. Complete DSNs retain their precedence; non-socket split settings retain
     existing TCP kwargs.
  3. `DatabasePool` and fixed `./ops db inspect` inherit the corrected path
     because both already call `connection_kwargs()`.
  4. The operational procedure authenticates the specific service OS account
     as `atlas` over the Unix socket before removing loopback `trust`, retains
     `postgres` peer recovery, and has a rollback order.
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
  - Cutover order is peer proof before removal of any trust rule; rollback
    restores saved HBA/ident/config state before restarting the application.

### Files touched

- `.agent/runbooks/database.md`
- `atlas_brain/storage/config.py`
- `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md`
- `plans/PR-Postgres-Loopback-Scram-Hardening.md`
- `tests/test_eom_render_profile.py`

## Mechanism

`DatabaseConfig` already owns the socket-path setting. Its split connection
path will consistently translate it into the asyncpg socket host plus the
configured port. No password is added: the post-merge server-side identity map
lets the `atlas-api` OS user authenticate as the existing `atlas` role with
peer authentication. The socket setting is a non-secret shared application
configuration value, which `atlas-api` and `./ops` already read.

The HBA sequence is staged. The exact `pg_ident.conf` mapping and specific HBA
rule are added and proved while current TCP rules remain available. Only after
the service genuinely uses the socket do all IPv4/IPv6 loopback TCP `trust`
rules change to `scram-sha-256`. This removes uncredentialed cross-user local
impersonation without a plaintext credential mechanism.

The migration-runbook change is documentation only. Raw discrepancies stay
visible and retain the existing forensic nonzero status; the document now
describes the runner's actual pending-admission condition.

## Intentional

- Raw migration mismatch/error logging and preflight exit semantics remain
  unchanged.
- `ops` source code remains unchanged: it already carries
  `ATLAS_DB_SOCKET_PATH` to the fixed inspection child.
- Existing app database role, database ownership, and non-loopback access
  remain unchanged.
- The endpoint-level CRM read is production verification, not a new feature.

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
  -k 'database_config'`.
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
| `.agent/runbooks/database.md` | 147 |
| `atlas_brain/storage/config.py` | 7 |
| `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` | 10 |
| `plans/PR-Postgres-Loopback-Scram-Hardening.md` | 203 |
| `tests/test_eom_render_profile.py` | 31 |
| **Total** | **398** |
