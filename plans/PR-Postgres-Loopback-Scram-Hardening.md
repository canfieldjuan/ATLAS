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
- A bare fixed inspection can select a worktree `.env` instead of the
  service's `EnvironmentFiles`, even when the worktree omits database keys.
- A socket target label omits its configured port even though the port selects
  the socket filename and the migration receipt confirms that label exactly.
- A successful service or inspector connection is not socket proof while a
  complete DSN retains deliberate precedence over split configuration.
- A final-only loopback-client inventory can lose a client observed at preflight
  before it is proved Unix-socket or SCRAM-ready, and a peer-map insertion
  without an exact loaded-map receipt can retain a broader OS-account
  authorization.
- A manual four-rule HBA conversion needs an exact loaded-tuple postcondition,
  derived coverage over every non-local rule that can match loopback, and
  active-file rollback; endpoint-only counts can miss a broader subnet or
  TLS-specific rule, while a disk-only restore leaves partial HBA rules active.
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
   service/CRM/backend-transport/inspection proof, reconciled initial/final
   loopback-client inventories, loopback-SCRAM replacement, a derived
   loaded-HBA coverage assertion plus exact four-tuple proof, and a rollback
   that restores TCP settings and reloads the restored HBA before restarting
   the service. The procedure must stop on unresolved or remaining loopback TCP
   clients and prove the loaded peer map has exactly the intended mapping. The fixed
   inspection must explicitly select the ordered `atlas-api.service`
   `EnvironmentFiles` while excluding ad hoc `ATLAS_DB_*` overrides.
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
- Adjacent configuration-context regression: a worktree file remains the
  default inspector context, while an ordered `ATLAS_OPS_ENV_FILES` override
  selects the intended service configuration.
- Cheap local gates: focused test target, `bash scripts/check_ascii_python.sh`,
  `git diff --check`, and `python scripts/sync_pr_plan.py ... --check`.
- GitHub remains the complete unit gate.
- Post-merge proof while existing TCP trust remains: deploy source; configure
  `ATLAS_DB_SOCKET_PATH` only when no complete DSN overrides it; add/reload the
  exact identity map and specific peer HBA rule; restart `atlas-api`; prove
  health, an authenticated EOM CRM read, the application's Unix-socket backend,
  fixed inspection selected from the service `EnvironmentFiles`, the exclusive
  loaded identity map, a Unix-socket or separately verified SCRAM receipt for
  every client observed in either inventory, and no remaining loopback TCP or
  replication client.
- Only then replace every loopback TCP `trust` rule with `scram-sha-256`,
  reload PostgreSQL, prove the derived loopback-network receipt is `4` and the
  exact loaded-HBA result is `1|1|1|1|0|0|0` (one exact application IPv4,
  application IPv6, replication IPv4, and replication IPv6 SCRAM tuple; zero
  unexpected endpoint-equal host rules, trust rows, and parser errors), repeat
  the proofs, prove passwordless TCP rejection, and prove
  `sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -Atc 'SELECT
  current_user'` succeeds.

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
     removing loopback `trust`, proves the exclusive loaded identity map, gives
     every initial/final loopback client a Unix-socket or verified-SCRAM receipt,
     leaves no TCP client, derives every loopback-covering HBA network rule,
     proves every exact tuple, retains `postgres` peer recovery, and reloads
     restored TCP authentication before a rollback restart.
  5. The migration runbook accurately distinguishes raw forensic output from
     attested admission without changing runner behavior.
- Reachability proof: `atlas_brain/storage/database.py` initializes the pool
  and raw connections from `db_settings.connection_kwargs()`; `ops` creates
  `DatabaseConfig(_env_file=None)` and passes those kwargs to its fixed asyncpg
  inspection. Direct maintenance scripts that call `db_settings.dsn` receive
  the corrected socket port as well.
- Affected surfaces: `DatabaseConfig` DSN/asyncpg construction and target
  labels; `DatabasePool`; the fixed `./ops db inspect` environment-selection
  seam; PostgreSQL HBA/ident operational procedure; migration target
  confirmation; and the authenticated EOM CRM read used as production proof.
- Risk areas: local role impersonation over loopback TCP, wrong-cluster
  inspection, service/inspector configuration skew, HBA parser failure,
  an overbroad peer map, active local client disconnect, duplicated/missing
  IPv4/IPv6/replication tuples, unreloaded failed HBA restoration,
  startup-migration availability, and rollback recovery.
- Reviewer rules triggered: R1, R2, R3, R11, R12, R14.
- Boundary-change enumeration:
  - `socket_path=None` continues to produce TCP host/port kwargs.
  - `connection_string` continues to win over split socket/TCP settings.
  - `socket_path` replaces only the host, retaining the configured port needed
    to select PostgreSQL's socket filename.
  - The socket target label includes that port, so confirmation cannot conflate
    same-directory, same-database clusters on distinct ports.
  - The fixed inspector uses only the service `EnvironmentFiles`, in service
    order, and removes ad hoc `ATLAS_DB_*` values before it constructs
    `DatabaseConfig`.
  - The cutover reconciles every client observed in both initial and final
    loopback TCP/replication inventories to a Unix-socket or verified-SCRAM
    receipt, requires no remaining final client, and requires an exact loaded
    `atlas_app | juan-canfield | atlas` identity map before HBA replacement.
  - The post-conversion HBA receipt derives every non-local rule whose network
    can cover either loopback endpoint, then requires one exact SCRAM tuple for
    each application/replication IPv4/IPv6 channel, no unexpected endpoint-equal
    host rule, no remaining trust row, and no parser error before an IPv4-only
    negative probe can be treated as sufficient.
  - Cutover order is peer proof before removal of any trust rule; rollback
    restores and reloads TCP authentication before restarting the application.

### Boundary-set closure declaration

- **`DatabaseConfig` connection-construction branches — CLOSED / DERIVED.**
  Membership is the three ordered outcomes in
  `atlas_brain/storage/config.py:80-136`: a nonblank complete DSN, a blank DSN
  with `socket_path`, or a blank DSN without `socket_path`. No fourth
  construction path exists. Any setting combination resolves through that
  existing order; a nonblank DSN is outside the socket-proof-admissible subset
  and stops this cutover rather than allowing HBA mutation.
- **Loopback HBA topology — CLOSED / DERIVED at cutover.** Membership comes
  from every non-local live `pg_hba_file_rules` record on the declared socket
  and port, not the four examples in this plan. A derived network predicate
  includes an address/netmask rule when it can cover either loopback endpoint
  and treats null, unparseable, or family-mismatched forms as candidates. The
  procedure admits exactly four candidates, each one exact
  application/replication IPv4/IPv6 tuple with the declared user, netmask, and
  SCRAM method. A missing, duplicate, broader, TLS-specific, or unlisted rule
  fails closed to no HBA change because an unknown authentication path is less
  safe than a deferred hardening run.
- **`atlas_app` peer-map membership — CLOSED / DERIVED at reload.** Membership
  comes from `pg_ident_file_mappings`; exactly one
  `atlas_app | juan-canfield | atlas` tuple and no mapping error is admitted.
  Any extra, missing, malformed, or unrecognized mapping stops before HBA
  replacement because it could broaden local role impersonation.
- **Loopback-client inventory — OPEN / DERIVED at both observation times.**
  `pg_stat_activity` and `pg_stat_replication` supply every current TCP
  loopback client at initial and final observation; application names and client
  identities are not enumerated. Every member of their union, including a
  client that disappears or newly appears, stops the cutover until it has a
  Unix-socket or separately verified SCRAM reconnect receipt; an empty final
  snapshot alone is not an admissible default.
- **Service environment-file list — CLOSED / DERIVED for the selected service
  unit.** Membership is the ordered `EnvironmentFiles` output of `./ops env
  systemd`. An absent, unreadable, or unlisted effective configuration file
  stops fixed inspection and HBA mutation, the safer side over inspecting or
  changing a guessed database target.

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

The fixed inspector keeps its existing source-selection semantics. The runbook
uses its documented explicit environment-file override, in the exact service
order and without inherited database settings, so the read-only proof observes
the same database target as `atlas-api.service`.

The migration-runbook change is documentation only: raw discrepancies stay
visible and retain their forensic nonzero status.

## Intentional

- Raw migration reporting/exit semantics, database role/ownership, and
  non-loopback access remain unchanged.
- `ops` already carries `ATLAS_DB_SOCKET_PATH`; CRM read is production proof,
  not a feature.
- `ops` context-selection code remains unchanged; the cutover invokes its
  existing explicit override rather than changing normal worktree behavior.

## Deferred

- Least-privilege database-role redesign (separate application ownership,
  migration DDL, and inspection authority).
- Refactoring historical maintenance scripts into a supported operations runner.
- Cross-host PostgreSQL authentication policy and TLS posture.

Parking predicate: role topology, maintenance-script ownership, or non-local
database access gets a new slice only when it blocks a future capability. None
blocks the socket-peer path.

## Verification

- `./ops test focused tests/test_eom_render_profile.py -q -k 'database_config
  or database_pool_uses_configured_connection_kwargs'` — 4 passed, 61
  deselected (local).
- `./ops test focused tests/test_agent_operations_contract.py -q -k
  'database_file_context_prefers_worktree_over_shared_and_systemd or
  database_file_context_honors_explicit_override_order'` — 2 passed, 39
  deselected (local).
- `bash scripts/check_ascii_python.sh` — passed (local).
- `git diff --check` — passed (local).
- `python scripts/sync_pr_plan.py
  plans/PR-Postgres-Loopback-Scram-Hardening.md --check` — passed (local).
- `python scripts/audit_pr_body.py --base-ref origin/main
  tmp/pr-body-postgres-loopback-peer-hardening.md` and the reconciliation/fix
  loop auditors — passed (local).
- Read-only derived-HBA probe against the current `trust` topology returned
  `4|0`: four loopback-network candidates and zero unexpected candidates; a
  synthetic broader subnet, `hostssl` rule, and non-IP form each became a
  candidate while the current remote `/12` rule did not (local).
- Guarded `scripts/push_pr.sh` local PR review — passed; GitHub owns the full
  unit gate.
- Post-merge: follow the exact peer-socket cutover/rollback procedure in
  `.agent/runbooks/database.md`; do not remove HBA trust until service-pinned
  inspection, CRM, transport, and loaded-HBA proofs all succeed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/runbooks/database.md` | 400 |
| `atlas_brain/storage/config.py` | 11 |
| `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` | 20 |
| `plans/PR-Postgres-Loopback-Scram-Hardening.md` | 304 |
| `tests/test_eom_render_profile.py` | 61 |
| **Total** | **796** |

## Diff budget

Diff-budget override: the socket construction, caller-seam evidence, exact
target identity, and staged peer/SCRAM procedure are one deployment-safety
boundary; splitting them would publish either executable configuration without
its safe cutover/rollback proof or a procedure that refers to unavailable
behavior.
