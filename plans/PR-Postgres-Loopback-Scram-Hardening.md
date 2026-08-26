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

This PR intentionally exceeds the normal diff budget because the configuration
and caller seams, deployment cutover/rollback procedure, and plan evidence are
one deployment-safety boundary. The migration-integrity correction must ship
with that same restart procedure: `atlas-api` startup runs migration checks,
and the former guidance stopped an operator on a raw `missing_source` state
that the runtime admits when matching current attestation exists. Splitting
either side would publish the socket transition without its proof/rollback or
leave its restart procedure with contradictory integrity instructions.

### Problem-derived contract

#### Root cause

- Loopback TCP `trust` lets a local process request any database role.
- `connection_kwargs()` ignores `socket_path`, while the pool and fixed
  inspection call it rather than `DatabaseConfig.dsn`.
- A bare fixed inspection can select a worktree `.env` instead of the
  service's `EnvironmentFiles`, even when the worktree omits database keys.
- The cutover rollback assumes that it created `ATLAS_DB_SOCKET_PATH`, but a
  pre-existing case-variant assignment in any selected service file has no
  baseline receipt; deleting or overriding it during rollback could change the
  service's original connection path.
- `DatabaseConfig` accepts case-variant database keys while the fixed inspector
  previously selected only canonical service-file keys; an alias or empty file
  list could therefore make fixed inspection use a fallback/different target
  from `atlas-api.service` before the cutover safety checks run.
- Rollback previously checked only one exact socket-assignment spelling in one
  file. A surviving case- or spacing-variant could keep the service on peer
  authentication after its identity map was restored, while fixed inspection
  would not necessarily observe that residual configuration.
- The peer-map receipt proved only that `atlas_app` exists; it did not prove the
  loaded `local atlas atlas peer map=atlas_app` rule is first among applicable
  local HBA rules. A preceding map or broad peer rule could therefore authorize
  a different OS account while the intended map and parser receipts still pass.
- `DatabaseConfig` resolves environment names case-insensitively, but `ops`
  previously removed only a subset of canonical uppercase database keys. A
  lower- or mixed-case inherited alias for any setting the inspector consumes
  could therefore override the selected service configuration in its subprocess.
- A socket target label omits its configured port even though the port selects
  the socket filename and the migration receipt confirms that label exactly.
- A socket path is interpolated raw into the direct-DSN query string while the
  pool path passes it as a raw asyncpg host, so URI delimiters can make the two
  supported connection forms select different targets.
- A successful service or inspector connection is not socket proof while a
  complete DSN retains deliberate precedence over split configuration.
- The transport receipt accepts one socket connection among all qualifying
  `atlas`/`atlas` backends, so it can pass while another qualifying backend
  remains on TCP.
- The HBA procedure backs up only the top-level file but did not prove the
  loopback rules it edits originate there; an included source could survive
  rollback.
- A final-only loopback-client inventory can lose a client observed at preflight
  before it is proved Unix-socket or SCRAM-ready, and a peer-map insertion
  without an exact loaded-map receipt can retain a broader OS-account
  authorization.
- A manual four-rule HBA conversion needs an exact loaded-tuple postcondition,
  derived coverage over every non-local rule that can match loopback, and
  active-file rollback; endpoint-only counts can miss a broader subnet or
  TLS-specific rule, while a disk-only restore leaves partial HBA rules active.
- The final exact-tuple HBA receipt consumes `covers_loopback` from a CTE that
  was scoped to the preceding, separate `psql` command, so PostgreSQL cannot
  evaluate the source-boundary column in the final receipt.
- Service-file admission for a pre-existing socket assignment or nonblank
  complete DSN was deferred until after backup, peer-map/HBA edits, and reload,
  so a rejected configuration could leave partial authentication state live.
- The effective service file was manually selected after authentication edits
  and only checked during rollback, so a mistyped or unlisted path could create
  a stray socket assignment that the rollback path refuses to remove.
- The loopback-client inventory sees only current PostgreSQL sessions. The
  repository-owned `eom-write-boundary-audit.timer` is an hourly one-shot and
  can be absent from both snapshots while its source default still selects
  loopback TCP, so its database path could be lost during trust removal without
  any client receipt.
- Its libpq helper begins with the inherited process environment, so ambient
  `PG*` settings can survive a socket-default change and make the monitor's
  effective transport depend on user-manager state rather than the admitted
  source URI.
- The monitor admission checks only whether effective `ExecStart` contains the
  installed script path. A drop-in can add `--atlas-dsn` or another
  target-affecting argument while passing that check, so its manual default
  dry-run can prove a different connection from the timer's scheduled command.
- Requiring an installed copy of the new socket-default source before the peer
  map/rule is loaded creates a rollout window in which the active timer runs as
  `juan-canfield` but the generic peer rule cannot authenticate it as `atlas`.
  A later rollback would also strand that new source if it restored the old HBA
  and identity files without restoring the pre-cutover installed monitor.
- The dormant-monitor admission rejects a scheduled DSN override but not its
  environment-selected `psql` executable, so the operator-shell dry run can
  prove a different binary from the one the next timer run invokes.
- That admission also reaches the first HBA/identity backup without verifying
  that both the installed monitor and the source it must stage are readable.
  A predictable staging failure is therefore discovered only after
  authentication mutation and leaves retry-blocking pre-peer backups behind.
- Clearing every inherited libpq setting for an explicit owner-provided audit
  DSN removes non-target TLS inputs such as root/client certificate paths that
  the prior monitor preserved, regressing supported explicit-DSN deployments.
- Two repository-owned typed maintenance callers construct asyncpg targets from
  `db_settings.host` directly rather than the corrected socket-aware
  `connection_kwargs()`/`dsn` seams, so they remain TCP clients after loopback
  trust is removed.
- Routing the chart rebuild through `connection_kwargs()` also injects its
  configured connection and command timeouts, although the prior `create_pool`
  call omitted both options; a long-running existing rebuild can therefore fail
  solely from this transport refactor.
- The colon-delimited service environment-file list silently skips empty
  members, while the effective-target sentinel treats an empty derived final
  member as present. A trailing delimiter can therefore pass pre-mutation
  admission, fail only at `sudoedit`, and block the rollback before it restores
  authentication files.
- The migration runbook confuses raw forensic output with attested admission.

#### Required change surface

1. Update `atlas_brain/storage/config.py` so both connection construction
   forms and its log-safe target label honour `socket_path` and the configured
   PostgreSQL port:
   - `dsn` includes the socket host and port for direct asyncpg callers.
   - The socket-host query value is URI-encoded so its decoded value remains
     identical to the asyncpg keyword host for delimiter-containing paths.
   - `connection_kwargs()` uses the socket directory as `host`, retaining the
     configured port, so `DatabasePool` and `./ops db inspect` reach the Unix
     socket rather than loopback TCP.
   - `target_label` includes the socket port so exact-target confirmation
     distinguishes same-directory, same-database PostgreSQL clusters.
2. Update focused `DatabaseConfig` tests in
   `tests/test_eom_render_profile.py` to pin both socket-path forms, distinct
   socket ports, a delimiter-containing socket path, and the existing
   TCP/complete-DSN precedence behavior, then assert the actual pool and
   raw-connection callers receive the socket kwargs.
3. Replace the provisional credential/SCRAM procedure with non-secret socket
   configuration, an exact `atlas-api` OS-user → `atlas` peer map, staged
   service/CRM/backend-transport/inspection proof, reconciled initial/final
   loopback-client inventories, loopback-SCRAM replacement, a derived
   loaded-HBA coverage assertion plus exact four-tuple proof, and a rollback
   that restores TCP settings and reloads the restored HBA before restarting
   the service. The procedure must stop on unresolved or remaining loopback TCP
   clients and prove the loaded peer map has exactly the intended mapping. The fixed
   inspection must explicitly select the ordered `atlas-api.service`
   `EnvironmentFiles` while the helper clears every exact-uppercase
   `ATLAS_DB_*` override and `ops` excludes every lower/mixed-case variant. It
   must reject a pre-existing case-variant `ATLAS_DB_SOCKET_PATH` in every
   selected service file before adding its single reversible assignment, and
   regressions must pin the helper's unset set to every `DatabaseConfig` field
   and its precondition's positive/negative matching boundaries. The helper
   must refuse an empty file set and any noncanonical-cased `ATLAS_DB_*` key
   before fixed inspection, so the full-DSN gate applies to the same source as
   the service. A shared no-socket-assignment guard must gate both the initial
   edit and rollback across every selected service file, including case and
   spacing variants, before rollback restores HBA/identity configuration.
   A combined new-socket admission gate must also reject every selected service
   file with a nonblank complete-DSN assignment, and must run before the first
   HBA/identity backup, edit, or reload. Each independent post-conversion HBA
   query must define its own derived loopback-coverage relation before it
   consumes that relation.
   The effective environment file must be derived from the ordered selected
   service-file list and have its nonempty absolute-path form and membership
   validated by the same pre-mutation gate and rollback path. Every selected
   list member must also be nonempty and absolute; no manual file selection may
   occur after authentication edits.
   The loaded peer receipt must also require exactly one intended local HBA rule
   and no preceding local rule except the retained `postgres` recovery rule.
   It must require at least one qualifying post-restart `atlas` backend and a
   null `client_addr` for every such backend before trust removal, and it must
   stop before mutation unless every loopback trust rule is sourced from the
   one top-level HBA file that rollback backs up and restores.
4. Correct `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` to require matching,
   currently attested evidence for every raw mismatch **and** missing-source
   item while preserving the raw report and its forensic nonzero exit.
5. Exclude the unviable encrypted-credential experiment from this PR; it would
   require an unsupported plaintext-secret fallback.
6. Update `ops` to admit only canonical uppercase runtime database overrides,
   remove every case-insensitive alias for every `DatabaseConfig` setting before
   its inspection child constructs that configuration, and add a regression that
   proves the selected service file wins for both target and timeout settings,
   plus a closure test that fails when a future `DatabaseConfig` field is not
   added to that scrub set.
7. Move the standalone EOM write-boundary monitor's source-owned default to a
   passwordless encoded Unix-socket URI, decode that URI host before passing it
   to libpq, and clear inherited `PG*` libpq settings only for that default.
   Explicit monitor DSNs retain their pre-existing owner-supplied libpq/TLS
   environment behavior. The cutover runbook must admit the dormant scheduled
   client before HBA mutation using its exact effective Python/script argv, an
   active timer/service identity, no `EnvironmentFile`, and no unit/user-manager
   DSN or `psql`-executable override without printing environment values, and
   verify both source-stage inputs are readable before the first authentication
   backup. It must stage the new installed monitor only after the scoped peer
   map/rule is loaded, retain a restorable pre-cutover copy through the cutover,
   and dry-run the admitted source with no alert/state mutation before trust
   replacement. Regression tests must cover the command-argument,
   environment-executable, source-stage/rollback, default-versus-explicit-
   libpq, and unreadable-route boundaries.
8. Route every repository-owned typed `DatabaseConfig` maintenance caller that
   still passes `db_settings.host` directly through `connection_kwargs()` while
   retaining its current command-timeout/pool-size intent. In particular, the
   chart-rebuild pool must omit the connection and command timeout options just
   as it did before this transport change, while retaining its existing pool
   bounds. Pin those two caller seams with socket-path regressions.

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
- Do not change the EOM audit's signals, alert delivery, timer cadence, state
  semantics, or any monitor override value; an explicit monitor DSN remains
  supported but is outside this cutover until separately migrated and proved.

#### Assumptions and blockers

- The service user is `juan-canfield` and the socket is `/var/run/postgresql`;
  recheck both immediately before cutover.
- Recheck the peer map and local replication activity before changing HBA.
- HBA/ident/shared configuration changes occur only after this source revision
  is deployed.
- The deployed audit unit executes the documented installed script path, both
  source-stage files are readable, and no `EnvironmentFile`,
  `ATLAS_EOM_AUDIT_ATLAS_DSN`, or `ATLAS_EOM_AUDIT_PSQL_BIN` value is inherited
  from its unit or user manager; otherwise the cutover stops rather than
  inferring its transport or executable.

#### Verification plan

- Focused regression: socket `dsn` and `connection_kwargs()` assertions,
  including a delimiter-containing socket path, distinct socket target-label
  assertions, existing TCP and complete-DSN assertions, and the pool/raw caller
  seam.
- Adjacent configuration-context regression: a worktree file remains the
  default inspector context, while an ordered `ATLAS_OPS_ENV_FILES` override
  selects the intended service configuration and lower/mixed-case inherited
  aliases for every consumed database setting cannot supersede it in the
  inspection child; the `DATABASE_CONFIG_KEYS` set must exactly match the
  current `DatabaseConfig` fields.
- Cheap local gates: focused test target, `bash scripts/check_ascii_python.sh`,
  `git diff --check`, and `python scripts/sync_pr_plan.py ... --check`.
- Guard boundary coverage must prove empty and quoted-empty complete-DSN
  assignments pass while literal and expansion-shaped values stop, prove the
  configuration-admission command precedes the first HBA backup, and prove each
  final HBA receipt independently defines `covers_loopback`.
- It must also prove the derived effective file passes only when it belongs to
  the ordered selected service-file list and stops before the gate otherwise,
  including when a leading, middle, or trailing delimiter creates an empty
  member or a relative member appears in the list.
- Monitor regressions must prove both default-socket selection and preserved
  explicit-override selection, including hostile inherited `PG*` settings, and
  the runbook guard must reject service and user-manager DSN and executable
  overrides plus a missing source or installed monitor before the first HBA
  backup.
- GitHub remains the complete unit gate.
- Post-merge proof while existing TCP trust remains: deploy source; configure
  `ATLAS_DB_SOCKET_PATH` only when no complete DSN overrides it; add/reload the
  exact identity map and specific peer HBA rule; restart `atlas-api`; prove
  health, an authenticated EOM CRM read, every qualifying application's
  Unix-socket backend,
  fixed inspection selected from the service `EnvironmentFiles`, the exclusive
  loaded identity map, a Unix-socket or separately verified SCRAM receipt for
  every client observed in either inventory, the dormant audit monitor's
  installed-source/no-override admission and no-alert socket dry-run receipt,
  and no remaining loopback TCP or replication client.
- Only then replace every loopback TCP `trust` rule with `scram-sha-256`,
  reload PostgreSQL, prove the derived loopback-network receipt is `4` and the
  exact loaded-HBA result is `1|1|1|1|0|0|0|0` (one exact application IPv4,
  application IPv6, replication IPv4, and replication IPv6 SCRAM tuple; zero
  unexpected endpoint-equal host rules, trust rows, parser errors, and
  loopback rules outside the backed-up top-level source), repeat
  the proofs, prove passwordless TCP rejection, and prove
  `sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -Atc 'SELECT
  current_user'` succeeds.

## Scope (this PR)

Ownership lane: eom-crm/runtime-security
Slice phase: production hardening
Max files: 12

### Review contract

- Acceptance criteria:
  1. A configured socket path reaches the configured PostgreSQL socket port in
     both `DatabaseConfig.dsn` and `connection_kwargs()`, URI delimiters round
     trip identically through the DSN host query, and that port appears in its
     log-safe exact-target label.
  2. Complete DSNs retain their precedence; non-socket split settings retain
     existing TCP kwargs.
  3. `DatabasePool` and fixed `./ops db inspect` inherit the corrected path
     because both already call `connection_kwargs()`, and the fixed inspector
     admits only selected service values or exact-uppercase runtime database
     overrides rather than case-variant inherited aliases.
  4. The operational procedure rejects an overriding complete DSN, independently
     admits the dormant EOM audit monitor's readable source and installed copy,
     service/timer identity, no `EnvironmentFile`, and no DSN or executable
     override on its default socket route before HBA mutation, then
     records its no-alert socket dry-run receipt before trust replacement,
     authenticates
     the specific service OS account as `atlas` over the Unix socket before
     removing loopback `trust`, proves the exclusive loaded identity map,
     requires every qualifying post-restart backend to use a Unix socket, gives
     every initial/final loopback client a Unix-socket or verified-SCRAM receipt,
     leaves no TCP client, derives every loopback-covering HBA network rule,
     proves every exact tuple, retains `postgres` peer recovery, and reloads
     restored TCP authentication before a rollback restart. It also proves every
     loopback rule it edits is in the backed-up top-level HBA file, and stops before
     any HBA/identity mutation when any selected service file already assigns the
     socket path or a nonblank complete DSN, so rollback removes only an
     assignment created by this cutover. Each final HBA receipt is independently
     executable with its own loopback-coverage relation, and the effective
     environment file is derived and its nonempty absolute form and membership
     are checked before any authentication mutation.
  5. The migration runbook accurately distinguishes raw forensic output from
     attested admission without changing runner behavior.
- Reachability proof: `atlas_brain/storage/database.py` initializes the pool
  and raw connections from `db_settings.connection_kwargs()`; `ops` creates
  `DatabaseConfig(_env_file=None)` and passes those kwargs to its fixed asyncpg
  inspection. Direct maintenance scripts that call `db_settings.dsn` receive
  the corrected socket port as well.
- Affected surfaces: `DatabaseConfig` DSN/asyncpg construction and target
  labels; `DatabasePool`; the fixed `./ops db inspect` environment-selection
  seam; the standalone EOM write-boundary monitor's libpq environment
  construction; PostgreSQL HBA/ident operational procedure; migration target
  confirmation; and the authenticated EOM CRM read used as production proof.
- Risk areas: local role impersonation over loopback TCP, wrong-cluster
  inspection, service/inspector configuration skew, HBA parser failure,
  an overbroad peer map, active local client disconnect, duplicated/missing
  IPv4/IPv6/replication tuples, unreloaded failed HBA restoration,
  dormant scheduled-client disconnect, startup-migration availability, and
  rollback recovery.
- Reviewer rules triggered: R1, R2, R3, R11, R12, R14.
- Boundary-change enumeration:
  - `socket_path=None` continues to produce TCP host/port kwargs.
  - `connection_string` continues to win over split socket/TCP settings.
  - `socket_path` replaces only the host, retaining the configured port needed
    to select PostgreSQL's socket filename.
  - A socket path containing URI delimiters is encoded only in `dsn`'s host
    query value; asyncpg kwargs retain the original filesystem path.
  - The socket target label includes that port, so confirmation cannot conflate
    same-directory, same-database clusters on distinct ports.
  - The standalone audit's absent monitor-DSN default selects the encoded Unix
    socket host and no password; its libpq environment decodes that host to the
    filesystem path and clears every inherited `PG*` setting first. An explicit
    `ATLAS_EOM_AUDIT_ATLAS_DSN` still wins for the monitor itself, but it stops
    this peer cutover until separately migrated and proved rather than becoming
    an unobserved TCP dependency.
  - The fixed inspector uses only the service `EnvironmentFiles`, in service
    order, and removes every case variant of every consumed `ATLAS_DB_*` value
    before it constructs `DatabaseConfig`; exact-uppercase runtime keys preserve
    the generic documented override behavior, while the service-pinned helper
    explicitly unsets all of them before its proof.
  - A pre-existing case-variant socket-path assignment in any selected service
   file is a stop condition; this cutover adds exactly one new assignment only
   after the absence precondition, so rollback never rewrites an earlier
   configuration.
  - A nonblank complete-DSN assignment in any selected service file is also a
    stop condition before authentication mutation. Empty, quoted-empty, and
    comment-only assignments remain admissible; a literal or expansion-shaped
    value requires a separate configuration migration.
  - The effective environment-file target is derived as the final selected
   service file and must be a nonempty absolute member of that exact ordered
   list before either the new-socket gate or rollback proceeds; a leading,
   middle, or trailing empty member, relative member, or unlisted target stops
   rather than creating a stray assignment.
  - Before fixed inspection or an edit, every selected service file must be
    readable, the selected set must be nonempty, and all `ATLAS_DB_*` keys must
    be canonical uppercase. This makes the `ops` configuration parser and the
    service's case-insensitive `DatabaseConfig` agree without exposing values.
  - The same full-service-file socket-assignment guard gates both adding and
    rollback. It permits restored HBA/identity configuration only after no
    canonical, case-variant, or spacing-variant socket assignment remains.
  - Before socket configuration, the loaded HBA receipt proves that exactly one
    intended local peer-map rule exists and no local rule that could supersede
    it precedes it; only the retained `postgres` recovery rule is admitted
    before the intended rule.
  - The cutover reconciles every client observed in both initial and final
    loopback TCP/replication inventories to a Unix-socket or verified-SCRAM
    receipt, requires every qualifying post-restart `atlas` backend to be a
    Unix socket, requires the known dormant audit monitor's independent
    installed-source/no-override and dry-run receipt, requires no remaining
    final client, and requires an exact loaded `atlas_app | juan-canfield |
    atlas` identity map before HBA replacement.
  - The post-conversion HBA receipt derives every non-local rule whose network
    can cover either loopback endpoint, then requires one exact SCRAM tuple for
    each application/replication IPv4/IPv6 channel, no unexpected endpoint-equal
    host rule, no remaining trust row, no parser error, and no loopback source
    outside the backed-up file. Each receipt carries its own `hba` CTE so it can
    execute independently before an IPv4-only
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
  from every non-local live `pg_hba_file_rules` record and its `file_name` on
  the declared socket and port, not the four examples in this plan. A derived
  network predicate
  includes an address/netmask rule when it can cover either loopback endpoint
  and treats null, unparseable, or family-mismatched forms as candidates. The
  procedure admits exactly four candidates, each one exact
  application/replication IPv4/IPv6 tuple with the declared user, netmask, and
  SCRAM method. A missing, duplicate, broader, TLS-specific, or unlisted rule
  fails closed to no HBA change because an unknown authentication path is less
  safe than a deferred hardening run. Before mutation, every trust rule must
  come from the top-level source the procedure backs up; the post-conversion
  receipt applies the same source check to every loopback rule.
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
- **EOM audit scheduled client — CLOSED / DERIVED before trust removal.**
  Membership is the one repository-owned `eom-write-boundary-audit.service`
  reached by its active hourly timer, its installed script at the documented
  path, any service `EnvironmentFile`, and any unit or user-manager
  `ATLAS_EOM_AUDIT_ATLAS_DSN` assignment.
  The admitted member has an exact effective Python/script argv with no extra
  target-affecting argument, active enabled timer, no `EnvironmentFile`, no
  override in either environment source, a byte-identical installed source only
  after the peer map/rule is live, an explicit-source-owned libpq target, and a
  post-peer no-alert dry run that does not report `COULD NOT MEASURE`. The
  pre-cutover installed source remains restorable until the cutover is closed.
  A clean result (`0`) and a measured boundary breach (`2`) both prove its
  socket transport; an absent timer, stale script, different executable,
  override, unreadable measurement, or other exit stops before HBA replacement.
- **Inherited libpq settings — CLOSED / DERIVED per monitor invocation.**
  Membership is every inherited environment key with the `PG` prefix, derived
  by `psql_environment()` from the process environment. The source-owned
  default removes every member, then derives the only admitted libpq settings
  from its parsed URI. An explicit audit DSN keeps its prior owner-controlled
  inherited libpq/TLS behavior and is rejected from this socket cutover by the
  unit/user-manager override admission; non-`PG` process environment remains
  available to locate and run `psql`.
- **Post-restart Atlas transport — CLOSED / DERIVED before trust removal.**
  Membership is every `pg_stat_activity` client backend for user/database
  `atlas` after the authenticated CRM read. The only admitted receipt has at
  least one member and null `client_addr` for every member; any TCP member or an
  empty set stops the cutover.
- **Service environment-file list — CLOSED / DERIVED for the selected service
  unit.** Membership is the ordered, nonempty absolute `EnvironmentFiles`
  output of `./ops env systemd`. The effective target is derived from the final
  member, then the same list validation and membership predicate gate new-socket
  admission and rollback. An empty, relative, absent, unreadable, or unlisted
  effective configuration file stops fixed inspection and HBA mutation, the
  safer side over inspecting or changing a guessed database target.
- **Pre-existing socket-path assignments — CLOSED / DERIVED before cutover.**
  Membership is every case-variant assignment matching `ATLAS_DB_SOCKET_PATH`
  in the ordered selected `EnvironmentFiles`. The only admitted set is empty;
  unreadable files or any empty, different, or matching-value assignment stop
  before editing. The safe default is a separate configuration migration with
  an exact baseline receipt, not an implicit overwrite or rollback rewrite.
- **Service configuration aliases — CLOSED / DERIVED before inspection.**
  Membership is each parsed `ATLAS_DB_*` assignment in the ordered selected
  `EnvironmentFiles`. Every file must be readable, the set must be nonempty,
  and every key must be canonical uppercase; lower/mixed variants stop before
  fixed inspection or the full-DSN precondition. The safe default is to
  normalize configuration in a separate migration, not to inspect a fallback
  or choose between alias collisions.
- **Complete-DSN admission — CLOSED / DERIVED before HBA mutation.** Membership
  is every canonical `ATLAS_DB_CONNECTION_STRING` assignment in the ordered
  selected `EnvironmentFiles`. The only admitted values are absent, empty,
  quoted-empty, or comment-only; any literal or expansion-shaped value stops
  before backup or reload because complete-DSN precedence could bypass the new
  socket setting. The safe default is a separate configuration migration, not a
  partial peer-authentication edit.
- **Rollback socket assignments — CLOSED / DERIVED before HBA restore.**
  Membership is every case-variant `ATLAS_DB_SOCKET_PATH` assignment in every
  ordered selected `EnvironmentFile`, regardless of value or spacing. The only
  admitted set is empty. A remaining assignment stops before HBA/identity
  restore; the safe default is to remove only the cutover-created assignment or
  use a separate configuration migration with an exact baseline receipt.
- **Peer-HBA precedence — CLOSED / DERIVED before socket configuration.**
  Membership is the ordered loaded `local` HBA rows preceding the exact
  `atlas | atlas | peer | map=atlas_app` row, ordered by
  `pg_hba_file_rules.rule_number` rather than source-local `line_number`. The
  receipt admits exactly one
  intended row and only the `all | postgres | peer` recovery row before it;
  every other preceding local rule stops the cutover for a separate HBA-policy
  migration.
- **Database environment-key aliases — CLOSED / DERIVED for fixed inspection.**
  Membership is every canonical `ATLAS_DB_*` setting consumed by
  `DatabaseConfig`, listed as `DATABASE_CONFIG_KEYS` in `ops`; its case-folded
  form is derived from that set. An exact-uppercase environment key is the only
  inherited database override admitted. Every lower- or mixed-case matching
  alias is removed before `DatabaseConfig` loads the child environment, while
  unrelated keys remain unchanged; this fails closed against inspecting a
  shadowed target or applying a shadowed timeout. The service-pinned helper
  clears every canonical key before it invokes the inspector, so it admits no
  inherited database override. A regression pins the membership equality so a
  new `DatabaseConfig` field cannot silently escape the scrub set.

### Files touched

- `.agent/runbooks/database.md`
- `atlas_brain/storage/config.py`
- `config/eom-write-boundary-audit.service`
- `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md`
- `ops`
- `plans/PR-Postgres-Loopback-Scram-Hardening.md`
- `scripts/backfill_community_buying_stage_defaults.py`
- `scripts/eom_write_boundary_audit.py`
- `scripts/rebuild_blog_charts.py`
- `tests/test_agent_operations_contract.py`
- `tests/test_eom_render_profile.py`
- `tests/test_eom_write_boundary_audit.py`

## Mechanism

`DatabaseConfig` now carries its existing socket path and port into asyncpg,
encoding the direct-DSN host query without changing the raw asyncpg kwargs.
No password is added: the post-merge identity map lets `atlas-api` authenticate
as `atlas` over peer. The map/rule are proved while TCP remains available;
only then do all loopback `trust` entries become `scram-sha-256`.

The fixed inspector keeps its existing source-selection semantics. The runbook
uses its documented explicit environment-file override in the exact service
order; it validates every selected member and the derived final target as a
nonempty absolute path before either admission or rollback, and `ops` discards
case-insensitive aliases for known database settings, so the read-only proof
observes the same database target as `atlas-api.service`.

The independent EOM audit monitor gains a source-owned passwordless socket
default without importing or depending on `atlas-api`. The cutover first
proves the timer's exact target-free command, readable staging inputs, and no
hidden DSN or executable override while the prior monitor remains live over
TCP; after the scoped peer map/rule is loaded it atomically stages the installed
source, retains a rollback copy, and then takes a measurement-only socket
receipt. Its subprocess removes every inherited `PG*` setting only for the
source-owned default, while explicit owner-provided DSNs retain their existing
libpq/TLS inputs and remain excluded from this socket cutover.

The two typed maintenance scripts that bypassed the shared connection seam now
use `DatabaseConfig.connection_kwargs()` with their existing timeout/pool-size
intent, so the same configured socket reaches application and maintenance
callers. The chart rebuild retains its previous no-explicit-timeout pool
behavior while it adopts only the corrected target construction.

The migration-runbook change is documentation only: raw discrepancies stay
visible and retain their forensic nonzero status.

## Intentional

- Raw migration reporting/exit semantics, database role/ownership, and
  non-loopback access remain unchanged.
- `ops` already carries `ATLAS_DB_SOCKET_PATH`; CRM read is production proof,
  not a feature.
- `ops` context-selection precedence remains unchanged; the cutover invokes
  its existing explicit override, while its inspection child now strips
  case-variant inherited database aliases rather than changing normal worktree
  selection behavior.
- The EOM audit's timer, signals, notification delivery, state persistence, and
  explicit DSN override precedence remain unchanged; only its source-owned
  absent-override default moves to the socket after peer authentication is
  staged. The scheduled cutover rejects rather than repurposes unit- or
  manager-level executable overrides.

## Deferred

- Least-privilege database-role redesign (separate application ownership,
  migration DDL, and inspection authority).
- Refactoring historical maintenance scripts into a supported operations runner.
- Cross-host PostgreSQL authentication policy and TLS posture.

Parking predicate: role topology, maintenance-script ownership, or non-local
database access gets a new slice only when it blocks a future capability. None
blocks the socket-peer path.

## Verification

- `./ops test focused tests/test_eom_write_boundary_audit.py
  tests/test_agent_operations_contract.py tests/test_eom_render_profile.py -q`
  — 203 passed, 2 skipped (local). This covers the source-owned default versus
  explicit TLS/libpq behavior, exact timer argv and service/user-manager
  executable-override boundaries, readable and unreadable source-stage inputs,
  post-peer source staging and rollback restoration, nonempty absolute
  service-environment list members and rollback-before-edit, and both typed
  maintenance caller seams.
- `bash scripts/check_ascii_python.sh` — passed (local).
- `python scripts/check_guard_class_closure.py --base origin/main --strict` —
  passed (local advisory guard-closure lint).
- The read-only loaded peer-HBA receipt returned `0|0|0|0|0|0` against the
  pre-cutover state, proving its negative boundary; the synthetic receipt
  returned `accepted|1|0` and `blocked|1|1` for the retained recovery-only and
  competing-local-rule cases, respectively (local).
- `git diff --check` — passed (local).
- `python scripts/sync_pr_plan.py
  plans/PR-Postgres-Loopback-Scram-Hardening.md origin/main --check` — passed
  (local).
- `python scripts/audit_pr_body.py --base-ref origin/main
  tmp/pr-body-postgres-loopback-peer-hardening.md` and the reconciliation/fix
  loop auditors — passed (local).
- Read-only derived-HBA probe against the current `trust` topology returned
  `4|0`: four loopback-network candidates and zero unexpected candidates; a
  synthetic broader subnet, `hostssl` rule, and non-IP form each became a
  candidate while the current remote `/12` rule did not (local).
- The read-only HBA source receipt returned `4|0|0`; the transport predicate
  returned `socket_only|1|1` and `mixed|1|0` against synthetic socket-only and
  mixed candidates (local).
- The exact final post-conversion HBA receipt executed read-only against the
  pre-conversion topology and returned `0|0|0|0|4|4|0|0`; that is intentionally
  not the cutover acceptance value, but proves the independently scoped
  `covers_loopback` query reaches PostgreSQL without a missing-column error
  (local).
- Guarded `scripts/push_pr.sh` local PR review — passed; GitHub owns the full
  unit gate.
- Post-merge: follow the exact peer-socket cutover/rollback procedure in
  `.agent/runbooks/database.md`; do not remove HBA trust until service-pinned
  inspection, CRM, transport, and loaded-HBA proofs all succeed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/runbooks/database.md` | 811 |
| `atlas_brain/storage/config.py` | 13 |
| `config/eom-write-boundary-audit.service` | 2 |
| `docs/MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md` | 20 |
| `ops` | 15 |
| `plans/PR-Postgres-Loopback-Scram-Hardening.md` | 663 |
| `scripts/backfill_community_buying_stage_defaults.py` | 8 |
| `scripts/eom_write_boundary_audit.py` | 59 |
| `scripts/rebuild_blog_charts.py` | 16 |
| `tests/test_agent_operations_contract.py` | 888 |
| `tests/test_eom_render_profile.py` | 122 |
| `tests/test_eom_write_boundary_audit.py` | 112 |
| **Total** | **2729** |

## Diff budget

The complete over-budget rationale is in **Why this slice exists**. This
footer is retained only as the plan's diff-budget marker.
