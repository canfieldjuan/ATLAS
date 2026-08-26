# Database operations

Use this runbook for PostgreSQL connectivity, fixed read-only inspections,
migration status, and database-backed tests.

## Architecture and safe inspection

Atlas uses native PostgreSQL 16 on `127.0.0.1:5433`, normally database `atlas`
and role `atlas`. Root Docker Compose connects to this host service; it does not
create PostgreSQL.

```bash
./ops db status
./ops db inspect connectivity
./ops db migrations
```

`./ops db inspect` accepts only the named, fixed `connectivity` and `migrations`
inspections. It executes them through the project Python and Atlas's
`DatabaseConfig`/asyncpg path, so a complete `ATLAS_DB_CONNECTION_STRING`
retains TLS, socket, and other connection parameters without appearing in a
process argument. The fixed SQL still runs inside a PostgreSQL `READ ONLY`
transaction with command, statement, and lock timeouts.

## Configuration context and precedence

Database inspection deliberately selects one application context instead of
merging every file that `./ops env keys` inventories:

1. `ATLAS_OPS_ENV_FILES`, when explicitly set, selects only those files in the
   listed order.
2. Otherwise, the presence of `.env` or `.env.local` in the current worktree
   selects that pair, matching `DatabaseConfig` in that working directory.
3. Without a worktree pair, the shared-root `.env`/`.env.local` pair is used.
4. Systemd `EnvironmentFiles` are the final live-service fallback.
5. Exported process values override the selected files.

A present worktree context remains selected even when it omits database keys;
do not silently fall through to shared production configuration. Tracked
examples and `.env.tailscale` remain useful for key inventory, but they are not
database value sources. Use `ATLAS_OPS_ENV_FILES` only to select an intentional
context, never to concatenate unrelated environments.

There is intentionally no arbitrary `./ops db query`: PostgreSQL permits
functions with operational side effects inside a `READ ONLY` transaction. A
generic query command is unavailable until Atlas has a privilege-restricted
inspection role. Do not paste customer content or identifiers into chat or
GitHub, and do not substitute the live application role as an ad hoc read role.

## Role-topology evidence preflight

This is a read-only prerequisite for a later least-privilege role cutover. It
does not create roles, grant or revoke privileges, transfer ownership, run a
migration, or restart Atlas. Keep the protected DBA DSN out of `.env`, service
environment files, shell history, Git, chat, and GitHub. The command accepts no
DSN argument and has no apply mode.

1. Record the live/runtime relationship first:

   ```bash
   ./ops deploy status
   git rev-parse HEAD
   ```

   A receipt describes only the target reached by its two connections. If the
   running `atlas-api.service` revision differs from the source revision that a
   future cutover would deploy, record that drift and do not use the receipt to
   authorize a role change. Converge the deployment, then collect a new receipt
   before the cutover review.

2. In a protected, operator-only environment, provide exactly one direct
   PostgreSQL-superuser DSN under
   `ATLAS_DATABASE_ROLE_TOPOLOGY_DBA_DATABASE_URL`. Do not reuse the normal
   `ATLAS_DB_*` application credentials. The normal runtime target continues to
   come only from `DatabaseConfig`; the command rejects a missing DBA value, a
   different database/schema/cluster, a switched `SET ROLE` session, or a
   non-superuser session before it reads the catalog.

3. From the Atlas worktree that supplies the intended runtime configuration,
   run the fixed evidence command:

   ```bash
   python scripts/check_database_role_topology.py
   ```

   It prints a redacted JSON receipt with target labels, role attributes,
   memberships, database/schema ownership, owner summaries, and effective ACL
   summaries (including PostgreSQL defaults and column grants), plus RLS policy
   role bindings. It never prints a password or DSN query values, but role and
   ownership metadata is still operationally sensitive: retain it in the
   protected cutover record rather than pasting it into public issues.

4. A nonzero exit means there is no valid receipt. Fix the configuration or
   target-attestation problem and rerun the read-only preflight; do not work
   around it with `./ops db inspect`, ad hoc SQL, or a role mutation. A later
   reviewed DBA-only slice owns any actual role, grant, ownership, migration,
   or service-credential change.

## Unix-socket peer and loopback SCRAM cutover

This is a production-mutating operation. Perform it only from an owned,
merged-and-deployed hardening slice. A Brain restart can run migration checks;
read the migration section below first. Do not add a database password or a
systemd credential for this cutover: the service is a non-root user service,
and the correct boundary is the operating-system identity already available to
PostgreSQL over its Unix socket.

The source revision for this runbook makes
`DatabaseConfig.connection_kwargs()` honour `ATLAS_DB_SOCKET_PATH`. That is
required before changing the live setting: the generic pool and fixed
`./ops db inspect` path both call that method. The service will connect as the
existing `atlas` PostgreSQL role through `/var/run/postgresql` on port `5433`.
`pg_ident.conf` maps only the actual service OS account to that role, and an
HBA rule scoped to database `atlas` and role `atlas` selects peer
authentication. The existing `postgres` Unix-socket peer rule remains the
break-glass path.

1. Revalidate the exact deployed source, service identity, PostgreSQL paths,
   active local clients, and current health. Record the service
   `EnvironmentFiles` in their printed order. Do not print `.env` values or a
   database URL:

   ```bash
   ./ops deploy status
   ./ops env systemd
   id -un
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SHOW hba_file; SHOW ident_file; SHOW unix_socket_directories;"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT backend_start, application_name, usename, datname, backend_type, \
             client_addr, state \
      FROM pg_stat_activity \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT backend_start, application_name, usename, client_addr, state, sync_state \
      FROM pg_stat_replication \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT map_name, sys_name, pg_username, error \
      FROM pg_ident_file_mappings \
      WHERE map_name = 'atlas_app' \
      ORDER BY line_number;"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT rule_number, file_name, line_number, type, database, user_name, \
             address, netmask, auth_method \
      FROM pg_hba_file_rules \
      WHERE auth_method = 'trust' \
      ORDER BY rule_number;"
   if [ "$(sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT count(*) FILTER (WHERE auth_method = 'trust'), \
             count(*) FILTER ( \
               WHERE auth_method = 'trust' \
                 AND file_name IS DISTINCT FROM '/etc/postgresql/16/main/pg_hba.conf' \
             ), \
             count(*) FILTER (WHERE error IS NOT NULL) \
      FROM pg_hba_file_rules;")" != '4|0|0' ]; then
     printf '%s\n' 'trust HBA source receipt was not 4|0|0; do not edit the top-level file' >&2
     exit 1
   fi
   ```

   Record every row from those two inventory queries in the cutover record as
   the **initial loopback-client inventory**. Keep its source
   (`pg_stat_activity` or `pg_stat_replication`), `backend_start`, application,
   user, database or replication state, address, and state/sync state. These
   are not diagnostic snapshots: every client identity observed here must later
   receive a Unix-socket or separately owner-verified SCRAM reconnect receipt
   before HBA conversion, even if its TCP session subsequently disappears. Do
   not infer that two rows are the same client from an application name or role
   alone; the owner must bind each initial row to its later receipt in the record.

   For the current single-user deployment, the service account is
   `juan-canfield`, the HBA file is `/etc/postgresql/16/main/pg_hba.conf`, the
   identity-map file is `/etc/postgresql/16/main/pg_ident.conf`, and the socket
   directory is `/var/run/postgresql`. Stop if the live results differ; derive
   the map/rules from the returned topology rather than copying these values
   blindly. The final query must show exactly four `host` trust rows: two
   `{all}` application rows and two `{replication}` rows, one for each of
   `127.0.0.1` and `::1`. The immediately preceding HBA source receipt must be
   `4|0|0`: four trust rows, none sourced outside
   `/etc/postgresql/16/main/pg_hba.conf`, and no parser error. Stop if any
   value differs; this runbook does not generalize a different HBA topology or
   edit an included HBA file. The `atlas_app` identity-map query must return no
   rows before editing: do not reuse a pre-existing map name.

   `eom-write-boundary-audit.timer` is a repository-owned hourly one-shot
   database client. It is normally absent from both live inventory queries, so
   its absence is not evidence that it has no TCP dependency. Before any HBA
   backup or authentication edit, admit its exact target-free timer command,
   active timer/service identity, and no unit/user-manager
   `ATLAS_EOM_AUDIT_ATLAS_DSN` override or `EnvironmentFile`. Do **not** require
   the installed source to be the new socket default yet: it would run before
   the peer map is loaded. After the map/rule succeeds, this procedure stages
   the current source atomically, retains the old installed copy for rollback,
   then obtains the socket receipt. The source-owned default clears inherited
   libpq settings; an explicit override retains its owner-managed TLS/libpq
   behavior and needs its own verified socket or SCRAM migration. The checks
   retain environment text only in shell variables and never print it.

   Set `SERVICE_ENV_FILES` to the absolute `EnvironmentFiles` paths printed by
   `./ops env systemd`, joined with `:` in that same order. Use this shell-local
   helper for every fixed inspection in this cutover. It selects the service's
   configuration without printing values. The inspector also removes every
   case variant of every Atlas database setting it can consume, so an inherited
   lower- or mixed-case alias cannot override those files:

   ```bash
   EOM_AUDIT_REPOSITORY_ROOT="$(git rev-parse --show-toplevel)" || exit 1
   EOM_AUDIT_SOURCE="$EOM_AUDIT_REPOSITORY_ROOT/scripts/eom_write_boundary_audit.py"
   EOM_AUDIT_INSTALLED="$HOME/.local/bin/eom-write-boundary-audit.py"
   EOM_AUDIT_PRE_PEER_SOURCE="$EOM_AUDIT_INSTALLED.pre-atlas-peer"
   EOM_AUDIT_SERVICE='eom-write-boundary-audit.service'
   EOM_AUDIT_TIMER='eom-write-boundary-audit.timer'
   SERVICE_ENV_FILES='/absolute/first.service.env:/absolute/second.service.env'
   EFFECTIVE_DB_ENV_FILE="${SERVICE_ENV_FILES##*:}"
   eom_audit_environment_has_dsn_override() {
     case "$1" in
       *'ATLAS_EOM_AUDIT_ATLAS_DSN='*) return 0 ;;
       *) return 1 ;;
     esac
   }
   eom_audit_environment_has_psql_bin_override() {
     case "$1" in
       *'ATLAS_EOM_AUDIT_PSQL_BIN='*) return 0 ;;
       *) return 1 ;;
     esac
   }
   eom_audit_require_unit_contract() {
     if ! systemctl --user is-enabled --quiet "$EOM_AUDIT_TIMER" \
       || ! systemctl --user is-active --quiet "$EOM_AUDIT_TIMER"; then
       printf '%s\n' 'EOM audit timer is not enabled and active; do not remove loopback trust' >&2
       return 1
     fi
     eom_audit_exec_start="$(systemctl --user show "$EOM_AUDIT_SERVICE" \
       --property=LoadState,ExecStart --value)" || {
       printf '%s\n' 'could not inspect EOM audit service identity; do not remove loopback trust' >&2
       return 1
     }
     case "$eom_audit_exec_start" in
       *'argv[]='*'argv[]='*)
         printf '%s\n' 'EOM audit service has multiple effective commands; do not remove loopback trust' >&2
         return 1
         ;;
       *'argv[]='*) ;;
       *)
         printf '%s\n' 'could not parse EOM audit service command; do not remove loopback trust' >&2
         return 1
         ;;
     esac
     eom_audit_argv="${eom_audit_exec_start#*argv[]=}"
     eom_audit_argv="${eom_audit_argv%% ;*}"
     if [ "$eom_audit_argv" != "/usr/bin/python3 $EOM_AUDIT_INSTALLED" ]; then
       printf '%s\n' 'EOM audit service command is not the admitted target-free argv; do not remove loopback trust' >&2
       return 1
     fi
     eom_audit_environment_files="$(systemctl --user show "$EOM_AUDIT_SERVICE" \
       --property=EnvironmentFiles --value)" || {
       printf '%s\n' 'could not inspect EOM audit service environment files; do not remove loopback trust' >&2
       return 1
     }
     if [ -n "$eom_audit_environment_files" ]; then
       printf '%s\n' 'EOM audit service uses an EnvironmentFile; inventory its transport separately before cutover' >&2
       return 1
     fi
     eom_audit_service_environment="$(systemctl --user show "$EOM_AUDIT_SERVICE" \
       --property=Environment --value)" || {
       printf '%s\n' 'could not inspect EOM audit service environment; do not remove loopback trust' >&2
       return 1
     }
     eom_audit_manager_environment="$(systemctl --user show-environment)" || {
       printf '%s\n' 'could not inspect EOM audit manager environment; do not remove loopback trust' >&2
       return 1
     }
     if eom_audit_environment_has_dsn_override "$eom_audit_service_environment" \
       || eom_audit_environment_has_dsn_override "$eom_audit_manager_environment"; then
       printf '%s\n' 'EOM audit DSN override found; migrate and prove it separately before cutover' >&2
       return 1
     fi
     if eom_audit_environment_has_psql_bin_override "$eom_audit_service_environment" \
       || eom_audit_environment_has_psql_bin_override "$eom_audit_manager_environment"; then
       printf '%s\n' 'EOM audit psql executable override found; remove it before cutover' >&2
       return 1
     fi
   }
   eom_audit_require_stage_inputs() {
     if ! test -f "$EOM_AUDIT_SOURCE" || ! test -r "$EOM_AUDIT_SOURCE" \
       || ! test -f "$EOM_AUDIT_INSTALLED" || ! test -r "$EOM_AUDIT_INSTALLED"; then
       printf '%s\n' 'EOM audit source or installed script is unreadable; do not remove loopback trust' >&2
       return 1
     fi
   }
   eom_audit_require_socket_default() {
     eom_audit_require_unit_contract || return 1
     eom_audit_require_stage_inputs || return 1
     if ! cmp -s "$EOM_AUDIT_SOURCE" "$EOM_AUDIT_INSTALLED"; then
       printf '%s\n' 'installed EOM audit script does not match deployed source; stage it after peer authentication' >&2
       return 1
     fi
   }
   eom_audit_stage_socket_source() {
     if test -e "$EOM_AUDIT_PRE_PEER_SOURCE"; then
       printf '%s\n' 'pre-peer EOM audit source backup already exists; inspect it before retrying' >&2
       return 1
     fi
     eom_audit_require_stage_inputs || return 1
     if ! cp --preserve=mode,ownership,timestamps "$EOM_AUDIT_INSTALLED" "$EOM_AUDIT_PRE_PEER_SOURCE"; then
       printf '%s\n' 'could not preserve the pre-peer EOM audit source; do not replace it' >&2
       return 1
     fi
     eom_audit_stage="$(mktemp "$EOM_AUDIT_INSTALLED.stage.XXXXXX")" || {
       printf '%s\n' 'could not create staged EOM audit source; do not replace it' >&2
       return 1
     }
     if ! install -m 755 "$EOM_AUDIT_SOURCE" "$eom_audit_stage" \
       || ! mv -f "$eom_audit_stage" "$EOM_AUDIT_INSTALLED"; then
       rm -f -- "$eom_audit_stage"
       cp --preserve=mode,ownership,timestamps "$EOM_AUDIT_PRE_PEER_SOURCE" "$EOM_AUDIT_INSTALLED" || return 1
       printf '%s\n' 'could not stage the socket-default EOM audit source; restored the prior source' >&2
       return 1
     fi
     eom_audit_require_socket_default
   }
   eom_audit_restore_pre_peer_source() {
     if ! test -e "$EOM_AUDIT_PRE_PEER_SOURCE"; then
       return 0
     fi
     cp --preserve=mode,ownership,timestamps "$EOM_AUDIT_PRE_PEER_SOURCE" "$EOM_AUDIT_INSTALLED"
   }
   service_db_has_case_variant_key() {
     sudo awk '
       BEGIN { found = 0 }
       {
         line = $0
         sub(/^[[:space:]]*/, "", line)
         sub(/^export[[:space:]]+/, "", line)
         if (line ~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/) {
           key = line
           sub(/[[:space:]]*=.*/, "", key)
           if (tolower(key) ~ /^atlas_db_/ && key != toupper(key)) {
             found = 1
             exit
           }
         }
       }
       END { exit(found ? 0 : 1) }
     ' "$1"
   }
   service_db_has_socket_assignment() {
     sudo grep -Eqi '^[[:space:]]*(export[[:space:]]+)?ATLAS_DB_SOCKET_PATH[[:space:]]*=' "$1"
   }
   service_db_has_nonempty_connection_string_assignment() {
     sudo awk '
       BEGIN { found = 0 }
       {
         line = $0
         sub(/^[[:space:]]*/, "", line)
         sub(/^export[[:space:]]+/, "", line)
         if (line !~ /^ATLAS_DB_CONNECTION_STRING[[:space:]]*=/) {
           next
         }
         sub(/^ATLAS_DB_CONNECTION_STRING[[:space:]]*=/, "", line)
         sub(/^[[:space:]]*/, "", line)
         if (line ~ /^#/) {
           line = ""
         } else {
           sub(/[[:space:]]+#.*/, "", line)
           sub(/[[:space:]]*$/, "", line)
         }
         if (line == "" || line == "\"\"" || line == sprintf("%c%c", 39, 39)) {
           next
         }
         found = 1
         exit
       }
       END { exit(found ? 0 : 1) }
     ' "$1"
   }
   service_db_require_canonical_keys() {
     if [ -z "$SERVICE_ENV_FILES" ]; then
       printf '%s\n' 'no service EnvironmentFiles selected; do not inspect a fallback context' >&2
       return 1
     fi
     saw_service_env=0
     while IFS= read -r service_env_file; do
       case "$service_env_file" in
         /*) ;;
         *)
           printf '%s\n' 'service EnvironmentFile must be a nonempty absolute path; do not inspect it' >&2
           return 1
           ;;
       esac
       saw_service_env=1
       if ! sudo test -r "$service_env_file"; then
         printf '%s\n' 'service EnvironmentFile is unreadable; do not inspect it' >&2
         return 1
       fi
       if service_db_has_case_variant_key "$service_env_file"; then
         printf '%s\n' 'case-variant ATLAS_DB_* key found; normalize it before inspection' >&2
         return 1
       else
         case_variant_status=$?
         if [ "$case_variant_status" -ne 1 ]; then
           printf '%s\n' 'could not inspect service EnvironmentFile keys; do not inspect it' >&2
           return 1
         fi
       fi
     done < <(printf '%s\n' "$SERVICE_ENV_FILES" | tr ':' '\n')
     if [ "$saw_service_env" -ne 1 ]; then
       printf '%s\n' 'no service EnvironmentFiles selected; do not inspect a fallback context' >&2
       return 1
     fi
   }
   service_db_require_no_socket_assignments() {
     service_db_require_canonical_keys || return 1
     while IFS= read -r service_env_file; do
       [ -n "$service_env_file" ] || continue
       if service_db_has_socket_assignment "$service_env_file"; then
         printf '%s\n' 'ATLAS_DB_SOCKET_PATH assignment remains; do not continue' >&2
         return 1
       else
         socket_status=$?
         if [ "$socket_status" -ne 1 ]; then
           printf '%s\n' 'could not inspect service EnvironmentFile socket assignments; do not continue' >&2
           return 1
         fi
       fi
     done < <(printf '%s\n' "$SERVICE_ENV_FILES" | tr ':' '\n')
   }
   service_db_require_effective_env_file() {
     service_db_require_canonical_keys || return 1
     case "$EFFECTIVE_DB_ENV_FILE" in
       /*) ;;
       *)
         printf '%s\n' 'effective database environment file must be a nonempty absolute path' >&2
         return 1
         ;;
     esac
     case ":$SERVICE_ENV_FILES:" in
       *":$EFFECTIVE_DB_ENV_FILE:"*) ;;
       *) printf '%s\n' 'effective database environment file is not a service EnvironmentFile' >&2; return 1 ;;
     esac
   }
   service_db_require_new_socket_configuration() {
     service_db_require_effective_env_file || return 1
     service_db_require_no_socket_assignments || return 1
     while IFS= read -r service_env_file; do
       [ -n "$service_env_file" ] || continue
       if service_db_has_nonempty_connection_string_assignment "$service_env_file"; then
         printf '%s\n' 'nonblank ATLAS_DB_CONNECTION_STRING assignment found; use a separate configuration migration' >&2
         return 1
       else
         connection_string_status=$?
         if [ "$connection_string_status" -ne 1 ]; then
           printf '%s\n' 'could not inspect service EnvironmentFile connection-string assignments; do not continue' >&2
           return 1
         fi
       fi
     done < <(printf '%s\n' "$SERVICE_ENV_FILES" | tr ':' '\n')
   }
   service_db_inspect() {
     service_db_require_canonical_keys || return 1
     env -u ATLAS_DB_ENABLED \
       -u ATLAS_DB_CONNECTION_STRING \
       -u ATLAS_DB_HOST \
       -u ATLAS_DB_PORT \
       -u ATLAS_DB_DATABASE \
       -u ATLAS_DB_USER \
       -u ATLAS_DB_PASSWORD \
       -u ATLAS_DB_MIN_POOL_SIZE \
       -u ATLAS_DB_MAX_POOL_SIZE \
       -u ATLAS_DB_SOCKET_PATH \
       -u ATLAS_DB_CONNECT_TIMEOUT \
       -u ATLAS_DB_COMMAND_TIMEOUT \
       ATLAS_OPS_ENV_FILES="$SERVICE_ENV_FILES" \
       ./ops db inspect connectivity
   }
   eom_audit_require_unit_contract || exit 1
   eom_audit_require_stage_inputs || exit 1
   service_db_require_new_socket_configuration || exit 1
   service_db_inspect
   ```

   Do not run a bare `./ops db inspect connectivity` in this procedure: a
   worktree `.env` takes precedence over the service files. If the service has
   no readable `EnvironmentFiles`, stop rather than substituting a worktree or
   shared configuration. The admission command must succeed before the first
   backup, HBA/identity edit, or PostgreSQL reload. It rejects an empty
   service-file set, any case-variant `ATLAS_DB_*` key, every existing socket
   assignment, and every nonblank complete-DSN assignment without printing a
   value. The dormant monitor admission and source-stage input receipt must
   also succeed before the first backup, HBA/identity edit, or PostgreSQL
   reload. A complete DSN deliberately takes precedence over
   `ATLAS_DB_SOCKET_PATH`; do not remove or rewrite one as part of this
   procedure. That needs a separate configuration-migration slice.

2. Only after the timer-contract, source-stage, and service-configuration
   receipts succeed, preserve the two PostgreSQL authentication files before
   editing them:

   ```bash
   if sudo test -e /etc/postgresql/16/main/pg_hba.conf.pre-atlas-peer \
     || sudo test -e /etc/postgresql/16/main/pg_ident.conf.pre-atlas-peer; then
     printf '%s\n' 'existing peer-cutover backup found; inspect it before retrying' >&2
     exit 1
   fi
   sudo cp --preserve=mode,ownership,timestamps /etc/postgresql/16/main/pg_hba.conf \
     /etc/postgresql/16/main/pg_hba.conf.pre-atlas-peer
   sudo cp --preserve=mode,ownership,timestamps /etc/postgresql/16/main/pg_ident.conf \
     /etc/postgresql/16/main/pg_ident.conf.pre-atlas-peer
   ```

   Then add this exact map with `sudoedit /etc/postgresql/16/main/pg_ident.conf`:

   ```text
   atlas_app    juan-canfield    atlas
   ```

   Add this exact rule with `sudoedit /etc/postgresql/16/main/pg_hba.conf`
   **above** the generic `local   all   all   peer` rule. Do not change the
   existing `local   all   postgres   peer` rule:

   ```text
   local   atlas   atlas                         peer map=atlas_app
   ```

   ```bash
   sudo systemctl reload postgresql@16-main
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -c \
     "WITH intended_peer_rule AS ( \
        SELECT rule_number \
        FROM pg_hba_file_rules \
        WHERE error IS NULL \
          AND type = 'local' \
          AND database = ARRAY['atlas']::text[] \
          AND user_name = ARRAY['atlas']::text[] \
          AND auth_method = 'peer' \
          AND options = ARRAY['map=atlas_app']::text[] \
      ), \
      preceding_local_rules AS ( \
        SELECT rules.rule_number \
        FROM pg_hba_file_rules AS rules \
        JOIN intended_peer_rule AS intended \
          ON rules.rule_number < intended.rule_number \
        WHERE rules.error IS NULL \
          AND rules.type = 'local' \
          AND NOT ( \
            rules.database = ARRAY['all']::text[] \
            AND rules.user_name = ARRAY['postgres']::text[] \
            AND rules.auth_method = 'peer' \
            AND COALESCE(rules.options, ARRAY[]::text[]) = ARRAY[]::text[] \
          ) \
      ) \
      SELECT \
        (SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL), \
        count(*) FILTER ( \
          WHERE map_name = 'atlas_app' \
            AND sys_name = 'juan-canfield' \
            AND pg_username = 'atlas' \
        ), \
        count(*) FILTER (WHERE map_name = 'atlas_app'), \
        count(*) FILTER (WHERE error IS NOT NULL), \
        (SELECT count(*) FROM intended_peer_rule), \
        (SELECT count(*) FROM preceding_local_rules) \
      FROM pg_ident_file_mappings;"
   ```

   The final query must return `0|1|1|0|1|0`: no HBA parser error, exactly one
   intended `atlas_app | juan-canfield | atlas` mapping, no additional
   `atlas_app` mapping, no identity-map error, exactly one loaded
   `local atlas atlas peer map=atlas_app` rule, and no preceding local rule in
   loaded `rule_number` order
   other than the retained `local all postgres peer` recovery rule. Otherwise
   restore both saved files and reload PostgreSQL before continuing. `atlas-api`
   still uses TCP at this point, so a valid scoped peer rule can be proved
   without removing the existing path. The separately admitted audit monitor is
   not yet its socket receipt; obtain that receipt after the application restart
   and before loopback trust is replaced.

3. The pre-mutation admission receipt derives and validates
   `EFFECTIVE_DB_ENV_FILE` as the final selected service file before any
   authentication edit. This procedure supports only a
   **new** socket setting. Do not replace, remove, or preserve an existing
   socket assignment or complete DSN here: its original behavior needs a
   separate configuration migration with an exact restoration receipt.

   Add exactly this non-secret setting with `sudoedit "$EFFECTIVE_DB_ENV_FILE"`;
   do not copy or print the rest of that file:

   ```bash
   sudoedit "$EFFECTIVE_DB_ENV_FILE"
   ```

   ```text
   ATLAS_DB_SOCKET_PATH=/var/run/postgresql
   ```

   Define this shell-local rollback helper before the restart. It is the only
   rollback path after the socket setting exists; when it opens the editor,
   remove the added `ATLAS_DB_SOCKET_PATH` assignment entirely without changing
   any other database key. The helper refuses to restore authentication while
   any case-variant socket assignment remains in any selected service file:

   ```bash
   rollback_peer_cutover() {
     eom_audit_restore_pre_peer_source || return 1
     service_db_require_effective_env_file || return 1
     sudoedit "$EFFECTIVE_DB_ENV_FILE" || return 1
     service_db_require_no_socket_assignments || return 1
     sudo cp --preserve=mode,ownership,timestamps /etc/postgresql/16/main/pg_hba.conf.pre-atlas-peer \
       /etc/postgresql/16/main/pg_hba.conf || return 1
     sudo cp --preserve=mode,ownership,timestamps /etc/postgresql/16/main/pg_ident.conf.pre-atlas-peer \
       /etc/postgresql/16/main/pg_ident.conf || return 1
     sudo systemctl reload postgresql@16-main || return 1
     if [ "$(sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c 'SELECT count(*) FILTER (WHERE auth_method = '\''trust'\''), count(*) FILTER (WHERE auth_method = '\''trust'\'' AND file_name IS DISTINCT FROM '\''/etc/postgresql/16/main/pg_hba.conf'\''), count(*) FILTER (WHERE error IS NOT NULL) FROM pg_hba_file_rules;')" != '4|0|0' ]; then
       printf '%s\n' 'restored HBA receipt was not 4|0|0; do not restart atlas-api' >&2
       return 1
     fi
     systemctl --user restart atlas-api.service || return 1
     service_db_inspect
   }
   ```

   Stage the socket-default monitor only after the previous peer-map/HBA receipt
   succeeded and the rollback helper exists. If staging cannot preserve or
   install the source, invoke that helper so the old source is restored before
   the peer authentication files are restored:

   ```bash
   eom_audit_stage_socket_source || {
     printf '%s\n' 'could not stage the EOM audit socket source; rollback required' >&2
     rollback_peer_cutover
     exit 1
   }
   ```

   Restart the user service, then prove that the application has changed its
   real connection path before any TCP trust rule is removed:

   ```bash
   systemctl --user restart atlas-api.service
   ./ops deploy status
   ```

   Perform one read-only authenticated EOM CRM Contacts-page refresh. A health
   response alone is insufficient: the generic pool can initialize lazily, so
   the CRM read proves the application data path. Immediately after that
   refresh, before running the fixed inspector, observe the application's
   PostgreSQL backends:

   ```bash
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT backend_start, usename, datname, \
             COALESCE(client_addr::text, '<unix>'), state \
      FROM pg_stat_activity \
      WHERE usename = 'atlas' \
        AND datname = 'atlas' \
        AND backend_type = 'client backend' \
      ORDER BY backend_start;"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "WITH atlas_backends AS ( \
        SELECT client_addr \
        FROM pg_stat_activity \
        WHERE usename = 'atlas' \
          AND datname = 'atlas' \
          AND backend_type = 'client backend' \
      ) \
      SELECT \
        CASE WHEN count(*) > 0 THEN 1 ELSE 0 END, \
        CASE WHEN count(*) FILTER (WHERE client_addr IS NULL) = count(*) \
          THEN 1 ELSE 0 END \
      FROM atlas_backends;"
   service_db_inspect
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT application_name, usename, client_addr, state \
      FROM pg_stat_activity \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT application_name, usename, client_addr, state, sync_state \
      FROM pg_stat_replication \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   EOM_AUDIT_DRY_RUN_DIR="$(mktemp -d)" || {
     printf '%s\n' 'could not create EOM audit dry-run state directory; rollback required' >&2
     rollback_peer_cutover
     exit 1
   }
   EOM_AUDIT_DRY_RUN_OUTPUT="$(env -u ATLAS_EOM_AUDIT_ATLAS_DSN \
     -u ATLAS_EOM_AUDIT_PSQL_BIN "$EOM_AUDIT_INSTALLED" \
     --state-dir "$EOM_AUDIT_DRY_RUN_DIR" \
     --ntfy-topic cutover-dry-run \
     --no-alert)"
   EOM_AUDIT_DRY_RUN_STATUS=$?
   rm -rf -- "$EOM_AUDIT_DRY_RUN_DIR" || {
     printf '%s\n' 'could not remove EOM audit dry-run state directory; rollback required' >&2
     rollback_peer_cutover
     exit 1
   }
   case "$EOM_AUDIT_DRY_RUN_OUTPUT" in
     *'COULD NOT MEASURE'*)
       printf '%s\n' 'EOM audit could not measure through its default socket route; rollback required' >&2
       rollback_peer_cutover
       exit 1
       ;;
   esac
   case "$EOM_AUDIT_DRY_RUN_STATUS" in
     0|2) ;;
     *)
       printf '%s\n' 'EOM audit dry run exited unexpectedly; rollback required' >&2
       rollback_peer_cutover
       exit 1
       ;;
   esac
   ```

   The immediate transport receipt must return `1|1`: the CRM read created at
   least one qualifying `atlas`/`atlas` client backend, and every qualifying
   backend has a null `client_addr` (a Unix socket). Do not accept one socket
   row while another qualifying backend is TCP; that would not prove the
   restarted application uses the socket. The EOM audit dry run must not report
   `COULD NOT MEASURE`; its `--no-alert` invocation intentionally leaves no
   notification or persistent alert-state change. Exit `0` means the audit was
   clean and exit `2` means it measured a boundary breach; either is a transport
   receipt. Any other exit is a failure. Compare the rows with the loopback
   clients from step 1. The fixed inspection must also succeed only after that
   transport proof. If any proof fails, remove only the exact non-secret setting
   just added, restore the two saved PostgreSQL files, reload PostgreSQL, and
   restart `atlas-api.service`.

   Record those two later queries as the **final loopback-client inventory**.
   Before step 4, reconcile every client identity in the union of the initial
   and final inventories. Each must have either a post-restart Unix-socket
   backend receipt or a separate, owner-verified SCRAM reconnect receipt for
   that client. Record the audit monitor's source-match/no-override admission
   and dry-run transport receipt alongside that union: its scheduled absence
   from both snapshots is admissible only with both receipts. A TCP session that
   disappeared between snapshots is not a receipt; stop and obtain one before
   conversion. Both final inventories must also be empty. A remaining loopback
   TCP client, including a replication client, is a stop condition even if it
   has a future SCRAM receipt: move that client to the Unix socket or complete
   its separate migration and disconnect it before re-running this cutover. Do
   not create a new credential or remove HBA trust while any row or receipt
   remains unresolved.

4. Only after the socket-peer proof succeeds, replace **all four** loopback
   TCP `trust` entries in `pg_hba.conf` with `scram-sha-256` using `sudoedit`:

   ```text
   host    all             all             127.0.0.1/32            scram-sha-256
   host    all             all             ::1/128                 scram-sha-256
   host    replication     all             127.0.0.1/32            scram-sha-256
   host    replication     all             ::1/128                 scram-sha-256
   ```

   Do not alter any non-loopback rule. Reload PostgreSQL, then verify the
   loaded rule set before repeating the service, fixed-inspection, and
   authenticated CRM proofs from step 3:

   ```bash
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "WITH hba AS ( \
        SELECT *, \
          CASE \
            WHEN address IS NULL OR netmask IS NULL THEN TRUE \
            WHEN NOT pg_input_is_valid(address, 'inet') \
              OR NOT pg_input_is_valid(netmask, 'inet') THEN TRUE \
            WHEN family(address::inet) <> family(netmask::inet) THEN TRUE \
            WHEN family(address::inet) = 4 THEN \
              host(address::inet & netmask::inet) \
                = host('127.0.0.1'::inet & netmask::inet) \
            WHEN family(address::inet) = 6 THEN \
              host(address::inet & netmask::inet) \
                = host('::1'::inet & netmask::inet) \
            ELSE TRUE \
          END AS covers_loopback \
        FROM pg_hba_file_rules \
      ) \
      SELECT count(*) FILTER ( \
        WHERE type <> 'local' \
          AND covers_loopback \
      ) AS loopback_network_rule_count \
      FROM hba;"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "WITH hba AS ( \
        SELECT *, \
          CASE \
            WHEN address IS NULL OR netmask IS NULL THEN TRUE \
            WHEN NOT pg_input_is_valid(address, 'inet') \
              OR NOT pg_input_is_valid(netmask, 'inet') THEN TRUE \
            WHEN family(address::inet) <> family(netmask::inet) THEN TRUE \
            WHEN family(address::inet) = 4 THEN \
              host(address::inet & netmask::inet) \
                = host('127.0.0.1'::inet & netmask::inet) \
            WHEN family(address::inet) = 6 THEN \
              host(address::inet & netmask::inet) \
                = host('::1'::inet & netmask::inet) \
            ELSE TRUE \
          END AS covers_loopback \
        FROM pg_hba_file_rules \
      ) \
      SELECT \
        count(*) FILTER ( \
          WHERE type = 'host' \
            AND database = ARRAY['all']::text[] \
            AND user_name = ARRAY['all']::text[] \
            AND address = '127.0.0.1' \
            AND netmask = '255.255.255.255' \
            AND auth_method = 'scram-sha-256' \
        ) AS application_ipv4_scram_rule, \
        count(*) FILTER ( \
          WHERE type = 'host' \
            AND database = ARRAY['all']::text[] \
            AND user_name = ARRAY['all']::text[] \
            AND address = '::1' \
            AND netmask = 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff' \
            AND auth_method = 'scram-sha-256' \
        ) AS application_ipv6_scram_rule, \
        count(*) FILTER ( \
          WHERE type = 'host' \
            AND database = ARRAY['replication']::text[] \
            AND user_name = ARRAY['all']::text[] \
            AND address = '127.0.0.1' \
            AND netmask = '255.255.255.255' \
            AND auth_method = 'scram-sha-256' \
        ) AS replication_ipv4_scram_rule, \
        count(*) FILTER ( \
          WHERE type = 'host' \
            AND database = ARRAY['replication']::text[] \
            AND user_name = ARRAY['all']::text[] \
            AND address = '::1' \
            AND netmask = 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff' \
            AND auth_method = 'scram-sha-256' \
        ) AS replication_ipv6_scram_rule, \
        count(*) FILTER ( \
          WHERE type = 'host' \
            AND address IN ('127.0.0.1', '::1') \
            AND NOT ( \
              (database = ARRAY['all']::text[] \
               AND user_name = ARRAY['all']::text[] \
               AND address = '127.0.0.1' \
               AND netmask = '255.255.255.255' \
               AND auth_method = 'scram-sha-256') \
              OR (database = ARRAY['all']::text[] \
                  AND user_name = ARRAY['all']::text[] \
                  AND address = '::1' \
                  AND netmask = 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff' \
                  AND auth_method = 'scram-sha-256') \
              OR (database = ARRAY['replication']::text[] \
                  AND user_name = ARRAY['all']::text[] \
                  AND address = '127.0.0.1' \
                  AND netmask = '255.255.255.255' \
                  AND auth_method = 'scram-sha-256') \
              OR (database = ARRAY['replication']::text[] \
                  AND user_name = ARRAY['all']::text[] \
                  AND address = '::1' \
                  AND netmask = 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff' \
                  AND auth_method = 'scram-sha-256') \
            ) \
        ) AS unexpected_loopback_host_rules, \
        count(*) FILTER (WHERE auth_method = 'trust') AS remaining_trust_rules, \
        count(*) FILTER (WHERE error IS NOT NULL) AS hba_errors, \
        count(*) FILTER ( \
          WHERE type <> 'local' \
            AND covers_loopback \
            AND file_name IS DISTINCT FROM '/etc/postgresql/16/main/pg_hba.conf' \
        ) AS loopback_rules_outside_backup \
      FROM hba;"
   ```

   The first query must return `4`: it derives every non-local HBA rule whose
   network can cover either loopback endpoint. This count fails closed for a
   broader subnet, `hostssl`/other host type, an unparseable address form, or a
   family mismatch. The second query must return `1|1|1|1|0|0|0|0`: one exact
   application IPv4 rule, one exact application IPv6 rule, one exact replication
   IPv4 rule, one exact replication IPv6 rule, no unexpected endpoint-equal host
   rule, no trust rule, no parser error, and no loopback rule sourced outside
   the backed-up top-level HBA file. If either receipt differs, do not proceed:
   run `rollback_peer_cutover` immediately so the restored HBA/ident files are
   reloaded before the procedure stops. Then repeat the service,
   `service_db_inspect`, and authenticated CRM proofs from step 3.

5. Verify both sides of the new boundary. The passwordless TCP probe must fail;
   the retained `postgres` socket peer probe must succeed:

   ```bash
   if env -u PGPASSWORD -u ATLAS_DB_PASSWORD PGPASSFILE=/dev/null \
     psql -w -h 127.0.0.1 -p 5433 -U atlas -d atlas -Atc 'SELECT 1'; then
     printf '%s\n' 'unexpected passwordless loopback TCP access' >&2
     rollback_peer_cutover
     exit 1
   fi
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -Atc \
     'SELECT current_user'
   ```

   The second command must print `postgres`. The first must reject
   authentication without prompting. Do not use an existing `.pgpass` file or
   a secret-bearing shell environment as a substitute test.

6. `rollback_peer_cutover` is mandatory when the loaded-HBA receipt, the
   passwordless TCP probe, or any later proof fails after HBA conversion. It
   removes only the added socket setting through `sudoedit`, restores and
   reloads HBA/ident, proves the restored `4|0|0` HBA receipt, then restarts
   `atlas-api.service` and re-runs fixed inspection. After it succeeds, re-run
   the CRM read before declaring rollback complete. Do not edit roles,
   passwords, migration ledger rows, or database data as part of rollback.

`service_db_inspect` remains fixed-query-only. It explicitly selects the same
service configuration as the application and uses a `READ ONLY` transaction.
This cutover does not create a generic command runner or expose a credential.

## Migrations

Atlas does not use Alembic even though an `alembic` executable is installed.
The canonical runner is `atlas_brain.storage.migrations.run_migrations`, backed
by versioned SQL under `atlas_brain/storage/migrations/` and a
`schema_migrations` ledger.

The full FastAPI lifespan initializes the pool and invokes the migration check
at startup. The runner holds a PostgreSQL advisory lock, re-snapshots the ledger
under that lock, and applies pending files. Therefore a service restart is also
a potential schema mutation.

Do not manually run the full chain against a fresh database. The runner's code
documents that migrations from `076` onward depend on an out-of-band
`product_metadata` table that no packaged migration creates. Components that
need one later prerequisite use the runner's bounded `only` mode. Never roll
back or edit the live migration ledger during discovery.

## Database-backed tests

CI creates disposable PostgreSQL 16 service databases and supplies one of:

- `ATLAS_MIGRATION_TEST_DATABASE_URL`
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL`
- `ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_TEST_DATABASE_URL`

For local work, create or obtain a disposable database outside `./ops`, verify
its exact name/host, then export exactly one matching test URL and acknowledge
the boundary. Integration mode rejects both zero and multiple active canonical
URLs before pytest starts:

```bash
export ATLAS_MIGRATION_TEST_DATABASE_URL='postgresql://.../disposable_db'
export ATLAS_CONFIRM_DISPOSABLE_TEST_DB=1
./ops test integration tests/test_migrations_runner.py -q
```

After confirmation, `./ops` constructs a database-isolated child environment.
It removes every inherited `DATABASE_URL`/`*_DATABASE_URL`, uppercase libpq
`PG*` variable, and Atlas application database setting, then exposes the one
confirmed DSN under the selected canonical key, `DATABASE_URL`,
`EXTRACTED_DATABASE_URL`, and `ATLAS_DB_CONNECTION_STRING`. These aliases
contain the same credential; they let current test consumers use their existing
interface without inheriting a second database. The DSN is never placed in
process arguments or output.

`./ops test focused ...` uses the removal half of the same boundary without
restoring any DSN. Database-backed focused targets therefore cannot inherit a
stale credential; rerun the exact file/node through `./ops test integration ...`
only after confirming a disposable database.

Never point those variables at the live `atlas` database. Run only a focused
database test target and do not run concurrent DB-backed suites against the
same disposable database; many tests create/drop or rewrite shared objects.
When no isolated local database is prepared, use the matching GitHub Actions
workflow as the canonical proof.

## Failure routing

- `pg_isready` fails: check `systemctl status postgresql@16-main` and whether
  port `5433` is listening; do not fall back to an unrelated `5432` database.
- Connectivity inspection fails: authentication or database selection is
  wrong. Inspect key names with `./ops env keys`; then verify the selected
  context above without printing the URL/password.
- Integration admission reports multiple URLs: unset every stale canonical
  test URL except the one belonging to the focused suite; never guess which
  disposable database should win.
- A test needs another database interface: add an explicit adapter from the
  already confirmed DSN and focused boundary proof; do not pass the parent
  environment through or add a second credential.
- `schema_migrations` is absent: stop. The target is probably a fresh or wrong
  database; do not “fix” it by applying the full chain.
- Startup logs show a migration/writer fence: follow
  `.agent/runbooks/deployment.md` and the specific product runbook. Do not
  bypass the fence or edit the ledger.
- NocoDB is unavailable: this does not mean PostgreSQL is down. NocoDB is an
  optional browser UI and has its own unprivileged credential prerequisite.
