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
     "SELECT application_name, usename, client_addr, state \
      FROM pg_stat_activity \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT application_name, usename, client_addr, state, sync_state \
      FROM pg_stat_replication \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT map_name, sys_name, pg_username, error \
      FROM pg_ident_file_mappings \
      WHERE map_name = 'atlas_app' \
      ORDER BY line_number;"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT type, database, user_name, address, netmask, auth_method \
      FROM pg_hba_file_rules \
      WHERE auth_method = 'trust' \
      ORDER BY line_number;"
   ```

   For the current single-user deployment, the service account is
   `juan-canfield`, the HBA file is `/etc/postgresql/16/main/pg_hba.conf`, the
   identity-map file is `/etc/postgresql/16/main/pg_ident.conf`, and the socket
   directory is `/var/run/postgresql`. Stop if the live results differ; derive
   the map/rules from the returned topology rather than copying these values
   blindly. The final query must show exactly four `host` trust rows: two
   `{all}` application rows and two `{replication}` rows, one for each of
   `127.0.0.1` and `::1`. Stop if any other trust row exists; this runbook does
   not generalize a different HBA topology. The `atlas_app` identity-map query
   must return no rows before editing: do not reuse a pre-existing map name.

   Set `SERVICE_ENV_FILES` to the absolute `EnvironmentFiles` paths printed by
   `./ops env systemd`, joined with `:` in that same order. Use this shell-local
   helper for every fixed inspection in this cutover. It selects the service's
   configuration without printing values and removes ad hoc Atlas database
   variables that would otherwise override those files:

   ```bash
   SERVICE_ENV_FILES='/absolute/first.service.env:/absolute/second.service.env'
   service_db_inspect() {
     env -u ATLAS_DB_CONNECTION_STRING \
       -u ATLAS_DB_HOST \
       -u ATLAS_DB_PORT \
       -u ATLAS_DB_DATABASE \
       -u ATLAS_DB_USER \
       -u ATLAS_DB_PASSWORD \
       -u ATLAS_DB_SOCKET_PATH \
       ATLAS_OPS_ENV_FILES="$SERVICE_ENV_FILES" \
       ./ops db inspect connectivity
   }
   service_db_inspect
   ```

   Do not run a bare `./ops db inspect connectivity` in this procedure: a
   worktree `.env` takes precedence over the service files. If the service has
   no readable `EnvironmentFiles`, stop rather than substituting a worktree or
   shared configuration.

2. Preserve only the two PostgreSQL authentication files before editing them:

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
     "SELECT \
        (SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL), \
        count(*) FILTER ( \
          WHERE map_name = 'atlas_app' \
            AND sys_name = 'juan-canfield' \
            AND pg_username = 'atlas' \
        ), \
        count(*) FILTER (WHERE map_name = 'atlas_app'), \
        count(*) FILTER (WHERE error IS NOT NULL) \
      FROM pg_ident_file_mappings;"
   ```

   The final query must return `0|1|1|0`: no HBA parser error, exactly one
   intended `atlas_app | juan-canfield | atlas` mapping, no additional
   `atlas_app` mapping, and no identity-map error. Otherwise restore both saved
   files and reload PostgreSQL before continuing. The service still uses TCP at
   this point, so a valid scoped peer rule can be proved without removing the
   existing path.

3. In the effective service configuration selected by `SERVICE_ENV_FILES`, use
   `./ops env keys --file <each service file>` and an editor to identify the
   final `ATLAS_DB_*` assignment in service-file order. Verify that the
   effective `ATLAS_DB_CONNECTION_STRING` is absent or empty. A nonblank full
   DSN deliberately takes precedence over `ATLAS_DB_SOCKET_PATH`, so this
   cutover must stop if one is effective. Do not remove or rewrite a full DSN as
   part of this procedure; that needs a separate configuration-migration slice.
   Set `EFFECTIVE_DB_ENV_FILE` to that one effective file, confirm it is one of
   the paths in `SERVICE_ENV_FILES`, and add exactly this non-secret setting
   with `sudoedit "$EFFECTIVE_DB_ENV_FILE"`; do not copy or print the rest of
   any file:

   ```bash
   EFFECTIVE_DB_ENV_FILE='/absolute/path/to/the-effective.service.env'
   sudoedit "$EFFECTIVE_DB_ENV_FILE"
   ```

   ```text
   ATLAS_DB_SOCKET_PATH=/var/run/postgresql
   ```

   Define this shell-local rollback helper before the restart. It is the only
   rollback path after the socket setting exists; when it opens the editor,
   remove only the exact `ATLAS_DB_SOCKET_PATH=/var/run/postgresql` line just
   added, save, and exit before the function restores the two authentication
   files:

   ```bash
   rollback_peer_cutover() {
     case ":$SERVICE_ENV_FILES:" in
       *":$EFFECTIVE_DB_ENV_FILE:"*) ;;
       *) printf '%s\n' 'effective database environment file is not a service EnvironmentFile' >&2; return 1 ;;
     esac
     sudoedit "$EFFECTIVE_DB_ENV_FILE" || return 1
     if sudo grep -Eq '^[[:space:]]*ATLAS_DB_SOCKET_PATH=/var/run/postgresql[[:space:]]*$' "$EFFECTIVE_DB_ENV_FILE"; then
       printf '%s\n' 'remove only the added ATLAS_DB_SOCKET_PATH line before rollback can continue' >&2
       return 1
     fi
     sudo cp --preserve=mode,ownership,timestamps /etc/postgresql/16/main/pg_hba.conf.pre-atlas-peer \
       /etc/postgresql/16/main/pg_hba.conf || return 1
     sudo cp --preserve=mode,ownership,timestamps /etc/postgresql/16/main/pg_ident.conf.pre-atlas-peer \
       /etc/postgresql/16/main/pg_ident.conf || return 1
     sudo systemctl reload postgresql@16-main || return 1
     if [ "$(sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c 'SELECT count(*) FILTER (WHERE auth_method = '\''trust'\''), count(*) FILTER (WHERE error IS NOT NULL) FROM pg_hba_file_rules;')" != '4|0' ]; then
       printf '%s\n' 'restored HBA receipt was not 4|0; do not restart atlas-api' >&2
       return 1
     fi
     systemctl --user restart atlas-api.service || return 1
     service_db_inspect
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
   service_db_inspect
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT application_name, usename, client_addr, state \
      FROM pg_stat_activity \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -h /var/run/postgresql -p 5433 -d atlas -At -F '|' -c \
     "SELECT application_name, usename, client_addr, state, sync_state \
      FROM pg_stat_replication \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   ```

   At least one backend with a `backend_start` after the just-issued service
   restart must show `<unix>` for `client_addr`; compare the rows with the
   loopback clients from step 1 and stop if the service's backend cannot be
   identified as a Unix socket connection. The fixed inspection must also
   succeed only after that transport proof. If any proof fails, remove only the
   exact non-secret setting just added, restore the two saved PostgreSQL files,
   reload PostgreSQL, and restart `atlas-api.service`.

   Before step 4, both final loopback inventories must be empty. A remaining
   loopback TCP client, including a replication client, is a stop condition:
   move that client to the Unix socket or complete a separate, owner-verified
   SCRAM reconnect migration before re-running this cutover. Do not create a
   new credential or remove HBA trust while any such row remains.

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
     "SELECT \
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
        count(*) FILTER (WHERE error IS NOT NULL) AS hba_errors \
      FROM pg_hba_file_rules;"
   ```

   The query must return `1|1|1|1|0|0|0`: one exact application IPv4 rule, one
   exact application IPv6 rule, one exact replication IPv4 rule, one exact
   replication IPv6 rule, no unexpected loopback host rule, no trust rule, and
   no parser error. If any field differs, do not proceed: run
   `rollback_peer_cutover` immediately so the restored HBA/ident files are
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
   reloads HBA/ident, proves the restored `4|0` HBA receipt, then restarts
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
