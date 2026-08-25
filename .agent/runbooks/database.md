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
   active local clients, and current health. Do not print `.env` values or a
   database URL:

   ```bash
   ./ops deploy status
   ./ops db inspect connectivity
   id -un
   sudo -u postgres psql -d atlas -At -F '|' -c \
     "SHOW hba_file; SHOW ident_file; SHOW unix_socket_directories;"
   sudo -u postgres psql -d atlas -At -F '|' -c \
     "SELECT application_name, usename, client_addr, state \
      FROM pg_stat_activity \
      WHERE client_addr IN ('127.0.0.1'::inet, '::1'::inet);"
   sudo -u postgres psql -d atlas -At -c \
     "SELECT count(*) FROM pg_stat_replication;"
   ```

   For the current single-user deployment, the service account is
   `juan-canfield`, the HBA file is `/etc/postgresql/16/main/pg_hba.conf`, the
   identity-map file is `/etc/postgresql/16/main/pg_ident.conf`, and the socket
   directory is `/var/run/postgresql`. Stop if the live results differ; derive
   the map/rules from the returned topology rather than copying these values
   blindly.

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
   sudo -u postgres psql -d atlas -At -c \
     "SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL;"
   ```

   The final query must return `0`; otherwise restore both saved files and
   reload PostgreSQL before continuing. The service still uses TCP at this
   point, so a valid scoped peer rule can be proved without removing the
   existing path.

3. In the shared application configuration read by `atlas-api.service`, add
   exactly this non-secret setting with an editor; do not copy or print the
   rest of the file:

   ```text
   ATLAS_DB_SOCKET_PATH=/var/run/postgresql
   ```

   Restart the user service, then prove that the application has changed its
   real connection path before any TCP trust rule is removed:

   ```bash
   systemctl --user restart atlas-api.service
   ./ops deploy status
   ./ops db inspect connectivity
   ```

   Perform one read-only authenticated EOM CRM Contacts-page refresh. A health
   response alone is insufficient: the generic pool can initialize lazily, so
   the CRM read proves the application data path. If either proof fails,
   restore the two saved PostgreSQL files, remove only the exact non-secret
   setting just added, reload PostgreSQL, and restart `atlas-api.service`.

4. Only after the socket-peer proof succeeds, replace **all four** loopback
   TCP `trust` entries in `pg_hba.conf` with `scram-sha-256` using `sudoedit`:

   ```text
   host    all             all             127.0.0.1/32            scram-sha-256
   host    all             all             ::1/128                 scram-sha-256
   host    replication     all             127.0.0.1/32            scram-sha-256
   host    replication     all             ::1/128                 scram-sha-256
   ```

   Do not alter any non-loopback rule. Reload PostgreSQL and repeat the
   service, fixed-inspection, and authenticated CRM proofs from step 3.

5. Verify both sides of the new boundary. The passwordless TCP probe must fail;
   the retained `postgres` socket peer probe must succeed:

   ```bash
   if env -u PGPASSWORD -u ATLAS_DB_PASSWORD PGPASSFILE=/dev/null \
     psql -w -h 127.0.0.1 -p 5433 -U atlas -d atlas -Atc 'SELECT 1'; then
     printf '%s\n' 'unexpected passwordless loopback TCP access' >&2
     exit 1
   fi
   sudo -u postgres psql -d atlas -Atc 'SELECT current_user'
   ```

   The second command must print `postgres`. The first must reject
   authentication without prompting. Do not use an existing `.pgpass` file or
   a secret-bearing shell environment as a substitute test.

6. If a step fails after the HBA conversion, restore both saved authentication
   files, reload PostgreSQL, and restart `atlas-api.service`. If the service
   still cannot reconnect, remove only the added `ATLAS_DB_SOCKET_PATH` line
   from shared configuration and restart again. Re-run the fixed inspection and
   CRM read before declaring rollback complete. Do not edit roles, passwords,
   migration ledger rows, or database data as part of rollback.

`./ops db inspect` remains fixed-query-only. It already selects the same
socket-path configuration as the application and uses a `READ ONLY`
transaction. This cutover does not create a generic command runner or expose a
credential.

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
