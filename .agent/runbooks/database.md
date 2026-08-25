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
It removes every inherited `DATABASE_URL`/`*_DATABASE_URL` and Atlas
application database setting, then exposes the one confirmed DSN under the
selected canonical key, `DATABASE_URL`, `EXTRACTED_DATABASE_URL`, and
`ATLAS_DB_CONNECTION_STRING`. These aliases contain the same credential; they
let current test consumers use their existing interface without inheriting a
second database. The DSN is never placed in process arguments or output.

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
