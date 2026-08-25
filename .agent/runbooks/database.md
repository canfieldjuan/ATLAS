# Database operations

Use this runbook for PostgreSQL connectivity, bounded read-only queries,
migration status, and database-backed tests.

## Architecture and safe inspection

Atlas uses native PostgreSQL 16 on `127.0.0.1:5433`, normally database `atlas`
and role `atlas`. Root Docker Compose connects to this host service; it does not
create PostgreSQL.

```bash
./ops db status
./ops db query "SELECT 1"
./ops db migrations
```

`./ops db query` accepts exactly one `SELECT`, `SHOW`, `TABLE`, or `VALUES`
statement. It refuses write/DDL leads, `SELECT INTO`, row-locking selects, and
multiple statements, then still runs the admitted SQL inside a PostgreSQL
`READ ONLY` transaction with statement and lock timeouts. It uses exported
`ATLAS_DB_*` variables or the same selected keys from discovered env files, but
never prints credentials.

The command is a write guard, not a data-privacy guard. Query only the columns
and rows needed, add a `LIMIT`, and do not paste customer content or identifiers
into chat or GitHub. There is intentionally no `./ops db write`.

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
its exact name/host, then export only the matching test URL and acknowledge the
boundary:

```bash
export ATLAS_MIGRATION_TEST_DATABASE_URL='postgresql://.../disposable_db'
export ATLAS_CONFIRM_DISPOSABLE_TEST_DB=1
./ops test integration tests/test_migrations_runner.py -q
```

Never point those variables at the live `atlas` database. Run only a focused
database test target and do not run concurrent DB-backed suites against the
same disposable database; many tests create/drop or rewrite shared objects.
When no isolated local database is prepared, use the matching GitHub Actions
workflow as the canonical proof.

## Failure routing

- `pg_isready` fails: check `systemctl status postgresql@16-main` and whether
  port `5433` is listening; do not fall back to an unrelated `5432` database.
- Readiness passes but the query fails: authentication or database selection is
  wrong. Inspect key names with `./ops env keys`; never print the URL/password.
- `schema_migrations` is absent: stop. The target is probably a fresh or wrong
  database; do not “fix” it by applying the full chain.
- Startup logs show a migration/writer fence: follow
  `.agent/runbooks/deployment.md` and the specific product runbook. Do not
  bypass the fence or edit the ledger.
- NocoDB is unavailable: this does not mean PostgreSQL is down. NocoDB is an
  optional browser UI and has its own unprivileged credential prerequisite.
