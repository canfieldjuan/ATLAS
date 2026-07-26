"""
Database migrations for Atlas Brain.

Tracks applied migrations in `schema_migrations` table to avoid re-running.
"""

import asyncio
import logging
import re
from collections.abc import Collection
from pathlib import Path

logger = logging.getLogger("atlas.storage.migrations")

MIGRATIONS_DIR = Path(__file__).parent


def _parse_migration_identity(filename: str) -> tuple[int, str]:
    """Extract the numeric prefix and canonical migration name."""
    prefix = filename.split("_", 1)[0]
    match = re.match(r"\d+", prefix)
    version = int(match.group()) if match else 0
    return version, filename.removesuffix(".sql")


def _find_duplicate_migration_prefixes(migration_files: list[Path]) -> dict[int, list[str]]:
    duplicates: dict[int, list[str]] = {}
    seen: dict[int, list[str]] = {}
    for migration_file in migration_files:
        version, _ = _parse_migration_identity(migration_file.name)
        seen.setdefault(version, []).append(migration_file.name)
    for version, names in seen.items():
        if version > 0 and len(names) > 1:
            duplicates[version] = sorted(names)
    return duplicates


async def _ensure_migrations_table(executor) -> None:
    """Create the migrations tracking table if it doesn't exist.

    ``executor`` is a pool OR a single acquired connection; both expose
    execute/fetch/fetchval. run_migrations passes a connection so the whole
    run needs exactly one, which is what makes a min=max=1 pool safe.
    """
    await executor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


async def _get_applied_migrations(executor) -> set[str]:
    """Get set of already applied migration names (e.g. '025_temporal_patterns')."""
    rows = await executor.fetch("SELECT name FROM schema_migrations")
    return {row["name"] for row in rows}


async def _record_migration(executor, filename: str) -> None:
    """Record that a migration has been applied.

    ``executor`` is a pool or a single acquired connection (see
    _ensure_migrations_table)."""
    version, name = _parse_migration_identity(filename)
    existing_version = await executor.fetchval(
        "SELECT version FROM schema_migrations WHERE name = $1",
        name,
    )
    if existing_version is not None:
        return

    record_version = version
    conflicting_name = await executor.fetchval(
        "SELECT name FROM schema_migrations WHERE version = $1",
        version,
    )
    if conflicting_name and conflicting_name != name:
        record_version = await executor.fetchval(
            """
            SELECT CASE
                WHEN COALESCE(MIN(version), 0) < 0 THEN MIN(version) - 1
                ELSE -1
            END
            FROM schema_migrations
            """
        )
        logger.warning(
            "Migration version collision for %s on prefix %d with existing %s; "
            "recording under synthetic version %d",
            name,
            version,
            conflicting_name,
            record_version,
        )

    await executor.execute(
        "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
        record_version,
        name,
    )


# Cluster-wide advisory key serializing migration runs. Concurrent first
# starts (main app + standalone MCP servers, or two replicas) otherwise
# snapshot the pending set independently and race the check-then-insert in
# _record_migration against the unique schema_migrations.version constraint.
_MIGRATIONS_ADVISORY_LOCK_KEY = 0x41544C41  # "ATLA"
_MIGRATIONS_LOCK_POLL_SECONDS = 0.25


async def run_migrations(
    pool,
    *,
    migrations_dir: Path | None = None,
    only: Collection[str] | None = None,
) -> None:
    """
    Run all pending migrations.

    Only runs migrations that haven't been applied yet.
    Tracks applied migrations in schema_migrations table.

    Concurrency: the whole run holds a SESSION-level advisory lock on one
    acquired connection, so simultaneous entrants (another replica, or the
    standalone MCP servers starting alongside the main app) queue at the lock
    and then re-snapshot, finding nothing pending.

    Two properties depend on doing it this way rather than with a transaction:

    * **Exactly one connection.** Every statement -- the lock, the bookkeeping,
      the migration SQL -- runs on the same acquired connection. A transaction
      that then reached back to the pool would deadlock a deployment configured
      with ``min_pool_size == max_pool_size == 1``.
    * **No open transaction.** Several migrations use ``CREATE INDEX
      CONCURRENTLY``, which Postgres refuses inside a transaction block. A
      session-level lock leaves the connection in autocommit, so they still run.

    Args:
        pool: The database pool to run migrations against
        migrations_dir: override for tests; defaults to the packaged dir
        only: restrict the run to these migration stems (e.g.
            ``{"350_scoped_mailbox_credentials"}``). A component that depends on
            one specific table can apply exactly that prerequisite instead of
            the whole chain. The full chain is NOT fresh-applicable -- 076+
            reference an out-of-band ``product_metadata`` table that no
            migration creates -- so a component which needs a later, otherwise
            self-contained migration cannot get it by running everything.
    """
    directory = migrations_dir if migrations_dir is not None else MIGRATIONS_DIR
    conn = await pool.acquire()
    try:
        # POLL with try-lock instead of blocking in pg_advisory_lock.
        #
        # A blocking waiter sits inside an open (implicit) transaction for as
        # long as it waits. CREATE INDEX CONCURRENTLY -- used by five packaged
        # migrations -- must wait for every concurrent transaction on the table
        # to drain before it can finish. So a blocked waiter and a holder
        # running CONCURRENTLY wait on each other and PostgreSQL kills one with
        # a deadlock. Two replicas starting together is enough to trigger it.
        #
        # try-lock returns immediately, so between polls this connection holds
        # no transaction and the holder's CONCURRENTLY build can complete.
        waited = 0.0
        while not await conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", _MIGRATIONS_ADVISORY_LOCK_KEY
        ):
            await asyncio.sleep(_MIGRATIONS_LOCK_POLL_SECONDS)
            waited += _MIGRATIONS_LOCK_POLL_SECONDS
            if waited % 10 < _MIGRATIONS_LOCK_POLL_SECONDS:
                logger.info(
                    "Waiting %.0fs for another process to finish migrations",
                    waited,
                )
        try:
            await _ensure_migrations_table(conn)

            # Snapshot under the lock, so a queued entrant re-reads AFTER the
            # winner recorded its work.
            applied = await _get_applied_migrations(conn)

            migration_files = sorted(directory.glob("*.sql"))
            if only is not None:
                requested = set(only)
                migration_files = [
                    f for f in migration_files if f.stem in requested
                ]
                missing = requested - {f.stem for f in migration_files}
                if missing:
                    raise FileNotFoundError(
                        f"requested migrations not found: {sorted(missing)}"
                    )

            if not migration_files:
                logger.info("No migration files found")
                return

            pending = [f for f in migration_files if f.stem not in applied]

            if not pending:
                logger.debug("All %d migrations already applied", len(migration_files))
                return

            logger.info("Running %d pending migrations (of %d total)", len(pending), len(migration_files))

            for migration_file in pending:
                logger.info("Running migration: %s", migration_file.name)

                sql = migration_file.read_text()

                try:
                    await conn.execute(sql)
                    await _record_migration(conn, migration_file.name)
                    logger.info("Migration %s completed successfully", migration_file.name)
                except Exception as e:
                    logger.error("Migration %s failed: %s", migration_file.name, e)
                    raise
        finally:
            await conn.execute(
                "SELECT pg_advisory_unlock($1)", _MIGRATIONS_ADVISORY_LOCK_KEY
            )
    finally:
        await pool.release(conn)


async def check_schema_exists(pool) -> bool:
    """
    Check if the database schema has been initialized.

    Returns:
        True if schema exists, False otherwise
    """
    try:
        result = await pool.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'sessions'
            )
            """
        )
        return result
    except Exception:
        return False
