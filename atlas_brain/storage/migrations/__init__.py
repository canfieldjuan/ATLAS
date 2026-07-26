"""
Database migrations for Atlas Brain.

Tracks applied migrations in `schema_migrations` table to avoid re-running.
"""

import logging
import re
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


async def _ensure_migrations_table(pool) -> None:
    """Create the migrations tracking table if it doesn't exist."""
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


async def _get_applied_migrations(pool) -> set[str]:
    """Get set of already applied migration names (e.g. '025_temporal_patterns')."""
    rows = await pool.fetch("SELECT name FROM schema_migrations")
    return {row["name"] for row in rows}


async def _record_migration(pool, filename: str) -> None:
    """Record that a migration has been applied."""
    version, name = _parse_migration_identity(filename)
    existing_version = await pool.fetchval(
        "SELECT version FROM schema_migrations WHERE name = $1",
        name,
    )
    if existing_version is not None:
        return

    record_version = version
    conflicting_name = await pool.fetchval(
        "SELECT name FROM schema_migrations WHERE version = $1",
        version,
    )
    if conflicting_name and conflicting_name != name:
        record_version = await pool.fetchval(
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

    await pool.execute(
        "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
        record_version,
        name,
    )


# Cluster-wide advisory key serializing migration runs. Concurrent first
# starts (main app + standalone MCP servers, or two replicas) otherwise
# snapshot the pending set independently and race the check-then-insert in
# _record_migration against the unique schema_migrations.version constraint.
_MIGRATIONS_ADVISORY_LOCK_KEY = 0x41544C41  # "ATLA"


async def run_migrations(pool, *, migrations_dir: Path | None = None) -> None:
    """
    Run all pending migrations.

    Only runs migrations that haven't been applied yet.
    Tracks applied migrations in schema_migrations table.

    The whole run happens under a Postgres advisory transaction lock, so
    concurrent entrants (another replica or another standalone MCP server
    starting at the same time) queue at the lock, then re-snapshot and find
    nothing pending. Migration SQL itself still executes on ordinary pool
    connections, outside the lock holder's transaction, preserving existing
    semantics for statements that cannot run transactionally.

    Args:
        pool: The database pool to run migrations against
        migrations_dir: override for tests; defaults to the packaged dir
    """
    directory = migrations_dir if migrations_dir is not None else MIGRATIONS_DIR
    async with pool.transaction() as conn:
        await conn.execute(
            "SELECT pg_advisory_xact_lock($1)", _MIGRATIONS_ADVISORY_LOCK_KEY
        )

        # Ensure tracking table exists
        await _ensure_migrations_table(pool)

        # Get already applied migrations -- snapshot under the lock, so a
        # queued entrant re-reads AFTER the winner recorded its work.
        applied = await _get_applied_migrations(pool)

        # Get list of SQL migration files
        migration_files = sorted(directory.glob("*.sql"))

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
                await pool.execute(sql)
                await _record_migration(pool, migration_file.name)
                logger.info("Migration %s completed successfully", migration_file.name)
            except Exception as e:
                logger.error("Migration %s failed: %s", migration_file.name, e)
                raise


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
