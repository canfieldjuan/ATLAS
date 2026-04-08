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


async def run_migrations(pool) -> None:
    """
    Run all pending migrations.

    Only runs migrations that haven't been applied yet.
    Tracks applied migrations in schema_migrations table.

    Args:
        pool: The database pool to run migrations against
    """
    # Ensure tracking table exists
    await _ensure_migrations_table(pool)

    # Get already applied migrations
    applied = await _get_applied_migrations(pool)

    # Get list of SQL migration files
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not migration_files:
        logger.info("No migration files found")
        return

    duplicate_prefixes = _find_duplicate_migration_prefixes(migration_files)
    if duplicate_prefixes:
        formatted = ", ".join(
            f"{version}: {', '.join(names)}"
            for version, names in sorted(duplicate_prefixes.items())
        )
        raise RuntimeError(f"Duplicate migration prefixes detected: {formatted}")

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
