"""
Database migrations for Atlas Brain.

Tracks applied migrations in `schema_migrations` table to avoid re-running.
"""

import logging
from pathlib import Path

logger = logging.getLogger("atlas.storage.migrations")

MIGRATIONS_DIR = Path(__file__).parent


async def _ensure_migrations_table(pool) -> None:
    """Create the migrations tracking table if it doesn't exist."""
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL UNIQUE,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


async def _get_applied_migrations(pool) -> set[str]:
    """Get set of already applied migration filenames."""
    rows = await pool.fetch("SELECT filename FROM schema_migrations")
    return {row["filename"] for row in rows}


async def _record_migration(pool, filename: str) -> None:
    """Record that a migration has been applied."""
    await pool.execute(
        "INSERT INTO schema_migrations (filename) VALUES ($1)",
        filename
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

    pending = [f for f in migration_files if f.name not in applied]

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
