"""
Database migrations for Atlas Brain.
"""

import logging
from pathlib import Path

logger = logging.getLogger("atlas.storage.migrations")

MIGRATIONS_DIR = Path(__file__).parent


async def run_migrations(pool) -> None:
    """
    Run all pending migrations.

    Args:
        pool: The database pool to run migrations against
    """
    # Get list of SQL migration files
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not migration_files:
        logger.info("No migration files found")
        return

    for migration_file in migration_files:
        logger.info("Running migration: %s", migration_file.name)

        sql = migration_file.read_text()

        try:
            await pool.execute(sql)
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
