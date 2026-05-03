"""Product-owned SQL migration runner for AI Content Ops."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
DEFAULT_MIGRATION_TABLE = "content_pipeline_schema_migrations"


@dataclass(frozen=True)
class MigrationFile:
    version: str
    path: Path
    checksum: str

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class MigrationRunEntry:
    version: str
    checksum: str
    path: str
    status: str


@dataclass(frozen=True)
class MigrationRunResult:
    applied: tuple[MigrationRunEntry, ...]
    skipped: tuple[MigrationRunEntry, ...]
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": [entry.__dict__ for entry in self.applied],
            "skipped": [entry.__dict__ for entry in self.skipped],
            "dry_run": self.dry_run,
            "applied_count": len(self.applied),
            "skipped_count": len(self.skipped),
        }


def list_content_pipeline_migrations(
    migrations_dir: str | Path = MIGRATIONS_DIR,
) -> tuple[MigrationFile, ...]:
    """Return packaged SQL migrations in filename order."""

    root = Path(migrations_dir)
    return tuple(
        MigrationFile(
            version=path.name,
            path=path,
            checksum=sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.glob("*.sql"))
    )


async def apply_content_pipeline_migrations(
    db: Any,
    *,
    migrations_dir: str | Path = MIGRATIONS_DIR,
    migration_table: str = DEFAULT_MIGRATION_TABLE,
    dry_run: bool = False,
) -> MigrationRunResult:
    """Apply pending content-pipeline migrations using a host pool/connection."""

    table = _identifier(migration_table)
    migrations = list_content_pipeline_migrations(migrations_dir)

    async def _run(conn: Any) -> MigrationRunResult:
        if not dry_run:
            await _ensure_migration_table(conn, table)
        applied_versions = await _read_applied_versions(conn, table, dry_run=dry_run)
        applied: list[MigrationRunEntry] = []
        skipped: list[MigrationRunEntry] = []
        for migration in migrations:
            entry = MigrationRunEntry(
                version=migration.version,
                checksum=migration.checksum,
                path=str(migration.path),
                status="dry_run" if dry_run else "applied",
            )
            if migration.version in applied_versions:
                skipped.append(
                    MigrationRunEntry(
                        version=migration.version,
                        checksum=migration.checksum,
                        path=str(migration.path),
                        status="skipped",
                    )
                )
                continue
            if dry_run:
                applied.append(entry)
                continue
            async with _transaction(conn):
                await conn.execute(migration.sql)
                await conn.execute(
                    f"""
                    INSERT INTO {table} (version, checksum)
                    VALUES ($1, $2)
                    ON CONFLICT (version) DO UPDATE
                       SET checksum = EXCLUDED.checksum,
                           applied_at = NOW()
                    """,
                    migration.version,
                    migration.checksum,
                )
            applied.append(entry)
        return MigrationRunResult(
            applied=tuple(applied),
            skipped=tuple(skipped),
            dry_run=dry_run,
        )

    return await _with_connection(db, _run)


async def _ensure_migration_table(conn: Any, table: str) -> None:
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            version TEXT PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


async def _read_applied_versions(conn: Any, table: str, *, dry_run: bool) -> set[str]:
    if dry_run:
        return set()
    rows = await conn.fetch(f"SELECT version FROM {table}")
    return {str(_row_value(row, "version")) for row in rows if _row_value(row, "version")}


async def _with_connection(
    db: Any,
    callback: Callable[[Any], Awaitable[MigrationRunResult]],
) -> MigrationRunResult:
    acquire = getattr(db, "acquire", None)
    if not callable(acquire):
        return await callback(db)

    acquired = acquire()
    if hasattr(acquired, "__aenter__"):
        async with acquired as conn:
            return await callback(conn)

    conn = await _maybe_await(acquired)
    try:
        return await callback(conn)
    finally:
        release = getattr(db, "release", None)
        if callable(release):
            await _maybe_await(release(conn))


@asynccontextmanager
async def _transaction(conn: Any):
    transaction = getattr(conn, "transaction", None)
    if not callable(transaction):
        yield
        return
    async with transaction():
        yield


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _row_value(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError):
        return getattr(row, key, None)


def _identifier(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned or not all(char.isalnum() or char == "_" for char in cleaned):
        raise ValueError(f"unsafe migration table name: {value!r}")
    return cleaned


__all__ = [
    "DEFAULT_MIGRATION_TABLE",
    "MIGRATIONS_DIR",
    "MigrationFile",
    "MigrationRunEntry",
    "MigrationRunResult",
    "apply_content_pipeline_migrations",
    "list_content_pipeline_migrations",
]
