from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.storage.migration_runner import (
    apply_content_pipeline_migrations,
    list_content_pipeline_migrations,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_extracted_content_pipeline_migrations.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_extracted_content_pipeline_migrations",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Transaction:
    def __init__(self, conn: "_Conn") -> None:
        self.conn = conn

    async def __aenter__(self):
        self.conn.transactions += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Conn:
    def __init__(self, *, applied_versions=None) -> None:
        self.applied_versions = set(applied_versions or ())
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.transactions = 0

    async def execute(self, query, *args):
        self.executed.append((str(query), args))
        if args and "INSERT INTO" in str(query):
            self.applied_versions.add(str(args[0]))
        return "EXECUTE"

    async def fetch(self, query):
        self.executed.append((str(query), ()))
        return [{"version": version} for version in sorted(self.applied_versions)]

    def transaction(self):
        return _Transaction(self)


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


def _write_migration(root: Path, name: str, sql: str) -> None:
    path = root / name
    path.write_text(sql, encoding="utf-8")


def test_list_content_pipeline_migrations_returns_sorted_sql_files(tmp_path) -> None:
    _write_migration(tmp_path, "002_second.sql", "SELECT 2;")
    _write_migration(tmp_path, "001_first.sql", "SELECT 1;")
    (tmp_path / "README.md").write_text("ignore", encoding="utf-8")

    migrations = list_content_pipeline_migrations(tmp_path)

    assert [migration.version for migration in migrations] == [
        "001_first.sql",
        "002_second.sql",
    ]
    assert all(len(migration.checksum) == 64 for migration in migrations)


@pytest.mark.asyncio
async def test_apply_content_pipeline_migrations_applies_pending_files(tmp_path) -> None:
    _write_migration(tmp_path, "001_first.sql", "CREATE TABLE example_one(id int);")
    _write_migration(tmp_path, "002_second.sql", "CREATE TABLE example_two(id int);")
    conn = _Conn()

    result = await apply_content_pipeline_migrations(conn, migrations_dir=tmp_path)

    assert [entry.version for entry in result.applied] == [
        "001_first.sql",
        "002_second.sql",
    ]
    assert result.skipped == ()
    assert conn.transactions == 2
    executed_sql = "\n".join(query for query, _ in conn.executed)
    assert "CREATE TABLE IF NOT EXISTS content_pipeline_schema_migrations" in executed_sql
    assert "CREATE TABLE example_one" in executed_sql
    assert "CREATE TABLE example_two" in executed_sql


@pytest.mark.asyncio
async def test_apply_content_pipeline_migrations_skips_applied_versions(tmp_path) -> None:
    _write_migration(tmp_path, "001_first.sql", "SELECT 1;")
    _write_migration(tmp_path, "002_second.sql", "SELECT 2;")
    conn = _Conn(applied_versions={"001_first.sql"})

    result = await apply_content_pipeline_migrations(conn, migrations_dir=tmp_path)

    assert [entry.version for entry in result.skipped] == ["001_first.sql"]
    assert [entry.version for entry in result.applied] == ["002_second.sql"]
    executed_sql = "\n".join(query for query, _ in conn.executed)
    assert "SELECT 1;" not in executed_sql
    assert "SELECT 2;" in executed_sql


@pytest.mark.asyncio
async def test_apply_content_pipeline_migrations_dry_run_does_not_execute_sql(tmp_path) -> None:
    _write_migration(tmp_path, "001_first.sql", "CREATE TABLE example_one(id int);")
    conn = _Conn()

    result = await apply_content_pipeline_migrations(
        conn,
        migrations_dir=tmp_path,
        dry_run=True,
    )

    assert result.dry_run is True
    assert [entry.status for entry in result.applied] == ["dry_run"]
    assert conn.executed == []


@pytest.mark.asyncio
async def test_apply_content_pipeline_migrations_accepts_pool(tmp_path) -> None:
    _write_migration(tmp_path, "001_first.sql", "SELECT 1;")
    conn = _Conn()

    result = await apply_content_pipeline_migrations(_Pool(conn), migrations_dir=tmp_path)

    assert [entry.version for entry in result.applied] == ["001_first.sql"]


@pytest.mark.asyncio
async def test_apply_content_pipeline_migrations_rejects_unsafe_table_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsafe migration table"):
        await apply_content_pipeline_migrations(
            object(),
            migrations_dir=tmp_path,
            migration_table="bad-name",
        )


@pytest.mark.asyncio
async def test_migration_cli_wires_pool_and_json_output(monkeypatch, capsys, tmp_path) -> None:
    cli = _load_cli_module()
    _write_migration(tmp_path, "001_first.sql", "SELECT 1;")
    conn = _Conn()
    pool = _Pool(conn)
    created_urls: list[str] = []

    async def create_pool(database_url):
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "run",
            "--database-url",
            "postgres://example",
            "--migrations-dir",
            str(tmp_path),
            "--json",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert output["applied_count"] == 1
    assert output["applied"][0]["version"] == "001_first.sql"


@pytest.mark.asyncio
async def test_migration_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.sys, "argv", ["run"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()
