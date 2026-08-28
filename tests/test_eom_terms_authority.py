"""Focused proof for the private EOM Terms version authority."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as funnel_auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.eom_terms_authority import (
    EOMTermsAuthority,
    EOMTermsConflictError,
    EOMTermsNotFoundError,
    EOMTermsUnavailableError,
    EOMTermsValidationError,
    canonical_eom_terms_documents,
    eom_terms_authority_schema_ready,
    normalize_eom_terms_documents,
)


_NOW = datetime(2026, 8, 27, 21, 0, tzinfo=timezone.utc)
_VERSION_ID = UUID("11111111-1111-4111-8111-111111111111")
_SERVICE = funnel_auth_mod.generate_eom_funnel_service_token()
_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/396_eom_terms_authority.sql"
)
_RUNTIME_DATABASE_URL_ENV = "ATLAS_EOM_FIRST_CLEAN_TEST_DATABASE_URL"
_DBA_DATABASE_URL_ENV = "ATLAS_EOM_FIRST_CLEAN_DBA_DATABASE_URL"


def _documents(marker: str = "approved") -> dict[str, Any]:
    return {
        audience: {
            locale: {
                "terms": f"{marker} {audience} {locale} terms",
                "servicesWeCannotProvide": f"{marker} {audience} {locale} services",
                "additionalWorkAcknowledgement": (
                    f"{marker} {audience} {locale} additional work"
                ),
            }
            for locale in ("en", "es")
        }
        for audience in ("residential", "commercial")
    }


def _row(
    *,
    status: str = "draft",
    documents: object | None = None,
    material_change: bool = True,
) -> dict[str, Any]:
    normalized, serialized, digest = canonical_eom_terms_documents(
        documents if documents is not None else _documents()
    )
    return {
        "id": _VERSION_ID,
        "version_label": "2026.1",
        "status": status,
        "material_change": material_change,
        "documents": serialized,
        "content_hash": digest,
        "created_by_id": 7,
        "created_by_name": "Juan",
        "created_at": _NOW,
        "published_by_id": 7 if status == "published" else None,
        "published_by_name": "Juan" if status == "published" else None,
        "published_at": _NOW if status == "published" else None,
        "normalized": normalized,
    }


class _Pool:
    is_initialized = True

    def __init__(self, connection: Any, *, current_row: dict[str, Any] | None = None):
        self.connection = connection
        self.current_row = current_row

    async def fetchval(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    async def fetchrow(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.current_row

    @asynccontextmanager
    async def transaction(self):
        yield self.connection


class _SchemaProbePool:
    def __init__(self, result: object) -> None:
        self.result = result
        self.query = ""

    async def fetchval(self, query: str) -> object:
        self.query = query
        return self.result


class _AsyncpgAuthorityPool:
    """Expose the service pool protocol over a real asyncpg pool."""

    is_initialized = True

    def __init__(self, pool: Any) -> None:
        self.raw_pool = pool

    @asynccontextmanager
    async def transaction(self):
        async with self.raw_pool.acquire() as connection:
            async with connection.transaction():
                yield connection

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        async with self.raw_pool.acquire() as connection:
            return await connection.fetchval(*args, **kwargs)

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:
        async with self.raw_pool.acquire() as connection:
            return await connection.fetchrow(*args, **kwargs)


class _FailBeforeCurrentConnection:
    """Inject one failure after publication UPDATE but before pointer UPSERT."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def execute(self, query: str, *args: Any) -> Any:
        if "INSERT INTO eom_terms_current_version" in query:
            raise OSError("injected pointer-write failure")
        return await self._connection.execute(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._connection.fetchval(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        return await self._connection.fetchrow(query, *args)


class _FailBeforeCurrentPool(_AsyncpgAuthorityPool):
    @asynccontextmanager
    async def transaction(self):
        async with self.raw_pool.acquire() as connection:
            async with connection.transaction():
                yield _FailBeforeCurrentConnection(connection)


class _RecordingCurrentConnection:
    """Record successful current-pointer writes while holding the DB lock."""

    def __init__(self, connection: Any, publication_order: list[UUID]) -> None:
        self._connection = connection
        self._publication_order = publication_order

    async def execute(self, query: str, *args: Any) -> Any:
        result = await self._connection.execute(query, *args)
        if "INSERT INTO eom_terms_current_version" in query:
            self._publication_order.append(UUID(str(args[0])))
        return result

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._connection.fetchval(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        return await self._connection.fetchrow(query, *args)


class _RecordingCurrentPool(_AsyncpgAuthorityPool):
    def __init__(self, pool: Any, publication_order: list[UUID]) -> None:
        super().__init__(pool)
        self._publication_order = publication_order

    @asynccontextmanager
    async def transaction(self):
        async with self.raw_pool.acquire() as connection:
            async with connection.transaction():
                yield _RecordingCurrentConnection(
                    connection,
                    self._publication_order,
                )


def _database_url_or_skip(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not configured")
    return value


async def _provision_terms_guard(dba_connection: Any) -> None:
    await dba_connection.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
            ) THEN
                ALTER ROLE atlas_eom_handoff_owner
                    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
                    NOREPLICATION NOBYPASSRLS;
            ELSE
                CREATE ROLE atlas_eom_handoff_owner
                    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
                    NOREPLICATION NOBYPASSRLS;
            END IF;
        END;
        $$;
        """
    )
    await dba_connection.execute("REVOKE atlas_eom_handoff_owner FROM atlas")


@asynccontextmanager
async def _real_terms_store():
    runtime_url = _database_url_or_skip(_RUNTIME_DATABASE_URL_ENV)
    dba_url = _database_url_or_skip(_DBA_DATABASE_URL_ENV)
    schema = f"atlas_eom_terms_{uuid4().hex}"
    dba_connection = await asyncpg.connect(dba_url)
    runtime_pool = None
    try:
        await _provision_terms_guard(dba_connection)
        await dba_connection.execute(
            f'CREATE SCHEMA "{schema}" AUTHORIZATION atlas_eom_handoff_owner'
        )
        await dba_connection.execute(
            f'GRANT USAGE, CREATE ON SCHEMA "{schema}" TO atlas'
        )
        await dba_connection.execute(f'SET search_path TO "{schema}", pg_catalog')
        await dba_connection.execute(_MIGRATION.read_text())
        runtime_pool = await asyncpg.create_pool(
            runtime_url,
            min_size=1,
            max_size=5,
            statement_cache_size=0,
            server_settings={"search_path": f'"{schema}", pg_catalog'},
        )
        yield _AsyncpgAuthorityPool(runtime_pool), dba_connection, schema
    finally:
        if runtime_pool is not None:
            await runtime_pool.close()
        try:
            await dba_connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await dba_connection.close()


class _CreateConnection:
    def __init__(
        self,
        *,
        inserted: dict[str, Any] | None,
        existing: dict[str, Any] | None = None,
    ) -> None:
        self.inserted = inserted
        self.existing = existing
        self.calls: list[str] = []

    async def fetchrow(self, query: str, *_args: Any) -> Any:
        self.calls.append(query)
        if "INSERT INTO eom_terms_versions" in query:
            return self.inserted
        return self.existing


class _PublishConnection:
    def __init__(self, source: dict[str, Any], *, current_id: UUID | None = None):
        self.source = source
        self.current_id = current_id
        self.calls: list[str] = []

    async def execute(self, query: str, *_args: Any) -> str:
        self.calls.append(query)
        return "OK"

    async def fetchval(self, query: str, *_args: Any) -> UUID | None:
        self.calls.append(query)
        return self.current_id

    async def fetchrow(self, query: str, *_args: Any) -> Any:
        self.calls.append(query)
        if query.lstrip().startswith("SELECT"):
            return self.source
        if "UPDATE eom_terms_versions" in query:
            return _row(status="published")
        raise AssertionError(f"Unexpected query: {query}")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("commercial"),
        lambda value: value.update({"other": {}}),
        lambda value: value["residential"].pop("es"),
        lambda value: value["residential"]["en"].pop("terms"),
        lambda value: value["residential"]["en"].update({"other": "x"}),
        lambda value: value["residential"]["en"].update({"terms": " "}),
        lambda value: value["residential"]["en"].update({"terms": "bad\x00text"}),
        lambda value: value["residential"]["en"].update({"terms": 7}),
        lambda value: value["residential"]["en"].update({"terms": "\ud800"}),
        lambda value: value["residential"]["en"].update({"terms": "x" * 100_001}),
    ],
)
def test_document_boundary_rejects_every_incomplete_or_extra_shape(mutate: Any) -> None:
    documents = _documents()
    mutate(documents)
    with pytest.raises(EOMTermsValidationError):
        normalize_eom_terms_documents(documents)


def test_document_section_accepts_exact_length_boundary() -> None:
    documents = _documents()
    documents["residential"]["en"]["terms"] = "x" * 100_000
    normalized = normalize_eom_terms_documents(documents)
    assert len(normalized["residential"]["en"]["terms"]) == 100_000


def test_canonical_hash_is_independent_of_mapping_order() -> None:
    documents = _documents()
    reversed_documents = dict(reversed(documents.items()))
    normalized, _, digest = canonical_eom_terms_documents(documents)
    normalized_reversed, _, reversed_digest = canonical_eom_terms_documents(
        reversed_documents
    )
    assert normalized_reversed == normalized
    assert reversed_digest == digest


@pytest.mark.asyncio
async def test_create_draft_writes_one_normalized_snapshot() -> None:
    connection = _CreateConnection(inserted=_row())
    result = await EOMTermsAuthority(pool=_Pool(connection)).create_draft(
        version_label=" 2026.1 ",
        material_change=True,
        documents=_documents(),
        actor_id=7,
        actor_name=" Juan ",
    )
    assert result["versionId"] == str(_VERSION_ID)
    assert result["status"] == "draft"
    assert result["idempotent"] is False
    assert len(connection.calls) == 1


@pytest.mark.asyncio
async def test_create_draft_replays_only_identical_content() -> None:
    existing = _row()
    connection = _CreateConnection(inserted=None, existing=existing)
    authority = EOMTermsAuthority(pool=_Pool(connection))
    replay = await authority.create_draft(
        version_label="2026.1",
        material_change=True,
        documents=_documents(),
        actor_id=7,
        actor_name="Juan",
    )
    assert replay["idempotent"] is True

    with pytest.raises(EOMTermsConflictError):
        await authority.create_draft(
            version_label="2026.1",
            material_change=True,
            documents=_documents("different"),
            actor_id=7,
            actor_name="Juan",
        )


@pytest.mark.asyncio
async def test_create_draft_accepts_false_but_rejects_non_boolean_material_flag() -> (
    None
):
    accepted_connection = _CreateConnection(inserted=_row(material_change=False))
    accepted = await EOMTermsAuthority(pool=_Pool(accepted_connection)).create_draft(
        version_label="2026.1",
        material_change=False,
        documents=_documents(),
        actor_id=7,
        actor_name="Juan",
    )
    assert accepted["materialChange"] is False

    rejected_connection = _CreateConnection(inserted=_row())
    with pytest.raises(EOMTermsValidationError, match="must be boolean"):
        await EOMTermsAuthority(pool=_Pool(rejected_connection)).create_draft(
            version_label="2026.1",
            material_change=0,
            documents=_documents(),
            actor_id=7,
            actor_name="Juan",
        )
    assert rejected_connection.calls == []


@pytest.mark.asyncio
async def test_publish_serializes_before_selecting_current_version() -> None:
    connection = _PublishConnection(_row())
    result = await EOMTermsAuthority(pool=_Pool(connection)).publish(
        version_id=_VERSION_ID,
        actor_id=7,
        actor_name="Juan",
    )
    assert result["status"] == "published"
    assert result["idempotent"] is False
    assert "pg_advisory_xact_lock" in connection.calls[0]
    assert "FOR UPDATE" in connection.calls[1]
    assert "eom_terms_current_version" in connection.calls[-1]


@pytest.mark.asyncio
async def test_publish_replays_only_when_version_is_still_current() -> None:
    published = _row(status="published")
    replay_connection = _PublishConnection(published, current_id=_VERSION_ID)
    replay = await EOMTermsAuthority(pool=_Pool(replay_connection)).publish(
        version_id=_VERSION_ID,
        actor_id=7,
        actor_name="Juan",
    )
    assert replay["idempotent"] is True
    assert not any(
        "UPDATE eom_terms_versions" in call for call in replay_connection.calls
    )

    stale_connection = _PublishConnection(published, current_id=uuid4())
    with pytest.raises(EOMTermsConflictError):
        await EOMTermsAuthority(pool=_Pool(stale_connection)).publish(
            version_id=_VERSION_ID,
            actor_id=7,
            actor_name="Juan",
        )


@pytest.mark.asyncio
async def test_current_read_fails_closed_without_schema_or_pointer() -> None:
    class _NoSchemaPool:
        is_initialized = True

        async def fetchval(self, *_args: Any, **_kwargs: Any) -> bool:
            return False

    with pytest.raises(EOMTermsUnavailableError):
        await EOMTermsAuthority(pool=_NoSchemaPool()).get_current()

    with pytest.raises(EOMTermsNotFoundError):
        await EOMTermsAuthority(pool=_Pool(connection=None)).get_current()


@pytest.mark.asyncio
async def test_schema_guard_requires_relations_triggers_unique_keys_and_fk() -> None:
    ready_pool = _SchemaProbePool(True)
    missing_boundary_pool = _SchemaProbePool(False)

    assert await eom_terms_authority_schema_ready(ready_pool) is True
    assert await eom_terms_authority_schema_ready(missing_boundary_pool) is False
    for expected_boundary in (
        "eom_terms_versions",
        "eom_terms_current_version",
        "trg_protect_eom_terms_version",
        "trg_protect_eom_terms_version_truncate",
        "trg_require_published_eom_terms_current_version",
        "trg_prevent_eom_terms_current_delete",
        "trg_prevent_eom_terms_current_truncate",
        "pg_index",
        "pg_constraint",
        "foreign_key.confdeltype = 'r'",
    ):
        assert expected_boundary in ready_pool.query


def test_slim_eom_profile_binds_terms_to_canonical_funnel_pool() -> None:
    from atlas_brain import main_eom

    assert main_eom.app.state.eom_funnel_terms_pool is main_eom.get_eom_funnel_db_pool


@pytest.mark.asyncio
async def test_real_postgres_enforces_guard_concurrency_and_rollback() -> None:
    async with _real_terms_store() as (pool, dba_connection, schema):
        authority = EOMTermsAuthority(pool=pool)

        assert await eom_terms_authority_schema_ready(pool) is True
        assert await pool.fetchval("SELECT COUNT(*) FROM eom_terms_versions") == 0
        relation_owners = await dba_connection.fetch(
            """
            SELECT relation.relname, owner.rolname AS owner
            FROM pg_class AS relation
            JOIN pg_roles AS owner ON owner.oid = relation.relowner
            WHERE relation.oid IN (
                'eom_terms_versions'::regclass,
                'eom_terms_current_version'::regclass
            )
            """
        )
        assert {
            (str(row["relname"]), str(row["owner"])) for row in relation_owners
        } == {
            ("eom_terms_versions", "atlas_eom_handoff_owner"),
            ("eom_terms_current_version", "atlas_eom_handoff_owner"),
        }
        assert (
            await pool.fetchval(
                """
            SELECT NOT has_table_privilege(
                       current_user, 'eom_terms_versions', 'UPDATE'
                   )
               AND NOT has_table_privilege(
                       current_user, 'eom_terms_versions', 'INSERT'
                   )
               AND has_column_privilege(
                       current_user, 'eom_terms_versions', 'status', 'UPDATE'
                   )
               AND NOT has_column_privilege(
                       current_user, 'eom_terms_versions', 'documents', 'UPDATE'
                   )
               AND NOT has_function_privilege(
                       current_user,
                       'protect_eom_terms_version()'::regprocedure,
                       'EXECUTE'
                   )
            """
            )
            is True
        )

        first = await authority.create_draft(
            version_label="2026.1",
            material_change=True,
            documents=_documents("first"),
            actor_id=7,
            actor_name="Juan",
        )
        first_id = UUID(first["versionId"])
        published = await authority.publish(
            version_id=first_id,
            actor_id=7,
            actor_name="Juan",
        )
        assert published["status"] == "published"
        assert published["idempotent"] is False

        async with pool.raw_pool.acquire() as runtime_connection:
            with pytest.raises(
                asyncpg.PostgresError,
                match="Published EOM Terms versions are immutable",
            ):
                await runtime_connection.execute(
                    """
                    UPDATE eom_terms_versions
                    SET published_by_name = 'tampered'
                    WHERE id = $1
                    """,
                    first_id,
                )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await runtime_connection.execute(
                    """
                    ALTER TABLE eom_terms_versions
                    DISABLE TRIGGER trg_protect_eom_terms_version
                    """
                )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await runtime_connection.execute(
                    """
                    CREATE OR REPLACE FUNCTION protect_eom_terms_version()
                    RETURNS TRIGGER LANGUAGE plpgsql AS $$
                    BEGIN
                        RETURN NEW;
                    END;
                    $$
                    """
                )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await runtime_connection.execute(
                    "DELETE FROM eom_terms_current_version"
                )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await runtime_connection.execute("TRUNCATE eom_terms_versions")

        duplicate = await authority.create_draft(
            version_label="2026.2",
            material_change=False,
            documents=_documents("duplicate"),
            actor_id=7,
            actor_name="Juan",
        )
        duplicate_id = UUID(duplicate["versionId"])
        async with pool.raw_pool.acquire() as runtime_connection:
            with pytest.raises(
                asyncpg.PostgresError,
                match="Current EOM Terms version must be published",
            ):
                await runtime_connection.execute(
                    """
                    UPDATE eom_terms_current_version
                    SET version_id = $1
                    WHERE singleton
                    """,
                    duplicate_id,
                )

        duplicate_results = await asyncio.gather(
            authority.publish(
                version_id=duplicate_id,
                actor_id=7,
                actor_name="Juan",
            ),
            authority.publish(
                version_id=duplicate_id,
                actor_id=7,
                actor_name="Juan",
            ),
        )
        assert sorted(result["idempotent"] for result in duplicate_results) == [
            False,
            True,
        ]

        distinct_versions = []
        for suffix in ("3", "4"):
            created = await authority.create_draft(
                version_label=f"2026.{suffix}",
                material_change=True,
                documents=_documents(f"distinct-{suffix}"),
                actor_id=7,
                actor_name="Juan",
            )
            distinct_versions.append(UUID(created["versionId"]))

        publication_order: list[UUID] = []
        recording_authority = EOMTermsAuthority(
            pool=_RecordingCurrentPool(pool.raw_pool, publication_order)
        )

        async def publish_recorded(version_id: UUID) -> dict[str, Any]:
            return await recording_authority.publish(
                version_id=version_id,
                actor_id=7,
                actor_name="Juan",
            )

        distinct_results = await asyncio.gather(
            *(publish_recorded(version_id) for version_id in distinct_versions)
        )
        assert all(result["status"] == "published" for result in distinct_results)
        assert len(publication_order) == 2
        current = await authority.get_current()
        assert UUID(current["versionId"]) == publication_order[-1]
        assert (
            await pool.fetchval(
                """
            SELECT COUNT(*)
            FROM eom_terms_versions
            WHERE id = ANY($1::uuid[]) AND status = 'published'
            """,
                distinct_versions,
            )
            == 2
        )
        with pytest.raises(EOMTermsConflictError):
            await authority.publish(
                version_id=publication_order[0],
                actor_id=7,
                actor_name="Juan",
            )

        rollback_draft = await authority.create_draft(
            version_label="2026.rollback",
            material_change=True,
            documents=_documents("rollback"),
            actor_id=7,
            actor_name="Juan",
        )
        rollback_id = UUID(rollback_draft["versionId"])
        current_before_failure = await pool.fetchval(
            "SELECT version_id FROM eom_terms_current_version WHERE singleton"
        )
        failing_authority = EOMTermsAuthority(
            pool=_FailBeforeCurrentPool(pool.raw_pool)
        )
        with pytest.raises(
            EOMTermsUnavailableError,
            match="could not be published",
        ):
            await failing_authority.publish(
                version_id=rollback_id,
                actor_id=7,
                actor_name="Juan",
            )
        assert (
            await pool.fetchval(
                "SELECT status FROM eom_terms_versions WHERE id = $1",
                rollback_id,
            )
            == "draft"
        )
        assert (
            await pool.fetchval(
                "SELECT version_id FROM eom_terms_current_version WHERE singleton"
            )
            == current_before_failure
        )

        rewrite_probe = await authority.create_draft(
            version_label="2026.rewrite",
            material_change=True,
            documents=_documents("rewrite"),
            actor_id=7,
            actor_name="Juan",
        )
        with pytest.raises(
            asyncpg.PostgresError,
            match="Publishing EOM Terms cannot rewrite draft content",
        ):
            await dba_connection.execute(
                """
                UPDATE eom_terms_versions
                SET status = 'published',
                    published_by_id = 7,
                    published_by_name = 'Juan',
                    published_at = CURRENT_TIMESTAMP,
                    documents = '{"tampered": true}'::jsonb
                WHERE id = $1
                """,
                UUID(rewrite_probe["versionId"]),
            )
        with pytest.raises(
            asyncpg.PostgresError,
            match="Current EOM Terms authority cannot be removed",
        ):
            await dba_connection.execute("DELETE FROM eom_terms_current_version")
        with pytest.raises(
            asyncpg.PostgresError,
            match=(
                "EOM Terms version history is append-only"
                "|Current EOM Terms authority cannot be removed"
            ),
        ):
            await dba_connection.execute(
                "TRUNCATE eom_terms_versions, eom_terms_current_version"
            )

        await dba_connection.execute(
            """
            ALTER TABLE eom_terms_versions
            DISABLE TRIGGER trg_protect_eom_terms_version
            """
        )
        assert await eom_terms_authority_schema_ready(pool) is False
        await dba_connection.execute(
            """
            ALTER TABLE eom_terms_versions
            ENABLE TRIGGER trg_protect_eom_terms_version
            """
        )
        assert await eom_terms_authority_schema_ready(pool) is True

        await dba_connection.execute(
            "GRANT DELETE ON TABLE eom_terms_versions TO atlas"
        )
        assert await eom_terms_authority_schema_ready(pool) is False
        await dba_connection.execute(
            "REVOKE DELETE ON TABLE eom_terms_versions FROM atlas"
        )
        assert await eom_terms_authority_schema_ready(pool) is True

        await dba_connection.execute(
            """
            ALTER FUNCTION protect_eom_terms_version()
            SET search_path TO pg_catalog
            """
        )
        assert await eom_terms_authority_schema_ready(pool) is False
        await dba_connection.execute(
            f"""
            ALTER FUNCTION protect_eom_terms_version()
            SET search_path TO pg_catalog, "{schema}", pg_temp
            """
        )
        assert await eom_terms_authority_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_current_read_rejects_documents_that_do_not_match_stored_hash() -> None:
    corrupt_row = _row(status="published")
    corrupt_row["content_hash"] = "0" * 64
    with pytest.raises(EOMTermsUnavailableError, match="content hash"):
        await EOMTermsAuthority(
            pool=_Pool(connection=None, current_row=corrupt_row)
        ).get_current()


class _RouteAuthority:
    def __init__(self, *, idempotent: bool = False) -> None:
        self.draft = _row()
        self.published = _row(status="published")
        self.idempotent = idempotent

    async def create_draft(self, **_kwargs: Any) -> dict[str, Any]:
        return _api_result(self.draft, idempotent=self.idempotent)

    async def publish(self, **_kwargs: Any) -> dict[str, Any]:
        return _api_result(self.published, idempotent=self.idempotent)

    async def get_current(self) -> dict[str, Any]:
        return _api_result(self.published, idempotent=True)


class _MalformedRouteAuthority(_RouteAuthority):
    async def create_draft(self, **_kwargs: Any) -> dict[str, Any]:
        result = await super().create_draft(**_kwargs)
        result["unexpected"] = "must not escape"
        return result


class _ConflictRouteAuthority(_RouteAuthority):
    async def create_draft(self, **_kwargs: Any) -> dict[str, Any]:
        raise EOMTermsConflictError("Terms version label belongs to different content")


def _api_result(row: dict[str, Any], *, idempotent: bool) -> dict[str, Any]:
    documents = json.loads(row["documents"])
    return {
        "versionId": str(row["id"]),
        "versionLabel": row["version_label"],
        "status": row["status"],
        "materialChange": row["material_change"],
        "documents": documents,
        "contentHash": row["content_hash"],
        "createdById": row["created_by_id"],
        "createdByName": row["created_by_name"],
        "createdAt": row["created_at"].isoformat(),
        "publishedById": row["published_by_id"],
        "publishedByName": row["published_by_name"],
        "publishedAt": (
            row["published_at"].isoformat() if row["published_at"] else None
        ),
        "idempotent": idempotent,
    }


def _app(authority: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router, prefix="/api/v1")
    app.dependency_overrides[funnel_auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(
            api_enabled=True,
            service_token_sha256=_SERVICE.sha256,
        )
    )
    app.dependency_overrides[funnel_mod._terms_authority_dependency] = lambda: authority
    return app


def _headers(*, actor: bool = True) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_SERVICE.token}"}
    if actor:
        headers.update({"X-EOM-Actor": "Juan", "X-EOM-Actor-ID": "7"})
    return headers


@pytest.mark.asyncio
async def test_private_routes_create_publish_and_read_current_version() -> None:
    app = _app(_RouteAuthority())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/api/v1/eom-funnel/terms/versions",
            headers=_headers(),
            json={
                "versionLabel": "2026.1",
                "materialChange": True,
                "documents": _documents(),
            },
        )
        published = await client.post(
            f"/api/v1/eom-funnel/terms/versions/{_VERSION_ID}/publish",
            headers=_headers(),
        )
        current = await client.get(
            "/api/v1/eom-funnel/terms/current", headers=_headers(actor=False)
        )

    assert created.status_code == 201
    assert created.json()["status"] == "draft"
    assert published.status_code == 201
    assert published.json()["status"] == "published"
    assert current.status_code == 200
    assert current.json()["versionId"] == str(_VERSION_ID)


@pytest.mark.asyncio
async def test_create_route_still_requires_actor_evidence() -> None:
    app = _app(_RouteAuthority())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/eom-funnel/terms/versions",
            headers=_headers(actor=False),
            json={
                "versionLabel": "2026.1",
                "materialChange": True,
                "documents": _documents(),
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_private_terms_routes_require_service_bearer() -> None:
    app = _app(_RouteAuthority())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/v1/eom-funnel/terms/current")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unchanged_create_and_publish_replays_return_200() -> None:
    app = _app(_RouteAuthority(idempotent=True))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/api/v1/eom-funnel/terms/versions",
            headers=_headers(),
            json={
                "versionLabel": "2026.1",
                "materialChange": True,
                "documents": _documents(),
            },
        )
        published = await client.post(
            f"/api/v1/eom-funnel/terms/versions/{_VERSION_ID}/publish",
            headers=_headers(),
        )
    assert created.status_code == 200
    assert published.status_code == 200


@pytest.mark.asyncio
async def test_create_route_does_not_bypass_closed_response_projection() -> None:
    app = _app(_MalformedRouteAuthority())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/eom-funnel/terms/versions",
            headers=_headers(),
            json={
                "versionLabel": "2026.1",
                "materialChange": True,
                "documents": _documents(),
            },
        )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_create_route_maps_typed_service_conflict() -> None:
    app = _app(_ConflictRouteAuthority())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/eom-funnel/terms/versions",
            headers=_headers(),
            json={
                "versionLabel": "2026.1",
                "materialChange": True,
                "documents": _documents(),
            },
        )
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "eom_terms_conflict",
        "message": "Terms version label belongs to different content",
    }


def test_migration_guards_published_history_and_seeds_no_terms_content() -> None:
    migration = _MIGRATION.read_text()
    assert "OLD.status = 'published'" in migration
    assert "Current EOM Terms version must be published" in migration
    assert "BEFORE TRUNCATE ON eom_terms_versions" in migration
    assert "Publishing EOM Terms cannot rewrite draft content" in migration
    assert "BEFORE DELETE ON eom_terms_current_version" in migration
    assert "INSERT INTO eom_terms_versions" not in migration
