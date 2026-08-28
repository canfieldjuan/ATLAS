"""Focused proof for the private EOM Terms version authority."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

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
