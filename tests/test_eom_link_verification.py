"""HTTP boundary proof for EOM contact-link verification.

The tracker stores its own copy of an Atlas contact id. Nothing on either side
notices when that copy stops resolving, which is the silent write-boundary
failure Slice 0 exists to close. These tests hold the route to answering that
one question and no more: which submitted ids name a live EOM contact.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
FOREIGN_TENANT = "churnsignals"

_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256


class _CRM:
    """Stands in for the EOM tenant slice of the contacts table."""

    def __init__(self, *, known: list[UUID] | None = None) -> None:
        self.known = known or []
        self.calls: list[list[UUID]] = []

    async def list_known_eom_contact_ids(
        self, *, contact_ids: list[UUID]
    ) -> list[UUID]:
        self.calls.append(list(contact_ids))
        return [value for value in self.known if value in set(contact_ids)]


def _app(crm: _CRM) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(
            api_enabled=True,
            service_token_sha256=_SERVICE_TOKEN_SHA256,
        )
    )
    return app


def _headers(token: str = _SERVICE_TOKEN) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EOM-Actor": "Juan Canfield",
        "X-EOM-Actor-ID": "1",
    }


async def _get(crm: _CRM, query: str, **kwargs: object) -> httpx.Response:
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get(
            f"/eom-funnel/known-contacts{query}",
            headers=_headers(),
            **kwargs,
        )


@pytest.mark.asyncio
async def test_a_dangling_link_is_reported_as_unknown():
    """The signal that matters: a stored id that no longer resolves."""
    live = uuid4()
    dangling = uuid4()
    crm = _CRM(known=[live])

    response = await _get(crm, f"?contact_id={live}&contact_id={dangling}")

    assert response.status_code == 200
    body = response.json()
    assert body["knownContactIds"] == [str(live)]
    assert str(dangling) not in body["knownContactIds"]
    assert body["checked"] == 2


@pytest.mark.asyncio
async def test_a_live_contact_is_not_reported_as_dangling():
    """The other direction -- a route that never confirms proves nothing."""
    first = uuid4()
    second = uuid4()
    crm = _CRM(known=[first, second])

    response = await _get(crm, f"?contact_id={first}&contact_id={second}")

    assert response.status_code == 200
    assert response.json()["knownContactIds"] == [str(first), str(second)]


@pytest.mark.asyncio
async def test_a_contact_in_another_tenant_is_not_known_here():
    """Cross-tenant ids read exactly like missing ones, by design.

    The provider is scoped to ``effingham_maids``, so an id from another
    business context never reaches the route. Reporting it any other way would
    turn an EOM-scoped credential into a cross-tenant existence oracle.
    """
    churnsignals_contact = uuid4()
    crm = _CRM(known=[])

    response = await _get(crm, f"?contact_id={churnsignals_contact}")

    assert response.status_code == 200
    assert response.json()["knownContactIds"] == []


@pytest.mark.asyncio
async def test_an_id_the_caller_never_submitted_is_never_returned():
    """A verdict the caller cannot attribute to a link it holds is not an answer."""
    asked = uuid4()
    never_asked = uuid4()

    class _LeakyCRM(_CRM):
        async def list_known_eom_contact_ids(
            self, *, contact_ids: list[UUID]
        ) -> list[UUID]:
            self.calls.append(list(contact_ids))
            return [asked, never_asked]

    response = await _get(_LeakyCRM(), f"?contact_id={asked}")

    assert response.status_code == 200
    assert response.json()["knownContactIds"] == [str(asked)]


@pytest.mark.asyncio
async def test_repeated_ids_are_asked_once_and_answered_once():
    live = uuid4()
    crm = _CRM(known=[live])

    response = await _get(crm, f"?contact_id={live}&contact_id={live}")

    assert response.status_code == 200
    assert response.json() == {
        "knownContactIds": [str(live)],
        "checked": 1,
        "limit": funnel_mod._MAX_KNOWN_CONTACT_IDS,
    }
    assert crm.calls == [[live]]


@pytest.mark.asyncio
async def test_an_empty_check_is_rejected_rather_than_answered_clean():
    """An empty request must not read as 'every link is fine'."""
    crm = _CRM(known=[])

    response = await _get(crm, "")

    assert response.status_code == 422
    assert crm.calls == []


@pytest.mark.asyncio
async def test_the_cap_is_enforced_at_the_boundary():
    crm = _CRM(known=[])
    over = funnel_mod._MAX_KNOWN_CONTACT_IDS + 1
    query = "?" + "&".join(f"contact_id={uuid4()}" for _ in range(over))

    response = await _get(crm, query)

    assert response.status_code == 422
    assert crm.calls == []


@pytest.mark.asyncio
async def test_exactly_the_cap_is_accepted():
    """max is allowed, max+1 is not -- the second side of the boundary."""
    crm = _CRM(known=[])
    ids = [uuid4() for _ in range(funnel_mod._MAX_KNOWN_CONTACT_IDS)]
    query = "?" + "&".join(f"contact_id={value}" for value in ids)

    response = await _get(crm, query)

    assert response.status_code == 200
    assert response.json()["checked"] == funnel_mod._MAX_KNOWN_CONTACT_IDS


@pytest.mark.asyncio
async def test_a_malformed_id_is_rejected_not_silently_dropped():
    """Dropping an unparseable id would report it as neither known nor asked."""
    crm = _CRM(known=[])

    response = await _get(crm, f"?contact_id={uuid4()}&contact_id=not-a-uuid")

    assert response.status_code == 422
    assert crm.calls == []


@pytest.mark.asyncio
async def test_the_route_refuses_an_unauthenticated_caller():
    crm = _CRM(known=[uuid4()])
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/eom-funnel/known-contacts?contact_id={uuid4()}")

    assert response.status_code in (401, 403)
    assert crm.calls == []


@pytest.mark.asyncio
async def test_a_wrong_service_token_is_refused():
    crm = _CRM(known=[uuid4()])
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/eom-funnel/known-contacts?contact_id={uuid4()}",
            headers=_headers(token="wrong-token"),
        )

    assert response.status_code in (401, 403)
    assert crm.calls == []


@pytest.mark.asyncio
async def test_the_route_discloses_no_contact_data_beyond_the_id():
    """Callers get resolution, not a read of the CRM record."""
    live = uuid4()
    crm = _CRM(known=[live])

    response = await _get(crm, f"?contact_id={live}")

    assert set(response.json()) == {"knownContactIds", "checked", "limit"}


def test_link_verification_is_advertised_in_the_capability_manifest():
    """Callers gate on the manifest, so an unlisted route is an unusable one."""
    assert "contact.link_verification" in funnel_mod.served_capabilities()


@pytest.mark.asyncio
async def test_the_real_aggregate_serves_the_route_at_its_deployed_path():
    """Prove reachability on the app that actually ships.

    Every other test in this file mounts ``funnel_mod.router`` on a fresh
    FastAPI instance, which proves the handler works but says nothing about
    whether the deployed application mounts it. If the aggregate stopped
    including the EOM funnel router, or mounted it under a different prefix,
    all of them would stay green while the caller got a 404 from the one path
    the review contract names. This calls the full deployed path instead.
    """
    from atlas_brain.main import app

    live = uuid4()
    dangling = uuid4()
    crm = _CRM(known=[live])

    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(
            api_enabled=True,
            service_token_sha256=_SERVICE_TOKEN_SHA256,
        )
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/eom-funnel/known-contacts"
                f"?contact_id={live}&contact_id={dangling}",
                headers=_headers(),
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200, (
        "the deployed path must serve this route, not 404"
    )
    body = response.json()
    assert body["knownContactIds"] == [str(live)]
    assert str(dangling) not in body["knownContactIds"]


@pytest.mark.asyncio
async def test_tenant_scope_holds_against_real_postgres():
    """The scope claim is only worth what the SQL does, so prove it there.

    Every test above stubs the provider, which means none of them can tell a
    tenant-scoped query from an unscoped one. This one seeds a real EOM contact
    beside a real churnsignals contact and asks for both: an unscoped
    ``WHERE c.id = ANY($1)`` would return the foreign row and fail here.
    """
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    schema = f"atlas_eom_known_contacts_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    try:
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        # 035 backfills contacts from appointments and call transcripts, so the
        # tables it reads have to exist even though this test only uses contacts.
        for name in (
            "001_initial_schema.sql",
            "012_appointments.sql",
            "030_call_transcripts.sql",
            "035_contacts.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())

        async def set_search_path(connection):
            await connection.execute(f'SET search_path TO "{schema}", public')

        pool = await asyncpg.create_pool(
            database_url, min_size=1, max_size=2, setup=set_search_path
        )
        # Injected through the constructor, which exists for exactly this.
        # Reaching into atlas_brain.storage.database._db_pool would test the
        # same SQL while coupling this file to a private module attribute.
        provider = DatabaseCRMProvider(pool=_PoolAdapter(pool))

        eom_row = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('EOM Customer', $1) RETURNING id
            """,
            TENANT,
        )
        foreign_row = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Churnsignals Customer', $1) RETURNING id
            """,
            FOREIGN_TENANT,
        )
        deleted_id = uuid4()

        known = await provider.list_known_eom_contact_ids(
            contact_ids=[eom_row["id"], foreign_row["id"], deleted_id]
        )

        assert eom_row["id"] in known, "a live EOM contact must resolve"
        assert foreign_row["id"] not in known, (
            "another tenant's contact must read exactly like a missing one"
        )
        assert deleted_id not in known
    finally:
        if pool is not None:
            await pool.close()
        await admin_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin_conn.close()


class _PoolAdapter:
    def __init__(self, pool):
        self._pool = pool
        self.is_initialized = True

    async def fetch(self, query, *args):
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query, *args):
        return await self._pool.fetchrow(query, *args)

    async def execute(self, query, *args):
        return await self._pool.execute(query, *args)
