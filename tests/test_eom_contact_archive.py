"""EOM contact archive/restore: canonical, receipted, reversible (website #253).

Archive is the status-axis sibling of the lost/reopen disposition pair: the
same advisory-lock, replay-receipt, key-ownership, and truthful-idempotency
machine, applied to ``contacts.status`` instead of ``lead_stage``. These tests
hold the new pair to that machine's standards at three layers: the HTTP
boundary (auth, actor, key discipline, error mapping, capability truth), the
admission guard (a generated matrix judged by a spec-derived oracle), and the
real-Postgres transaction (receipts, replays, ABA supersession, the won-loss
cancellation fence, and the directory's disjoint lifecycle views).
"""

from __future__ import annotations

import os
import uuid
from itertools import product
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
FOREIGN_TENANT = "churnsignals"

_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256


class _TransitionCRM:
    """Spy provider for the route layer: records calls, echoes results."""

    def __init__(self) -> None:
        self.archive_calls: list[dict[str, object]] = []
        self.restore_calls: list[dict[str, object]] = []
        self.archive_result: dict[str, object] | None = None
        self.restore_result: dict[str, object] | None = None
        self.error: Exception | None = None

    async def archive_eom_contact(self, **kwargs: object) -> dict[str, object]:
        self.archive_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(
            self.archive_result
            or {
                "contact_id": kwargs["contact_id"],
                "contact_type": "customer",
                "lead_stage": None,
                "status": "archived",
                "idempotent": False,
            }
        )

    async def restore_eom_contact(self, **kwargs: object) -> dict[str, object]:
        self.restore_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(
            self.restore_result
            or {
                "contact_id": kwargs["contact_id"],
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
                "idempotent": False,
            }
        )

    async def list_eom_new_lead_review_items(self, **kwargs: object) -> list[dict]:
        return []

    def __getattr__(self, name: str):
        # Any other provider access from these routes is a scope leak; make it
        # loud instead of silently succeeding.
        raise AssertionError(f"archive/restore must not touch crm.{name}")


def _app(crm: _TransitionCRM) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(api_enabled=True, service_token_sha256=_SERVICE_TOKEN_SHA256)
    )
    return app


def _operation_key() -> str:
    return f"office-archive-{uuid4().hex}"


def _headers(
    token: str = _SERVICE_TOKEN, operation_key: str | None = None
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EOM-Actor": "Juan Canfield",
        "X-EOM-Actor-ID": "1",
        "Idempotency-Key": operation_key or _operation_key(),
    }


async def _post(
    crm: _TransitionCRM,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(
            path, headers=_headers() if headers is None else headers
        )


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    funnel_mod._served_capabilities_cache = None
    yield
    funnel_mod._served_capabilities_cache = None


# ---------------------------------------------------------------------------
# Route boundary: auth, actor, key discipline, echo, error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_routes_refuse_unauthenticated_and_wrong_token_callers():
    wrong = auth_mod.generate_eom_funnel_service_token().token
    for action in ("archive", "restore"):
        path = f"/eom-funnel/contacts/{uuid4()}/{action}"
        crm = _TransitionCRM()
        app = _app(crm)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post(path)).status_code == 401
            assert (
                await client.post(path, headers=_headers(token=wrong))
            ).status_code == 401
        assert crm.archive_calls == [] and crm.restore_calls == []


@pytest.mark.asyncio
async def test_both_routes_require_an_actor_and_a_wellformed_key():
    for action in ("archive", "restore"):
        path = f"/eom-funnel/contacts/{uuid4()}/{action}"
        missing_actor = _headers()
        missing_actor.pop("X-EOM-Actor")
        assert (
            await _post(_TransitionCRM(), path, headers=missing_actor)
        ).status_code == 422
        short_key = _headers(operation_key="too-short")
        assert (
            await _post(_TransitionCRM(), path, headers=short_key)
        ).status_code == 422


@pytest.mark.asyncio
async def test_archive_returns_201_and_echoes_identity_status_and_actor():
    crm = _TransitionCRM()
    contact_id = uuid4()
    op_key = _operation_key()
    response = await _post(
        crm,
        f"/eom-funnel/contacts/{contact_id}/archive",
        headers=_headers(operation_key=op_key),
    )
    assert response.status_code == 201
    assert response.json() == {
        "success": True,
        "contact_id": str(contact_id),
        "contact_type": "customer",
        "lead_stage": None,
        "status": "archived",
        "idempotent": False,
    }
    assert crm.archive_calls == [
        {
            "contact_id": str(contact_id),
            "operation_key": op_key,
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_restore_returns_201_and_echoes_identity_status_and_actor():
    crm = _TransitionCRM()
    contact_id = uuid4()
    op_key = _operation_key()
    response = await _post(
        crm,
        f"/eom-funnel/contacts/{contact_id}/restore",
        headers=_headers(operation_key=op_key),
    )
    assert response.status_code == 201
    assert response.json() == {
        "success": True,
        "contact_id": str(contact_id),
        "contact_type": "customer",
        "lead_stage": None,
        "status": "active",
        "idempotent": False,
    }
    assert crm.restore_calls == [
        {
            "contact_id": str(contact_id),
            "operation_key": op_key,
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_idempotent_replays_return_200_not_201():
    crm = _TransitionCRM()
    crm.archive_result = {
        "contact_id": "ignored",
        "contact_type": "lead",
        "lead_stage": "lost",
        "status": "archived",
        "idempotent": True,
    }
    crm.restore_result = {
        "contact_id": "ignored",
        "contact_type": "lead",
        "lead_stage": "lost",
        "status": "active",
        "idempotent": True,
    }
    for action in ("archive", "restore"):
        response = await _post(crm, f"/eom-funnel/contacts/{uuid4()}/{action}")
        assert response.status_code == 200, action
        assert response.json()["idempotent"] is True, action


@pytest.mark.asyncio
async def test_conversion_errors_map_to_their_status_and_detail():
    for action, code, message in (
        ("archive", 404, "EOM contact was not found"),
        ("archive", 409, "EOM contact is already archived"),
        ("restore", 404, "EOM contact was not found"),
        ("restore", 409, "EOM contact is already active"),
    ):
        crm = _TransitionCRM()
        crm.error = EOMLeadConversionError(code, message)
        response = await _post(crm, f"/eom-funnel/contacts/{uuid4()}/{action}")
        assert response.status_code == code, (action, code)
        assert response.json()["detail"] == message, (action, code)


# ---------------------------------------------------------------------------
# Capability truth
# ---------------------------------------------------------------------------


def test_the_transitions_and_archived_view_are_advertised_from_real_routes():
    served = funnel_mod.served_capabilities()
    for name in ("contact.archive", "contact.restore", "contact.directory.archived"):
        assert name in served, name
    routes = funnel_mod.served_capability_routes()
    assert ("POST", "/eom-funnel/contacts/{contact_id}/archive") in routes
    assert ("POST", "/eom-funnel/contacts/{contact_id}/restore") in routes
    # The archived-view proof rides the directory's registered GET: the name
    # exists only in builds whose directory understands `lifecycle`.
    assert ("GET", "/eom-funnel/contact-directory") in routes


@pytest.mark.asyncio
async def test_the_lead_review_envelope_advertises_all_three_names():
    crm = _TransitionCRM()
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads", headers=_headers())
    body = response.json()
    for name in ("contact.archive", "contact.restore", "contact.directory.archived"):
        assert name in body["capabilities"], name
    for signature in (
        {"method": "POST", "path": "/eom-funnel/contacts/{contact_id}/archive"},
        {"method": "POST", "path": "/eom-funnel/contacts/{contact_id}/restore"},
    ):
        assert signature in body["capabilityRoutes"], signature


def _deployed_apps():
    """Both application objects a real deployment starts (see the directory
    tests for why proving only one of them is not a reachability proof)."""
    from atlas_brain.main import app as aggregate_app
    from atlas_brain.main_eom import app as eom_app

    return [("main", aggregate_app), ("main_eom", eom_app)]


@pytest.mark.asyncio
async def test_every_deployed_entrypoint_serves_both_transition_routes():
    for name, app in _deployed_apps():
        for action in ("archive", "restore"):
            crm = _TransitionCRM()
            original_overrides = dict(app.dependency_overrides)
            app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
            app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
                EOMFunnelConfig(
                    api_enabled=True, service_token_sha256=_SERVICE_TOKEN_SHA256
                )
            )
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        f"/api/v1/eom-funnel/contacts/{uuid4()}/{action}",
                        headers=_headers(),
                    )
            finally:
                app.dependency_overrides.clear()
                app.dependency_overrides.update(original_overrides)
            assert response.status_code == 201, (
                f"{name} must serve the deployed {action} path, not 404"
            )


# ---------------------------------------------------------------------------
# Real-Postgres transaction proofs
# ---------------------------------------------------------------------------


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


class _PoolAdapter:
    """Delegates reads AND exposes ``acquire`` so ``_transaction_connection``
    runs each transition inside a real transaction -- the advisory xact locks
    and multi-statement receipts under test are meaningless in autocommit."""

    def __init__(self, pool):
        self._pool = pool
        self.is_initialized = True

    def acquire(self):
        return self._pool.acquire()

    async def fetch(self, query, *args):
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query, *args):
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query, *args):
        return await self._pool.fetchval(query, *args)

    async def execute(self, query, *args):
        return await self._pool.execute(query, *args)


@pytest.fixture()
async def _archive_provider():
    """A DatabaseCRMProvider over a disposable schema with contacts AND the
    lifecycle-events ledger (351) plus its append sequence (363)."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url_or_skip()

    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    schema = f"atlas_eom_contact_archive_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    try:
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        for name in (
            "001_initial_schema.sql",
            "012_appointments.sql",
            "030_call_transcripts.sql",
            "035_contacts.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
            "363_eom_lead_lifecycle_sequence.sql",
            "366_contacts_customer_type.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())

        async def set_search_path(connection):
            await connection.execute(f'SET search_path TO "{schema}", public')

        pool = await asyncpg.create_pool(
            database_url, min_size=1, max_size=2, setup=set_search_path
        )
        provider = DatabaseCRMProvider(pool=_PoolAdapter(pool))
        yield provider, pool
    finally:
        if pool is not None:
            await pool.close()
        await admin_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin_conn.close()


async def _seed(pool, **overrides):
    fields = {
        "full_name": "Archive Subject",
        "business_context_id": TENANT,
        "contact_type": "customer",
        "status": "active",
        "email": None,
        "phone": None,
    }
    fields.update({k: v for k, v in overrides.items() if k != "lead_stage"})
    row = await pool.fetchrow(
        """
        INSERT INTO contacts (full_name, business_context_id, contact_type,
                              status, email, phone)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        fields["full_name"],
        fields["business_context_id"],
        fields["contact_type"],
        fields["status"],
        fields["email"],
        fields["phone"],
    )
    if overrides.get("lead_stage") is not None:
        await pool.execute(
            "UPDATE contacts SET lead_stage = $2 WHERE id = $1",
            row["id"],
            overrides["lead_stage"],
        )
    return row["id"]


async def _events(pool, contact_id, event_type):
    return await pool.fetch(
        """
        SELECT event_type, from_stage, to_stage, actor, source, operation_key,
               metadata, lifecycle_sequence
        FROM eom_lead_lifecycle_events
        WHERE contact_id = $1 AND event_type = $2
        ORDER BY lifecycle_sequence
        """,
        contact_id,
        event_type,
    )


_ACTOR = {"actor_id": 1, "actor_name": "Juan Canfield"}

# Grammar axes for the admission matrix. Tuples, never flat scalar
# assignments: each axis is a closed family the oracle reasons over.
_OPERATION_FAMILIES = ("archive", "restore")
_TENANT_CONTAINERS = (TENANT, FOREIGN_TENANT)
_STATUS_CONTAINERS = ("active", "archived")
_KIND_STAGE_TOKENS = (
    ("customer", None),
    ("lead", "new"),
    ("lead", "estimate_booked"),
    ("lead", "lost"),
    ("lead", "won"),
    ("vendor", None),
)


def _transition_oracle(operation, tenant, status, contact_type, lead_stage):
    """Spec-derived expected verdict, independent of implementation branch
    order: foreign tenants read as missing; archive admits active directory
    kinds except a won-stage lead (the lost flow owns that teardown); restore
    admits exactly the archived directory kinds."""
    if tenant != TENANT:
        return ("error", 404)
    if contact_type not in ("lead", "customer"):
        return ("error", 409)
    if operation == "archive":
        if status != "active":
            return ("error", 409)
        if contact_type == "lead" and lead_stage == "won":
            return ("error", 409)
        return ("ok", "archived")
    if status != "archived":
        return ("error", 409)
    return ("ok", "active")


@pytest.mark.asyncio
async def test_transition_admission_holds_across_operations_tenants_and_kinds(
    _archive_provider,
):
    """Class-closure proof for the transition admission guard: cases are
    GENERATED over four grammar axes -- operation families x tenant containers
    x status containers x kind/stage tokens -- and every generated case is
    judged by the spec-derived oracle above, not by a sampled fixture list."""
    provider, pool = _archive_provider
    for operation, tenant, status, (contact_type, lead_stage) in product(
        _OPERATION_FAMILIES,
        _TENANT_CONTAINERS,
        _STATUS_CONTAINERS,
        _KIND_STAGE_TOKENS,
    ):
        case = (operation, tenant, status, contact_type, lead_stage)
        contact_id = await _seed(
            pool,
            business_context_id=tenant,
            status=status,
            contact_type=contact_type,
            lead_stage=lead_stage,
        )
        method = (
            provider.archive_eom_contact
            if operation == "archive"
            else provider.restore_eom_contact
        )
        expected = _transition_oracle(
            operation, tenant, status, contact_type, lead_stage
        )
        if expected[0] == "error":
            with pytest.raises(EOMLeadConversionError) as err:
                await method(
                    contact_id=str(contact_id),
                    operation_key=f"matrix-{uuid4().hex}",
                    **_ACTOR,
                )
            assert err.value.status_code == expected[1], case
            unchanged = await pool.fetchrow(
                "SELECT status FROM contacts WHERE id = $1", contact_id
            )
            assert unchanged["status"] == status, case
        else:
            result = await method(
                contact_id=str(contact_id),
                operation_key=f"matrix-{uuid4().hex}",
                **_ACTOR,
            )
            assert result == {
                "contact_id": str(contact_id),
                "contact_type": contact_type,
                "lead_stage": lead_stage,
                "status": expected[1],
                "idempotent": False,
            }, case
            moved = await pool.fetchrow(
                "SELECT status, lead_stage FROM contacts WHERE id = $1", contact_id
            )
            assert moved["status"] == expected[1], case
            assert moved["lead_stage"] == lead_stage, (
                "the stage axis must never move",
                case,
            )


@pytest.mark.asyncio
async def test_archive_writes_a_sequenced_receipt_and_replays_truthfully(
    _archive_provider,
):
    provider, pool = _archive_provider
    contact_id = await _seed(pool, contact_type="lead", lead_stage="lost")
    key = f"archive-{uuid4().hex}"

    fresh = await provider.archive_eom_contact(
        contact_id=str(contact_id), operation_key=key, **_ACTOR
    )
    assert fresh["idempotent"] is False

    events = await _events(pool, contact_id, "contact_archived")
    assert len(events) == 1
    event = events[0]
    assert (event["from_stage"], event["to_stage"]) == ("active", "archived")
    assert event["actor"] == "employee:1:Juan Canfield"
    assert event["source"] == "eom_office"
    assert event["operation_key"] == key
    assert event["lifecycle_sequence"] is not None
    import json as _json

    metadata = _json.loads(event["metadata"])
    assert metadata["previous_status"] == "active"
    assert metadata["resulting_status"] == "archived"
    assert metadata["contact_type"] == "lead"
    assert metadata["lead_stage"] == "lost"
    assert metadata["archived_by_employee_id"] == 1

    replay = await provider.archive_eom_contact(
        contact_id=str(contact_id), operation_key=key, **_ACTOR
    )
    assert replay == {
        "contact_id": str(contact_id),
        "contact_type": "lead",
        "lead_stage": "lost",
        "status": "archived",
        "idempotent": True,
    }
    assert len(await _events(pool, contact_id, "contact_archived")) == 1, (
        "a replay must not write a second receipt"
    )

    with pytest.raises(EOMLeadConversionError) as err:
        await provider.archive_eom_contact(
            contact_id=str(contact_id),
            operation_key=f"archive-{uuid4().hex}",
            **_ACTOR,
        )
    assert err.value.status_code == 409, "already archived under a different key"


@pytest.mark.asyncio
async def test_restore_round_trip_and_aba_replays_are_refused(_archive_provider):
    provider, pool = _archive_provider
    contact_id = await _seed(pool)
    archive_key = f"archive-{uuid4().hex}"
    restore_key = f"restore-{uuid4().hex}"

    await provider.archive_eom_contact(
        contact_id=str(contact_id), operation_key=archive_key, **_ACTOR
    )
    restored = await provider.restore_eom_contact(
        contact_id=str(contact_id), operation_key=restore_key, **_ACTOR
    )
    assert restored["status"] == "active"
    assert restored["idempotent"] is False

    events = await _events(pool, contact_id, "contact_restored")
    assert len(events) == 1
    assert (events[0]["from_stage"], events[0]["to_stage"]) == ("archived", "active")

    restore_replay = await provider.restore_eom_contact(
        contact_id=str(contact_id), operation_key=restore_key, **_ACTOR
    )
    assert restore_replay["idempotent"] is True

    # The archive replay now reports a status the row no longer has.
    with pytest.raises(EOMLeadConversionError) as err:
        await provider.archive_eom_contact(
            contact_id=str(contact_id), operation_key=archive_key, **_ACTOR
        )
    assert err.value.status_code == 409

    # Re-archive under a new key, completing the ABA shape: the row's status
    # matches the ORIGINAL archive receipt again, but a later disposition owns
    # it, so both original keys must refuse rather than report ownership.
    await provider.archive_eom_contact(
        contact_id=str(contact_id),
        operation_key=f"archive-{uuid4().hex}",
        **_ACTOR,
    )
    with pytest.raises(EOMLeadConversionError) as aba_archive:
        await provider.archive_eom_contact(
            contact_id=str(contact_id), operation_key=archive_key, **_ACTOR
        )
    assert aba_archive.value.status_code == 409
    with pytest.raises(EOMLeadConversionError) as aba_restore:
        await provider.restore_eom_contact(
            contact_id=str(contact_id), operation_key=restore_key, **_ACTOR
        )
    assert aba_restore.value.status_code == 409


@pytest.mark.asyncio
async def test_an_operation_key_belongs_to_exactly_one_contact(_archive_provider):
    provider, pool = _archive_provider
    first = await _seed(pool)
    second = await _seed(pool, full_name="Second Subject")
    key = f"archive-{uuid4().hex}"

    await provider.archive_eom_contact(
        contact_id=str(first), operation_key=key, **_ACTOR
    )
    with pytest.raises(EOMLeadConversionError) as err:
        await provider.archive_eom_contact(
            contact_id=str(second), operation_key=key, **_ACTOR
        )
    assert err.value.status_code == 409
    unchanged = await pool.fetchrow(
        "SELECT status FROM contacts WHERE id = $1", second
    )
    assert unchanged["status"] == "active"


@pytest.mark.asyncio
async def test_the_wonloss_cancellation_fence_blocks_both_transitions(
    _archive_provider,
):
    """An unresolved first-clean cancellation is the authoritative fence for
    ANY status flip -- the same evidence delete_contact honors. No current
    writer can produce this state (the fence blocks archive during it), so it
    is constructed directly; the guard must still fail closed on it."""
    provider, pool = _archive_provider

    async def _fence(contact_id):
        await pool.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, source, operation_key
            )
            VALUES ($1, 'first_clean_cancellation_requested', 'eom_office', $2)
            """,
            contact_id,
            f"wonloss-{uuid4().hex}",
        )

    active = await _seed(pool)
    await _fence(active)
    with pytest.raises(EOMLeadConversionError) as archive_err:
        await provider.archive_eom_contact(
            contact_id=str(active), operation_key=f"archive-{uuid4().hex}", **_ACTOR
        )
    assert archive_err.value.status_code == 409
    assert "reconciliation" in str(archive_err.value)

    archived = await _seed(pool, status="archived")
    await _fence(archived)
    with pytest.raises(EOMLeadConversionError) as restore_err:
        await provider.restore_eom_contact(
            contact_id=str(archived), operation_key=f"restore-{uuid4().hex}", **_ACTOR
        )
    assert restore_err.value.status_code == 409
    assert "reconciliation" in str(restore_err.value)


@pytest.mark.asyncio
async def test_directory_lifecycle_views_are_disjoint_and_restore_returns_once(
    _archive_provider,
):
    provider, pool = _archive_provider
    staying = await _seed(pool, full_name="Staying Active")
    moving = await _seed(pool, full_name="Moving Contact")
    parked = await _seed(pool, full_name="Already Parked", status="archived")

    def ids(rows):
        return [row["contact_id"] for row in rows]

    active_view = ids(await provider.list_eom_contact_directory(limit=50))
    archived_view = ids(
        await provider.list_eom_contact_directory(limit=50, lifecycle="archived")
    )
    assert moving in active_view and parked not in active_view
    assert parked in archived_view and moving not in archived_view

    await provider.archive_eom_contact(
        contact_id=str(moving), operation_key=f"archive-{uuid4().hex}", **_ACTOR
    )
    active_view = ids(await provider.list_eom_contact_directory(limit=50))
    archived_view = ids(
        await provider.list_eom_contact_directory(limit=50, lifecycle="archived")
    )
    assert moving not in active_view
    assert archived_view.count(moving) == 1
    assert all(
        row["status"] == "archived"
        for row in await provider.list_eom_contact_directory(
            limit=50, lifecycle="archived"
        )
    )

    await provider.restore_eom_contact(
        contact_id=str(moving), operation_key=f"restore-{uuid4().hex}", **_ACTOR
    )
    active_view = ids(await provider.list_eom_contact_directory(limit=50))
    archived_view = ids(
        await provider.list_eom_contact_directory(limit=50, lifecycle="archived")
    )
    assert active_view.count(moving) == 1, "restore returns the row exactly once"
    assert moving not in archived_view
    assert staying in active_view

    with pytest.raises(ValueError):
        await provider.list_eom_contact_directory(limit=50, lifecycle="junk")


# Grammar axes for the disclosure-ordering matrix: target containers x key
# provenance tokens x operation families. Separate from the admission matrix
# above because its oracle is about WHICH refusal wins, not whether one fires.
_TARGET_CONTAINERS = ("eom", "foreign", "missing")
_KEY_PROVENANCE_TOKENS = ("fresh", "foreign-owned")


def _ordering_oracle(target, provenance):
    """Spec-derived verdict: tenancy resolves before key ownership, so a
    foreign or missing target reads 404 regardless of the key's provenance;
    only a legitimate EOM target may learn of an ownership conflict."""
    if target in ("foreign", "missing"):
        return ("error", 404)
    if provenance == "foreign-owned":
        return ("error", 409)
    return ("ok", None)


@pytest.mark.asyncio
async def test_tenancy_resolves_before_key_ownership_across_targets_and_keys(
    _archive_provider,
):
    """Class-closure proof for the refusal ORDER: cases are generated over
    operation families x target containers x key-provenance tokens and judged
    by the oracle above. A foreign-owned key must never convert a 404 target
    into a 409 -- that would disclose the key's existence to a caller with no
    right to the target."""
    provider, pool = _archive_provider
    for operation, target, provenance in product(
        _OPERATION_FAMILIES, _TARGET_CONTAINERS, _KEY_PROVENANCE_TOKENS
    ):
        case = (operation, target, provenance)
        key = f"order-{uuid4().hex}"
        if provenance == "foreign-owned":
            # The key already belongs to ANOTHER EOM contact's receipt of
            # this same operation.
            owner = await _seed(
                pool,
                full_name="Key Owner",
                status="active" if operation == "archive" else "archived",
            )
            method = (
                provider.archive_eom_contact
                if operation == "archive"
                else provider.restore_eom_contact
            )
            await method(contact_id=str(owner), operation_key=key, **_ACTOR)
        if target == "missing":
            target_id = uuid4()
        else:
            target_id = await _seed(
                pool,
                business_context_id=TENANT if target == "eom" else FOREIGN_TENANT,
                status="active" if operation == "archive" else "archived",
            )
        method = (
            provider.archive_eom_contact
            if operation == "archive"
            else provider.restore_eom_contact
        )
        expected = _ordering_oracle(target, provenance)
        if expected[0] == "error":
            with pytest.raises(EOMLeadConversionError) as err:
                await method(contact_id=str(target_id), operation_key=key, **_ACTOR)
            assert err.value.status_code == expected[1], case
            if expected[1] == 404:
                assert "was not found" in str(err.value), case
        else:
            result = await method(
                contact_id=str(target_id), operation_key=key, **_ACTOR
            )
            assert result["idempotent"] is False, case


def _split_migration_statements(sql: str) -> list[str]:
    """Strip comments and split on ';' so CONCURRENTLY statements can run
    outside a transaction block, mirroring the runner's own handling."""
    lines = [
        line for line in sql.splitlines() if not line.strip().startswith("--")
    ]
    return [
        statement.strip()
        for statement in "\n".join(lines).split(";")
        if statement.strip()
    ]


@pytest.mark.asyncio
async def test_migration_388_replaces_the_disposition_index_and_replays():
    """Real-PostgreSQL proof for the 362 -> 388 index replacement: the
    resulting partial index is VALID and its predicate covers all four
    disposition event types, and the drop-then-recreate pair replays cleanly
    (the canceled-startup recovery path the header documents)."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_m388_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        for name in (
            "001_initial_schema.sql",
            "012_appointments.sql",
            "030_call_transcripts.sql",
            "035_contacts.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
        ):
            await conn.execute((MIGRATIONS / name).read_text())

        async def apply_split(name):
            for statement in _split_migration_statements(
                (MIGRATIONS / name).read_text()
            ):
                await conn.execute(statement)

        async def index_row():
            return await conn.fetchrow(
                """
                SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) AS indexdef
                FROM pg_index AS i
                JOIN pg_class AS c ON c.oid = i.indexrelid
                JOIN pg_namespace AS n ON n.oid = c.relnamespace
                WHERE n.nspname = $1
                  AND c.relname = 'idx_eom_lead_lifecycle_disposition_operation_key'
                """,
                schema,
            )

        await apply_split("362_eom_lead_disposition_operation_key_index.sql")
        before = await index_row()
        assert before is not None and before["indisvalid"] is True
        assert "contact_archived" not in before["indexdef"]

        await apply_split("388_eom_contact_archive_disposition_index.sql")
        after = await index_row()
        assert after is not None, "388 must leave the disposition index present"
        assert after["indisvalid"] is True, "the replacement must be VALID"
        for event_type in (
            "lead_lost",
            "lead_reopened",
            "contact_archived",
            "contact_restored",
        ):
            assert event_type in after["indexdef"], event_type
        assert "operation_key IS NOT NULL" in after["indexdef"]

        # Replay: the drop-then-recreate pair must recover cleanly when run
        # again (the same guarantee the runner relies on after a canceled
        # startup left any prior state behind).
        await apply_split("388_eom_contact_archive_disposition_index.sql")
        replayed = await index_row()
        assert replayed is not None and replayed["indisvalid"] is True
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
