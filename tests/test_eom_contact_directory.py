"""EOM contact-directory read: bounded, tenant-scoped, read-only (website #240).

The directory is the discovery boundary the operator mutation never had: the
pipeline read admits only stage-active leads, so a ``customer`` created or
matched through ``/operator-contacts`` was unreachable from every portal
surface. These tests hold the new read to the same closure standards as the
routes beside it: authenticated, tenant-scoped, closed filters, closed
projection, deterministic keyset pagination, and no writes of any kind.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.crm_provider import (
    _eom_directory_phone_search_digits,
    _escape_eom_directory_like_pattern,
)
from atlas_brain.services.eom_crm_mutations import (
    get_eom_operator_contact_editability,
)

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
FOREIGN_TENANT = "churnsignals"

_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256

_BASE_CREATED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _row(
    *,
    contact_type: str = "customer",
    full_name: str = "Directory Contact",
    lead_stage: str | None = None,
    created_at: datetime | None = None,
    contact_id: UUID | None = None,
    status: str = "active",
    customer_type: str = "unknown",
) -> dict[str, object]:
    row: dict[str, object] = {
        "contact_id": contact_id or uuid4(),
        "full_name": full_name,
        "email": "directory@example.test",
        "phone": "2175550100",
        "address": "1 Directory Way",
        "contact_type": contact_type,
        "customer_type": customer_type,
        "lead_stage": lead_stage,
        "status": status,
        "source": "manual",
        "created_at": created_at or _BASE_CREATED_AT,
        "updated_at": created_at or _BASE_CREATED_AT,
    }
    editability = get_eom_operator_contact_editability(row)
    row["editable"] = editability.editable
    row["edit_block_reason"] = editability.edit_block_reason
    return row


class _CRM:
    """Spy provider: records every method invoked plus directory kwargs."""

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def list_eom_contact_directory(self, **kwargs: object) -> list[dict]:
        self.calls.append(("list_eom_contact_directory", dict(kwargs)))
        return [dict(row) for row in self.rows]

    async def list_eom_new_lead_review_items(self, **kwargs: object) -> list[dict]:
        self.calls.append(("list_eom_new_lead_review_items", dict(kwargs)))
        return []

    def __getattr__(self, name: str):
        # Any OTHER provider access from the directory read is a write-path
        # or scope leak; make it loud instead of silently succeeding.
        raise AssertionError(f"directory read must not touch crm.{name}")


def _app(crm: _CRM) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(api_enabled=True, service_token_sha256=_SERVICE_TOKEN_SHA256)
    )
    return app


def _headers(token: str = _SERVICE_TOKEN) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EOM-Actor": "Juan Canfield",
        "X-EOM-Actor-ID": "1",
    }


async def _get(
    crm: _CRM, query: str = "", *, raise_app_exceptions: bool = True, **kwargs: object
) -> httpx.Response:
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=app, raise_app_exceptions=raise_app_exceptions
        ),
        base_url="http://test",
    ) as client:
        return await client.get(
            f"/eom-funnel/contact-directory{query}", headers=_headers(), **kwargs
        )


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    funnel_mod._served_capabilities_cache = None
    yield
    funnel_mod._served_capabilities_cache = None


# ---------------------------------------------------------------------------
# Authentication and actor boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_route_refuses_an_unauthenticated_caller():
    app = _app(_CRM())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/contact-directory")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_wrong_service_token_is_refused():
    app = _app(_CRM())
    wrong = auth_mod.generate_eom_funnel_service_token().token
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/eom-funnel/contact-directory", headers=_headers(token=wrong)
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_missing_actor_header_is_refused():
    app = _app(_CRM())
    headers = _headers()
    headers.pop("X-EOM-Actor")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/contact-directory", headers=headers)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Filter closure: kind, search, cursor, unknown parameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_contact_kinds_come_back_and_the_projection_is_closed():
    lead = _row(contact_type="lead", full_name="Lead Person", lead_stage="new")
    customer = _row(contact_type="customer", full_name="Customer Person")
    response = await _get(_CRM([lead, customer]))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"contacts", "limit", "cursor", "hasMore", "nextCursor"}
    kinds = {item["contactType"] for item in body["contacts"]}
    assert kinds == {"lead", "customer"}
    for item in body["contacts"]:
        assert set(item) == {
            "contactId",
            "fullName",
            "email",
            "phone",
            "address",
            "contactType",
            "customerType",
            "leadStage",
            "status",
            "source",
            "createdAt",
            "updatedAt",
            "editable",
            "editBlockedReason",
        }
        assert item["status"] == "active"
        assert item["editable"] is True
        assert item["editBlockedReason"] is None


@pytest.mark.asyncio
async def test_the_kind_filter_is_closed():
    response = await _get(_CRM(), "?kind=prospect")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_each_admitted_kind_is_forwarded_verbatim():
    for kind in ("all", "lead", "customer"):
        crm = _CRM()
        response = await _get(crm, f"?kind={kind}")
        assert response.status_code == 200
        assert crm.calls[-1][1]["kind"] == kind


@pytest.mark.asyncio
async def test_an_unknown_query_parameter_is_rejected_not_ignored():
    """A typoed filter silently ignored would return the unfiltered directory
    while looking filtered -- the wrong rows with a confident face."""
    response = await _get(_CRM(), "?serach=smith")
    assert response.status_code == 422
    assert "serach" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_blank_search_is_rejected():
    response = await _get(_CRM(), "?search=%20%20")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_an_overlong_search_is_rejected():
    response = await _get(_CRM(), f"?search={'x' * 121}")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_database_unrepresentable_search_fails_closed():
    """NUL cannot live in a Postgres text parameter: it must 422 at the
    boundary, never 500 mid-query."""
    crm = _CRM()
    response = await _get(crm, "?search=%00")
    assert response.status_code == 422
    embedded = await _get(crm, "?search=ada%00lovelace")
    assert embedded.status_code == 422
    assert crm.calls == [], "an invalid search must never reach the provider"


def test_the_directory_kind_set_is_derived_from_the_operator_boundary():
    """The route and provider read the canonical mutation kind set, so a kind
    the write boundary admits can never be born write-only in the directory."""
    from atlas_brain.services.eom_crm_mutations import EOM_OPERATOR_CONTACT_TYPES

    assert funnel_mod._CONTACT_DIRECTORY_KINDS == ("all", *EOM_OPERATOR_CONTACT_TYPES)


@pytest.mark.asyncio
async def test_search_is_forwarded_stripped():
    crm = _CRM()
    response = await _get(crm, "?search=%20Ada%20Operator%20")
    assert response.status_code == 200
    assert crm.calls[-1][1]["search"] == "Ada Operator"


@pytest.mark.asyncio
async def test_a_malformed_cursor_is_rejected():
    response = await _get(_CRM(), "?cursor=%2Bnot-base64-at-all%2B")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_short_cursor_is_rejected():
    response = await _get(_CRM(), "?cursor=abc")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Pagination mechanics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagination_reports_has_more_and_a_cursor_that_round_trips():
    rows = [
        _row(created_at=_BASE_CREATED_AT - timedelta(minutes=index))
        for index in range(3)
    ]
    crm = _CRM(rows)
    first = await _get(crm, "?limit=2")

    assert first.status_code == 200
    body = first.json()
    assert crm.calls[-1][1]["limit"] == 3, "route must overfetch by one"
    assert len(body["contacts"]) == 2
    assert body["hasMore"] is True
    assert body["nextCursor"]

    crm_second = _CRM(rows[2:])
    second = await _get(crm_second, f"?limit=2&cursor={body['nextCursor']}")
    assert second.status_code == 200
    forwarded = crm_second.calls[-1][1]
    assert forwarded["cursor_contact_id"] == rows[1]["contact_id"]
    assert forwarded["cursor_created_at"] == rows[1]["created_at"]
    assert second.json()["hasMore"] is False
    assert second.json()["nextCursor"] is None


@pytest.mark.asyncio
async def test_a_full_final_page_reports_no_more():
    rows = [_row(), _row()]
    response = await _get(_CRM(rows), "?limit=2")
    body = response.json()
    assert len(body["contacts"]) == 2
    assert body["hasMore"] is False
    assert body["nextCursor"] is None


# ---------------------------------------------------------------------------
# Read-only and projection guarantees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_directory_read_touches_no_other_provider_method():
    """The spy raises on any attribute except the two reads it defines, so a
    write (or any second read) reaching the provider fails this test."""
    crm = _CRM([_row()])
    response = await _get(crm, "?search=Ada&kind=customer")
    assert response.status_code == 200
    assert [name for name, _ in crm.calls] == ["list_eom_contact_directory"]


@pytest.mark.asyncio
async def test_a_row_outside_the_directory_kinds_can_never_be_emitted():
    """A legacy 'prospect' row must 500 loudly, not render as a directory
    contact -- the projection is the second enforcement of admission."""
    response = await _get(
        _CRM([_row(contact_type="prospect")]), raise_app_exceptions=False
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_an_archived_row_can_never_be_emitted_by_the_active_view():
    """With the status Literal widened for the archived view, the route's
    page-homogeneity check is what keeps the views disjoint: an archived row
    under the default (active) lifecycle must 500 loudly, not render."""
    response = await _get(
        _CRM([_row(status="archived")]), raise_app_exceptions=False
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_an_active_row_can_never_be_emitted_by_the_archived_view():
    response = await _get(
        _CRM([_row(status="active")]),
        "?lifecycle=archived",
        raise_app_exceptions=False,
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_the_lifecycle_filter_is_closed_and_forwarded_verbatim():
    for lifecycle in ("active", "archived"):
        crm = _CRM([_row(status=lifecycle)])
        response = await _get(crm, f"?lifecycle={lifecycle}")
        assert response.status_code == 200
        assert crm.calls[-1][1]["lifecycle"] == lifecycle
        assert all(
            item["status"] == lifecycle
            for item in response.json()["contacts"]
        )
    assert (await _get(_CRM(), "?lifecycle=deleted")).status_code == 422
    assert (await _get(_CRM(), "?lifecycle=")).status_code == 422


@pytest.mark.asyncio
async def test_omitting_lifecycle_defaults_to_the_active_view():
    crm = _CRM([_row()])
    response = await _get(crm)
    assert response.status_code == 200
    assert crm.calls[-1][1]["lifecycle"] == "active"


@pytest.mark.asyncio
async def test_a_junk_customer_type_can_never_be_emitted():
    response = await _get(
        _CRM([_row(customer_type="franchise")]), raise_app_exceptions=False
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_a_lost_lead_is_rendered_with_its_stage():
    """Lost leads are pipeline-hidden but DB-active; the directory is exactly
    where they must remain findable, labelled truthfully."""
    response = await _get(_CRM([_row(contact_type="lead", lead_stage="lost")]))
    assert response.status_code == 200
    item = response.json()["contacts"][0]
    assert item["contactType"] == "lead"
    assert item["leadStage"] == "lost"
    assert item["editable"] is False
    assert item["editBlockedReason"] == "not_editable_stage"


@pytest.mark.parametrize(
    ("contact", "expected"),
    [
        ({"contact_type": "customer", "status": "active"}, (True, None)),
        (
            {"contact_type": "lead", "status": "active", "lead_stage": "new"},
            (True, None),
        ),
        (
            {
                "contact_type": "lead",
                "status": "active",
                "lead_stage": "estimate_booked",
            },
            (True, None),
        ),
        (
            {"contact_type": "lead", "status": "active", "lead_stage": "won"},
            (True, None),
        ),
        (
            {"contact_type": "lead", "status": "active", "lead_stage": "lost"},
            (False, "not_editable_stage"),
        ),
        (
            {"contact_type": "lead", "status": "inactive", "lead_stage": "new"},
            (False, "not_editable_lead_status"),
        ),
        (
            {"contact_type": "customer", "status": "archived"},
            (False, "not_editable_archived"),
        ),
        (
            {"contact_type": "vendor", "status": "active"},
            (False, "not_editable_contact_type"),
        ),
    ],
)
def test_the_editability_policy_is_closed_and_preserves_write_boundary_order(
    contact: dict[str, str], expected: tuple[bool, str | None]
):
    decision = get_eom_operator_contact_editability(contact)

    assert (decision.editable, decision.edit_block_reason) == expected


@pytest.mark.asyncio
async def test_an_archived_directory_row_has_a_closed_non_editable_verdict():
    response = await _get(_CRM([_row(status="archived")]), "?lifecycle=archived")

    assert response.status_code == 200
    item = response.json()["contacts"][0]
    assert item["status"] == "archived"
    assert item["editable"] is False
    assert item["editBlockedReason"] == "not_editable_archived"


def test_the_directory_schema_refuses_incoherent_or_unknown_editability_values():
    mismatched = _row(contact_type="lead", lead_stage="lost")
    mismatched["editable"] = True
    with pytest.raises(ValueError, match="editable"):
        funnel_mod.EOMContactDirectoryItem.model_validate(mismatched)

    unknown_reason = _row()
    unknown_reason["editable"] = False
    unknown_reason["edit_block_reason"] = "not_a_reason"
    with pytest.raises(ValueError, match="edit_block_reason"):
        funnel_mod.EOMContactDirectoryItem.model_validate(unknown_reason)


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


def test_the_directory_is_advertised_in_the_capability_manifest():
    assert "contact.directory" in funnel_mod.served_capabilities()
    assert "contact.directory.editability" in funnel_mod.served_capabilities()
    assert ("GET", "/eom-funnel/contact-directory") in funnel_mod.served_capability_routes()


@pytest.mark.asyncio
async def test_the_lead_review_response_advertises_the_directory():
    """Callers derive the deployment proof from the manifest on the pipeline
    read, so the name and the exact method/path must both appear there."""
    crm = _CRM()
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads", headers=_headers())
    body = response.json()
    assert "contact.directory" in body["capabilities"]
    assert "contact.directory.editability" in body["capabilities"]
    assert {"method": "GET", "path": "/eom-funnel/contact-directory"} in body[
        "capabilityRoutes"
    ]


def _deployed_apps():
    """Every application object a real deployment starts.

    ``atlas_brain.main:app`` is what the live systemd unit runs today
    (atlas-api.service, uvicorn atlas_brain.main:app); ``main_eom:app`` is the
    slim EOM topology render.eom.yaml starts and the one that pins the
    dedicated funnel CRM provider. A reachability proof that exercises only
    one of them stays green while the other drops the route.
    """
    from atlas_brain.main import app as aggregate_app
    from atlas_brain.main_eom import app as eom_app

    return [("main", aggregate_app), ("main_eom", eom_app)]


@pytest.mark.asyncio
async def test_every_deployed_entrypoint_serves_the_route_at_its_path():
    """Every other test mounts the router on a fresh app; this one proves the
    applications that actually ship mount it under /api/v1."""
    for name, app in _deployed_apps():
        crm = _CRM([_row()])
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
                response = await client.get(
                    "/api/v1/eom-funnel/contact-directory", headers=_headers()
                )
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_overrides)

        assert response.status_code == 200, (
            f"{name} must serve the deployed path, not 404"
        )
        contacts = response.json()["contacts"]
        assert len(contacts) == 1, name
        assert contacts[0]["editable"] is True, name
        assert contacts[0]["editBlockedReason"] is None, name


# ---------------------------------------------------------------------------
# Search-grammar helpers (pure, no DB)
# ---------------------------------------------------------------------------


def test_search_admission_grammar_holds_across_tokens_containers_and_families():
    """Class-closure proof for the search-grammar guard (GUARD_CLASS_CLOSURE
    req 3): inputs are GENERATED over three grammar axes -- digit tokens x
    punctuation containers x query families -- rather than sampled as a
    fixture list, and every generated query is judged by a spec-derived
    oracle.

    The oracle pins the CONTRACT values as literals here -- the phone
    punctuation family " ()+.-" and the 4-digit minimum -- independent of the
    implementation constants, so widening or narrowing the implemented set
    (adding '#', dropping '.') or moving the threshold breaks this test even
    though every fixture a reviewer might list still passes.
    """
    spec_phone_punctuation = " ()+.-"  # contract literal, deliberately not imported
    spec_minimum_digits = 4  # contract literal, deliberately not imported

    def oracle_expected_digits(query: str) -> str | None:
        if not query:
            return None
        if any(
            not (char.isdigit() or char in spec_phone_punctuation) for char in query
        ):
            return None
        digits = "".join(char for char in query if char.isdigit())
        return digits if len(digits) >= spec_minimum_digits else None

    digit_runs = ("", "21", "217", "5550", "2175550142")

    def container_plain(run: str) -> str:
        return run

    def container_parenthesized(run: str) -> str:
        return f"({run[:3]}) {run[3:]}" if len(run) > 3 else f"({run})"

    def container_dotted(run: str) -> str:
        return ".".join(part for part in (run[:3], run[3:6], run[6:]) if part) or run

    def container_international(run: str) -> str:
        return f"+1 {run}".rstrip()

    query_family_suffixes = ("", "x", "@example.test", " Suite B", "#42")

    checked = 0
    for run, container, family_suffix in product(
        digit_runs,
        (
            container_plain,
            container_parenthesized,
            container_dotted,
            container_international,
        ),
        query_family_suffixes,
    ):
        query = container(run) + family_suffix
        if not query:
            continue
        checked += 1
        assert _eom_directory_phone_search_digits(query) == oracle_expected_digits(
            query
        ), f"grammar case diverged from the spec oracle: {query!r}"
    assert checked > 80, "the generator must actually cover the grammar"


def test_escaped_patterns_never_leave_an_active_like_metacharacter():
    """Invariant oracle for the escaping guard, over generated inputs: after
    removing every backslash-escaped pair from the escaped pattern, no bare
    %, _, or backslash may remain, and unescaping must restore the original
    text exactly. This holds for the whole input class, not a sample of
    reported strings.
    """
    for prefix, metacharacter, suffix in product(
        ("", "a", "5", "\\"),
        ("%", "_", "\\", "%_", "\\%", "%%"),
        ("", "b", "%", "_x"),
    ):
        original = prefix + metacharacter + suffix
        escaped = _escape_eom_directory_like_pattern(original)
        remainder = re.sub(r"\\.", "", escaped)
        assert not any(char in remainder for char in "%_\\"), (
            f"active metacharacter survives escaping: {original!r} -> {escaped!r}"
        )
        unescaped = re.sub(r"\\(.)", r"\1", escaped)
        assert unescaped == original, "escaping must be reversible"


def test_like_metacharacters_are_escaped_literally():
    assert _escape_eom_directory_like_pattern("50%_off\\x") == "50\\%\\_off\\\\x"


def test_phone_shaped_queries_yield_their_digit_run():
    assert _eom_directory_phone_search_digits("(217) 555-0100") == "2175550100"
    assert _eom_directory_phone_search_digits("217.555") == "217555"


def test_non_phone_shaped_queries_get_no_digit_fallback():
    # An incidental digit run inside a name or email must never reach the
    # phone comparison (the client-side search closed this in website #232).
    assert _eom_directory_phone_search_digits("client2026@example.com") is None
    assert _eom_directory_phone_search_digits("Suite 2026") is None


def test_short_digit_runs_get_no_fallback():
    assert _eom_directory_phone_search_digits("217") is None


# ---------------------------------------------------------------------------
# Real-Postgres proofs: the scope claims are only worth what the SQL does
# ---------------------------------------------------------------------------


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


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


@pytest.fixture()
async def _directory_provider():
    """A DatabaseCRMProvider over a disposable schema with the contacts DDL."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url_or_skip()

    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    schema = f"atlas_eom_contact_directory_{uuid.uuid4().hex}"
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
        "full_name": "Seeded Contact",
        "business_context_id": TENANT,
        "contact_type": "customer",
        "status": "active",
        "email": None,
        "phone": None,
    }
    fields.update(overrides)
    row = await pool.fetchrow(
        """
        INSERT INTO contacts (full_name, business_context_id, contact_type,
                              status, email, phone)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, created_at
        """,
        fields["full_name"],
        fields["business_context_id"],
        fields["contact_type"],
        fields["status"],
        fields["email"],
        fields["phone"],
    )
    if "lead_stage" in overrides:
        await pool.execute(
            "UPDATE contacts SET lead_stage = $2 WHERE id = $1",
            row["id"],
            overrides["lead_stage"],
        )
    return row["id"]


@pytest.mark.asyncio
async def test_tenant_scope_and_lifecycle_hold_against_real_postgres(
    _directory_provider,
):
    """Seeds every excluded neighbor beside every admitted one and asks for
    the directory: an unscoped or status-blind query fails here."""
    provider, pool = _directory_provider
    eom_customer = await _seed(pool, full_name="EOM Customer")
    eom_lead = await _seed(
        pool, full_name="EOM Lead", contact_type="lead", lead_stage="new"
    )
    eom_lost_lead = await _seed(
        pool, full_name="EOM Lost Lead", contact_type="lead", lead_stage="lost"
    )
    archived = await _seed(pool, full_name="EOM Archived", status="archived")
    foreign = await _seed(
        pool, full_name="Foreign Customer", business_context_id=FOREIGN_TENANT
    )
    vendor = await _seed(pool, full_name="EOM Vendor", contact_type="vendor")

    rows = await provider.list_eom_contact_directory(limit=50)
    ids = {row["contact_id"] for row in rows}

    assert eom_customer in ids
    assert eom_lead in ids
    assert eom_lost_lead in ids, "a lost lead is DB-active and must stay findable"
    assert archived not in ids, "archived rows are excluded in this slice"
    assert foreign not in ids, "another tenant's contact must never appear"
    assert vendor not in ids, "kinds outside lead/customer are not directory rows"
    lost_row = next(row for row in rows if row["contact_id"] == eom_lost_lead)
    assert lost_row["lead_stage"] == "lost"
    assert lost_row["editable"] is False
    assert lost_row["edit_block_reason"] == "not_editable_stage"


@pytest.mark.asyncio
async def test_the_kind_filter_narrows_against_real_postgres(_directory_provider):
    provider, pool = _directory_provider
    customer = await _seed(pool, full_name="Only Customer")
    lead = await _seed(
        pool, full_name="Only Lead", contact_type="lead", lead_stage="new"
    )

    leads = await provider.list_eom_contact_directory(limit=50, kind="lead")
    customers = await provider.list_eom_contact_directory(limit=50, kind="customer")

    assert {row["contact_id"] for row in leads} == {lead}
    assert {row["contact_id"] for row in customers} == {customer}


@pytest.mark.asyncio
async def test_search_matches_name_email_and_phone_against_real_postgres(
    _directory_provider,
):
    provider, pool = _directory_provider
    by_name = await _seed(pool, full_name="Ada Lovelace")
    by_email = await _seed(pool, full_name="Email Row", email="unique.needle@example.test")
    by_phone = await _seed(pool, full_name="Phone Row", phone="(217) 555-0142")
    bystander = await _seed(pool, full_name="Bystander")

    named = await provider.list_eom_contact_directory(limit=50, search="lovelace")
    mailed = await provider.list_eom_contact_directory(limit=50, search="unique.needle")
    dialed = await provider.list_eom_contact_directory(limit=50, search="2175550142")

    assert {row["contact_id"] for row in named} == {by_name}
    assert {row["contact_id"] for row in mailed} == {by_email}
    assert {row["contact_id"] for row in dialed} == {by_phone}, (
        "a digits-only query must match a formatted stored phone"
    )
    assert bystander not in {row["contact_id"] for row in named}


@pytest.mark.asyncio
async def test_a_like_metacharacter_searches_literally_against_real_postgres(
    _directory_provider,
):
    provider, pool = _directory_provider
    # A discriminating pair: unescaped, the pattern '%5%%' also matches
    # 'Percent 55' (contains a 5 followed by anything); escaped, only the
    # literal '5%' row can match. A pair the wildcard cannot reach would
    # pass with or without escaping and prove nothing.
    literal = await _seed(pool, full_name="Percent 5% Off")
    wildcard_bait = await _seed(pool, full_name="Percent 55 Co")

    rows = await provider.list_eom_contact_directory(limit=50, search="5%")

    ids = {row["contact_id"] for row in rows}
    assert literal in ids
    assert wildcard_bait not in ids, "% must not act as a wildcard"


@pytest.mark.asyncio
async def test_keyset_traversal_neither_drops_nor_duplicates(_directory_provider):
    """Walk the whole directory two rows at a time and require the union to be
    exact -- an OFFSET or a mutable-column keyset fails this under skew, and a
    broken tuple comparison fails it immediately."""
    provider, pool = _directory_provider
    seeded = set()
    for index in range(5):
        seeded.add(await _seed(pool, full_name=f"Page Contact {index}"))

    collected: list[UUID] = []
    cursor_created_at = None
    cursor_contact_id = None
    for _ in range(10):
        rows = await provider.list_eom_contact_directory(
            limit=2 + 1,
            cursor_created_at=cursor_created_at,
            cursor_contact_id=cursor_contact_id,
        )
        page = rows[:2]
        collected.extend(row["contact_id"] for row in page)
        if len(rows) <= 2:
            break
        cursor_created_at = page[-1]["created_at"]
        cursor_contact_id = page[-1]["contact_id"]

    assert len(collected) == len(set(collected)), "no row may repeat"
    assert set(collected) == seeded, "no row may be dropped"


@pytest.mark.asyncio
async def test_directory_rows_carry_the_projection_columns(_directory_provider):
    provider, pool = _directory_provider
    await _seed(
        pool,
        full_name="Projection Row",
        email="projection@example.test",
        phone="2175550199",
    )
    rows = await provider.list_eom_contact_directory(limit=5)
    row = rows[0]
    assert set(row) == {
        "contact_id",
        "full_name",
        "email",
        "phone",
        "address",
        "contact_type",
        "customer_type",
        "lead_stage",
        "status",
        "source",
        "created_at",
        "updated_at",
        "editable",
        "edit_block_reason",
    }
    assert row["customer_type"] == "unknown", "migration 366 default must surface"
    assert row["editable"] is True
    assert row["edit_block_reason"] is None
