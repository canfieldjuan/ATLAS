"""Capability manifest on the EOM funnel lead-review read (website #112, Slice 0E).

The manifest exists so a caller can disable a control instead of shipping a
button that 404s. Website (Vercel) and tracker (Render) auto-deploy from `main`;
Atlas is deployed by hand, so callers routinely run ahead of it.

The property that matters is one-directional: the manifest must never advertise
a capability this build does not serve. Over-advertising re-creates exactly the
failure the slice exists to prevent -- a visible control whose backend returns
404. Under-advertising merely disables a control that would have worked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain import main_eom
from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig

_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256
_TERMS_CAPABILITY_ROUTES = {
    "terms.invitation.issue": ("POST", "/eom-funnel/terms/invitations"),
    "terms.invitation.revoke": (
        "POST",
        "/eom-funnel/terms/invitations/{invitation_id}/revoke",
    ),
    "terms.readiness.read": (
        "GET",
        "/eom-funnel/terms/readiness/{contact_id}",
    ),
    "terms.delivery.confirm_sent": (
        "POST",
        "/eom-funnel/terms/deliveries/{delivery_id}/confirm-sent",
    ),
    "terms.public.session": ("POST", "/eom-funnel/terms/public/session"),
    "terms.public.accept": ("POST", "/eom-funnel/terms/public/accept"),
}


class _CRM:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []

    async def list_eom_new_lead_review_items(self, **_kwargs: object) -> list[dict]:
        return list(self.rows)


def _app(crm: _CRM) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(api_enabled=True, service_token_sha256=_SERVICE_TOKEN_SHA256)
    )
    return app


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_SERVICE_TOKEN}",
        "X-EOM-Actor": "Juan Canfield",
        "X-EOM-Actor-ID": "1",
        "Idempotency-Key": f"office-handoff-{uuid4().hex}",
    }


def _lead_row() -> dict[str, object]:
    return {
        "contact_id": uuid4(),
        "full_name": "Test Lead",
        "email": None,
        "phone": None,
        "address": None,
        "source": "website_form",
        "lead_stage": "new",
        "created_at": datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc),
    }


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    """The manifest memoizes; a stale entry would leak across tests."""
    funnel_mod._served_capabilities_cache = None
    yield
    funnel_mod._served_capabilities_cache = None


def _registered_routes() -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in funnel_mod.router.routes
        for method in (getattr(route, "methods", None) or ())
    }


def test_every_advertised_capability_has_a_registered_route() -> None:
    """The load-bearing direction: never claim what this build cannot serve."""
    registered = _registered_routes()
    for name in funnel_mod.served_capabilities():
        assert funnel_mod._CAPABILITY_ROUTES[name] in registered, name


def test_every_mapped_and_registered_route_is_advertised() -> None:
    """The other direction, so the map cannot silently rot into a no-op."""
    registered = _registered_routes()
    expected = {
        name
        for name, signature in funnel_mod._CAPABILITY_ROUTES.items()
        if signature in registered
    }
    assert set(funnel_mod.served_capabilities()) == expected
    assert expected, "capability map matched no registered route at all"


def test_capability_map_has_no_entry_for_an_unregistered_route() -> None:
    """A typo in the map is invisible without this: the name would just never
    appear, which reads identically to 'this build does not serve it'."""
    registered = _registered_routes()
    unmatched = {
        name: signature
        for name, signature in funnel_mod._CAPABILITY_ROUTES.items()
        if signature not in registered
    }
    assert unmatched == {}, f"capability map references unregistered routes: {unmatched}"


def test_terms_capabilities_pin_the_existing_route_contract() -> None:
    """The Tracker contract names each existing Terms route exactly."""
    registered = _registered_routes()
    for name, signature in _TERMS_CAPABILITY_ROUTES.items():
        assert funnel_mod._CAPABILITY_ROUTES[name] == signature
        assert signature in registered


def test_manifest_omits_a_capability_whose_route_is_not_registered(monkeypatch) -> None:
    """The degradation path itself, forced.

    This is the shape a caller running ahead of Atlas actually sees, reproduced
    without needing an old build: a capability the map knows about but this
    deployment does not serve must be absent, not advertised.
    """
    monkeypatch.setitem(
        funnel_mod._CAPABILITY_ROUTES,
        "lead.not_deployed_yet",
        ("POST", "/eom-funnel/leads/{contact_id}/not-deployed-yet"),
    )
    funnel_mod._served_capabilities_cache = None

    served = funnel_mod.served_capabilities()

    assert "lead.not_deployed_yet" not in served
    assert "lead.lost" in served, "unrelated capabilities must be unaffected"


@pytest.mark.asyncio
async def test_lead_review_response_advertises_lost_and_reopen() -> None:
    """The specific shape website #112 names: the Mark-lost control.

    Atlas 497d3155f serves /lost and /reopen, so the manifest must say so --
    otherwise the portal would hide a button that works.
    """
    app = _app(_CRM([_lead_row()]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads", headers=_headers())

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert "lead.lost" in capabilities
    assert "lead.reopen" in capabilities
    assert "onboarding.public_link.list" in capabilities
    assert "onboarding.public_link.revoke" in capabilities
    assert "onboarding.public_handoff.recover" in capabilities
    capability_routes = {
        (route["method"], route["path"])
        for route in response.json()["capabilityRoutes"]
    }
    assert capability_routes == {
        funnel_mod._CAPABILITY_ROUTES[name] for name in capabilities
    }
    assert (
        "GET",
        "/eom-funnel/public-onboarding/issued-links",
    ) in capability_routes
    assert (
        "POST",
        "/eom-funnel/onboarding-drafts/{draft_id}/revoke-link",
    ) in capability_routes
    assert (
        "POST",
        "/eom-funnel/public-onboarding/recover",
    ) in capability_routes


@pytest.mark.asyncio
async def test_manifest_is_present_even_when_the_queue_is_empty() -> None:
    """An empty queue is the common case and must still carry the manifest.

    Deriving capabilities from rows rather than routes would make an idle
    portal disable every control.
    """
    app = _app(_CRM([]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads", headers=_headers())

    body = response.json()
    assert body["leads"] == []
    assert "lead.lost" in body["capabilities"]


@pytest.mark.asyncio
async def test_lead_review_response_advertises_terms_routes() -> None:
    """The deployed slim-app entrypoint exposes the Terms bridge contract."""
    app = main_eom.app
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: _CRM([])
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        EOMFunnelConfig(api_enabled=True, service_token_sha256=_SERVICE_TOKEN_SHA256)
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/eom-funnel/leads", headers=_headers()
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200
    body = response.json()
    assert set(_TERMS_CAPABILITY_ROUTES).issubset(body["capabilities"])
    advertised_routes = {
        (route["method"], route["path"]) for route in body["capabilityRoutes"]
    }
    assert set(_TERMS_CAPABILITY_ROUTES.values()).issubset(advertised_routes)


@pytest.mark.asyncio
async def test_pre_manifest_caller_still_reads_every_field_it_knew() -> None:
    """Backward compatibility, stated as the old caller's own contract.

    A caller written before this slice reads these keys and ignores the rest;
    the tracker's `_parse_atlas_lead_review_response` does exactly that. Adding
    a key is safe, but renaming or dropping one of these is not, so pin them.
    """
    app = _app(_CRM([_lead_row()]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads", headers=_headers())

    body = response.json()
    for key in ("leads", "limit", "cursor", "hasMore", "nextCursor"):
        assert key in body, key
    assert "capabilityRoutes" in body
    lead = body["leads"][0]
    for key in (
        "contactId",
        "fullName",
        "email",
        "phone",
        "address",
        "source",
        "leadStage",
        "createdAt",
    ):
        assert key in lead, key


def test_response_model_defaults_capabilities_for_a_caller_that_omits_it() -> None:
    """Constructing the envelope without the field must not raise.

    Any code path building this response that predates the manifest keeps
    working and simply advertises nothing, which callers read as 'degrade'.
    """
    response = funnel_mod.EOMLeadReviewResponse(
        leads=[], limit=25, cursor=None, has_more=False, next_cursor=None
    )
    assert response.capabilities == []
    assert response.capability_routes == []


def test_field_clear_capability_versions_the_operator_mutation_semantics() -> None:
    """`contact.field_clear` shares the operator-mutation route ON PURPOSE.

    The name versions the route's SEMANTICS (audited present-null clearing,
    website #254), not its existence: an older build serves the same POST
    route but never advertises this name because the dict entry ships with
    the clearing contract. Pin the pairing so a route move or a well-meaning
    'dedupe' of the shared signature breaks loudly instead of silently
    un-advertising field-clearing.
    """
    assert funnel_mod._CAPABILITY_ROUTES["contact.field_clear"] == (
        "POST",
        "/eom-funnel/operator-contacts",
    )
    assert (
        funnel_mod._CAPABILITY_ROUTES["contact.field_clear"]
        == funnel_mod._CAPABILITY_ROUTES["contact.operator_mutation"]
    )
    served = funnel_mod.served_capabilities()
    assert ("contact.field_clear" in served) == (
        "contact.operator_mutation" in served
    ), "shared route means shared availability"
    assert "contact.field_clear" in served
