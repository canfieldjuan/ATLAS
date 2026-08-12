"""EOM Slice 1A: which Atlas contact may receive an invoice, and who is it.

The guarantee is negative as much as positive. An ineligible contact must
never carry a name or an address out of this boundary, and a contact under a
different tenant must be indistinguishable from one that does not exist -- the
lookup is tenant-scoped in SQL so this service never learns the difference in
the first place. Several tests below assert the ABSENCE of fields for that
reason; a suite that only checked the happy path would pass against exactly
the disclosure this route exists to avoid.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

EOM = "effingham_maids"


def _database_url() -> str:
    url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")
    return url


async def _seed_contact(conn, *, name, status="active", email="ap@example.test",
                        tenant=EOM):
    contact_id = uuid4()
    await conn.execute(
        """
        INSERT INTO contacts (id, full_name, email, status, contact_type,
                              business_context_id)
        VALUES ($1, $2, $3, $4, 'customer', $5)
        """,
        contact_id, name, email, status, tenant,
    )
    return contact_id


async def _service_for(conn_pool):
    from atlas_brain.services.receivables import ReceivablesService

    return ReceivablesService(pool=conn_pool)


@pytest.mark.asyncio
async def test_the_billing_projection_answers_every_eligibility_case():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()

    from atlas_brain.services import receivables as svc

    schema = f"billing_recipients_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            """
            CREATE TABLE contacts (
                id UUID PRIMARY KEY,
                full_name VARCHAR(256) NOT NULL,
                email VARCHAR(256),
                status VARCHAR(32) NOT NULL,
                contact_type VARCHAR(32) NOT NULL,
                business_context_id VARCHAR(64)
            )
            """
        )

        eligible = await _seed_contact(
            conn, name="Accounts Payable", email="invoice.usa@example.test")
        archived = await _seed_contact(
            conn, name="Old AP", status="archived", email="old@example.test")
        inactive = await _seed_contact(
            conn, name="Paused AP", status="inactive", email="paused@example.test")
        no_email = await _seed_contact(conn, name="No Address", email=None)
        blank_email = await _seed_contact(conn, name="Blank Address", email="   ")
        other_tenant = await _seed_contact(
            conn, name="Someone Else's AP", email="ap@other.test",
            tenant="churnsignals")
        missing = uuid4()

        class _Pool:
            is_initialized = True

            async def fetch(self, query, *args):
                return await conn.fetch(query, *args)

            async def fetchrow(self, query, *args):
                return await conn.fetchrow(query, *args)

        service = svc.ReceivablesService(pool=_Pool())

        # --- the positive case -------------------------------------------
        ok = await service.get_billing_recipient(eligible)
        assert ok == {
            "contactId": str(eligible),
            "displayName": "Accounts Payable",
            "email": "invoice.usa@example.test",
            "eligible": True,
            "reason": None,
        }

        # --- every ineligible cause, and what it may NOT disclose ---------
        for contact_id, expected_reason in (
            (archived, "inactive"),
            (inactive, "inactive"),
            (no_email, "no_email"),
            (blank_email, "no_email"),
            (missing, "not_found"),
        ):
            verdict = await service.get_billing_recipient(contact_id)
            assert verdict["eligible"] is False, contact_id
            assert verdict["reason"] == expected_reason, contact_id
            assert verdict["displayName"] is None, (
                f"an ineligible verdict leaked a name for {expected_reason}")
            assert verdict["email"] is None, (
                f"an ineligible verdict leaked an address for {expected_reason}")
            assert set(verdict) == {
                "contactId", "displayName", "email", "eligible", "reason"}

        # --- the tenant probe --------------------------------------------
        # A contact under another tenant must be reported EXACTLY as one that
        # does not exist. Any difference makes this a cross-tenant existence
        # oracle for an EOM-scoped credential.
        foreign = await service.get_billing_recipient(other_tenant)
        absent = await service.get_billing_recipient(missing)
        assert foreign == {**absent, "contactId": str(other_tenant)}, (
            "another tenant's contact is distinguishable from a missing one")
        assert foreign["reason"] == "not_found"

        # --- the list exposes eligible rows only --------------------------
        listed = await service.list_billing_recipients()
        ids = {item["contactId"] for item in listed}
        assert str(eligible) in ids
        for excluded, label in (
            (archived, "archived"), (inactive, "inactive"),
            (no_email, "no email"), (blank_email, "blank email"),
            (other_tenant, "another tenant"),
        ):
            assert str(excluded) not in ids, f"{label} contact was offered"
        assert all(item["eligible"] is True for item in listed)
        assert all(
            set(item) == {"contactId", "displayName", "email", "eligible", "reason"}
            for item in listed
        )

        # --- search narrows without widening disclosure -------------------
        found = await service.list_billing_recipients(search="invoice.usa")
        assert [item["contactId"] for item in found] == [str(eligible)]
        assert await service.list_billing_recipients(search="nothing-matches") == []
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_the_routes_sit_behind_the_receivables_credential():
    """The funnel token must not open a billing capability.

    This is why the projection lives in the receivables router rather than
    beside /eom-funnel/known-contacts: the funnel credential is broad, and
    recipient identity is billing data.
    """
    from atlas_brain.eom_api import auth as receivables_auth
    from atlas_brain.eom_api import receivables as routes

    generated = receivables_auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )

    class _Service:
        async def list_billing_recipients(self, **_):
            return [{"contactId": str(uuid4()), "displayName": "AP",
                     "email": "ap@example.test", "eligible": True, "reason": None}]

        async def get_billing_recipient(self, contact_id):
            return {"contactId": str(contact_id), "displayName": "AP",
                    "email": "ap@example.test", "eligible": True, "reason": None}

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = (
        lambda: config
    )
    app.dependency_overrides[routes.get_receivables_service] = lambda: _Service()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://receivables.test"
    ) as client:
        target = f"/receivables/billing-recipients/{uuid4()}"

        assert (await client.get("/receivables/billing-recipients")).status_code == 401
        assert (await client.get(target)).status_code == 401

        bad = {"Authorization": "Bearer eomrx_v1_not-the-right-token"}
        assert (
            await client.get("/receivables/billing-recipients", headers=bad)
        ).status_code == 401
        assert (await client.get(target, headers=bad)).status_code == 401

        good = {"Authorization": f"Bearer {generated.token}"}
        listed = await client.get("/receivables/billing-recipients", headers=good)
        assert listed.status_code == 200, listed.text
        detail = await client.get(target, headers=good)
        assert detail.status_code == 200, detail.text
        # An ineligible verdict is a domain answer, not a transport failure --
        # so the detail route answers 200 rather than 404 and the caller must
        # read `eligible`.
        assert detail.json()["eligible"] is True


@pytest.mark.asyncio
async def test_an_ineligible_verdict_is_200_with_a_reason_not_404():
    from atlas_brain.eom_api import auth as receivables_auth
    from atlas_brain.eom_api import receivables as routes

    generated = receivables_auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    contact_id = uuid4()

    class _Service:
        async def get_billing_recipient(self, requested):
            assert isinstance(requested, UUID)
            return {"contactId": str(requested), "displayName": None,
                    "email": None, "eligible": False, "reason": "no_email"}

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = (
        lambda: config
    )
    app.dependency_overrides[routes.get_receivables_service] = lambda: _Service()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://receivables.test"
    ) as client:
        response = await client.get(
            f"/receivables/billing-recipients/{contact_id}",
            headers={"Authorization": f"Bearer {generated.token}"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"contactId": str(contact_id), "displayName": None,
                    "email": None, "eligible": False, "reason": "no_email"}


def test_wrong_tenant_is_not_a_public_reason():
    """The public reason set must not carry a token callers may never receive.

    An earlier draft of this contract listed `wrong_tenant` publicly while also
    requiring that callers never learn it -- a contradiction. The lookup is
    tenant-scoped instead, so the distinction is never computed.
    """
    from atlas_brain.services.receivables import BILLING_RECIPIENT_REASONS

    assert "wrong_tenant" not in BILLING_RECIPIENT_REASONS
    assert set(BILLING_RECIPIENT_REASONS) == {"not_found", "inactive", "no_email"}
