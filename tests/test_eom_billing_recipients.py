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


@pytest.mark.asyncio
async def test_the_billing_projection_answers_every_eligibility_case():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()

    from atlas_brain.services import crm_provider as svc

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
        tab_email = await _seed_contact(conn, name="Tab Address", email="\t")
        newline_email = await _seed_contact(conn, name="Newline Address", email="\n ")
        malformed = await _seed_contact(
            conn, name="Bad Address", email="not-an-address")
        lead = await _seed_contact(
            conn, name="A Lead With Email", email="lead@example.test")
        await conn.execute(
            "UPDATE contacts SET contact_type = 'lead' WHERE id = $1", lead)
        other_tenant = await _seed_contact(
            conn, name="Someone Else's AP", email="ap@other.test",
            tenant="churnsignals")
        missing = uuid4()

        class _Pool:
            is_initialized = True

            @property
            def pool(self):
                return self

            async def fetch(self, query, *args):
                return await conn.fetch(query, *args)

            async def fetchrow(self, query, *args):
                return await conn.fetchrow(query, *args)

        service = svc.DatabaseCRMProvider(pool=_Pool())

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
            (tab_email, "no_email"),
            (newline_email, "no_email"),
            (malformed, "no_email"),
            (lead, "inactive"),
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
            (tab_email, "tab-only email"), (newline_email, "newline-only email"),
            (malformed, "malformed email"), (lead, "a lead, not a customer"),
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
    app.dependency_overrides[routes._billing_crm_dependency] = lambda: _Service()

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
    app.dependency_overrides[routes._billing_crm_dependency] = lambda: _Service()

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
    from atlas_brain.services.crm_provider import BILLING_RECIPIENT_REASONS

    assert "wrong_tenant" not in BILLING_RECIPIENT_REASONS
    assert set(BILLING_RECIPIENT_REASONS) == {"not_found", "inactive", "no_email"}


def _canonical_config(dsn: str, *, api_enabled: bool = False):
    return SimpleNamespace(api_enabled=api_enabled, db_connection_string=dsn)


@pytest.mark.parametrize(
    "receivables_enabled, dsn, api_enabled, confirmed, expect_raise",
    [
        # Receivables opening the pool is admission-bearing on its own.
        (True, "postgresql://host/db", False, False, True),
        (True, "postgresql://host/db", False, True, False),
        # No DSN: receivables never opens the pool, so nothing to admit.
        (True, "", False, False, False),
        (True, "   ", False, False, False),
        # Receivables off: unchanged from before this slice.
        (False, "postgresql://host/db", False, False, False),
        # The funnel flag keeps its own admission regardless of receivables.
        (False, "postgresql://host/db", True, False, True),
        (True, "postgresql://host/db", True, False, True),
    ],
)
def test_canonical_admission_is_owed_by_whoever_opens_the_pool(
    monkeypatch, receivables_enabled, dsn, api_enabled, confirmed, expect_raise,
):
    """Admission follows the pool, not the flag that first needed it.

    Gating on `api_enabled` alone let receivables open a configured DSN
    unadmitted. Pointed at a reachable non-canonical Atlas database holding
    `effingham_maids` contacts, that would disclose their names and addresses
    to the receivables bearer through the very routes this slice adds.
    """
    from atlas_brain.eom_api import config as eom_config
    from atlas_brain.eom_api import funnel_database as fd

    monkeypatch.setattr(
        eom_config.invoicing_settings,
        "receivables_api_enabled",
        receivables_enabled,
    )
    config = _canonical_config(dsn, api_enabled=api_enabled)

    if expect_raise:
        with pytest.raises(RuntimeError) as excinfo:
            fd.validate_eom_funnel_canonical_crm_config(
                config, canonical_crm_database_confirmed=confirmed)
        message = str(excinfo.value)
        assert "ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true" in message
        # The message must name the trigger that actually applies, or an
        # operator sets the wrong variable and the pool stays shut.
        expected_trigger = (
            "ATLAS_EOM_FUNNEL_API_ENABLED=true"
            if api_enabled
            else "ATLAS_EOM_RECEIVABLES_API_ENABLED=true"
        )
        assert expected_trigger in message
    else:
        fd.validate_eom_funnel_canonical_crm_config(
            config, canonical_crm_database_confirmed=confirmed)


def test_admission_condition_matches_the_condition_that_opens_the_pool():
    """The validator and the initializer must agree on when the pool opens.

    They are two separate predicates over the same facts. If they drift, a
    pool opens that the validator never demanded admission for -- which is
    exactly the hole this pair of changes closes.
    """
    import inspect

    from atlas_brain.eom_api import funnel_database as fd

    init_src = inspect.getsource(fd.init_eom_funnel_database)
    validate_src = inspect.getsource(fd.validate_eom_funnel_canonical_crm_config)
    for source in (init_src, validate_src):
        assert "invoicing_settings.receivables_api_enabled" in source
        assert "db_connection_string.strip()" in source


def test_the_real_app_fails_closed_when_the_contact_pool_is_unconfigured(tmp_path):
    """No dependency override -- the production wiring, end to end.

    The earlier version of this guard ran AFTER the app-state factory branch,
    and the real app installs that factory unconditionally, so the guard sat
    on a path production never takes: the fetch raised an untranslated
    RuntimeError and answered 500. Every route test here overrides
    `_billing_crm_dependency`, so none of them could see it. This one runs the
    real app in a subprocess and asserts the override count is zero.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from atlas_brain.eom_api import auth as receivables_auth

    generated = receivables_auth.generate_receivables_service_token()
    probe = """
import json
import os

from fastapi.testclient import TestClient

from atlas_brain import main_eom
from atlas_brain.eom_api import receivables


# Only the receivables DB is stubbed -- never the billing dependency. This
# profile runs with ATLAS_DB_ENABLED=false, so /ready would 503 on the global
# database before it ever reported the contact pool.
class _ReadyService:
    async def is_ready(self):
        return True


receivables.get_receivables_service = lambda: _ReadyService()
if main_eom.app.dependency_overrides:
    raise AssertionError("no dependency may be overridden in this probe")

with TestClient(main_eom.app) as client:
    headers = {"Authorization": f"Bearer {os.environ['EOM_TEST_CALLER_TOKEN']}"}
    listed = client.get("/api/v1/receivables/billing-recipients", headers=headers)
    detail = client.get(
        "/api/v1/receivables/billing-recipients/"
        "00000000-0000-4000-8000-000000000000",
        headers=headers,
    )
    ready = client.get("/api/v1/receivables/ready", headers=headers)

print(json.dumps({
    "listed_status": listed.status_code,
    "listed_code": listed.json().get("detail", {}).get("code"),
    "detail_status": detail.status_code,
    "ready_billing": ready.json().get("billingRecipients"),
    "dependency_overrides": len(main_eom.app.dependency_overrides),
}))
"""
    env = os.environ.copy()
    for key in list(env):
        if key.upper().startswith(("ATLAS_INVOICING_", "ATLAS_EOM_FUNNEL_")):
            env.pop(key, None)
    repo_root = Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root) if not existing else f"{repo_root}{os.pathsep}{existing}"
    )
    env["ATLAS_DB_ENABLED"] = "false"
    env["ATLAS_EOM_FUNNEL_API_ENABLED"] = "false"
    # Receivables on, no funnel DSN: the deployment shape the finding names.
    env["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] = "true"
    env["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"] = generated.sha256
    env["EOM_TEST_CALLER_TOKEN"] = generated.token

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False, capture_output=True, cwd=tmp_path, env=env, text=True,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        # 503 with a named cause, NOT a 500 from an untranslated RuntimeError.
        "listed_status": 503,
        "listed_code": "billing_recipients_unavailable",
        "detail_status": 503,
        # Readiness says so too, rather than reporting a healthy service whose
        # every recipient read fails.
        "ready_billing": "unconfigured",
        "dependency_overrides": 0,
    }, observed


def test_admission_holds_over_the_whole_configuration_grammar(monkeypatch):
    """Every point in the config space, checked against a spec-derived oracle.

    The rule this encodes is one sentence: **admission is owed exactly when
    the pool is opened**, and the pool is opened by the funnel API being on,
    or by receivables being on with a usable DSN.

    The oracle is derived from that sentence rather than from the
    implementation, so a change that redefines when the pool opens breaks this
    test instead of quietly agreeing with itself. Enumerating the grammar
    matters here because the defect was a whole FAMILY of configurations --
    every (receivables on, DSN set, funnel off) point -- being admitted
    without confirmation, not one special case.
    """
    from itertools import product

    from atlas_brain.eom_api import config as eom_config
    from atlas_brain.eom_api import funnel_database as fd

    # Containers for a DSN: absent, several shapes of blank, and present.
    # `strip()` decides usability, so whitespace families are the interesting
    # boundary -- a tab-only DSN must count as absent, not as configured.
    blank_dsns = ["", " ", "   ", "\t", "\n", "\r\n", " \t\n "]
    real_dsns = ["postgresql://h/d", "  postgresql://h/d  ", "postgres://u:p@h:5432/d"]

    space = list(product(
        (True, False),                 # api_enabled
        (True, False),                 # receivables_api_enabled
        blank_dsns + real_dsns,        # DSN containers
        (True, False),                 # canonical admission confirmed
    ))
    for api_enabled, receivables_enabled, dsn, confirmed in space:
        monkeypatch.setattr(
            eom_config.invoicing_settings,
            "receivables_api_enabled",
            receivables_enabled,
        )
        config = _canonical_config(dsn, api_enabled=api_enabled)

        # --- the oracle, straight from the rule ------------------------
        usable_dsn = bool(dsn.strip())
        pool_opens = api_enabled or (receivables_enabled and usable_dsn)
        # A DSN is also independently required once the funnel API is on,
        # which is the pre-existing second failure mode.
        expect_raise = pool_opens and (
            not confirmed or (api_enabled and not usable_dsn)
        )

        try:
            fd.validate_eom_funnel_canonical_crm_config(
                config, canonical_crm_database_confirmed=confirmed)
            raised = False
        except RuntimeError:
            raised = True

        assert raised == expect_raise, (
            f"api_enabled={api_enabled} receivables={receivables_enabled} "
            f"dsn={dsn!r} confirmed={confirmed}: "
            f"expected raise={expect_raise}, got {raised}"
        )

    # Guards against the enumeration silently collapsing to a few cases.
    assert len(space) == 2 * 2 * (len(blank_dsns) + len(real_dsns)) * 2 == 80
