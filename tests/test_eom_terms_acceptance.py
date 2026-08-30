"""Focused proof for customer-bound EOM Terms acceptance evidence."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
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
from atlas_brain.services.eom_public_onboarding_tokens import (
    format_eom_public_onboarding_token,
)
from atlas_brain.services.eom_terms_acceptance import (
    AuthenticatedEOMTermsToken,
    EOMTermsAcceptanceConflictError,
    EOMTermsAcceptanceNotFoundError,
    EOMTermsAcceptanceService,
    EOMTermsAcceptanceUnavailableError,
    EOMTermsAcceptanceValidationError,
    append_eom_terms_acceptance_link,
    authenticate_eom_terms_token,
    build_eom_terms_link,
    eom_terms_acceptance_schema_ready,
    format_eom_terms_token,
    render_eom_terms_executed_copy,
    render_eom_terms_invitation,
)
from atlas_brain.services.eom_card_vault import eom_card_vault_schema_ready
from atlas_brain.services.eom_terms_authority import (
    EOMTermsAuthority,
    eom_terms_authority_schema_ready,
)


_NOW = datetime(2026, 8, 28, 14, 30, tzinfo=timezone.utc)
_CONTACT_ID = UUID("11111111-1111-4111-8111-111111111111")
_INVITATION_ID = UUID("22222222-2222-4222-8222-222222222222")
_VERSION_ID = UUID("33333333-3333-4333-8333-333333333333")
_ACCEPTANCE_ID = UUID("44444444-4444-4444-8444-444444444444")
_DELIVERY_ID = UUID("55555555-5555-4555-8555-555555555555")
_SECRET = "current-eom-public-key-material-1234567890"
_PREVIOUS_SECRET = "previous-eom-public-key-material-123456"
_SERVICE = funnel_auth_mod.generate_eom_funnel_service_token()
_RUNTIME_DATABASE_URL_ENV = "ATLAS_EOM_FIRST_CLEAN_TEST_DATABASE_URL"
_DBA_DATABASE_URL_ENV = "ATLAS_EOM_FIRST_CLEAN_DBA_DATABASE_URL"
_AUTHORITY_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/396_eom_terms_authority.sql"
)
_ACCEPTANCE_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/397_eom_terms_acceptance.sql"
)
_CANDIDATE_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/395_eom_post_clean_onboarding_candidates.sql"
)
_CARD_VAULT_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/398_eom_card_vault.sql"
)


def _documents(marker: str = "approved") -> dict[str, Any]:
    return {
        audience: {
            locale: {
                "terms": f"{marker} {audience} {locale} terms",
                "servicesWeCannotProvide": (f"{marker} {audience} {locale} services"),
                "additionalWorkAcknowledgement": (
                    f"{marker} {audience} {locale} additional work"
                ),
            }
            for locale in ("en",)
        }
        for audience in ("residential", "commercial")
    }


class _AsyncpgServicePool:
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


class _RecordingSender:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.fail:
            raise OSError("injected transport uncertainty")
        return {
            "message_id": f"message-{len(self.calls)}",
            "idempotent_replay": False,
        }


class _IdempotentReplaySender:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"message_id": None, "idempotent_replay": True}


class _BlockingSender:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        self.started.set()
        await self.release.wait()
        return {"message_id": "message-after-lock", "idempotent_replay": False}


class _NoDeliveryService(EOMTermsAcceptanceService):
    """Model process loss after evidence commit and before transport claim."""

    async def _deliver(self, **kwargs: Any) -> tuple[dict[str, Any], bool]:
        delivery_id = UUID(str(kwargs["delivery_id"]))
        async with self.pool.transaction() as connection:
            row = await self._delivery_by_id(connection, delivery_id)
        assert row is not None
        return dict(row), False


def _database_url_or_skip(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not configured")
    return value


async def _provision_guard_role(connection: Any) -> None:
    await connection.execute(
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
    await connection.execute("REVOKE atlas_eom_handoff_owner FROM atlas")


@asynccontextmanager
async def _real_terms_store():
    runtime_url = _database_url_or_skip(_RUNTIME_DATABASE_URL_ENV)
    dba_url = _database_url_or_skip(_DBA_DATABASE_URL_ENV)
    schema = f"atlas_eom_terms_acceptance_{uuid4().hex}"
    dba_connection = await asyncpg.connect(dba_url)
    runtime_pool = None
    try:
        await _provision_guard_role(dba_connection)
        await dba_connection.execute(
            f'CREATE SCHEMA "{schema}" AUTHORIZATION atlas_eom_handoff_owner'
        )
        await dba_connection.execute(
            f'GRANT USAGE, CREATE ON SCHEMA "{schema}" TO atlas'
        )
        await dba_connection.execute(f'SET search_path TO "{schema}", pg_catalog')
        await dba_connection.execute(
            """
            CREATE TABLE contacts (
                id UUID PRIMARY KEY,
                business_context_id VARCHAR(64) NOT NULL,
                contact_type VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                customer_type VARCHAR(16) NOT NULL,
                full_name VARCHAR(256) NOT NULL,
                email VARCHAR(256)
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                content_sha256 VARCHAR(64),
                applied_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await dba_connection.execute(_AUTHORITY_MIGRATION.read_text())
        await dba_connection.execute(
            """
            INSERT INTO schema_migrations (version, name)
            VALUES (396, '396_eom_terms_authority')
            """
        )
        await dba_connection.execute(_ACCEPTANCE_MIGRATION.read_text())
        # The production runtime already reads and row-locks canonical contacts.
        # This minimal fixture creates that prerequisite as the DBA, so restore
        # only those pre-existing capabilities after the controlled migrations
        # have normalized their own object ACLs. PostgreSQL requires UPDATE as
        # well as SELECT for the service's SELECT ... FOR SHARE lock.
        await dba_connection.execute(
            f'GRANT SELECT, UPDATE ON TABLE "{schema}".contacts TO atlas'
        )
        assert (
            await dba_connection.fetchval(
                "SELECT has_table_privilege('atlas', $1, 'SELECT')",
                f'"{schema}".contacts',
            )
            is True
        )
        runtime_pool = await asyncpg.create_pool(
            runtime_url,
            min_size=1,
            max_size=8,
            statement_cache_size=0,
            server_settings={"search_path": f'"{schema}", pg_catalog'},
        )
        async with runtime_pool.acquire() as runtime_connection:
            runtime_boundary = await runtime_connection.fetchrow(
                """
                SELECT current_user AS current_user,
                       current_schema() AS current_schema,
                       namespace.nspname AS relation_schema,
                       has_table_privilege(
                           current_user, relation.oid, 'SELECT'
                       ) AS can_select,
                       has_table_privilege(
                           current_user, relation.oid, 'UPDATE'
                       ) AS can_lock
                FROM pg_class AS relation
                JOIN pg_namespace AS namespace
                  ON namespace.oid = relation.relnamespace
                WHERE relation.oid = to_regclass('contacts')
                """
            )
        assert runtime_boundary is not None
        assert runtime_boundary["current_user"] == "atlas"
        assert runtime_boundary["current_schema"] == schema
        assert runtime_boundary["relation_schema"] == schema
        assert runtime_boundary["can_select"] is True
        assert runtime_boundary["can_lock"] is True
        yield _AsyncpgServicePool(runtime_pool), dba_connection, schema
    finally:
        if runtime_pool is not None:
            await runtime_pool.close()
        try:
            await dba_connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await dba_connection.close()


async def _seed_customer(
    connection: Any,
    *,
    contact_id: UUID,
    audience: str = "residential",
    email: str = "customer@example.com",
) -> None:
    await connection.execute(
        """
        INSERT INTO contacts (
            id, business_context_id, contact_type, status, customer_type,
            full_name, email
        ) VALUES ($1, 'effingham_maids', 'customer', 'active', $2, $3, $4)
        """,
        contact_id,
        audience,
        f"Customer {str(contact_id)[:8]}",
        email,
    )


async def _publish(
    authority: EOMTermsAuthority,
    *,
    label: str,
    material: bool,
    marker: str,
) -> dict[str, Any]:
    draft = await authority.create_draft(
        version_label=label,
        material_change=material,
        documents=_documents(marker),
        actor_id=7,
        actor_name="Juan",
    )
    return await authority.publish(
        version_id=draft["versionId"],
        actor_id=7,
        actor_name="Juan",
    )


def test_terms_bearer_is_domain_separated_fragment_only_and_rotatable() -> None:
    token = format_eom_terms_token(
        invitation_id=_INVITATION_ID,
        secret=_PREVIOUS_SECRET,
    )
    authenticated = authenticate_eom_terms_token(
        token=token,
        secret=_SECRET,
        previous_secret=_PREVIOUS_SECRET,
    )
    assert authenticated.invitation_id == _INVITATION_ID
    assert token.startswith("eomt1.")
    assert "?" not in build_eom_terms_link(
        base_url="https://example.test/onboarding",
        token=token,
    )
    assert build_eom_terms_link(
        base_url="https://example.test/onboarding",
        token=token,
    ).endswith(f"#termsToken={token}")

    profile_token = format_eom_public_onboarding_token(
        token_id=_INVITATION_ID,
        secret=_PREVIOUS_SECRET,
    )
    with pytest.raises(EOMTermsAcceptanceNotFoundError):
        authenticate_eom_terms_token(
            token=profile_token,
            secret=_SECRET,
            previous_secret=_PREVIOUS_SECRET,
        )
    with pytest.raises(EOMTermsAcceptanceNotFoundError):
        authenticate_eom_terms_token(
            token=token,
            secret=_SECRET,
        )


@pytest.mark.parametrize("token", [None, False, 0, "", "eomt1.bad", "x" * 1000])
def test_terms_bearer_rejects_every_noncanonical_boundary(token: object) -> None:
    with pytest.raises(EOMTermsAcceptanceNotFoundError):
        authenticate_eom_terms_token(token=token, secret=_SECRET)


def test_renderers_use_only_the_selected_published_document_text() -> None:
    selected = _documents("selected")["commercial"]["en"]
    subject, body = render_eom_terms_invitation(
        full_name="Customer",
        version_label="2026.1",
        content_hash="a" * 64,
        documents=selected,
        locale="en",
    )
    linked = append_eom_terms_acceptance_link(
        body=body,
        link="https://example.test/#termsToken=opaque",
        locale="en",
    )
    executed_subject, executed_body = render_eom_terms_executed_copy(
        full_name="Customer",
        signer_name="Customer",
        accepted_at=_NOW,
        version_label="2026.1",
        content_hash="a" * 64,
        documents=selected,
        locale="en",
    )
    assert subject == "Review and accept Effingham Office Maids terms"
    assert executed_subject == "Accepted copy of Effingham Office Maids terms"
    assert all(value in body for value in selected.values())
    assert all(value in executed_body for value in selected.values())
    assert "termsToken=opaque" not in body
    assert "termsToken=opaque" in linked
    assert "Accept the terms securely:" in linked
    assert "Additional-work acknowledgement accepted: Yes" in executed_body


def test_customer_renderers_reject_spanish_locale() -> None:
    selected = _documents("selected")["commercial"]["en"]
    with pytest.raises(EOMTermsAcceptanceValidationError, match="locale must be en"):
        render_eom_terms_invitation(
            full_name="Customer",
            version_label="2026.1",
            content_hash="a" * 64,
            documents=selected,
            locale="es",
        )
    with pytest.raises(EOMTermsAcceptanceValidationError, match="locale must be en"):
        append_eom_terms_acceptance_link(
            body="Terms",
            link="https://example.test/#termsToken=opaque",
            locale="es",
        )
    with pytest.raises(EOMTermsAcceptanceValidationError, match="locale must be en"):
        render_eom_terms_executed_copy(
            full_name="Customer",
            signer_name="Customer",
            accepted_at=_NOW,
            version_label="2026.1",
            content_hash="a" * 64,
            documents=selected,
            locale="es",
        )


@pytest.mark.asyncio
async def test_invitation_rejects_spanish_before_dependencies_or_delivery() -> None:
    sender = _RecordingSender()
    with pytest.raises(EOMTermsAcceptanceValidationError, match="locale must be en"):
        await EOMTermsAcceptanceService(pool=object()).issue_and_send(
            request_key="terms-request-spanish-rejected",
            contact_id=_CONTACT_ID,
            locale="es",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=sender,
        )
    assert sender.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terms_accepted", "additional_work_accepted"),
    [(False, True), (True, False), (1, True), (True, "true")],
)
async def test_acceptance_requires_two_literal_true_acknowledgements(
    terms_accepted: object,
    additional_work_accepted: object,
) -> None:
    service = EOMTermsAcceptanceService(pool=object())
    token = AuthenticatedEOMTermsToken(
        invitation_id=_INVITATION_ID,
        signing_key_fingerprint="a" * 64,
    )
    with pytest.raises(EOMTermsAcceptanceValidationError):
        await service.accept_and_send(
            token=token,
            signer_name="Customer",
            terms_accepted=terms_accepted,
            additional_work_accepted=additional_work_accepted,
            client_ip="192.0.2.1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signer_name", "client_ip"),
    [
        ("", "192.0.2.1"),
        (None, "192.0.2.1"),
        ("Customer\nInjected", "192.0.2.1"),
        ("Customer\rInjected", "192.0.2.1"),
        ("Customer\tInjected", "192.0.2.1"),
        ("Customer\u2028Injected", "192.0.2.1"),
        ("Customer", "not-an-ip"),
        ("Customer", ""),
    ],
)
async def test_acceptance_rejects_invalid_signer_or_client_ip_before_database_use(
    signer_name: object,
    client_ip: object,
) -> None:
    service = EOMTermsAcceptanceService(pool=object())
    token = AuthenticatedEOMTermsToken(
        invitation_id=_INVITATION_ID,
        signing_key_fingerprint="a" * 64,
    )
    with pytest.raises(EOMTermsAcceptanceValidationError):
        await service.accept_and_send(
            token=token,
            signer_name=signer_name,
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip=client_ip,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"business_context_id": "other"},
        {"contact_type": "lead"},
        {"status": "archived"},
        {"customer_type": "unknown"},
        {"email": None},
        {"full_name": " "},
        {"full_name": "Customer\nInjected"},
    ],
)
def test_customer_boundary_rejects_every_noncanonical_account(
    changes: dict[str, object],
) -> None:
    row: dict[str, object] = {
        "id": _CONTACT_ID,
        "business_context_id": "effingham_maids",
        "contact_type": "customer",
        "status": "active",
        "customer_type": "residential",
        "full_name": "Customer",
        "email": "customer@example.com",
    }
    row.update(changes)
    with pytest.raises(EOMTermsAcceptanceConflictError):
        EOMTermsAcceptanceService._customer(row)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actor_name",
    ["Juan\nInjected", "Juan\rInjected", "Juan\tInjected", "Juan\u2028Injected"],
)
async def test_actor_identity_rejects_line_controls_before_database_use(
    actor_name: str,
) -> None:
    service = EOMTermsAcceptanceService(pool=object())
    with pytest.raises(EOMTermsAcceptanceValidationError):
        await service.revoke(
            invitation_id=_INVITATION_ID,
            actor_id=7,
            actor_name=actor_name,
        )


class _RouteAcceptance:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def issue_and_send(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("issue", kwargs))
        return _invitation_api_result()

    async def revoke(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("revoke", kwargs))
        return _invitation_api_result(status="revoked", revoked_at=_NOW)

    async def get_readiness(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("readiness", kwargs))
        return {
            "contactId": str(_CONTACT_ID),
            "audience": "residential",
            "ready": True,
            "reason": "accepted",
            "currentVersionId": str(_VERSION_ID),
            "currentVersionLabel": "2026.1",
            "currentContentHash": "a" * 64,
            "acceptedVersionId": str(_VERSION_ID),
            "acceptedVersionLabel": "2026.1",
            "acceptedAt": _NOW.isoformat(),
            "executedCopyDeliveryStatus": "sent",
        }

    async def confirm_delivery_sent(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("confirm", kwargs))
        return {
            "deliveryId": str(_DELIVERY_ID),
            "kind": "executed_copy",
            "status": "sent",
            "sentAt": _NOW.isoformat(),
            "idempotent": False,
        }

    async def get_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("session", kwargs))
        return {
            "status": "ready",
            "invitationId": str(_INVITATION_ID),
            "versionId": str(_VERSION_ID),
            "versionLabel": "2026.1",
            "contentHash": "a" * 64,
            "audience": "residential",
            "locale": "en",
            "customerName": "Customer",
            "documents": _documents()["residential"]["en"],
            "expiresAt": _NOW.isoformat(),
        }

    async def accept_and_send(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("accept", kwargs))
        return {
            "acceptanceId": str(_ACCEPTANCE_ID),
            "invitationId": str(_INVITATION_ID),
            "contactId": str(_CONTACT_ID),
            "versionId": str(_VERSION_ID),
            "versionLabel": "2026.1",
            "contentHash": "a" * 64,
            "audience": "residential",
            "locale": "en",
            "signerName": "Customer",
            "termsAccepted": True,
            "additionalWorkAccepted": True,
            "acceptedAt": _NOW.isoformat(),
            "deliveryId": str(_DELIVERY_ID),
            "executedCopyDeliveryStatus": "sent",
            "deliveryNeedsReconciliation": False,
            "deliveryError": False,
            "idempotent": False,
        }


def _invitation_api_result(
    *, status: str = "issued", revoked_at: datetime | None = None
) -> dict[str, Any]:
    return {
        "invitationId": str(_INVITATION_ID),
        "contactId": str(_CONTACT_ID),
        "versionId": str(_VERSION_ID),
        "versionLabel": "2026.1",
        "contentHash": "a" * 64,
        "audience": "residential",
        "locale": "en",
        "recipientEmail": "customer@example.com",
        "status": status,
        "issuedAt": _NOW.isoformat(),
        "expiresAt": _NOW.isoformat(),
        "revokedAt": revoked_at.isoformat() if revoked_at else None,
        "acceptanceId": None,
        "deliveryId": str(_DELIVERY_ID),
        "deliveryStatus": "sent",
        "deliveryNeedsReconciliation": False,
        "deliveryError": False,
        "idempotent": False,
    }


def _route_app(service: Any, *, issuance_enabled: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router, prefix="/api/v1")
    config = EOMFunnelConfig(
        api_enabled=True,
        service_token_sha256=_SERVICE.sha256,
        public_onboarding_enabled=True,
        public_onboarding_issuance_enabled=issuance_enabled,
        public_onboarding_url="https://example.test/onboarding",
        public_onboarding_hmac_secret=_SECRET,
        public_onboarding_previous_hmac_secret=_PREVIOUS_SECRET,
    )
    app.dependency_overrides[funnel_auth_mod.get_eom_funnel_api_config] = lambda: config
    app.dependency_overrides[funnel_mod._terms_acceptance_dependency] = lambda: service
    app.dependency_overrides[funnel_mod._onboarding_sender_dependency] = lambda: (
        object()
    )
    return app


def _headers(*, actor: bool = False, client_ip: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_SERVICE.token}"}
    if actor:
        headers.update({"X-EOM-Actor": "Juan", "X-EOM-Actor-ID": "7"})
    if client_ip:
        headers["X-EOM-Client-IP"] = "192.0.2.9"
    return headers


@pytest.mark.asyncio
async def test_mounted_routes_enforce_trust_boundaries_and_closed_projections() -> None:
    service = _RouteAcceptance()
    app = _route_app(service)
    token = format_eom_terms_token(invitation_id=_INVITATION_ID, secret=_SECRET)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        unauthorized = await client.get(
            f"/api/v1/eom-funnel/terms/readiness/{_CONTACT_ID}"
        )
        missing_actor = await client.post(
            "/api/v1/eom-funnel/terms/invitations",
            headers=_headers(),
            json={
                "requestKey": "terms-request-0001",
                "contactId": str(_CONTACT_ID),
                "locale": "en",
            },
        )
        issued = await client.post(
            "/api/v1/eom-funnel/terms/invitations",
            headers=_headers(actor=True),
            json={
                "requestKey": "terms-request-0001",
                "contactId": str(_CONTACT_ID),
                "locale": "en",
            },
        )
        spanish = await client.post(
            "/api/v1/eom-funnel/terms/invitations",
            headers=_headers(actor=True),
            json={
                "requestKey": "terms-request-spanish",
                "contactId": str(_CONTACT_ID),
                "locale": "es",
            },
        )
        missing_locale = await client.post(
            "/api/v1/eom-funnel/terms/invitations",
            headers=_headers(actor=True),
            json={
                "requestKey": "terms-request-missing-locale",
                "contactId": str(_CONTACT_ID),
            },
        )
        session = await client.post(
            "/api/v1/eom-funnel/terms/public/session",
            headers=_headers(),
            json={"token": token},
        )
        missing_ip = await client.post(
            "/api/v1/eom-funnel/terms/public/accept",
            headers=_headers(),
            json={
                "token": token,
                "signerName": "Customer",
                "termsAccepted": True,
                "additionalWorkAccepted": True,
            },
        )
        accepted = await client.post(
            "/api/v1/eom-funnel/terms/public/accept",
            headers=_headers(client_ip=True),
            json={
                "token": token,
                "signerName": "Customer",
                "termsAccepted": True,
                "additionalWorkAccepted": True,
            },
        )
        revoked = await client.post(
            f"/api/v1/eom-funnel/terms/invitations/{_INVITATION_ID}/revoke",
            headers=_headers(actor=True),
        )
        readiness = await client.get(
            f"/api/v1/eom-funnel/terms/readiness/{_CONTACT_ID}",
            headers=_headers(),
        )
        confirmed = await client.post(
            f"/api/v1/eom-funnel/terms/deliveries/{_DELIVERY_ID}/confirm-sent",
            headers=_headers(actor=True),
        )

    assert unauthorized.status_code == 401
    assert missing_actor.status_code == 422
    assert missing_ip.status_code == 422
    assert issued.status_code == 201
    assert spanish.status_code == 422
    assert missing_locale.status_code == 422
    assert session.status_code == 200
    assert accepted.status_code == 201
    assert revoked.status_code == 200
    assert readiness.json()["ready"] is True
    assert confirmed.json()["status"] == "sent"
    accept_call = next(kwargs for name, kwargs in service.calls if name == "accept")
    assert accept_call["client_ip"] == "192.0.2.9"
    assert isinstance(accept_call["token"], AuthenticatedEOMTermsToken)
    issue_call = next(kwargs for name, kwargs in service.calls if name == "issue")
    assert sum(name == "issue" for name, _kwargs in service.calls) == 1
    assert "audience" not in issue_call
    assert "recipient_email" not in issue_call
    assert "version_id" not in issue_call


@pytest.mark.asyncio
async def test_invitation_route_obeys_the_separate_issuance_pause() -> None:
    service = _RouteAcceptance()
    app = _route_app(service, issuance_enabled=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/eom-funnel/terms/invitations",
            headers=_headers(actor=True),
            json={
                "requestKey": "terms-request-0001",
                "contactId": str(_CONTACT_ID),
                "locale": "en",
            },
        )
    assert response.status_code == 503
    assert service.calls == []


@pytest.mark.asyncio
async def test_real_postgres_end_to_end_preserves_evidence_and_readiness() -> None:
    async with _real_terms_store() as (pool, dba, _schema):
        await _seed_customer(dba, contact_id=_CONTACT_ID)
        authority = EOMTermsAuthority(pool=pool)
        version_one = await _publish(
            authority,
            label="2026.1",
            material=True,
            marker="version-one",
        )
        service = EOMTermsAcceptanceService(pool=pool)
        invitation_sender = _RecordingSender()
        invitation = await service.issue_and_send(
            request_key="terms-request-real-0001",
            contact_id=_CONTACT_ID,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            previous_hmac_secret=_PREVIOUS_SECRET,
            sender=invitation_sender,
        )
        token = format_eom_terms_token(
            invitation_id=UUID(invitation["invitationId"]),
            secret=_SECRET,
        )
        session = await service.get_session(
            token=authenticate_eom_terms_token(token=token, secret=_SECRET)
        )
        assert session["documents"] == _documents("version-one")["residential"]["en"]
        assert len(invitation_sender.calls) == 1
        assert f"#termsToken={token}" in invitation_sender.calls[0]["body"]
        unknown_token = format_eom_terms_token(
            invitation_id=uuid4(),
            secret=_SECRET,
        )
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.get_session(
                token=authenticate_eom_terms_token(
                    token=unknown_token,
                    secret=_SECRET,
                )
            )
        wrong_key_token = format_eom_terms_token(
            invitation_id=UUID(invitation["invitationId"]),
            secret=_PREVIOUS_SECRET,
        )
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.get_session(
                token=authenticate_eom_terms_token(
                    token=wrong_key_token,
                    secret=_SECRET,
                    previous_secret=_PREVIOUS_SECRET,
                )
            )

        await _publish(
            authority,
            label="2026.2",
            material=False,
            marker="non-material",
        )
        replay = await service.issue_and_send(
            request_key="terms-request-real-0001",
            contact_id=_CONTACT_ID,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            previous_hmac_secret=_PREVIOUS_SECRET,
            sender=invitation_sender,
        )
        assert replay["versionId"] == version_one["versionId"]
        assert replay["idempotent"] is True
        assert len(invitation_sender.calls) == 1

        no_delivery = _NoDeliveryService(pool=pool)
        materially_stale = await no_delivery.issue_and_send(
            request_key="terms-request-materially-stale",
            contact_id=_CONTACT_ID,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        materially_stale_token = authenticate_eom_terms_token(
            token=format_eom_terms_token(
                invitation_id=UUID(materially_stale["invitationId"]),
                secret=_SECRET,
            ),
            secret=_SECRET,
        )

        receipt_sender = _RecordingSender(fail=True)
        acceptance = await service.accept_and_send(
            token=authenticate_eom_terms_token(token=token, secret=_SECRET),
            signer_name="Customer Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="2001:db8::1",
            sender=receipt_sender,
        )
        assert acceptance["deliveryError"] is True
        assert acceptance["executedCopyDeliveryStatus"] == "sending"
        assert len(receipt_sender.calls) == 1
        accepted_replay = await service.accept_and_send(
            token=authenticate_eom_terms_token(token=token, secret=_SECRET),
            signer_name="Customer Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="2001:db8::2",
            sender=receipt_sender,
        )
        assert accepted_replay["acceptanceId"] == acceptance["acceptanceId"]
        assert accepted_replay["idempotent"] is True
        assert len(receipt_sender.calls) == 1
        with pytest.raises(EOMTermsAcceptanceConflictError):
            await service.accept_and_send(
                token=authenticate_eom_terms_token(token=token, secret=_SECRET),
                signer_name="Different Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="2001:db8::2",
                sender=receipt_sender,
            )
        readiness = await service.get_readiness(contact_id=_CONTACT_ID)
        assert readiness["ready"] is True
        assert readiness["reason"] == "accepted"

        stored = await dba.fetchrow(
            """
            SELECT invitation.request_key, invitation.signing_key_fingerprint,
                   invitation_delivery.body AS invitation_body,
                   acceptance.signer_name, acceptance.client_ip::text,
                   acceptance.terms_accepted,
                   acceptance.additional_work_accepted,
                   executed.body AS executed_body,
                   executed.status AS executed_status
            FROM eom_terms_invitations AS invitation
            JOIN eom_terms_deliveries AS invitation_delivery
              ON invitation_delivery.invitation_id = invitation.id
             AND invitation_delivery.kind = 'invitation'
            JOIN eom_terms_acceptances AS acceptance
              ON acceptance.invitation_id = invitation.id
            JOIN eom_terms_deliveries AS executed
              ON executed.acceptance_id = acceptance.id
            WHERE invitation.id = $1
            """,
            UUID(invitation["invitationId"]),
        )
        assert stored is not None
        assert token not in stored["invitation_body"]
        assert token not in stored["executed_body"]
        assert stored["signer_name"] == "Customer Signer"
        assert stored["client_ip"] == "2001:db8::1/128"
        assert stored["terms_accepted"] is True
        assert stored["additional_work_accepted"] is True
        assert stored["executed_status"] == "sending"
        assert all(
            value in stored["executed_body"]
            for value in _documents("version-one")["residential"]["en"].values()
        )

        confirmed = await service.confirm_delivery_sent(
            delivery_id=acceptance["deliveryId"],
            actor_id=7,
            actor_name="Juan",
        )
        assert confirmed["status"] == "sent"

        await dba.execute(
            """
            ALTER TABLE eom_terms_versions
            DISABLE TRIGGER trg_protect_eom_terms_version
            """
        )
        await dba.execute(
            """
            UPDATE eom_terms_versions
            SET published_at = clock_timestamp() + INTERVAL '1 day'
            WHERE status = 'published'
            """
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_versions
            ENABLE TRIGGER trg_protect_eom_terms_version
            """
        )
        prior_release = await dba.fetchrow(
            """
            SELECT max(publication_order) AS publication_order,
                   min(published_at) AS published_at
            FROM eom_terms_versions
            WHERE status = 'published'
            """
        )
        assert prior_release is not None
        material_release = await _publish(
            authority,
            label="2026.3",
            material=True,
            marker="material",
        )
        material_release_row = await dba.fetchrow(
            """
            SELECT publication_order, published_at
            FROM eom_terms_versions
            WHERE id = $1
            """,
            UUID(material_release["versionId"]),
        )
        assert material_release_row is not None
        assert (
            material_release_row["publication_order"]
            > prior_release["publication_order"]
        )
        assert material_release_row["published_at"] < prior_release["published_at"]
        stale = await service.get_readiness(contact_id=_CONTACT_ID)
        assert stale["ready"] is False
        assert stale["reason"] == "reacceptance_required"
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.accept_and_send(
                token=materially_stale_token,
                signer_name="Customer Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="192.0.2.30",
                sender=_RecordingSender(),
            )
        async with pool.raw_pool.acquire() as runtime:
            with pytest.raises(
                asyncpg.PostgresError,
                match="superseded by a material release",
            ):
                await runtime.execute(
                    """
                    INSERT INTO eom_terms_acceptances (
                        id, invitation_id, signer_name, terms_accepted,
                        additional_work_accepted, client_ip
                    ) VALUES ($1, $2, $3, TRUE, TRUE, $4::inet)
                    """,
                    uuid4(),
                    UUID(materially_stale["invitationId"]),
                    "Database Boundary Probe",
                    "192.0.2.31",
                )

        current_invitation = await service.issue_and_send(
            request_key="terms-request-current-version-order",
            contact_id=_CONTACT_ID,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        current_acceptance = await service.accept_and_send(
            token=authenticate_eom_terms_token(
                token=format_eom_terms_token(
                    invitation_id=UUID(current_invitation["invitationId"]),
                    secret=_SECRET,
                ),
                secret=_SECRET,
            ),
            signer_name="Current Version Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="192.0.2.32",
            sender=_RecordingSender(),
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_acceptances
            DISABLE TRIGGER trg_protect_eom_terms_acceptance
            """
        )
        await dba.execute(
            """
            UPDATE eom_terms_acceptances
            SET accepted_at = TIMESTAMPTZ '2000-01-01 00:00:00+00'
            WHERE id = $1
            """,
            UUID(current_acceptance["acceptanceId"]),
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_acceptances
            ENABLE TRIGGER trg_protect_eom_terms_acceptance
            """
        )
        current_readiness = await service.get_readiness(contact_id=_CONTACT_ID)
        assert current_readiness["ready"] is True
        assert current_readiness["reason"] == "accepted"
        assert current_readiness["acceptedVersionId"] == material_release["versionId"]

        await dba.execute(
            "UPDATE contacts SET customer_type = 'commercial' WHERE id = $1",
            _CONTACT_ID,
        )
        changed_audience_readiness = await service.get_readiness(contact_id=_CONTACT_ID)
        assert changed_audience_readiness["ready"] is False
        assert changed_audience_readiness["reason"] == "audience_changed"
        assert changed_audience_readiness["audience"] == "commercial"
        commercial_invitation = await service.issue_and_send(
            request_key="terms-request-current-version-commercial",
            contact_id=_CONTACT_ID,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        commercial_acceptance = await service.accept_and_send(
            token=authenticate_eom_terms_token(
                token=format_eom_terms_token(
                    invitation_id=UUID(commercial_invitation["invitationId"]),
                    secret=_SECRET,
                ),
                secret=_SECRET,
            ),
            signer_name="Commercial Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="192.0.2.35",
            sender=_RecordingSender(),
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_acceptances
            DISABLE TRIGGER trg_protect_eom_terms_acceptance
            """
        )
        await dba.execute(
            """
            UPDATE eom_terms_acceptances
            SET accepted_at = TIMESTAMPTZ '1999-01-01 00:00:00+00'
            WHERE id = $1
            """,
            UUID(commercial_acceptance["acceptanceId"]),
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_acceptances
            ENABLE TRIGGER trg_protect_eom_terms_acceptance
            """
        )
        commercial_readiness = await service.get_readiness(contact_id=_CONTACT_ID)
        assert commercial_readiness["ready"] is True
        assert commercial_readiness["reason"] == "accepted"
        assert commercial_readiness["audience"] == "commercial"
        assert (
            commercial_readiness["acceptedVersionId"] == material_release["versionId"]
        )

        non_material_release = await _publish(
            authority,
            label="2026.4",
            material=False,
            marker="non-material-after-reclassification",
        )
        later_commercial_invitation = await service.issue_and_send(
            request_key="terms-request-later-non-material-commercial",
            contact_id=_CONTACT_ID,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        await service.accept_and_send(
            token=authenticate_eom_terms_token(
                token=format_eom_terms_token(
                    invitation_id=UUID(later_commercial_invitation["invitationId"]),
                    secret=_SECRET,
                ),
                secret=_SECRET,
            ),
            signer_name="Later Commercial Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="192.0.2.36",
            sender=_RecordingSender(),
        )
        await dba.execute(
            "UPDATE contacts SET customer_type = 'residential' WHERE id = $1",
            _CONTACT_ID,
        )
        returned_audience_readiness = await service.get_readiness(
            contact_id=_CONTACT_ID
        )
        assert returned_audience_readiness["ready"] is True
        assert returned_audience_readiness["reason"] == "accepted"
        assert returned_audience_readiness["audience"] == "residential"
        assert (
            returned_audience_readiness["currentVersionId"]
            == non_material_release["versionId"]
        )
        assert (
            returned_audience_readiness["acceptedVersionId"]
            == material_release["versionId"]
        )

        with pytest.raises(EOMTermsAcceptanceConflictError):
            await service.revoke(
                invitation_id=invitation["invitationId"],
                actor_id=7,
                actor_name="Juan",
            )

        assert await eom_terms_acceptance_schema_ready(pool) is True
        assert await eom_terms_authority_schema_ready(pool) is True
        await dba.execute(
            """
            ALTER TABLE eom_terms_versions
            DISABLE TRIGGER trg_protect_eom_terms_version
            """
        )
        assert await eom_terms_acceptance_schema_ready(pool) is False
        assert await eom_terms_authority_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_terms_versions
            ENABLE TRIGGER trg_protect_eom_terms_version
            """
        )
        assert await eom_terms_acceptance_schema_ready(pool) is True
        assert await eom_terms_authority_schema_ready(pool) is True
        await dba.execute(
            """
            ALTER TABLE eom_terms_deliveries
            DISABLE TRIGGER trg_protect_eom_terms_delivery
            """
        )
        assert await eom_terms_acceptance_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_terms_deliveries
            ENABLE TRIGGER trg_protect_eom_terms_delivery
            """
        )
        assert await eom_terms_acceptance_schema_ready(pool) is True
        owners = await dba.fetch(
            """
            SELECT relation.relname, owner.rolname AS owner
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_roles AS owner ON owner.oid = relation.relowner
            WHERE namespace.nspname = current_schema()
              AND relation.relname IN (
                  'eom_terms_invitations',
                  'eom_terms_acceptances',
                  'eom_terms_deliveries'
              )
            """
        )
        assert {row["owner"] for row in owners} == {"atlas_eom_handoff_owner"}
        async with pool.raw_pool.acquire() as runtime:
            assert (
                await runtime.fetchval(
                    "SELECT has_table_privilege(current_user, "
                    "'eom_terms_acceptances', 'DELETE')"
                )
                is False
            )
            assert (
                await runtime.fetchval(
                    "SELECT has_function_privilege(current_user, "
                    "'protect_eom_terms_acceptance()', 'EXECUTE')"
                )
                is False
            )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute(
                    "UPDATE eom_terms_acceptances SET signer_name = 'Changed'"
                )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute("TRUNCATE eom_terms_deliveries")
        with pytest.raises(asyncpg.PostgresError):
            await dba.execute(
                "UPDATE eom_terms_acceptances SET signer_name = 'Changed'"
            )
        with pytest.raises(asyncpg.PostgresError):
            await dba.execute("UPDATE eom_terms_deliveries SET body = 'Changed'")


@pytest.mark.asyncio
async def test_real_postgres_serializes_duplicate_and_opposing_operations() -> None:
    async with _real_terms_store() as (pool, dba, _schema):
        contact_duplicate = uuid4()
        contact_opposing = uuid4()
        contact_revoked = uuid4()
        contact_invalid = uuid4()
        contact_expired = uuid4()
        contact_delivery_lock = uuid4()
        contact_changed_before_delivery = uuid4()
        contact_executed_copy_changed = uuid4()
        contact_replay = uuid4()
        contact_confirmation_failure = uuid4()
        contact_acceptance_confirmation_failure = uuid4()
        await _seed_customer(dba, contact_id=contact_duplicate)
        await _seed_customer(
            dba,
            contact_id=contact_opposing,
            email="opposing@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_revoked,
            email="revoked@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_invalid,
            email="invalid@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_expired,
            email="expired@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_delivery_lock,
            email="delivery-lock@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_changed_before_delivery,
            email="changed-before-delivery@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_executed_copy_changed,
            email="executed-copy-before-change@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_replay,
            email="replay@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_confirmation_failure,
            email="confirmation-failure@example.com",
        )
        await _seed_customer(
            dba,
            contact_id=contact_acceptance_confirmation_failure,
            email="acceptance-confirmation-failure@example.com",
        )
        await dba.execute(
            "UPDATE contacts SET status = 'archived' WHERE id = $1",
            contact_invalid,
        )
        authority = EOMTermsAuthority(pool=pool)
        await _publish(
            authority,
            label="race.1",
            material=True,
            marker="race",
        )
        concurrent_releases = await asyncio.gather(
            _publish(
                authority,
                label="race.2",
                material=False,
                marker="race-two",
            ),
            _publish(
                authority,
                label="race.3",
                material=False,
                marker="race-three",
            ),
        )
        concurrent_ids = [UUID(release["versionId"]) for release in concurrent_releases]
        concurrent_orders = await dba.fetch(
            """
            SELECT id, publication_order
            FROM eom_terms_versions
            WHERE id = ANY($1::uuid[])
            ORDER BY publication_order
            """,
            concurrent_ids,
        )
        assert len(concurrent_orders) == 2
        assert (
            concurrent_orders[0]["publication_order"]
            < concurrent_orders[1]["publication_order"]
        )
        current_order = await dba.fetchval(
            """
            SELECT version.publication_order
            FROM eom_terms_current_version AS selected
            JOIN eom_terms_versions AS version ON version.id = selected.version_id
            WHERE selected.singleton
            """
        )
        assert current_order == concurrent_orders[-1]["publication_order"]
        service = EOMTermsAcceptanceService(pool=pool)
        no_delivery = _NoDeliveryService(pool=pool)
        invitation_sender = _RecordingSender()
        invalid_sender = _RecordingSender()
        with pytest.raises(EOMTermsAcceptanceConflictError):
            await service.issue_and_send(
                request_key="terms-request-invalid-contact",
                contact_id=contact_invalid,
                locale="en",
                actor_id=7,
                actor_name="Juan",
                public_base_url="https://example.test/onboarding",
                hmac_secret=_SECRET,
                sender=invalid_sender,
            )
        assert invalid_sender.calls == []

        async def issue_duplicate() -> dict[str, Any]:
            return await service.issue_and_send(
                request_key="terms-request-race-duplicate",
                contact_id=contact_duplicate,
                locale="en",
                actor_id=7,
                actor_name="Juan",
                public_base_url="https://example.test/onboarding",
                hmac_secret=_SECRET,
                sender=invitation_sender,
            )

        duplicate_results = await asyncio.gather(
            issue_duplicate(),
            issue_duplicate(),
        )
        assert {result["invitationId"] for result in duplicate_results} == {
            duplicate_results[0]["invitationId"]
        }
        assert {result["idempotent"] for result in duplicate_results} == {False, True}
        assert len(invitation_sender.calls) == 1
        with pytest.raises(EOMTermsAcceptanceConflictError):
            await service.issue_and_send(
                request_key="terms-request-race-duplicate",
                contact_id=contact_opposing,
                locale="en",
                actor_id=7,
                actor_name="Juan",
                public_base_url="https://example.test/onboarding",
                hmac_secret=_SECRET,
                sender=invitation_sender,
            )
        with pytest.raises(EOMTermsAcceptanceValidationError):
            await service.issue_and_send(
                request_key="terms-request-race-duplicate",
                contact_id=contact_duplicate,
                locale="es",
                actor_id=7,
                actor_name="Juan",
                public_base_url="https://example.test/onboarding",
                hmac_secret=_SECRET,
                sender=invitation_sender,
            )

        duplicate_token = format_eom_terms_token(
            invitation_id=UUID(duplicate_results[0]["invitationId"]),
            secret=_SECRET,
        )
        authenticated = authenticate_eom_terms_token(
            token=duplicate_token,
            secret=_SECRET,
        )
        receipt_sender = _RecordingSender()

        async def accept_duplicate() -> dict[str, Any]:
            return await service.accept_and_send(
                token=authenticated,
                signer_name="Same Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="192.0.2.20",
                sender=receipt_sender,
            )

        acceptance_results = await asyncio.gather(
            accept_duplicate(),
            accept_duplicate(),
        )
        assert {result["acceptanceId"] for result in acceptance_results} == {
            acceptance_results[0]["acceptanceId"]
        }
        assert {result["idempotent"] for result in acceptance_results} == {False, True}
        assert len(receipt_sender.calls) == 1

        executed_copy_invitation = await service.issue_and_send(
            request_key="terms-request-executed-copy-contact-change",
            contact_id=contact_executed_copy_changed,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        executed_copy_token = authenticate_eom_terms_token(
            token=format_eom_terms_token(
                invitation_id=UUID(executed_copy_invitation["invitationId"]),
                secret=_SECRET,
            ),
            secret=_SECRET,
        )
        executed_copy_acceptance = await no_delivery.accept_and_send(
            token=executed_copy_token,
            signer_name="Executed Copy Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="192.0.2.34",
            sender=_RecordingSender(),
        )
        await dba.execute(
            "UPDATE contacts SET email = $2 WHERE id = $1",
            contact_executed_copy_changed,
            "executed-copy-after-change@example.com",
        )
        executed_copy_sender = _RecordingSender()
        accepted_after_contact_change = await service.accept_and_send(
            token=executed_copy_token,
            signer_name="Executed Copy Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="192.0.2.34",
            sender=executed_copy_sender,
        )
        assert (
            accepted_after_contact_change["acceptanceId"]
            == executed_copy_acceptance["acceptanceId"]
        )
        assert accepted_after_contact_change["idempotent"] is True
        assert accepted_after_contact_change["deliveryError"] is True
        assert accepted_after_contact_change["deliveryNeedsReconciliation"] is True
        assert accepted_after_contact_change["executedCopyDeliveryStatus"] == "pending"
        assert executed_copy_sender.calls == []
        assert (
            await dba.fetchval(
                "SELECT status FROM eom_terms_deliveries WHERE id = $1",
                UUID(executed_copy_acceptance["deliveryId"]),
            )
            == "pending"
        )
        with pytest.raises(
            asyncpg.PostgresError,
            match="executed-copy delivery is no longer valid",
        ):
            await dba.execute(
                "UPDATE eom_terms_deliveries SET status = 'sending' WHERE id = $1",
                UUID(executed_copy_acceptance["deliveryId"]),
            )

        acceptance_confirmation_invitation = await service.issue_and_send(
            request_key="terms-request-acceptance-confirmation-failure",
            contact_id=contact_acceptance_confirmation_failure,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        acceptance_confirmation_token = authenticate_eom_terms_token(
            token=format_eom_terms_token(
                invitation_id=UUID(acceptance_confirmation_invitation["invitationId"]),
                secret=_SECRET,
            ),
            secret=_SECRET,
        )
        pending_acceptance = await no_delivery.accept_and_send(
            token=acceptance_confirmation_token,
            signer_name="Confirmation Failure Signer",
            terms_accepted=True,
            additional_work_accepted=True,
            client_ip="192.0.2.37",
            sender=_RecordingSender(),
        )

        class _AcceptanceConfirmationFailureService(EOMTermsAcceptanceService):
            async def _deliver(self, **kwargs: Any) -> tuple[dict[str, Any], bool]:
                await dba.execute(
                    """
                    CREATE FUNCTION test_reject_eom_terms_acceptance_confirmation()
                    RETURNS TRIGGER
                    LANGUAGE plpgsql
                    AS $$
                    BEGIN
                        IF OLD.status = 'sending' AND NEW.status = 'sent' THEN
                            RAISE EXCEPTION
                                'injected acceptance confirmation failure';
                        END IF;
                        RETURN NEW;
                    END;
                    $$;
                    CREATE TRIGGER trg_test_reject_eom_terms_acceptance_confirmation
                    BEFORE UPDATE ON eom_terms_deliveries
                    FOR EACH ROW EXECUTE FUNCTION
                        test_reject_eom_terms_acceptance_confirmation();
                    """
                )
                try:
                    return await super()._deliver(**kwargs)
                finally:
                    await dba.execute(
                        """
                        DROP TRIGGER
                            trg_test_reject_eom_terms_acceptance_confirmation
                            ON eom_terms_deliveries;
                        DROP FUNCTION
                            test_reject_eom_terms_acceptance_confirmation();
                        """
                    )

        confirmation_failure_service = _AcceptanceConfirmationFailureService(pool=pool)
        acceptance_confirmation_sender = _RecordingSender()
        accepted_after_confirmation_failure = (
            await confirmation_failure_service.accept_and_send(
                token=acceptance_confirmation_token,
                signer_name="Confirmation Failure Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="192.0.2.37",
                sender=acceptance_confirmation_sender,
            )
        )
        assert (
            accepted_after_confirmation_failure["acceptanceId"]
            == pending_acceptance["acceptanceId"]
        )
        assert accepted_after_confirmation_failure["idempotent"] is True
        assert accepted_after_confirmation_failure["deliveryError"] is True
        assert (
            accepted_after_confirmation_failure["deliveryNeedsReconciliation"] is True
        )
        assert (
            accepted_after_confirmation_failure["executedCopyDeliveryStatus"]
            == "sending"
        )
        assert len(acceptance_confirmation_sender.calls) == 1
        assert (
            await dba.fetchval(
                "SELECT status FROM eom_terms_deliveries WHERE id = $1",
                UUID(pending_acceptance["deliveryId"]),
            )
            == "sending"
        )

        changed_invitation = await no_delivery.issue_and_send(
            request_key="terms-request-contact-changed-before-delivery",
            contact_id=contact_changed_before_delivery,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        await dba.execute(
            "UPDATE contacts SET email = $2 WHERE id = $1",
            contact_changed_before_delivery,
            "changed-after-issue@example.com",
        )
        changed_contact_sender = _RecordingSender()
        with pytest.raises(
            EOMTermsAcceptanceConflictError,
            match="customer changed before Terms delivery",
        ):
            await service._deliver(
                delivery_id=UUID(changed_invitation["deliveryId"]),
                secret=_SECRET,
                previous_secret=None,
                sender=changed_contact_sender,
            )
        assert changed_contact_sender.calls == []
        assert (
            await dba.fetchval(
                "SELECT status FROM eom_terms_deliveries WHERE id = $1",
                UUID(changed_invitation["deliveryId"]),
            )
            == "pending"
        )

        locked_invitation = await no_delivery.issue_and_send(
            request_key="terms-request-delivery-lock",
            contact_id=contact_delivery_lock,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        blocking_sender = _BlockingSender()
        delivery_task = asyncio.create_task(
            service._deliver(
                delivery_id=UUID(locked_invitation["deliveryId"]),
                secret=_SECRET,
                previous_secret=None,
                sender=blocking_sender,
            )
        )
        await asyncio.wait_for(blocking_sender.started.wait(), timeout=2)
        revoke_task = asyncio.create_task(
            service.revoke(
                invitation_id=locked_invitation["invitationId"],
                actor_id=7,
                actor_name="Juan",
            )
        )

        contact_update_started = asyncio.Event()

        async def update_locked_contact() -> None:
            async with pool.raw_pool.acquire() as connection:
                contact_update_started.set()
                await connection.execute(
                    "UPDATE contacts SET email = $2 WHERE id = $1",
                    contact_delivery_lock,
                    "delivery-lock-after-send@example.com",
                )

        contact_update_task = asyncio.create_task(update_locked_contact())
        try:
            await asyncio.wait_for(contact_update_started.wait(), timeout=2)
            await asyncio.sleep(0.05)
            assert revoke_task.done() is False
            assert contact_update_task.done() is False
        finally:
            blocking_sender.release.set()
        locked_delivery, locked_delivery_error = await delivery_task
        locked_revocation = await revoke_task
        await contact_update_task
        assert locked_delivery["status"] == "sent"
        assert locked_delivery_error is False
        assert locked_revocation["revokedAt"] is not None
        assert len(blocking_sender.calls) == 1

        replay_invitation = await no_delivery.issue_and_send(
            request_key="terms-request-provider-replay",
            contact_id=contact_replay,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        replay_sender = _IdempotentReplaySender()
        replay_delivery, replay_delivery_error = await service._deliver(
            delivery_id=UUID(replay_invitation["deliveryId"]),
            secret=_SECRET,
            previous_secret=None,
            sender=replay_sender,
        )
        assert replay_delivery["status"] == "sent"
        assert replay_delivery_error is False
        assert len(replay_sender.calls) == 1
        replay_evidence = await dba.fetchrow(
            """
            SELECT resend_message_id, transport_idempotent_replay
            FROM eom_terms_deliveries
            WHERE id = $1
            """,
            UUID(replay_invitation["deliveryId"]),
        )
        assert replay_evidence is not None
        assert replay_evidence["resend_message_id"] is None
        assert replay_evidence["transport_idempotent_replay"] is True

        confirmation_failure = await no_delivery.issue_and_send(
            request_key="terms-request-confirmation-failure",
            contact_id=contact_confirmation_failure,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        await dba.execute(
            """
            CREATE FUNCTION test_reject_eom_terms_confirmation()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF OLD.status = 'sending' AND NEW.status = 'sent' THEN
                    RAISE EXCEPTION 'injected confirmation failure';
                END IF;
                RETURN NEW;
            END;
            $$;
            CREATE TRIGGER trg_test_reject_eom_terms_confirmation
            BEFORE UPDATE ON eom_terms_deliveries
            FOR EACH ROW EXECUTE FUNCTION test_reject_eom_terms_confirmation();
            """
        )
        confirmation_sender = _RecordingSender()
        try:
            with pytest.raises(EOMTermsAcceptanceUnavailableError):
                await service._deliver(
                    delivery_id=UUID(confirmation_failure["deliveryId"]),
                    secret=_SECRET,
                    previous_secret=None,
                    sender=confirmation_sender,
                )
        finally:
            await dba.execute(
                """
                DROP TRIGGER trg_test_reject_eom_terms_confirmation
                    ON eom_terms_deliveries;
                DROP FUNCTION test_reject_eom_terms_confirmation();
                """
            )
        assert len(confirmation_sender.calls) == 1
        assert (
            await dba.fetchval(
                "SELECT status FROM eom_terms_deliveries WHERE id = $1",
                UUID(confirmation_failure["deliveryId"]),
            )
            == "sending"
        )
        confirmation_retry_sender = _RecordingSender()
        confirmation_retry, confirmation_retry_error = await service._deliver(
            delivery_id=UUID(confirmation_failure["deliveryId"]),
            secret=_SECRET,
            previous_secret=None,
            sender=confirmation_retry_sender,
        )
        assert confirmation_retry["status"] == "sending"
        assert confirmation_retry_error is False
        assert confirmation_retry_sender.calls == []

        invalid_replay = await no_delivery.issue_and_send(
            request_key="terms-request-invalid-provider-replay",
            contact_id=contact_replay,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        uncertain_delivery, uncertain_error = await service._deliver(
            delivery_id=UUID(invalid_replay["deliveryId"]),
            secret=_SECRET,
            previous_secret=None,
            sender=_RecordingSender(fail=True),
        )
        assert uncertain_delivery["status"] == "sending"
        assert uncertain_error is True
        with pytest.raises(
            asyncpg.PostgresError,
            match="requires transport or actor evidence",
        ):
            await dba.execute(
                """
                UPDATE eom_terms_deliveries
                SET status = 'sent',
                    resend_message_id = NULL,
                    transport_idempotent_replay = FALSE
                WHERE id = $1
                """,
                UUID(invalid_replay["deliveryId"]),
            )
        with pytest.raises(
            EOMTermsAcceptanceConflictError,
            match="delivery requires reconciliation",
        ):
            await service.revoke(
                invitation_id=invalid_replay["invitationId"],
                actor_id=7,
                actor_name="Juan",
            )
        with pytest.raises(
            asyncpg.PostgresError,
            match="delivery requires reconciliation",
        ):
            await dba.execute(
                """
                UPDATE eom_terms_invitations
                SET revoked_at = clock_timestamp(),
                    revoked_by_id = 7,
                    revoked_by_name = 'Juan'
                WHERE id = $1
                """,
                UUID(invalid_replay["invitationId"]),
            )
        with pytest.raises(
            asyncpg.PostgresError,
            match="delivery requires reconciliation",
        ):
            await dba.execute(
                """
                INSERT INTO eom_terms_acceptances (
                    id, invitation_id, signer_name, terms_accepted,
                    additional_work_accepted, client_ip
                ) VALUES ($1, $2, $3, TRUE, TRUE, $4::inet)
                """,
                uuid4(),
                UUID(invalid_replay["invitationId"]),
                "Sending Boundary Probe",
                "192.0.2.33",
            )

        opposing = await no_delivery.issue_and_send(
            request_key="terms-request-race-opposing",
            contact_id=contact_opposing,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        opposing_token = authenticate_eom_terms_token(
            token=format_eom_terms_token(
                invitation_id=UUID(opposing["invitationId"]),
                secret=_SECRET,
            ),
            secret=_SECRET,
        )
        opposing_results = await asyncio.gather(
            service.accept_and_send(
                token=opposing_token,
                signer_name="Opposing Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="192.0.2.21",
                sender=_RecordingSender(),
            ),
            service.revoke(
                invitation_id=opposing["invitationId"],
                actor_id=7,
                actor_name="Juan",
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(result, Exception) for result in opposing_results) == 1
        opposing_row = await dba.fetchrow(
            """
            SELECT invitation.revoked_at,
                   acceptance.id AS acceptance_id
            FROM eom_terms_invitations AS invitation
            LEFT JOIN eom_terms_acceptances AS acceptance
              ON acceptance.invitation_id = invitation.id
            WHERE invitation.id = $1
            """,
            UUID(opposing["invitationId"]),
        )
        assert opposing_row is not None
        assert (opposing_row["revoked_at"] is None) != (
            opposing_row["acceptance_id"] is None
        )

        revoked = await no_delivery.issue_and_send(
            request_key="terms-request-revoked-unsent",
            contact_id=contact_revoked,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        await service.revoke(
            invitation_id=revoked["invitationId"],
            actor_id=7,
            actor_name="Juan",
        )
        after_revoke_sender = _RecordingSender()
        delivery, delivery_error = await service._deliver(
            delivery_id=UUID(revoked["deliveryId"]),
            secret=_SECRET,
            previous_secret=None,
            sender=after_revoke_sender,
        )
        assert delivery["status"] == "pending"
        assert delivery_error is False
        assert after_revoke_sender.calls == []
        revoked_token = authenticate_eom_terms_token(
            token=format_eom_terms_token(
                invitation_id=UUID(revoked["invitationId"]),
                secret=_SECRET,
            ),
            secret=_SECRET,
        )
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.get_session(token=revoked_token)
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.accept_and_send(
                token=revoked_token,
                signer_name="Revoked Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="192.0.2.22",
                sender=_RecordingSender(),
            )

        expired = await no_delivery.issue_and_send(
            request_key="terms-request-expired",
            contact_id=contact_expired,
            locale="en",
            actor_id=7,
            actor_name="Juan",
            public_base_url="https://example.test/onboarding",
            hmac_secret=_SECRET,
            sender=_RecordingSender(),
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_invitations
            DISABLE TRIGGER trg_protect_eom_terms_invitation
            """
        )
        await dba.execute(
            """
            WITH authoritative_time AS (
                SELECT clock_timestamp() AS value
            )
            UPDATE eom_terms_invitations
            SET issued_at = authoritative_time.value - INTERVAL '31 days',
                expires_at = authoritative_time.value - INTERVAL '1 day'
            FROM authoritative_time
            WHERE id = $1
            """,
            UUID(expired["invitationId"]),
        )
        await dba.execute(
            """
            ALTER TABLE eom_terms_invitations
            ENABLE TRIGGER trg_protect_eom_terms_invitation
            """
        )
        assert await eom_terms_acceptance_schema_ready(pool) is True
        expired_token = authenticate_eom_terms_token(
            token=format_eom_terms_token(
                invitation_id=UUID(expired["invitationId"]),
                secret=_SECRET,
            ),
            secret=_SECRET,
        )
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.get_session(token=expired_token)
        with pytest.raises(EOMTermsAcceptanceNotFoundError):
            await service.accept_and_send(
                token=expired_token,
                signer_name="Expired Signer",
                terms_accepted=True,
                additional_work_accepted=True,
                client_ip="192.0.2.23",
                sender=_RecordingSender(),
            )


@pytest.mark.asyncio
async def test_card_vault_migration_applies_with_guarded_runtime_acl() -> None:
    async with _real_terms_store() as (pool, dba, schema):
        await dba.execute(_CANDIDATE_MIGRATION.read_text())
        await dba.execute(
            """
            INSERT INTO schema_migrations (version, name)
            VALUES
                (395, '395_eom_post_clean_onboarding_candidates'),
                (397, '397_eom_terms_acceptance')
            ON CONFLICT (version) DO NOTHING
            """
        )
        await dba.execute(_CARD_VAULT_MIGRATION.read_text())
        await dba.execute(
            """
            INSERT INTO schema_migrations (version, name)
            VALUES (398, '398_eom_card_vault')
            """
        )

        assert await eom_card_vault_schema_ready(pool) is True
        original_enrollment_guard = await dba.fetchval(
            """
            SELECT pg_get_functiondef(
                to_regprocedure('protect_eom_card_vault_enrollment()')
            )
            """
        )
        assert isinstance(original_enrollment_guard, str)
        await dba.execute(
            f"""
            CREATE OR REPLACE FUNCTION protect_eom_card_vault_enrollment()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            SET search_path TO pg_catalog, "{schema}", pg_temp
            AS $$
            BEGIN
                RETURN NEW;
            END;
            $$
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(original_enrollment_guard)
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_sessions
            ALTER COLUMN created_at DROP DEFAULT
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_sessions
            ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP
            """
        )
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_events
            ALTER COLUMN received_at DROP NOT NULL
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_events
            ALTER COLUMN received_at SET NOT NULL
            """
        )
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_events
            ALTER COLUMN received_at DROP DEFAULT;
            ALTER TABLE eom_card_vault_events
            ALTER COLUMN received_at TYPE TIMESTAMP WITHOUT TIME ZONE
                USING received_at AT TIME ZONE 'UTC'
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_events
            ALTER COLUMN received_at TYPE TIMESTAMPTZ
                USING received_at AT TIME ZONE 'UTC';
            ALTER TABLE eom_card_vault_events
            ALTER COLUMN received_at SET DEFAULT CURRENT_TIMESTAMP
            """
        )
        assert await eom_card_vault_schema_ready(pool) is True
        relations = await dba.fetch(
            """
            SELECT relation.relname,
                   owner.rolname AS owner,
                   has_table_privilege('atlas', relation.oid, 'SELECT') AS can_select,
                   has_table_privilege('atlas', relation.oid, 'DELETE') AS can_delete
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_roles AS owner ON owner.oid = relation.relowner
            WHERE namespace.nspname = $1
              AND relation.relname LIKE 'eom_card_vault_%'
              AND relation.relkind = 'r'
            ORDER BY relation.relname
            """,
            schema,
        )
        assert [row["relname"] for row in relations] == [
            "eom_card_vault_enrollments",
            "eom_card_vault_events",
            "eom_card_vault_sessions",
        ]
        assert all(row["owner"] == "atlas_eom_handoff_owner" for row in relations)
        assert all(row["can_select"] is True for row in relations)
        assert all(row["can_delete"] is False for row in relations)

        await dba.execute(
            """
            ALTER TABLE eom_card_vault_events
            DISABLE TRIGGER trg_protect_eom_card_vault_event
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_events
            ENABLE TRIGGER trg_protect_eom_card_vault_event
            """
        )
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            """
            CREATE OR REPLACE TRIGGER trg_protect_eom_card_vault_enrollment
            BEFORE INSERT ON eom_card_vault_enrollments
            FOR EACH ROW EXECUTE FUNCTION protect_eom_card_vault_enrollment()
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            """
            CREATE OR REPLACE TRIGGER trg_protect_eom_card_vault_enrollment
            BEFORE INSERT OR UPDATE OR DELETE ON eom_card_vault_enrollments
            FOR EACH ROW EXECUTE FUNCTION protect_eom_card_vault_enrollment()
            """
        )
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_enrollments
            RENAME CONSTRAINT uq_eom_card_vault_enrollment_candidate
            TO drifted_eom_card_vault_enrollment_candidate
            """
        )
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_enrollments
            ADD CONSTRAINT uq_eom_card_vault_enrollment_candidate
            UNIQUE (contact_id)
            """
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_enrollments
            DROP CONSTRAINT uq_eom_card_vault_enrollment_candidate
            """
        )
        await dba.execute(
            """
            ALTER TABLE eom_card_vault_enrollments
            RENAME CONSTRAINT drifted_eom_card_vault_enrollment_candidate
            TO uq_eom_card_vault_enrollment_candidate
            """
        )
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            "REVOKE UPDATE (ready_at) ON eom_card_vault_enrollments FROM atlas"
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            "GRANT UPDATE (ready_at) ON eom_card_vault_enrollments TO atlas"
        )
        assert await eom_card_vault_schema_ready(pool) is True
        await dba.execute(
            "GRANT UPDATE (contact_id) ON eom_card_vault_enrollments TO atlas"
        )
        assert await eom_card_vault_schema_ready(pool) is False
        await dba.execute(
            "REVOKE UPDATE (contact_id) ON eom_card_vault_enrollments FROM atlas"
        )
        assert await eom_card_vault_schema_ready(pool) is True
        with pytest.raises(asyncpg.RaiseError, match="event evidence is append-only"):
            await dba.execute("TRUNCATE eom_card_vault_events")


def test_migration_is_guard_owned_append_only_and_seeds_no_content() -> None:
    migration = _ACCEPTANCE_MIGRATION.read_text()
    assert "database administrator must run 397_eom_terms_acceptance" in migration
    assert "validate_eom_terms_invitation" in migration
    assert "validate_eom_terms_acceptance" in migration
    assert "protect_eom_terms_delivery" in migration
    assert "CREATE OR REPLACE FUNCTION protect_eom_terms_version()" in migration
    assert "trg_assign_eom_terms_publication_order" not in migration
    assert "ADD COLUMN publication_order BIGINT" in migration
    assert "later.publication_order > invitation_row.publication_order" in migration
    assert "BEFORE TRUNCATE ON eom_terms_acceptances" in migration
    assert "GRANT INSERT (id, invitation_id, signer_name" in migration
    assert "GRANT UPDATE (status, claimed_at, sent_at" in migration
    assert "GRANT DELETE" not in migration
    assert "GRANT UPDATE (publication_order" not in migration
    assert "raw bearer tokens are never stored" in migration
    assert "INSERT INTO eom_terms_versions" not in migration
