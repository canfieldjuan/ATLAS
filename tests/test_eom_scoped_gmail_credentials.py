"""Real-Postgres lifecycle proof for exact-context scoped Gmail reads."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs
import uuid

import httpx
import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.config import InboxMailboxBinding  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.storage.repositories.scoped_mailbox_credential import (  # noqa: E402
    ScopedMailboxCredentialRepository,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
OTHER_TENANT = "churnsignals"
TEST_KEK = "test:DEj0-fNH6mOs5JYXn3Uv6ejEfP4PQ6XIqWla36eIR_U="


class _PoolAdapter:
    def __init__(self, pool):
        self._pool = pool
        self.is_initialized = True

    async def fetch(self, query, *args):
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query, *args):
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query, *args):
        return await self._pool.fetchval(query, *args)

    async def execute(self, query, *args):
        return await self._pool.execute(query, *args)

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield connection


class _GoogleHTTP:
    def __init__(self) -> None:
        self.token_inputs: list[tuple[str, str, str]] = []
        self.active_token_requests = 0
        self.max_active_token_requests = 0
        self.gmail_requests = 0

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL(
            "https://oauth2.googleapis.com/token"
        ):
            form = parse_qs(request.content.decode("ascii"))
            self.active_token_requests += 1
            self.max_active_token_requests = max(
                self.max_active_token_requests,
                self.active_token_requests,
            )
            try:
                await asyncio.sleep(0.03)
                call_number = len(self.token_inputs) + 1
                self.token_inputs.append(
                    (
                        form["client_id"][0],
                        form["client_secret"][0],
                        form["refresh_token"][0],
                    )
                )
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "access_token": f"access-{call_number}",
                        "expires_in": 3600,
                        "refresh_token": f"rotated-{call_number}",
                    },
                )
            finally:
                self.active_token_requests -= 1

        self.gmail_requests += 1
        assert request.headers["Authorization"].startswith("Bearer access-")
        if request.url.path.endswith("/users/me/messages"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "messages": [
                        {"id": "collision"},
                        {"id": "exact"},
                    ]
                },
            )
        if request.url.path.endswith("/messages/collision"):
            headers = [
                {"name": "From", "value": "customer@example.com"},
                {"name": "From", "value": "attacker@example.com"},
                {"name": "Subject", "value": "Ambiguous"},
            ]
        elif request.url.path.endswith("/messages/exact"):
            headers = [
                {
                    "name": "from",
                    "value": "Customer <customer@example.com>",
                },
                {"name": "Subject", "value": "Bound reply"},
                {"name": "Date", "value": "Fri, 25 Jul 2026 10:00:00 -0500"},
            ]
        else:
            raise AssertionError(f"Unexpected Google request: {request.url}")
        return httpx.Response(
            200,
            request=request,
            json={
                "id": request.url.path.rsplit("/", 1)[-1],
                "payload": {"headers": headers},
                "snippet": "Scoped Gmail proof",
            },
        )


@pytest.mark.asyncio
async def test_scoped_rotation_finishes_before_cancellation_propagates():
    import atlas_brain.autonomous.tasks.gmail_digest as gmail_mod

    started = asyncio.Event()
    release = asyncio.Event()
    persisted: list[str] = []
    lease_exited = asyncio.Event()

    class _Lease:
        credentials = SimpleNamespace(
            client_id="client",
            client_secret="secret",
            refresh_token="old-refresh",
        )

        async def persist_refresh_token(self, token):
            persisted.append(token)
            return 2

    class _Source:
        @asynccontextmanager
        async def locked_credentials(self):
            try:
                yield _Lease()
            finally:
                lease_exited.set()

    async def token_handler(request):
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": "access",
                "expires_in": 3600,
                "refresh_token": "new-refresh",
            },
        )

    client = gmail_mod.GmailClient(_Source())
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(token_handler)
    )
    refresh = asyncio.create_task(client._refresh_token())
    await started.wait()
    refresh.cancel()
    await asyncio.sleep(0)
    assert not refresh.done()
    refresh.cancel()
    await asyncio.sleep(0)
    assert not refresh.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await refresh

    assert persisted == ["new-refresh"]
    assert lease_exited.is_set()
    assert client._access_token == "access"
    await client.close()


@pytest.mark.asyncio
async def test_scoped_refresh_failure_preserves_recorded_cancellation():
    import atlas_brain.autonomous.tasks.gmail_digest as gmail_mod

    started = asyncio.Event()
    release = asyncio.Event()
    lease_exited = asyncio.Event()

    class _Lease:
        credentials = SimpleNamespace(
            client_id="client",
            client_secret="secret",
            refresh_token="old-refresh",
        )

        async def persist_refresh_token(self, _token):
            raise AssertionError("A failed refresh must not persist")

    class _Source:
        @asynccontextmanager
        async def locked_credentials(self):
            try:
                yield _Lease()
            finally:
                lease_exited.set()

    async def token_handler(request):
        started.set()
        await release.wait()
        return httpx.Response(
            503,
            request=request,
            json={"error": "temporarily_unavailable"},
        )

    client = gmail_mod.GmailClient(_Source())
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(token_handler)
    )
    refresh = asyncio.create_task(client._refresh_token())
    await started.wait()
    refresh.cancel()
    await asyncio.sleep(0)
    assert not refresh.done()
    release.set()

    with pytest.raises(asyncio.CancelledError) as captured:
        await refresh

    assert isinstance(captured.value.__cause__, httpx.HTTPStatusError)
    assert lease_exited.is_set()
    await client.close()


@pytest.mark.asyncio
async def test_unconfigured_global_gmail_does_not_allocate_http_client(
    monkeypatch,
    tmp_path,
):
    import atlas_brain.autonomous.tasks.gmail_digest as gmail_mod
    import atlas_brain.services.google_oauth as google_oauth_mod

    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "google_token_file",
        str(tmp_path / "missing-global-google-token.json"),
    )
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "gmail_client_id",
        "",
    )
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "gmail_client_secret",
        "",
    )
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "gmail_refresh_token",
        "",
    )
    previous_store = google_oauth_mod._store
    google_oauth_mod._store = None
    try:
        client = gmail_mod.GmailClient()
        with pytest.raises(RuntimeError, match="Gmail OAuth not configured"):
            await client._refresh_token()
        assert client._client is None
    finally:
        google_oauth_mod._store = previous_store


@pytest.mark.asyncio
async def test_unscoped_gmail_keeps_global_store_rotation_contract(
    monkeypatch,
    tmp_path,
):
    import atlas_brain.autonomous.tasks.gmail_digest as gmail_mod
    import atlas_brain.services.google_oauth as google_oauth_mod

    token_path = tmp_path / "global-google-token.json"
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "google_token_file",
        str(token_path),
    )
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "gmail_client_id",
        "global-client",
    )
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "gmail_client_secret",
        "global-secret",
    )
    monkeypatch.setattr(
        google_oauth_mod.settings.tools,
        "gmail_refresh_token",
        "global-refresh",
    )
    previous_store = google_oauth_mod._store
    google_oauth_mod._store = None

    async def token_handler(request):
        form = parse_qs(request.content.decode("ascii"))
        assert form["refresh_token"] == ["global-refresh"]
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": "global-access",
                "expires_in": 3600,
                "refresh_token": "global-rotated",
            },
        )

    try:
        client = gmail_mod.GmailClient()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(token_handler)
        )
        assert await client._refresh_token() == "global-access"
        persisted = json.loads(token_path.read_text())
        assert persisted["services"]["gmail"]["refresh_token"] == (
            "global-rotated"
        )
        await client.close()
    finally:
        google_oauth_mod._store = previous_store


@pytest.mark.asyncio
async def test_scoped_gmail_mcp_lifecycle_is_encrypted_durable_and_serial(
    caplog,
    monkeypatch,
    tmp_path,
):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_scoped_gmail_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    crm_server = None
    context_mod = None
    db_mod = None
    original_db_pool = None
    previous_service = None
    google_oauth_mod = None
    previous_google_store = None
    gmail_mod = None
    previous_global_gmail_client = None
    try:
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        for name in (
            "001_initial_schema.sql",
            "012_appointments.sql",
            "016_sent_emails.sql",
            "030_call_transcripts.sql",
            "035_contacts.sql",
            "036_call_contact_link.sql",
            "044_sms_messages.sql",
            "045_invoices.sql",
            "348_appointment_operating_fields.sql",
            "349_sent_emails_business_context.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())
        migration = (
            MIGRATIONS / "350_scoped_mailbox_credentials.sql"
        ).read_text()
        await admin_conn.execute(migration)
        await admin_conn.execute(migration)

        with pytest.raises(asyncpg.CheckViolationError):
            await admin_conn.execute(
                """
                INSERT INTO scoped_mailbox_credentials (
                    business_context_id,
                    provider,
                    encrypted_credentials,
                    encryption_kid
                )
                VALUES (' ', 'gmail', '\\x01', 'test')
                """
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await admin_conn.execute(
                """
                INSERT INTO scoped_mailbox_credentials (
                    business_context_id,
                    provider,
                    encrypted_credentials,
                    encryption_kid
                )
                VALUES ('foreign', 'imap', '\\x01', 'test')
                """
            )

        async def set_search_path(connection):
            await connection.execute(
                f'SET search_path TO "{schema}", public'
            )

        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=12,
            setup=set_search_path,
        )
        adapter = _PoolAdapter(pool)

        import atlas_brain.autonomous.tasks.gmail_digest as gmail_mod
        import atlas_brain.config as config_mod
        import atlas_brain.mcp.crm_server as crm_server
        import atlas_brain.services.crm_provider as crm_provider_mod
        import atlas_brain.services.customer_context as context_mod
        import atlas_brain.services.google_oauth as google_oauth_mod
        import atlas_brain.storage.database as db_mod

        original_db_pool = db_mod._db_pool
        db_mod._db_pool = adapter
        provider = DatabaseCRMProvider()
        crm_server.set_provider_override(lambda: provider)
        monkeypatch.setattr(crm_provider_mod, "_crm_provider", provider)
        previous_service = context_mod._customer_context_service
        context_mod._customer_context_service = (
            context_mod.CustomerContextService()
        )
        monkeypatch.setattr(
            config_mod.settings.email,
            "inbox_context_bindings",
            {
                TENANT: InboxMailboxBinding(provider="gmail"),
                OTHER_TENANT: InboxMailboxBinding(provider="gmail"),
            },
        )
        monkeypatch.setattr(
            config_mod.settings.saas_auth,
            "byok_encryption_kek",
            TEST_KEK,
        )
        monkeypatch.setattr(
            config_mod.settings.tools,
            "google_token_file",
            str(tmp_path / "unused-global-google-token.json"),
        )
        monkeypatch.setattr(
            config_mod.settings.tools,
            "gmail_client_id",
            "forbidden-global-client",
        )
        monkeypatch.setattr(
            config_mod.settings.tools,
            "gmail_client_secret",
            "forbidden-global-secret",
        )
        monkeypatch.setattr(
            config_mod.settings.tools,
            "gmail_refresh_token",
            "forbidden-global-refresh",
        )
        previous_google_store = google_oauth_mod._store
        google_oauth_mod._store = None
        previous_global_gmail_client = gmail_mod._gmail_client
        gmail_mod._gmail_client = None
        google = _GoogleHTTP()
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            gmail_mod.httpx,
            "AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(google.handle),
                **kwargs,
            ),
        )

        contact_id = await pool.fetchval(
            """
            INSERT INTO contacts (
                full_name, email, phone, business_context_id
            )
            VALUES (
                'Scoped Gmail Customer',
                'customer@example.com',
                '217-555-0196',
                $1
            )
            RETURNING id
            """,
            TENANT,
        )
        foreign_contact_id = await pool.fetchval(
            """
            INSERT INTO contacts (
                full_name, email, phone, business_context_id
            )
            VALUES (
                'Unbound Gmail Customer',
                'customer@example.com',
                '217-555-0197',
                $1
            )
            RETURNING id
            """,
            OTHER_TENANT,
        )
        repository = ScopedMailboxCredentialRepository()
        assert await repository.bind_gmail(
            business_context_id=TENANT,
            client_id="client-1",
            client_secret="secret-1",
            refresh_token="refresh-1",
        ) == 1

        stored = await pool.fetchrow(
            """
            SELECT encrypted_credentials, encryption_kid, generation
            FROM scoped_mailbox_credentials
            WHERE business_context_id = $1 AND provider = 'gmail'
            """,
            TENANT,
        )
        ciphertext = bytes(stored["encrypted_credentials"])
        assert stored["encryption_kid"] == "test"
        assert stored["generation"] == 1
        for plaintext in (b"client-1", b"secret-1", b"refresh-1"):
            assert plaintext not in ciphertext

        malformed_kek = "review-kek-secret-without-kid"
        monkeypatch.setattr(
            config_mod.settings.saas_auth,
            "byok_encryption_kek",
            malformed_kek,
        )
        caplog.set_level(logging.WARNING)
        caplog.clear()
        malformed_config = json.loads(
            await crm_server.get_customer_context(
                contact_id=str(contact_id),
                business_context_id=TENANT,
                max_emails=1,
            )
        )
        assert malformed_config["inbox_emails"] == []
        assert malformed_config["email_sources_omitted_under_scope"] == [
            "inbox_emails"
        ]
        assert google.token_inputs == []
        assert google.gmail_requests == 0
        for forbidden in (
            malformed_kek,
            "client-1",
            "secret-1",
            "refresh-1",
            ciphertext.decode("ascii"),
        ):
            assert forbidden not in caplog.text
        monkeypatch.setattr(
            config_mod.settings.saas_auth,
            "byok_encryption_kek",
            TEST_KEK,
        )
        caplog.clear()

        unbound = json.loads(
            await crm_server.get_customer_context(
                contact_id=str(foreign_contact_id),
                business_context_id=OTHER_TENANT,
                max_emails=1,
            )
        )
        assert unbound["inbox_emails"] == []
        assert unbound["email_sources_omitted_under_scope"] == [
            "inbox_emails"
        ]
        assert google.token_inputs == []
        assert google.gmail_requests == 0

        assert await repository.bind_gmail(
            business_context_id=OTHER_TENANT,
            client_id="foreign-client",
            client_secret="foreign-secret",
            refresh_token="foreign-refresh",
        ) == 1
        await pool.execute(
            """
            UPDATE scoped_mailbox_credentials
            SET encryption_kid = 'retired-test-key'
            WHERE business_context_id = $1 AND provider = 'gmail'
            """,
            OTHER_TENANT,
        )
        undecryptable = json.loads(
            await crm_server.get_customer_context(
                contact_id=str(foreign_contact_id),
                business_context_id=OTHER_TENANT,
                max_emails=1,
            )
        )
        assert undecryptable["inbox_emails"] == []
        assert undecryptable["email_sources_omitted_under_scope"] == [
            "inbox_emails"
        ]
        assert google.token_inputs == []
        assert google.gmail_requests == 0

        async def read_context() -> dict:
            return json.loads(
                await crm_server.get_customer_context(
                    contact_id=str(contact_id),
                    business_context_id=TENANT,
                    max_emails=1,
                )
            )

        first = await read_context()
        assert [item["id"] for item in first["inbox_emails"]] == ["exact"]
        assert "_atlas_from_header_values" not in first["inbox_emails"][0]
        assert first["email_sources_omitted_under_scope"] == []
        assert google.token_inputs == [
            ("client-1", "secret-1", "refresh-1")
        ]
        assert (
            await repository.get_active_gmail(TENANT)
        ).refresh_token == "rotated-1"

        context_mod._customer_context_service = (
            context_mod.CustomerContextService()
        )
        restarted = await read_context()
        assert [item["id"] for item in restarted["inbox_emails"]] == [
            "exact"
        ]
        assert google.token_inputs[-1] == (
            "client-1",
            "secret-1",
            "rotated-1",
        )

        google.max_active_token_requests = 0
        concurrent = await asyncio.gather(read_context(), read_context())
        assert all(
            [item["id"] for item in result["inbox_emails"]] == ["exact"]
            for result in concurrent
        )
        assert google.max_active_token_requests == 1
        assert [item[2] for item in google.token_inputs[-2:]] == [
            "rotated-2",
            "rotated-3",
        ]

        assert await repository.bind_gmail(
            business_context_id=TENANT,
            client_id="client-2",
            client_secret="secret-2",
            refresh_token="refresh-2",
        ) == 6
        rebound = await read_context()
        assert [item["id"] for item in rebound["inbox_emails"]] == ["exact"]
        assert google.token_inputs[-1] == (
            "client-2",
            "secret-2",
            "refresh-2",
        )

        token_calls_before_revoke = len(google.token_inputs)
        gmail_calls_before_revoke = google.gmail_requests
        assert await repository.revoke_gmail(TENANT) == 8
        revoked = await read_context()
        assert revoked["inbox_emails"] == []
        assert revoked["email_sources_omitted_under_scope"] == [
            "inbox_emails"
        ]
        assert len(google.token_inputs) == token_calls_before_revoke
        assert google.gmail_requests == gmail_calls_before_revoke
        assert await repository.get_active_gmail(TENANT) is None
    finally:
        if crm_server is not None:
            crm_server.set_provider_override(None)
        if context_mod is not None:
            context_mod._customer_context_service = previous_service
        if db_mod is not None:
            db_mod._db_pool = original_db_pool
        if gmail_mod is not None:
            if (
                gmail_mod._gmail_client is not None
                and gmail_mod._gmail_client is not previous_global_gmail_client
            ):
                await gmail_mod._gmail_client.close()
            gmail_mod._gmail_client = previous_global_gmail_client
        if google_oauth_mod is not None:
            google_oauth_mod._store = previous_google_store
        if pool is not None:
            await pool.close()
        await admin_conn.execute(
            f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'
        )
        await admin_conn.close()
