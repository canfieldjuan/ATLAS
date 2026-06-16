from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest

from atlas_brain import _content_ops_zendesk_credentials as service


MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "330_content_ops_zendesk_credentials.sql"
)


class _Pool:
    def __init__(self, *, fetchrow_result=None, fetch_rows=None) -> None:
        self.fetchrow_result = fetchrow_result
        self.fetch_rows = list(fetch_rows or [])
        self.execute_calls: list[dict] = []
        self.fetchrow_calls: list[dict] = []
        self.fetch_calls: list[dict] = []

    def transaction(self):
        return _Transaction(self)

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": query, "args": args})
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows


class _Transaction:
    def __init__(self, pool: _Pool) -> None:
        self.pool = pool

    async def __aenter__(self):
        return self.pool

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RawAsyncpgPool:
    def __init__(self, conn: "_RawAsyncpgConnection") -> None:
        self.conn = conn
        self.acquire_calls = 0

    def acquire(self):
        self.acquire_calls += 1
        return _AcquireContext(self.conn)


class _AcquireContext:
    def __init__(self, conn: "_RawAsyncpgConnection") -> None:
        self.conn = conn

    async def __aenter__(self):
        self.conn.acquired = True
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.released = True
        return False


class _RawAsyncpgConnection:
    def __init__(self, *, fetchrow_result=None) -> None:
        self.fetchrow_result = fetchrow_result
        self.execute_calls: list[dict] = []
        self.fetchrow_calls: list[dict] = []
        self.transaction_calls = 0
        self.acquired = False
        self.released = False

    def transaction(self):
        self.transaction_calls += 1
        return _ConnectionTransaction(self)

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": query, "args": args})
        return self.fetchrow_result


class _ConnectionTransaction:
    def __init__(self, conn: _RawAsyncpgConnection) -> None:
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _row(**overrides):
    row = {
        "id": uuid.uuid4(),
        "account_id": uuid.uuid4(),
        "email": "agent@example.com",
        "encrypted_api_token": b"ciphertext",
        "encryption_kid": "v1",
        "api_token_prefix": "secret-t",
        "subdomain": "acme",
        "base_url": "",
        "label": "Primary",
        "added_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "last_used_at": None,
        "revoked_at": None,
    }
    row.update(overrides)
    return row


def test_content_ops_zendesk_credentials_migration_is_scoped_and_encrypted() -> None:
    sql = MIGRATION.read_text()

    assert "CREATE TABLE IF NOT EXISTS content_ops_zendesk_credentials" in sql
    assert "account_id          UUID NOT NULL REFERENCES saas_accounts(id)" in sql
    assert "encrypted_api_token BYTEA NOT NULL" in sql
    assert "encryption_kid      VARCHAR(64) NOT NULL" in sql
    assert "api_token_prefix    VARCHAR(16) NOT NULL" in sql
    assert "uq_content_ops_zendesk_credentials_one_active" in sql
    assert "WHERE revoked_at IS NULL" in sql


@pytest.mark.asyncio
async def test_upsert_zendesk_credentials_encrypts_token_and_returns_display_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetchrow_result=_row(account_id=account_id))
    monkeypatch.setattr(
        service,
        "encrypt_secret",
        lambda raw: (f"encrypted:{raw}".encode("utf-8"), "kid-1"),
    )

    record = await service.upsert_zendesk_credentials(
        pool,
        account_id=account_id,
        email=" agent@example.com ",
        api_token=" secret-token ",
        subdomain="acme",
        label=" Primary ",
    )

    insert_call = pool.fetchrow_calls[0]
    assert "INSERT INTO content_ops_zendesk_credentials" in insert_call["query"]
    assert insert_call["args"][0] == account_id
    assert insert_call["args"][1] == "agent@example.com"
    assert insert_call["args"][2] == b"encrypted:secret-token"
    assert insert_call["args"][3] == "kid-1"
    assert insert_call["args"][4] == "secret-t"
    assert insert_call["args"][5] == "acme"
    assert insert_call["args"][7] == "Primary"
    assert record.api_token_prefix == "secret-t"
    assert not hasattr(record, "api_token")
    assert not hasattr(record, "encrypted_api_token")
    assert not hasattr(record, "encryption_kid")


@pytest.mark.asyncio
async def test_upsert_zendesk_credentials_accepts_raw_asyncpg_pool_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = uuid.uuid4()
    conn = _RawAsyncpgConnection(fetchrow_result=_row(account_id=account_id))
    pool = _RawAsyncpgPool(conn)
    monkeypatch.setattr(
        service,
        "encrypt_secret",
        lambda raw: (f"encrypted:{raw}".encode("utf-8"), "kid-1"),
    )

    record = await service.upsert_zendesk_credentials(
        pool,
        account_id=account_id,
        email="agent@example.com",
        api_token="secret-token",
        subdomain="acme",
        label="Primary",
    )

    assert pool.acquire_calls == 1
    assert conn.acquired is True
    assert conn.released is True
    assert conn.transaction_calls == 1
    assert conn.execute_calls[0]["query"].strip().startswith("SELECT id FROM saas_accounts")
    insert_call = conn.fetchrow_calls[0]
    assert "INSERT INTO content_ops_zendesk_credentials" in insert_call["query"]
    assert insert_call["args"][0] == account_id
    assert insert_call["args"][2] == b"encrypted:secret-token"
    assert record.account_id == account_id


@pytest.mark.asyncio
async def test_upsert_zendesk_credentials_rejects_unsupported_pool_before_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnsupportedPool:
        def __init__(self) -> None:
            self.execute_calls = 0

        async def execute(self, query, *args):
            self.execute_calls += 1

    pool = _UnsupportedPool()
    monkeypatch.setattr(service, "encrypt_secret", lambda raw: (b"encrypted", "kid-1"))

    with pytest.raises(
        service.ZendeskCredentialWriteError,
        match="zendesk_credentials_write_pool_unsupported",
    ):
        await service.upsert_zendesk_credentials(
            pool,
            account_id=uuid.uuid4(),
            email="agent@example.com",
            api_token="secret-token",
            subdomain="acme",
        )

    assert pool.execute_calls == 0


@pytest.mark.asyncio
async def test_lookup_zendesk_credentials_decrypts_tenant_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = uuid.uuid4()
    row = _row(
        id=uuid.uuid4(),
        account_id=account_id,
        encrypted_api_token=b"ciphertext",
        encryption_kid="kid-1",
    )
    pool = _Pool(fetchrow_result=row)
    monkeypatch.setattr(
        service,
        "decrypt_secret",
        lambda ciphertext, kid: "secret-token"
        if ciphertext == b"ciphertext" and kid == "kid-1"
        else None,
    )

    credentials = await service.lookup_zendesk_credentials(
        pool,
        account_id=str(account_id),
    )

    assert credentials is not None
    assert credentials.email == "agent@example.com"
    assert credentials.api_token == "secret-token"
    assert credentials.normalized_base_url() == "https://acme.zendesk.com"
    assert pool.fetchrow_calls[0]["args"] == (account_id,)
    assert "WHERE account_id = $1 AND revoked_at IS NULL" in pool.fetchrow_calls[0]["query"]
    assert pool.execute_calls[0]["args"] == (row["id"],)


@pytest.mark.asyncio
async def test_lookup_zendesk_credentials_raises_on_decrypt_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool(fetchrow_result=_row())
    monkeypatch.setattr(service, "decrypt_secret", lambda ciphertext, kid: None)

    with pytest.raises(
        service.ZendeskCredentialLookupError,
        match="zendesk_credentials_unavailable",
    ):
        await service.lookup_zendesk_credentials(
            pool,
            account_id=str(uuid.uuid4()),
        )

    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_lookup_zendesk_credentials_raises_on_fetch_failure() -> None:
    class _FailingPool(_Pool):
        async def fetchrow(self, query, *args):
            raise RuntimeError("db down secret-token")

    with pytest.raises(
        service.ZendeskCredentialLookupError,
        match="zendesk_credentials_unavailable",
    ) as exc:
        await service.lookup_zendesk_credentials(
            _FailingPool(),
            account_id=str(uuid.uuid4()),
        )

    assert "secret-token" not in str(exc.value)


@pytest.mark.asyncio
async def test_list_zendesk_credentials_is_display_safe() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetch_rows=[_row(account_id=account_id)])

    records = await service.list_zendesk_credentials(pool, account_id=account_id)

    assert records[0].account_id == account_id
    assert records[0].api_token_prefix == "secret-t"
    assert "encrypted_api_token" not in records[0].__dict__
    assert "encryption_kid" not in records[0].__dict__
