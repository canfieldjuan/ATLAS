from __future__ import annotations

import json
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from atlas_brain.mcp.auth import BearerAuthMiddleware
from atlas_brain.mcp import invoicing_draft_writer_server as draft_writer


DRAFT_WRITER_TOOLS = {
    "create_draft_invoice",
    "update_draft_invoice",
    "get_invoice",
    "list_pending_drafts",
}

DENIED_TOOLS = {
    "approve_and_send",
    "create_invoice",
    "create_service",
    "customer_balance",
    "export_invoice_pdf",
    "list_invoices",
    "list_services",
    "mark_void",
    "payment_history",
    "record_payment",
    "search_invoices",
    "send_invoice",
    "set_service_status",
    "update_invoice",
    "update_service",
}


class _FakeRepo:
    def __init__(self, existing=None, invoice=None):
        self.existing = existing
        self.invoice = invoice
        self.create_calls = []
        self.source_refs = []

    async def get_by_source_ref(self, source_ref):
        self.source_refs.append(source_ref)
        return self.existing

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return {
            "id": "invoice-1",
            "invoice_number": "INV-2026-May-0001",
            "status": "draft",
            **kwargs,
        }

    async def get_by_number(self, invoice_number):
        return self.invoice

    async def get_by_id(self, invoice_id):
        return self.invoice


def _tool_names() -> set[str]:
    return set(draft_writer.mcp._tool_manager._tools)


class _SchemaPool:
    is_initialized = True

    def __init__(self, connection, schema: str) -> None:
        self.connection = connection
        self.schema = schema

    async def acquire(self):
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        return self.connection

    async def release(self, released) -> None:
        assert released is self.connection

    async def fetchval(self, query, *args):
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        return await self.connection.fetchval(query, *args)


def test_invoicing_draft_writer_exposes_exact_safe_tool_surface():
    assert _tool_names() == DRAFT_WRITER_TOOLS
    assert _tool_names().isdisjoint(DENIED_TOOLS)


@pytest.mark.asyncio
async def test_create_draft_invoice_creates_draft_without_full_mcp_side_effects(monkeypatch):
    repo = _FakeRepo()
    monkeypatch.setattr(draft_writer, "_repo", lambda: repo)

    def fail_crm():
        raise AssertionError("CRM touched")

    monkeypatch.setattr(draft_writer._full, "_crm", fail_crm)

    result = json.loads(
        await draft_writer.create_draft_invoice(
            customer_name="Mid Illinois Concrete",
            line_items=json.dumps(
                [{"description": "Monthly service", "quantity": 2, "unit_price": 125}]
            ),
            idempotency_key="mid-il-concrete-may-2026",
            customer_email="billing@example.com",
            invoice_for="May 2026 service",
            business_context_id="mid-illinois-concrete",
        )
    )

    assert result["success"] is True
    assert result["created"] is True
    assert result["invoice"]["status"] == "draft"
    assert len(repo.create_calls) == 1
    call = repo.create_calls[0]
    assert call["source"] == "chatgpt_draft_writer"
    assert call["source_ref"].startswith("chatgpt_draft_writer:")
    assert call["metadata"] == {
        "mcp_connector": "invoicing_draft_writer",
        "operator_review_required": True,
        "created_by_remote_connector": True,
        "idempotency_key": "mid-il-concrete-may-2026",
    }
    assert call["line_items"][0]["amount"] == 250.0


@pytest.mark.asyncio
async def test_create_draft_invoice_is_idempotent(monkeypatch):
    existing = {"id": "invoice-existing", "status": "draft"}
    repo = _FakeRepo(existing=existing)
    monkeypatch.setattr(draft_writer, "_repo", lambda: repo)

    result = json.loads(
        await draft_writer.create_draft_invoice(
            customer_name="Firefly Grill",
            line_items=json.dumps([{"description": "Hours", "quantity": 1, "unit_price": 50}]),
            idempotency_key="firefly-may-2026",
        )
    )

    assert result == {"success": True, "created": False, "invoice": existing}
    assert repo.create_calls == []
    assert repo.source_refs == [draft_writer._source_ref_for_idempotency_key("firefly-may-2026")]


@pytest.mark.asyncio
async def test_create_draft_invoice_rejects_missing_or_invalid_inputs(monkeypatch):
    repo = _FakeRepo()
    monkeypatch.setattr(draft_writer, "_repo", lambda: repo)

    missing_key = json.loads(
        await draft_writer.create_draft_invoice(
            customer_name="Customer",
            line_items=json.dumps([{"description": "Service"}]),
            idempotency_key="",
        )
    )
    invalid_item = json.loads(
        await draft_writer.create_draft_invoice(
            customer_name="Customer",
            line_items=json.dumps([{"quantity": -1, "unit_price": 20}]),
            idempotency_key="key-1",
        )
    )

    assert missing_key == {"success": False, "error": "idempotency_key is required"}
    assert invalid_item == {
        "success": False,
        "error": "Line item 1 description is required",
    }
    assert repo.create_calls == []


@pytest.mark.asyncio
async def test_create_draft_invoice_rejects_non_finite_numbers(monkeypatch):
    repo = _FakeRepo()
    monkeypatch.setattr(draft_writer, "_repo", lambda: repo)

    bad_line_item = json.loads(
        await draft_writer.create_draft_invoice(
            customer_name="Customer",
            line_items=json.dumps(
                [{"description": "Service", "quantity": "NaN", "unit_price": 20}]
            ),
            idempotency_key="key-1",
        )
    )
    bad_tax = json.loads(
        await draft_writer.create_draft_invoice(
            customer_name="Customer",
            line_items=json.dumps([{"description": "Service", "quantity": 1, "unit_price": 20}]),
            idempotency_key="key-2",
            tax_rate=float("nan"),
        )
    )

    assert bad_line_item == {"success": False, "error": "Line item 1 quantity must be finite"}
    assert bad_tax == {"success": False, "error": "tax_rate must be finite"}
    assert repo.create_calls == []


@pytest.mark.asyncio
async def test_update_draft_invoice_delegates_to_existing_draft_only_update(monkeypatch):
    calls = []
    repo = _FakeRepo(invoice={"id": "invoice-1", "status": "draft"})
    monkeypatch.setattr(draft_writer, "_repo", lambda: repo)

    async def fake_update_invoice(**kwargs):
        calls.append(kwargs)
        return '{"success": true}'

    monkeypatch.setattr(draft_writer._full, "update_invoice", fake_update_invoice)

    result = await draft_writer.update_draft_invoice(
        invoice_id="INV-2026-May-0001",
        line_items='[{"description":"Service","quantity":1,"unit_price":10}]',
        due_date="2026-06-01",
        notes="Operator reviewed",
        tax_rate=0.0,
        discount_amount=0.0,
        invoice_for="June service",
        contact_name="Jane",
    )

    assert result == '{"success": true}'
    assert calls == [
        {
            "invoice_id": "INV-2026-May-0001",
            "line_items": '[{"description":"Service","quantity":1,"unit_price":10}]',
            "due_date": "2026-06-01",
            "notes": "Operator reviewed",
            "tax_rate": 0.0,
            "discount_amount": 0.0,
            "invoice_for": "June service",
            "contact_name": "Jane",
        }
    ]


@pytest.mark.asyncio
async def test_update_draft_invoice_rejects_non_draft_before_full_update(monkeypatch):
    calls = []
    repo = _FakeRepo(invoice={"id": "invoice-1", "status": "sent"})
    monkeypatch.setattr(draft_writer, "_repo", lambda: repo)

    async def fail_update_invoice(**kwargs):
        calls.append(kwargs)
        raise AssertionError("full update touched")

    monkeypatch.setattr(draft_writer._full, "update_invoice", fail_update_invoice)

    result = json.loads(await draft_writer.update_draft_invoice(invoice_id="INV-2026-May-0001"))

    assert result == {"success": False, "error": "Invoice is not in draft status"}
    assert calls == []


def test_invoicing_draft_writer_http_requires_bearer_token(monkeypatch):
    monkeypatch.delenv("ATLAS_MCP_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="ATLAS_MCP_AUTH_TOKEN is required"):
        draft_writer._streamable_http_app()


def test_invoicing_draft_writer_http_wraps_with_bearer_auth(monkeypatch):
    monkeypatch.setenv("ATLAS_MCP_AUTH_TOKEN", "test-token-with-enough-entropy")

    app = draft_writer._streamable_http_app()

    assert isinstance(app, BearerAuthMiddleware)


@pytest.mark.parametrize("token", ["<token>", "token", "test-token"])
def test_invoicing_draft_writer_http_rejects_placeholder_tokens(monkeypatch, token):
    monkeypatch.setenv("ATLAS_MCP_AUTH_TOKEN", token)

    with pytest.raises(RuntimeError, match="must not be a placeholder"):
        draft_writer._streamable_http_app()


def test_invoicing_draft_writer_http_rejects_short_tokens(monkeypatch):
    monkeypatch.setenv("ATLAS_MCP_AUTH_TOKEN", "short-token-value")

    with pytest.raises(RuntimeError, match="at least 24 characters"):
        draft_writer._streamable_http_app()


@pytest.mark.asyncio
async def test_draft_invoice_schema_ready_requires_runtime_dependencies():
    class _Pool:
        is_initialized = True

        async def fetchval(self, query):
            assert "invoice_number_seq" in query
            assert "invoice_payments" in query
            assert "invoice_number" in query
            return True

    assert await draft_writer._draft_invoice_schema_ready(_Pool()) is True


@pytest.mark.asyncio
async def test_draft_writer_lifespan_now_migrates_and_verifies_readiness(monkeypatch):
    """Before ATLAS #2448's startup-fence extension, this server's _lifespan
    did nothing but init_database() -- no migration run, no schema check --
    even though its create_draft_invoice/update_draft_invoice tools reach
    InvoiceRepository.create(). It now uses a curated invoice-schema
    migration set and a draft-invoice readiness probe, not the full
    receivables chain."""
    import atlas_brain.storage.database as database_mod
    import atlas_brain.storage.migrations as migrations_mod

    events: list[str] = []
    pool = SimpleNamespace(is_initialized=True)

    async def initialize() -> None:
        events.append("initialize")

    def get_pool() -> object:
        events.append("get-pool")
        return pool

    async def migrate(candidate: object, *, only=None) -> None:
        assert candidate is pool
        assert tuple(only) == draft_writer._DRAFT_WRITER_INVOICE_MIGRATIONS
        events.append("migrate")

    async def close() -> None:
        events.append("close")

    async def ready(candidate: object) -> bool:
        assert candidate is pool
        events.append("ready")
        return True

    monkeypatch.setattr(database_mod, "init_database", initialize)
    monkeypatch.setattr(database_mod, "get_db_pool", get_pool)
    monkeypatch.setattr(database_mod, "close_database", close)
    monkeypatch.setattr(migrations_mod, "run_migrations", migrate)
    monkeypatch.setattr(draft_writer, "_draft_invoice_schema_ready", ready)

    async with draft_writer._lifespan(object()):
        events.append("serving")

    assert events == [
        "initialize",
        "get-pool",
        "migrate",
        "ready",
        "serving",
        "close",
    ]


@pytest.mark.asyncio
async def test_draft_writer_lifespan_blocks_on_incomplete_schema(monkeypatch):
    """Negative control: an incomplete draft invoice schema now aborts this
    server's startup instead of silently serving."""
    import atlas_brain.storage.database as database_mod
    import atlas_brain.storage.migrations as migrations_mod

    pool = SimpleNamespace(is_initialized=True)
    closed: list[bool] = []

    async def initialize() -> None:
        pass

    def get_pool() -> object:
        return pool

    async def migrate(candidate: object, *, only=None) -> None:
        assert candidate is pool
        assert tuple(only) == draft_writer._DRAFT_WRITER_INVOICE_MIGRATIONS

    async def close() -> None:
        closed.append(True)

    async def not_ready(candidate: object) -> bool:
        assert candidate is pool
        return False

    monkeypatch.setattr(database_mod, "init_database", initialize)
    monkeypatch.setattr(database_mod, "get_db_pool", get_pool)
    monkeypatch.setattr(database_mod, "close_database", close)
    monkeypatch.setattr(migrations_mod, "run_migrations", migrate)
    monkeypatch.setattr(draft_writer, "_draft_invoice_schema_ready", not_ready)

    with pytest.raises(RuntimeError, match="complete draft invoice schema"):
        async with draft_writer._lifespan(object()):
            raise AssertionError("incomplete schema must not serve")

    assert closed == [True]


@pytest.mark.asyncio
async def test_draft_writer_curated_migrations_apply_to_empty_schema():
    """The curated draft-writer migration set is dependency-closed on a clean
    component schema, including the 012_appointments prerequisite that
    035_contacts alters."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain.storage.migrations import run_migrations

    schema = f"draft_writer_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    pool = _SchemaPool(connection, schema)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await run_migrations(
            pool,
            only=draft_writer._DRAFT_WRITER_INVOICE_MIGRATIONS,
        )

        assert await draft_writer._draft_invoice_schema_ready(pool) is True
    finally:
        await connection.execute("SET search_path TO public")
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()
