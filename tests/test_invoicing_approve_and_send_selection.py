"""approve_and_send: which invoices a call selects, and what the mail says.

Widening `invoice_ids` to accept a native list made `[]` a valid argument. An
explicit selection -- however empty -- must select exactly those invoices; only
an omitted argument means "every invoice matching status_filter". Getting that
wrong mails up to 200 drafts to customers on a call that meant to send none.

Every case goes through the published MCP boundary (`mcp.call_tool`, and
`mcp.list_tools` for the schema), because the defect being fixed lived in the
tool annotation that FastMCP validates arguments against, not in the Python
body. The repository and PDF renderer are real: the tests run on a throwaway
schema built from the invoice migrations inside the Postgres named by
`ATLAS_RECEIVABLES_TEST_DATABASE_URL`, the same database the receivables tests
use, and drop it afterwards. The only substitution is the outbound email port,
replaced by a capturing provider so nothing leaves the machine; the attachment
type it captures is proven to reach Gmail's raw message by
test_gmail_attachment_mime.py.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from atlas_brain.mcp import invoicing_server
from atlas_brain.templates.email.invoice import BUSINESS_EMAIL, BUSINESS_NAME, BUSINESS_PHONE

pytestmark = pytest.mark.integration

INVOICE_MIGRATIONS = ("045_invoices.sql", "047_invoice_extra_fields.sql")


class CapturingProvider:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, *, to, subject, body, attachments=None, **_):
        self.sent.append({"to": to, "subject": subject, "body": body, "attachments": attachments})
        return {"id": "m-1"}


@pytest.fixture
async def ledger(monkeypatch, tmp_path):
    """Two real draft invoices in an isolated schema; mail captured; PDFs in tmp."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    import atlas_brain.storage.database as db_module
    from atlas_brain.config import settings
    from atlas_brain.storage.database import DatabasePool
    from atlas_brain.storage.repositories.invoice import get_invoice_repo

    provider = CapturingProvider()
    monkeypatch.setattr("atlas_brain.services.email_provider.get_email_provider", lambda: provider)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_save_path", str(tmp_path))

    schema = f"approve_and_send_{uuid4().hex}"
    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    admin = await asyncpg.connect(database_url)
    pool = None
    previous_pool = db_module._db_pool
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.execute(f'SET search_path TO "{schema}"')
        await admin.execute("CREATE TABLE contacts (id UUID PRIMARY KEY)")
        for name in INVOICE_MIGRATIONS:
            await admin.execute((migrations / name).read_text())

        # The real application pool, pointed at the throwaway schema.
        wrapped = DatabasePool()
        pool = await asyncpg.create_pool(
            database_url, min_size=1, max_size=2, server_settings={"search_path": schema}
        )
        wrapped._pool = pool
        wrapped._initialized = True
        db_module._db_pool = wrapped

        repo = get_invoice_repo()
        invoices = []
        for n in (1, 2):
            invoices.append(await repo.create(
                customer_name="Brookstone Test",
                customer_email="ap@example.test",
                due_date=date(2026, 9, 30),
                issue_date=date(2026, 9, 1),
                line_items=[{"date": "09/01/2026", "description": f"Cleaning visit {n}", "quantity": 1, "unit_price": 135.0}],
                invoice_for="August cleaning",
                source="test",
                source_ref=f"test_approve_and_send_selection_{uuid4()}",
            ))
        yield {"invoices": invoices, "provider": provider, "repo": repo}
    finally:
        db_module._db_pool = previous_pool
        if pool is not None:
            await pool.close()
        try:
            await admin.execute("SET search_path TO public")
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def _call(arguments: dict) -> dict:
    """Invoke the published tool exactly as an MCP client would, and decode its JSON reply."""
    result = await invoicing_server.mcp.call_tool("approve_and_send", arguments)
    content = result[0] if isinstance(result, tuple) else result
    (block,) = content
    return json.loads(block.text)


def _standard_body(inv_num: str) -> str:
    # The body a customer received before this change, written out independently
    # of the tool so a drifted template cannot pass by agreeing with itself.
    return (
        f"Please find attached invoice {inv_num} for August cleaning.\n\n"
        f"Amount Due: $135.00\n"
        f"Due Date: 09/30/2026\n\n"
        f"Make all checks payable to {BUSINESS_NAME}.\n\n"
        f"Thank you for your business!\n\n"
        f"{BUSINESS_NAME}\n"
        f"{BUSINESS_PHONE}\n"
        f"{BUSINESS_EMAIL}"
    )


async def _status(repo, invoice) -> str:
    return (await repo.get_by_id(invoice["id"]))["status"]


# --- the published schema ----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_published_tool_schema_accepts_a_list_or_a_json_string_and_an_optional_note():
    tools = await invoicing_server.mcp.list_tools()
    (tool,) = [t for t in tools if t.name == "approve_and_send"]
    properties = tool.inputSchema["properties"]

    branches = {json.dumps(b, sort_keys=True) for b in properties["invoice_ids"]["anyOf"]}
    assert json.dumps({"type": "string"}, sort_keys=True) in branches
    assert json.dumps({"type": "array", "items": {"type": "string"}}, sort_keys=True) in branches
    assert "invoice_ids" not in tool.inputSchema.get("required", [])
    assert json.dumps({"type": "null"}, sort_keys=True) not in branches, "an explicit null must not be admitted"
    assert "note" in properties and "note" not in tool.inputSchema.get("required", [])
    assert json.dumps({"type": "string"}, sort_keys=True) in {
        json.dumps(b, sort_keys=True) for b in properties["note"]["anyOf"]
    }


# --- selection -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [[], "[]"], ids=["native-list", "json-string"])
async def test_an_explicit_empty_selection_sends_nothing(ledger, empty):
    result = await _call({"invoice_ids": empty, "dry_run": False})

    assert result == {"success": True, "message": "No invoices selected", "processed": 0}
    assert ledger["provider"].sent == [], "an empty selection must not widen to every draft"
    for inv in ledger["invoices"]:
        assert await _status(ledger["repo"], inv) == "draft"


@pytest.mark.asyncio
async def test_an_omitted_selection_still_covers_every_matching_draft(ledger):
    result = await _call({"dry_run": True})

    listed = {d["invoice"]: d["status"] for d in result["details"]}
    assert result["processed"] == 2
    for inv in ledger["invoices"]:
        assert listed[inv["invoice_number"]] == "would_send"
    assert ledger["provider"].sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["native-list", "json-string"])
async def test_both_selection_forms_send_exactly_the_named_invoice_as_a_declared_pdf(ledger, form):
    first, second = ledger["invoices"]
    number = first["invoice_number"]
    selection = [number] if form == "native-list" else json.dumps([number])

    result = await _call({"invoice_ids": selection, "dry_run": False})

    assert result["sent"] == 1 and result["processed"] == 1
    sent = ledger["provider"].sent
    assert [m["to"] for m in sent] == [["ap@example.test"]]
    (attachment,) = sent[0]["attachments"]
    assert attachment["filename"] == f"{number}.pdf"
    assert attachment["mime_type"] == "application/pdf"
    assert await _status(ledger["repo"], first) == "sent"
    assert await _status(ledger["repo"], second) == "draft"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    ["not json", '"INV-2026-0456"', "42", "INV-2026-0456"],
    ids=["invalid-json", "json-scalar-string", "json-number", "bare-number-string"],
)
async def test_a_string_that_is_not_a_json_array_of_strings_is_refused_and_sends_nothing(ledger, bad):
    result = await _call({"invoice_ids": bad, "dry_run": False})

    assert result["success"] is False
    assert ledger["provider"].sent == []
    for inv in ledger["invoices"]:
        assert await _status(ledger["repo"], inv) == "draft"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [None, [1, 2], ["INV-2026-0456", None], 42, {"a": 1}, '{"a": 1}'],
    ids=["explicit-null", "list-of-ints", "list-with-none", "bare-number", "object", "json-object-string"],
)
async def test_the_boundary_rejects_a_selection_outside_the_schema_before_the_tool_runs(ledger, bad):
    # FastMCP pre-parses a string argument that decodes to a JSON object or array
    # before validating it, so a JSON-object string arrives as a dict and is
    # refused here; the JSON-array string form reaches the tool as a list.
    with pytest.raises(ToolError):
        await invoicing_server.mcp.call_tool("approve_and_send", {"invoice_ids": bad, "dry_run": False})

    assert ledger["provider"].sent == []
    for inv in ledger["invoices"]:
        assert await _status(ledger["repo"], inv) == "draft"


# --- the body --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_note_the_body_is_byte_identical_to_the_standard_body(ledger):
    number = ledger["invoices"][0]["invoice_number"]
    await _call({"invoice_ids": [number], "dry_run": False})

    (message,) = ledger["provider"].sent
    assert message["body"] == _standard_body(number)
    assert message["subject"] == f"Invoice {number} - {BUSINESS_NAME} - $135.00"


@pytest.mark.asyncio
@pytest.mark.parametrize("note", ["", "   ", "\n\t"], ids=["empty", "spaces", "whitespace"])
async def test_a_blank_note_leaves_the_body_unchanged(ledger, note):
    number = ledger["invoices"][0]["invoice_number"]
    await _call({"invoice_ids": [number], "dry_run": False, "note": note})

    assert ledger["provider"].sent[0]["body"] == _standard_body(number)


@pytest.mark.asyncio
async def test_a_note_is_placed_above_the_standard_body_and_stripped(ledger):
    number = ledger["invoices"][0]["invoice_number"]
    await _call({"invoice_ids": [number], "dry_run": False, "note": "  Resending per your request.  \n"})

    assert ledger["provider"].sent[0]["body"] == "Resending per your request.\n\n" + _standard_body(number)


@pytest.mark.asyncio
async def test_a_note_outside_the_schema_is_rejected_at_the_boundary(ledger):
    number = ledger["invoices"][0]["invoice_number"]
    with pytest.raises(ToolError):
        await invoicing_server.mcp.call_tool(
            "approve_and_send", {"invoice_ids": [number], "dry_run": False, "note": ["not", "a", "string"]}
        )
    assert ledger["provider"].sent == []
