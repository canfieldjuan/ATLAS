"""approve_and_send: which invoices a call selects, and what the mail says.

Widening `invoice_ids` to accept a native list made `[]` a valid argument. An
explicit selection -- however empty -- must select exactly those invoices; only
an omitted argument means "every invoice matching status_filter". Getting that
wrong mails up to 200 drafts to customers on a call that meant to send none.

These tests run the real tool against the real invoice repository on Postgres
and the real PDF renderer. The only substitution is the outbound email port,
replaced by a capturing provider so nothing leaves the machine; the attachment
type it captures is proven to reach Gmail's raw message by
test_gmail_attachment_mime.py.
"""

from __future__ import annotations

import json
from datetime import date
from uuid import uuid4

import pytest

from atlas_brain.mcp.invoicing_server import approve_and_send
from atlas_brain.templates.email.invoice import BUSINESS_EMAIL, BUSINESS_NAME, BUSINESS_PHONE


class CapturingProvider:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, *, to, subject, body, attachments=None, **_):
        self.sent.append({"to": to, "subject": subject, "body": body, "attachments": attachments})
        return {"id": "m-1"}


@pytest.fixture
async def ledger(monkeypatch, tmp_path):
    """Two real draft invoices, voided again afterwards; mail captured, PDFs in tmp."""
    from atlas_brain.config import settings
    from atlas_brain.storage.database import close_database, get_db_pool, init_database
    from atlas_brain.storage.repositories.invoice import get_invoice_repo

    provider = CapturingProvider()
    monkeypatch.setattr("atlas_brain.services.email_provider.get_email_provider", lambda: provider)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_save_path", str(tmp_path))

    await init_database()
    pool = get_db_pool()
    repo = get_invoice_repo()
    source_ref = f"test_approve_and_send_selection_{uuid4()}"
    try:
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
                source_ref=source_ref,
            ))
        yield {"invoices": invoices, "provider": provider, "repo": repo}
    finally:
        try:
            await pool.execute(
                "UPDATE invoices SET status = 'void', void_reason = 'test cleanup' WHERE source_ref = $1",
                source_ref,
            )
        finally:
            await close_database()


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


# --- selection ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [[], "[]"], ids=["native-list", "json-string"])
async def test_an_explicit_empty_selection_sends_nothing(ledger, empty):
    result = json.loads(await approve_and_send(invoice_ids=empty, dry_run=False))

    assert result == {"success": True, "message": "No invoices selected", "processed": 0}
    assert ledger["provider"].sent == [], "an empty selection must not widen to every draft"
    for inv in ledger["invoices"]:
        assert await _status(ledger["repo"], inv) == "draft"


@pytest.mark.asyncio
async def test_an_omitted_selection_still_covers_every_matching_draft(ledger):
    result = json.loads(await approve_and_send(dry_run=True))

    listed = {d["invoice"]: d["status"] for d in result["details"]}
    for inv in ledger["invoices"]:
        assert listed[inv["invoice_number"]] == "would_send"
    assert ledger["provider"].sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["native-list", "json-string"])
async def test_both_selection_forms_send_exactly_the_named_invoice_as_a_declared_pdf(ledger, form):
    first, second = ledger["invoices"]
    number = first["invoice_number"]
    selection = [number] if form == "native-list" else json.dumps([number])

    result = json.loads(await approve_and_send(invoice_ids=selection, dry_run=False))

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
    ["not json", '"INV-2026-0456"', '{"a": 1}', "42", [1, 2], ["INV-2026-0456", None], "INV-2026-0456"],
    ids=["invalid-json", "json-scalar-string", "json-object", "json-number", "list-of-ints", "list-with-none", "bare-number-string"],
)
async def test_a_selection_that_is_not_a_list_of_strings_is_refused_and_sends_nothing(ledger, bad):
    result = json.loads(await approve_and_send(invoice_ids=bad, dry_run=False))

    assert result["success"] is False
    assert ledger["provider"].sent == []
    for inv in ledger["invoices"]:
        assert await _status(ledger["repo"], inv) == "draft"


# --- the body ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_note_the_body_is_byte_identical_to_the_standard_body(ledger):
    number = ledger["invoices"][0]["invoice_number"]
    await approve_and_send(invoice_ids=[number], dry_run=False)

    (message,) = ledger["provider"].sent
    assert message["body"] == _standard_body(number)
    assert message["subject"] == f"Invoice {number} - {BUSINESS_NAME} - $135.00"


@pytest.mark.asyncio
@pytest.mark.parametrize("note", ["", "   ", "\n\t"], ids=["empty", "spaces", "whitespace"])
async def test_a_blank_note_leaves_the_body_unchanged(ledger, note):
    number = ledger["invoices"][0]["invoice_number"]
    await approve_and_send(invoice_ids=[number], dry_run=False, note=note)

    assert ledger["provider"].sent[0]["body"] == _standard_body(number)


@pytest.mark.asyncio
async def test_a_note_is_placed_above_the_standard_body_and_stripped(ledger):
    number = ledger["invoices"][0]["invoice_number"]
    await approve_and_send(invoice_ids=[number], dry_run=False, note="  Resending per your request.  \n")

    assert ledger["provider"].sent[0]["body"] == "Resending per your request.\n\n" + _standard_body(number)
