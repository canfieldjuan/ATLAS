"""Gmail attachment MIME typing, proven through the transport's real entrypoints.

A PDF declared as application/octet-stream is stripped or quarantined by
recipient mail gateways, which is how a batch of invoice attachments left
here intact and arrived unreadable. The filename said .pdf; the declared
type never did.

Two things are pinned. Input that should resolve to a real type does, and it
reaches the raw message Gmail receives -- the tests decode the posted payload
rather than trusting the helper. And a declared type is admitted only when it
is one legal type/subtype pair: the declaration is caller input that lands in
a raw MIME header, so anything malformed refuses the whole send before any
HTTP request, never a repaired or partially built message.
"""

from __future__ import annotations

import base64
import json
import time
from email import message_from_bytes

import httpx
import pytest

from atlas_brain.tools.gmail import (
    GmailDraftCreateError,
    GmailSendError,
    GmailTransport,
    _attachment_type,
)

PDF_BYTES = b"%PDF-1.4\n%fake invoice body\n%%EOF\n"
PDF_B64 = base64.b64encode(PDF_BYTES).decode("ascii")


# --- the resolution rule, at the helper -----------------------------------


def test_explicit_mime_type_wins_over_the_extension():
    assert _attachment_type({"mime_type": "application/pdf"}, "invoice.bin") == (
        "application",
        "pdf",
    )


def test_pdf_extension_is_inferred_when_no_type_is_declared():
    assert _attachment_type({}, "INV-2026-Aug-0456.pdf") == ("application", "pdf")


@pytest.mark.parametrize("filename", ["invoice", "invoice.unknownext"])
def test_unresolvable_filename_falls_back_to_octet_stream(filename):
    assert _attachment_type({}, filename) == ("application", "octet-stream")


@pytest.mark.parametrize("declared", ["", None])
def test_empty_declared_type_defers_to_the_extension(declared):
    assert _attachment_type({"mime_type": declared}, "invoice.pdf") == (
        "application",
        "pdf",
    )


# The admitted class is RFC 6838 restricted-name "/" restricted-name and nothing
# else. The oracle below is derived from that grammar, not from the helper:
# every legal pair admits (case-folded), every mutation that leaves the grammar
# refuses. A fixture of the reported strings would prove those strings only.
LEGAL_PAIRS = [
    ("application/pdf", ("application", "pdf")),
    ("Application/PDF", ("application", "pdf")),
    ("image/png", ("image", "png")),
    ("text/plain", ("text", "plain")),
    ("application/vnd.ms-excel", ("application", "vnd.ms-excel")),
    ("application/x-custom+json", ("application", "x-custom+json")),
    ("x-atlas/" + "a" * 127, ("x-atlas", "a" * 127)),
]

MALFORMED = [
    "application/pdf\nX-Foo: injected",   # header injection via LF
    "application/pdf\r\nX-Foo: injected", # header injection via CRLF
    "application/pdf; charset=utf-8",     # parameters are not part of the pair
    "text/plain/extra",                   # a second slash
    "application",                        # no subtype
    "application/",                       # empty subtype
    "/pdf",                               # empty type
    " application/pdf",                   # leading whitespace
    "application/pdf ",                   # trailing whitespace
    "application /pdf",                   # inner whitespace
    "application/pdf\x00",                # NUL
    "-application/pdf",                   # name must start alphanumeric
    "application/" + "a" * 128,           # over the 127-character name limit
    "application/p\u00e9df",           # non-ASCII
]


@pytest.mark.parametrize("declared,expected", LEGAL_PAIRS)
def test_every_legal_pair_is_admitted_and_case_folded(declared, expected):
    assert _attachment_type({"mime_type": declared}, "whatever.bin") == expected


@pytest.mark.parametrize("declared", MALFORMED)
def test_every_malformed_declaration_is_refused_not_repaired(declared):
    with pytest.raises(ValueError, match="mime_type is invalid"):
        _attachment_type({"mime_type": declared}, "invoice.pdf")


@pytest.mark.parametrize("declared", [1, b"application/pdf", ["application/pdf"], {"a": 1}])
def test_a_non_string_declaration_is_refused(declared):
    with pytest.raises(ValueError, match="mime_type is invalid"):
        _attachment_type({"mime_type": declared}, "invoice.pdf")


# --- through the real transport --------------------------------------------


def _transport(requests: list[httpx.Request], response_json: dict) -> GmailTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_json)

    transport = GmailTransport()
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return transport


def _attachment_part(raw_b64: str):
    raw = base64.urlsafe_b64decode(raw_b64 + "=" * (-len(raw_b64) % 4))
    message = message_from_bytes(raw)
    parts = [p for p in message.walk() if p.get_content_disposition() == "attachment"]
    assert len(parts) == 1
    return parts[0]


@pytest.mark.asyncio
async def test_send_declares_the_pdf_in_the_raw_message_gmail_receives():
    requests: list[httpx.Request] = []
    transport = _transport(requests, {"id": "m-1", "threadId": "t-1"})
    try:
        await transport.send(
            to=["ap@example.test"],
            subject="Invoice INV-2026-0456",
            body="Please find attached.",
            attachments=[{"filename": "INV-2026-0456.pdf", "mime_type": "application/pdf", "content": PDF_B64}],
        )
    finally:
        await transport.close()

    assert len(requests) == 1 and requests[0].url.path.endswith("/users/me/messages/send")
    part = _attachment_part(json.loads(requests[0].content)["raw"])
    assert part.get_content_type() == "application/pdf"
    assert part.get_filename() == "INV-2026-0456.pdf"
    assert part.get_payload(decode=True) == PDF_BYTES


@pytest.mark.asyncio
async def test_send_infers_the_pdf_from_the_filename_when_nothing_is_declared():
    """The exact defect: filename says PDF, so the delivered part must not say otherwise."""
    requests: list[httpx.Request] = []
    transport = _transport(requests, {"id": "m-1", "threadId": "t-1"})
    try:
        await transport.send(
            to=["ap@example.test"],
            subject="Invoice",
            body="Body",
            attachments=[{"filename": "INV-2026-0456.pdf", "content": PDF_B64}],
        )
    finally:
        await transport.close()

    part = _attachment_part(json.loads(requests[0].content)["raw"])
    assert part.get_content_type() == "application/pdf"
    assert part.get_content_type() != "application/octet-stream"


@pytest.mark.asyncio
async def test_create_draft_declares_the_pdf_in_the_raw_message_gmail_receives():
    requests: list[httpx.Request] = []
    transport = _transport(requests, {"id": "d-1", "message": {"id": "m-1", "threadId": "t-1"}})
    try:
        await transport.create_draft(
            to=["ap@example.test"],
            subject="Invoice",
            body="Body",
            attachments=[{"filename": "INV-2026-0456.pdf", "mime_type": "application/pdf", "content": PDF_B64}],
        )
    finally:
        await transport.close()

    assert len(requests) == 1 and requests[0].url.path.endswith("/users/me/drafts")
    part = _attachment_part(json.loads(requests[0].content)["message"]["raw"])
    assert part.get_content_type() == "application/pdf"
    assert part.get_payload(decode=True) == PDF_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", MALFORMED)
async def test_send_refuses_a_malformed_declaration_before_any_request(declared):
    requests: list[httpx.Request] = []
    transport = _transport(requests, {"id": "m-1", "threadId": "t-1"})
    try:
        with pytest.raises(GmailSendError) as excinfo:
            await transport.send(
                to=["ap@example.test"],
                subject="Invoice",
                body="Body",
                attachments=[{"filename": "INV.pdf", "mime_type": declared, "content": PDF_B64}],
            )
    finally:
        await transport.close()

    assert excinfo.value.definitely_not_sent is True
    assert requests == [], "a refused declaration must never reach Gmail"


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", MALFORMED)
async def test_create_draft_refuses_a_malformed_declaration_before_any_request(declared):
    requests: list[httpx.Request] = []
    transport = _transport(requests, {"id": "d-1", "message": {"id": "m-1", "threadId": "t-1"}})
    try:
        with pytest.raises(GmailDraftCreateError) as excinfo:
            await transport.create_draft(
                to=["ap@example.test"],
                subject="Invoice",
                body="Body",
                attachments=[{"filename": "INV.pdf", "mime_type": declared, "content": PDF_B64}],
            )
    finally:
        await transport.close()

    assert excinfo.value.definitely_not_created is True
    assert requests == []
