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
    GmailSendInputError,
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


@pytest.mark.parametrize("declared", ["application/octet-stream", "Application/Octet-Stream"])
def test_an_explicit_octet_stream_declaration_does_not_defeat_pdf_inference(declared):
    """The generic default carries no information, so the filename still decides."""
    assert _attachment_type({"mime_type": declared}, "invoice.pdf") == ("application", "pdf")
    assert _attachment_type({"mime_type": declared}, "blob.unknownext") == ("application", "octet-stream")


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


# --- through the production entrypoints ------------------------------------------
#
# The composite provider and the email tool fall back to Resend when Gmail
# fails. A refusal of the caller's input is not a Gmail failure: Resend would be
# handed the same input. These tests drive the real CompositeEmailProvider and
# the real EmailTool with fakes only at the two external edges (the Gmail HTTP
# transport and the Resend HTTP client) and prove that a refused declaration
# never reaches either edge. (Invalid caller headers are the same class and
# raise the same input error at the transport; they are a direct-transport
# feature the composite does not forward, so they are proven there.)


class _RecordingResend:
    """Stand-in for the Resend provider at the composite's port."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "resend-1", "transport": "resend"}


class _Credentialed:
    def get_credentials(self, _name):
        return "cred"


@pytest.fixture
def gmail_edge(monkeypatch):
    """The real Gmail transport singleton, with its HTTP edge replaced and restored."""
    from atlas_brain.services import google_oauth
    from atlas_brain.tools import gmail as gmail_mod

    monkeypatch.setattr(google_oauth, "get_google_token_store", lambda: _Credentialed())
    transport = gmail_mod.get_gmail_transport()
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "m-1", "threadId": "t-1"})

    saved = (transport._client, transport._access_token, transport._token_expires)
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    try:
        yield requests
    finally:
        transport._client, transport._access_token, transport._token_expires = saved


@pytest.mark.asyncio
async def test_composite_provider_refuses_a_malformed_declaration_before_choosing_a_provider(gmail_edge):
    from atlas_brain.services.email_provider import CompositeEmailProvider

    composite = CompositeEmailProvider()
    resend = _RecordingResend()
    composite._resend = resend

    with pytest.raises(GmailSendInputError, match="mime_type is invalid") as excinfo:
        await composite.send(
            to=["ap@example.test"],
            subject="Invoice",
            body="Body",
            attachments=[{"filename": "INV.pdf", "mime_type": "application/pdf\nX: y", "content": PDF_B64}],
        )

    assert excinfo.value.definitely_not_sent is True
    assert gmail_edge == [], "Gmail must not have been asked"
    assert resend.calls == [], "a refused declaration must never fall back to Resend"


@pytest.mark.asyncio
async def test_composite_provider_forced_resend_still_goes_to_resend_only(gmail_edge):
    from atlas_brain.services.email_provider import CompositeEmailProvider

    composite = CompositeEmailProvider()
    resend = _RecordingResend()
    composite._resend = resend

    # A forced Resend send never touches Gmail; Resend receives the attachment
    # as given (its own API validates it). What this slice guarantees is that
    # a Gmail refusal is never *converted* into a Resend send.
    await composite.send(
        to=["ap@example.test"],
        subject="Invoice",
        body="Body",
        provider="resend",
        attachments=[{"filename": "INV.pdf", "mime_type": "application/pdf", "content": PDF_B64}],
    )

    assert len(resend.calls) == 1 and gmail_edge == []


@pytest.mark.asyncio
async def test_composite_provider_delivers_a_declared_pdf_through_the_real_gmail_transport(gmail_edge):
    from atlas_brain.services.email_provider import CompositeEmailProvider

    composite = CompositeEmailProvider()
    resend = _RecordingResend()
    composite._resend = resend

    await composite.send(
        to=["ap@example.test"],
        subject="Invoice",
        body="Body",
        attachments=[{"filename": "INV-2026-0456.pdf", "mime_type": "application/pdf", "content": PDF_B64}],
    )

    assert resend.calls == []
    assert len(gmail_edge) == 1
    part = _attachment_part(json.loads(gmail_edge[0].content)["raw"])
    assert part.get_content_type() == "application/pdf"


class _RecordingResendClient:
    """Stand-in for httpx.AsyncClient -- the email tool's external Resend edge."""

    def __init__(self) -> None:
        self.posted: list = []

    async def post(self, url, json=None, headers=None):
        self.posted.append(json)
        return httpx.Response(200, json={"id": "resend-1"}, request=httpx.Request("POST", url))

    async def aclose(self):
        return None


def _email_tool():
    from atlas_brain.tools.email import EmailTool

    tool = EmailTool()
    tool._config = tool._config.model_copy(
        update={"enabled": True, "api_key": "re_test_key", "gmail_send_enabled": True}
    )
    tool._client = _RecordingResendClient()
    return tool


@pytest.mark.asyncio
async def test_email_tool_refuses_a_malformed_declaration_without_trying_either_transport(gmail_edge):
    tool = _email_tool()

    result = await tool.execute({
        "action": "send",
        "from_email": "billing@example.test",
        "to": "ap@example.test",
        "subject": "Invoice",
        "body": "Body",
        "attachments": [{"filename": "INV.pdf", "mime_type": "application/pdf; charset=x", "content": PDF_B64}],
    })

    assert result.success is False and result.error == "INVALID_PARAMETER"
    assert gmail_edge == []
    assert tool._client.posted == [], "a refused declaration must never fall back to Resend"


@pytest.mark.asyncio
async def test_email_tool_delivers_a_declared_pdf_through_the_real_gmail_transport(gmail_edge):
    tool = _email_tool()

    result = await tool.execute({
        "action": "send",
        "from_email": "billing@example.test",
        "to": "ap@example.test",
        "subject": "Invoice",
        "body": "Body",
        "attachments": [{"filename": "INV-2026-0456.pdf", "mime_type": "application/pdf", "content": PDF_B64}],
    })

    assert result.success is True, result
    assert tool._client.posted == []
    assert len(gmail_edge) == 1
    part = _attachment_part(json.loads(gmail_edge[0].content)["raw"])
    assert part.get_content_type() == "application/pdf"


# --- class closure: the whole grammar, not the reported strings ----------------------
#
# docs/GUARD_CLASS_CLOSURE.md req 3. The inputs below are GENERATED from the
# RFC 6838 restricted-name grammar and its mutations, crossed with the
# declaration families the port accepts and the container shapes it is handed,
# and every verdict is checked against an oracle written from the contract,
# not against `_attachment_type` itself. Two layers: the oracle anchors
# correctness at the scalar, and the container product proves that wrapping a
# declaration in an attachment list, alone or beside a valid neighbour, never
# changes the verdict of `validate_attachment_types`.

import itertools
import string

from atlas_brain.tools.gmail import validate_attachment_types

_NAME_FIRST = string.ascii_letters + string.digits
_NAME_REST = _NAME_FIRST + "!#$&^_.+-"


def _legal_name_tokens():
    """Names the grammar admits: first character alphanumeric, then up to 126 of the name alphabet."""
    for first, length in itertools.product(_NAME_FIRST[::9], (1, 2, 9, 127)):
        yield (first + (_NAME_REST * 3))[:length]


def _mutations():
    """Ways of leaving the grammar, each applied to a legal pair."""
    yield "append-lf", lambda pair: pair + "\nX-Injected: 1"
    yield "append-crlf", lambda pair: pair + "\r\nX-Injected: 1"
    yield "parameters", lambda pair: pair + "; charset=utf-8"
    yield "second-slash", lambda pair: pair + "/extra"
    yield "no-slash", lambda pair: pair.replace("/", "")
    yield "empty-subtype", lambda pair: pair.split("/")[0] + "/"
    yield "empty-type", lambda pair: "/" + pair.split("/")[1]
    yield "leading-space", lambda pair: " " + pair
    yield "trailing-space", lambda pair: pair + " "
    yield "inner-space", lambda pair: pair.replace("/", " /")
    yield "nul", lambda pair: pair + "\x00"
    yield "non-ascii", lambda pair: pair + "\u00e9"
    yield "bad-first-char", lambda pair: "-" + pair
    yield "over-length", lambda pair: pair.split("/")[0] + "/" + "a" * 128
    yield "not-a-string", lambda pair: pair.encode("ascii")


def _families():
    """How a caller can express the type: the declaration key, or only the filename."""
    yield "declared", "invoice.unknownext"          # the declaration alone decides
    yield "declared-over-pdf-name", "invoice.pdf"   # a legal declaration beats the extension
    yield "declared-over-png-name", "invoice.png"


def _containers(att, valid_neighbour):
    """The shapes validate_attachment_types is handed: alone, wrapped, and mixed."""
    yield "single", [att]
    yield "after-valid", [valid_neighbour, att]
    yield "before-valid", [att, valid_neighbour]


def _expected_verdict(declared, filename):
    """Spec-derived oracle, written from the contract rather than from the code.

    A non-empty declaration is admitted only if it is name "/" name with each
    name one alphanumeric then up to 126 name characters; anything else is
    refused. An admitted octet-stream, an empty declaration, or no declaration
    defers to the filename extension, and an extension that resolves to nothing
    lands on octet-stream.
    """
    def is_name(part):
        return (
            1 <= len(part) <= 127
            and part[0] in _NAME_FIRST
            and all(ch in _NAME_REST for ch in part)
        )

    inferred = {"pdf": ("application", "pdf"), "png": ("image", "png")}.get(
        filename.rsplit(".", 1)[-1], ("application", "octet-stream")
    )
    if declared is None or declared == "":
        return "admit", inferred
    if not isinstance(declared, str) or declared.count("/") != 1:
        return "refuse", None
    maintype, subtype = declared.split("/")
    if not (is_name(maintype) and is_name(subtype)):
        return "refuse", None
    pair = (maintype.lower(), subtype.lower())
    if pair == ("application", "octet-stream"):
        return "admit", inferred
    return "admit", pair


def _verdict(att, filename):
    try:
        return "admit", _attachment_type(att, filename)
    except ValueError:
        return "refuse", None


def _list_verdict(attachments):
    try:
        validate_attachment_types(attachments)
    except ValueError:
        return "refuse"
    return "admit"


def test_every_generated_legal_pair_is_admitted_as_the_oracle_says():
    tokens = list(_legal_name_tokens())
    families = list(_families())
    checked = 0
    for (maintype, subtype), (family, filename) in itertools.product(
        itertools.product(tokens, tokens[::2]), families
    ):
        declared = f"{maintype}/{subtype}"
        att = {"mime_type": declared, "filename": filename, "content": PDF_B64}
        expected = _expected_verdict(declared, filename)
        assert _verdict(att, filename) == expected, (family, declared)
        checked += 1
    assert checked > 100


def test_every_generated_mutation_is_refused_and_containers_do_not_change_the_verdict():
    tokens = list(_legal_name_tokens())
    families = list(_families())
    valid_neighbour = {"mime_type": "application/pdf", "filename": "ok.pdf", "content": PDF_B64}
    refused = 0
    for (maintype, subtype), (mutation, mutate), (family, filename) in itertools.product(
        itertools.product(tokens[::3], tokens[1::5]), _mutations(), families
    ):
        declared = mutate(f"{maintype}/{subtype}")
        att = {"mime_type": declared, "filename": filename, "content": PDF_B64}
        expected = _expected_verdict(declared, filename)
        assert expected[0] == "refuse", (mutation, declared)
        assert _verdict(att, filename) == expected, (mutation, family, declared)
        # representation parity: alone, after a valid neighbour, before one
        for container, attachments in _containers(att, valid_neighbour):
            assert _list_verdict(attachments) == "refuse", (mutation, family, container)
        refused += 1
    assert refused > 100


def test_no_declaration_and_octet_stream_declaration_defer_to_the_filename_across_containers():
    families = list(_families())
    valid_neighbour = {"mime_type": "application/pdf", "filename": "ok.pdf", "content": PDF_B64}
    for (declared, key_present), (family, filename) in itertools.product(
        ((None, True), ("", True), (None, False), ("application/octet-stream", True), ("Application/OCTET-stream", True)),
        families,
    ):
        att = {"filename": filename, "content": PDF_B64}
        if key_present:
            att["mime_type"] = declared
        expected = _expected_verdict(declared, filename)
        assert expected[0] == "admit"
        assert _verdict(att, filename) == expected, (declared, family)
        for container, attachments in _containers(att, valid_neighbour):
            assert _list_verdict(attachments) == "admit", (declared, family, container)
