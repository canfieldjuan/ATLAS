"""Regression tests for Gmail attachment MIME typing.

A PDF declared as application/octet-stream is stripped or quarantined by
recipient mail gateways, which is how a batch of invoice attachments left
here intact and arrived unreadable. The filename said .pdf; the declared
type never did.

These pin both sides of the resolution: input that should resolve to a real
type does, and input that cannot resolve still lands on a safe fallback
rather than an empty or malformed Content-Type.
"""

from __future__ import annotations

from email.mime.base import MIMEBase

import pytest

from atlas_brain.tools.gmail import _attachment_type


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


def test_declared_type_without_a_subtype_does_not_yield_an_empty_subtype():
    assert _attachment_type({"mime_type": "application"}, "invoice.pdf") == (
        "application",
        "octet-stream",
    )


def test_resolved_pair_builds_a_part_that_declares_the_pdf_content_type():
    maintype, subtype = _attachment_type({"mime_type": "application/pdf"}, "x.pdf")
    part = MIMEBase(maintype, subtype)
    assert part.get_content_type() == "application/pdf"


def test_a_pdf_attachment_is_never_built_as_octet_stream():
    """The exact defect: filename says PDF, so the part must not say otherwise."""
    maintype, subtype = _attachment_type({}, "INV-2026-Aug-0456.pdf")
    part = MIMEBase(maintype, subtype)
    assert part.get_content_type() != "application/octet-stream"
