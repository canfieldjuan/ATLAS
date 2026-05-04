from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from extracted_competitive_intelligence._standalone.email_provider import (
    EmailProvider,
    EmailProviderNotConfigured,
    configure_email_provider,
    get_email_provider,
)


class EmailAdapter:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        from_email: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to: str | None = None,
        html: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        references: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "to": to,
            "subject": subject,
            "body": body,
            "from_email": from_email,
            "cc": cc,
            "bcc": bcc,
            "reply_to": reply_to,
            "html": html,
            "attachments": attachments,
            "thread_id": thread_id,
            "in_reply_to": in_reply_to,
            "references": references,
        }
        self.sent.append(payload)
        return {"id": "email-1", "payload": payload}


def teardown_function() -> None:
    configure_email_provider(None)


def test_standalone_email_provider_fails_closed_until_configured() -> None:
    configure_email_provider(None)

    with pytest.raises(EmailProviderNotConfigured):
        get_email_provider()


@pytest.mark.asyncio
async def test_standalone_email_provider_returns_configured_adapter() -> None:
    adapter = EmailAdapter()

    configure_email_provider(adapter)
    result = await get_email_provider().send(
        to=["buyer@example.com"],
        subject="Subscription Confirmed",
        body="Plain text body",
        html="<p>HTML body</p>",
        reply_to="outreach@example.com",
    )

    assert isinstance(adapter, EmailProvider)
    assert result["id"] == "email-1"
    assert adapter.sent[0]["to"] == ["buyer@example.com"]
    assert adapter.sent[0]["html"] == "<p>HTML body</p>"
    assert adapter.sent[0]["reply_to"] == "outreach@example.com"


def test_service_module_uses_standalone_port_without_atlas(monkeypatch) -> None:
    module_name = "extracted_competitive_intelligence.services.email_provider"
    atlas_module_name = "atlas_brain.services.email_provider"
    monkeypatch.setenv("EXTRACTED_COMP_INTEL_STANDALONE", "1")
    sys.modules.pop(module_name, None)
    sys.modules.pop(atlas_module_name, None)

    module = importlib.import_module(module_name)

    try:
        assert module.EmailProvider.__module__ == (
            "extracted_competitive_intelligence._standalone.email_provider"
        )
        with pytest.raises(module.EmailProviderNotConfigured):
            module.get_email_provider()

        adapter = EmailAdapter()
        module.configure_email_provider(adapter)
        assert module.get_email_provider() is adapter
        assert atlas_module_name not in sys.modules
    finally:
        module.configure_email_provider(None)
        sys.modules.pop(module_name, None)

