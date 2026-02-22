"""
Email provider abstraction for Atlas.

Provider-agnostic interface for email send + read operations.

Two concrete providers:
  - GmailEmailProvider   — Gmail API (OAuth2); read + send.
  - ResendEmailProvider  — Resend REST API; send-only fallback.

CompositeEmailProvider (default) tries Gmail first for both reading and
sending; falls back to Resend for send when Gmail is unavailable.

Usage:
    from atlas_brain.services.email_provider import get_email_provider

    provider = get_email_provider()
    await provider.send(to=["alice@example.com"], subject="Hi", body="Hello")
    messages = await provider.list_messages("is:unread newer_than:1d")
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("atlas.services.email_provider")


# ---------------------------------------------------------------------------
# GmailEmailProvider
# ---------------------------------------------------------------------------

class GmailEmailProvider:
    """
    Email provider backed by the Gmail API.

    Reuses:
      - GmailClient (atlas_brain/autonomous/tasks/gmail_digest.py) for reading
      - GmailTransport (atlas_brain/tools/gmail.py) for sending
    """

    async def is_available(self) -> bool:
        try:
            from .google_oauth import get_google_token_store

            return get_google_token_store().get_credentials("gmail") is not None
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Send
    # -----------------------------------------------------------------------

    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        cc: Optional[list[str]] = None,
        bcc: Optional[list[str]] = None,
        reply_to: Optional[str] = None,
        html: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
    ) -> dict[str, Any]:
        from ..tools.gmail import get_gmail_transport

        return await get_gmail_transport().send(
            to=to,
            subject=subject,
            body=body,
            from_email=from_email,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            html=html,
            attachments=attachments,
            thread_id=thread_id,
            in_reply_to=in_reply_to,
            references=references,
        )

    # -----------------------------------------------------------------------
    # Read (reuses GmailClient from the gmail_digest autonomous task)
    # -----------------------------------------------------------------------

    async def _client(self):
        from ..autonomous.tasks.gmail_digest import _get_gmail_client

        return await _get_gmail_client()

    async def list_messages(
        self, query: str = "is:unread", max_results: int = 20
    ) -> list[dict[str, Any]]:
        client = await self._client()
        return await client.list_messages(query=query, max_results=max_results)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        client = await self._client()
        return await client.get_message_full(message_id)

    async def get_message_metadata(self, message_id: str) -> dict[str, Any]:
        client = await self._client()
        return await client.get_message_metadata(message_id)

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        """Return a Gmail thread with message list (metadata format)."""
        from ..autonomous.tasks.gmail_digest import GMAIL_API_BASE

        try:
            client = await self._client()
            http = await client._ensure_client()
            headers = await client._get_headers()
            resp = await http.get(
                f"{GMAIL_API_BASE}/users/me/threads/{thread_id}",
                headers=headers,
                params={"format": "metadata"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("get_thread failed for %s: %s", thread_id, exc)
            return {}


# ---------------------------------------------------------------------------
# ResendEmailProvider  (send-only fallback)
# ---------------------------------------------------------------------------

class ResendEmailProvider:
    """
    Send-only email provider backed by Resend API.

    Delegates directly to the existing EmailTool to avoid duplicating the
    Resend request logic and attachment handling.
    """

    async def is_available(self) -> bool:
        try:
            from ..config import settings

            return settings.email.enabled and bool(settings.email.api_key)
        except Exception:
            return False

    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        cc: Optional[list[str]] = None,
        bcc: Optional[list[str]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        from ..tools.email import email_tool

        params: dict[str, Any] = {
            "to": ", ".join(to),
            "subject": subject,
            "body": body,
        }
        if from_email:
            params["from_email"] = from_email
        if cc:
            params["cc"] = ", ".join(cc)
        if bcc:
            params["bcc"] = ", ".join(bcc)
        if reply_to:
            params["reply_to"] = reply_to
        if attachments:
            params["attachments"] = attachments

        result = await email_tool.execute(params)
        if result.success:
            return result.data
        raise RuntimeError(result.message or "Resend send failed")

    # Read methods are not supported by Resend
    async def list_messages(self, **_kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("Resend does not support reading inbox")

    async def get_message(self, message_id: str, **_kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Resend does not support reading messages")

    async def get_thread(self, thread_id: str, **_kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Resend does not support reading threads")


# ---------------------------------------------------------------------------
# CompositeEmailProvider  (default)
# ---------------------------------------------------------------------------

class CompositeEmailProvider:
    """
    Composite email provider: Gmail preferred, Resend fallback for sending.

    - Sending:  tries Gmail first; if Gmail is unavailable or fails, falls
                back to Resend.
    - Reading:  Gmail only.  Raises a clear error when Gmail is not configured.
    """

    def __init__(self) -> None:
        self._gmail = GmailEmailProvider()
        self._resend = ResendEmailProvider()

    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if await self._gmail.is_available():
            try:
                return await self._gmail.send(to=to, subject=subject, body=body, **kwargs)
            except Exception as exc:
                logger.warning("Gmail send failed, trying Resend: %s", exc)
        return await self._resend.send(to=to, subject=subject, body=body, **kwargs)

    async def list_messages(
        self, query: str = "is:unread", max_results: int = 20
    ) -> list[dict[str, Any]]:
        return await self._gmail.list_messages(query=query, max_results=max_results)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        return await self._gmail.get_message(message_id)

    async def get_message_metadata(self, message_id: str) -> dict[str, Any]:
        return await self._gmail.get_message_metadata(message_id)

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        return await self._gmail.get_thread(thread_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_email_provider: Optional[CompositeEmailProvider] = None


def get_email_provider() -> CompositeEmailProvider:
    """Return the global CompositeEmailProvider singleton."""
    global _email_provider
    if _email_provider is None:
        _email_provider = CompositeEmailProvider()
    return _email_provider
