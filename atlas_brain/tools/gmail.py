"""
Gmail send transport.

Sends email via the Gmail API using OAuth2 credentials
from GoogleTokenStore. Used as an alternative to Resend
when gmail_send_enabled is True.
"""

import base64
import logging
import time
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

from ..services.google_oauth import get_google_token_store

logger = logging.getLogger("atlas.tools.gmail")

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
TOKEN_URL = "https://oauth2.googleapis.com/token"


class GmailTransport:
    """Send emails via the Gmail API."""

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._token_expires: float = 0.0
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        store = get_google_token_store()
        creds = store.get_credentials("gmail")
        if not creds:
            raise RuntimeError(
                "Gmail OAuth not configured. "
                "Run: python scripts/setup_google_oauth.py"
            )

        client = await self._ensure_client()
        data = {
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "refresh_token": creds.refresh_token,
            "grant_type": "refresh_token",
        }

        response = await client.post(TOKEN_URL, data=data)
        if response.status_code in (400, 401):
            raise RuntimeError(
                f"Gmail refresh token rejected (HTTP {response.status_code}). "
                "Re-run: python scripts/setup_google_oauth.py"
            )
        response.raise_for_status()
        token_data = response.json()

        self._access_token = token_data["access_token"]
        self._token_expires = time.time() + token_data.get("expires_in", 3600)

        # Auto-persist rotated refresh token
        new_refresh = token_data.get("refresh_token")
        if new_refresh and new_refresh != creds.refresh_token:
            store.persist_refresh_token("gmail", new_refresh)

        return self._access_token

    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        from_email: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        html: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an email via Gmail API.

        Args:
            to: List of recipient addresses.
            subject: Email subject.
            body: Plain text body.
            from_email: Sender (uses Gmail account if None).
            cc: CC addresses.
            bcc: BCC addresses.
            reply_to: Reply-to address.
            attachments: List of {"filename": str, "content": str (base64)}.
            html: Optional HTML body (used instead of plain text if provided).

        Returns:
            Dict with "id" (Gmail message ID) and "threadId".
        """
        # Build MIME message
        if attachments:
            msg = MIMEMultipart("mixed")
            if html:
                msg.attach(MIMEText(html, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))
        else:
            if html:
                msg = MIMEText(html, "html")
            else:
                msg = MIMEText(body, "plain")

        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        if from_email:
            msg["From"] = from_email
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)
        if reply_to:
            msg["Reply-To"] = reply_to

        # Add attachments
        if attachments:
            for att in attachments:
                filename = att.get("filename", "attachment")
                content_b64 = att.get("content", "")
                content_bytes = base64.b64decode(content_b64)

                part = MIMEBase("application", "octet-stream")
                part.set_payload(content_bytes)
                part.add_header(
                    "Content-Disposition", "attachment", filename=filename
                )
                part.add_header("Content-Transfer-Encoding", "base64")
                part.set_payload(base64.b64encode(content_bytes).decode("ascii"))
                msg.attach(part)

        # Base64url encode the message
        raw_bytes = msg.as_bytes()
        raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode("ascii")

        # Send via Gmail API
        token = await self._get_access_token()
        client = await self._ensure_client()

        response = await client.post(
            f"{GMAIL_API_BASE}/users/me/messages/send",
            json={"raw": raw_b64},
            headers={"Authorization": f"Bearer {token}"},
        )

        if response.status_code == 403:
            raise RuntimeError(
                "Gmail send permission denied. Re-run setup with gmail.send scope: "
                "python scripts/setup_google_oauth.py"
            )
        response.raise_for_status()

        result = response.json()
        logger.info(
            "Email sent via Gmail: id=%s, to=%s, subject=%s",
            result.get("id"),
            to,
            subject[:50],
        )
        return result


# Module-level singleton
_transport: GmailTransport | None = None


def get_gmail_transport() -> GmailTransport:
    """Get or create the Gmail transport singleton."""
    global _transport
    if _transport is None:
        _transport = GmailTransport()
    return _transport
