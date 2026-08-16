"""
Gmail send transport.

Sends email via the Gmail API using OAuth2 credentials
from GoogleTokenStore. Used as an alternative to Resend
when gmail_send_enabled is True.
"""

import base64
import logging
import re
import time
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Mapping

import httpx

from ..services.google_oauth import get_google_token_store

logger = logging.getLogger("atlas.tools.gmail")

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
TOKEN_URL = "https://oauth2.googleapis.com/token"
_MAX_GMAIL_DRAFT_LOOKUP_PAGES = 20
_MAX_GMAIL_SENT_LOOKUP_PAGES = 20
_MAX_GMAIL_RFC_MESSAGE_ID_LENGTH = 320
_MAX_GMAIL_EXTRA_HEADER_VALUE_LENGTH = 998
_EXTRA_HEADER_NAME = re.compile(r"^(?:Message-ID|X-[A-Za-z0-9-]{1,72})$")
_PROTECTED_HEADER_NAMES = frozenset(
    {
        "bcc",
        "cc",
        "from",
        "reply-to",
        "subject",
        "to",
    }
)


class GmailDraftLookupError(RuntimeError):
    """A Gmail draft lookup cannot safely name one external draft."""


class GmailSentMessageLookupError(RuntimeError):
    """A Gmail Sent lookup cannot safely name one external message."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class GmailDraftCreateError(RuntimeError):
    """A Gmail draft create failed with known or uncertain delivery state."""

    def __init__(self, message: str, *, definitely_not_created: bool) -> None:
        super().__init__(message)
        self.definitely_not_created = definitely_not_created


class GmailSendError(RuntimeError):
    """A Gmail send failed with known or uncertain acceptance state."""

    def __init__(self, message: str, *, definitely_not_sent: bool) -> None:
        super().__init__(message)
        self.definitely_not_sent = definitely_not_sent


def _extra_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Accept the narrow immutable headers a server-side draft may add.

    Gmail's raw-message API accepts arbitrary message headers.  This transport
    is shared, so only Message-ID and namespaced X- headers are admitted here;
    recipients and normal mail headers remain explicit method arguments.  A
    newline would permit header injection into the raw MIME message.
    """

    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise ValueError("Gmail draft headers must be a mapping")
    result: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not _EXTRA_HEADER_NAME.fullmatch(name):
            raise ValueError("Gmail draft header name is invalid")
        if name.casefold() in _PROTECTED_HEADER_NAMES:
            raise ValueError("Gmail draft header name is invalid")
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _MAX_GMAIL_EXTRA_HEADER_VALUE_LENGTH
            or "\r" in value
            or "\n" in value
            or "\x00" in value
        ):
            raise ValueError("Gmail draft header value is invalid")
        result[name] = value
    return result


def _rfc_message_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_GMAIL_RFC_MESSAGE_ID_LENGTH
        or len(value) < 5
        or "\r" in value
        or "\n" in value
        or "\x00" in value
        or not value.startswith("<")
        or not value.endswith(">")
        or value.count("@") != 1
        or any(char.isspace() for char in value)
    ):
        raise GmailDraftLookupError("Gmail draft RFC Message-ID is invalid")
    return value


def _gmail_message_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise GmailSentMessageLookupError(
            f"Gmail sent-message {field} is invalid"
        )
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or "\r" in normalized
        or "\n" in normalized
        or "\x00" in normalized
    ):
        raise GmailSentMessageLookupError(
            f"Gmail sent-message {field} is invalid"
        )
    return normalized


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
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        references: str | None = None,
        headers: Mapping[str, str] | None = None,
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
            thread_id: Gmail thread ID for threading replies.
            in_reply_to: Message-ID of the email being replied to.
            references: Message-ID references for threading.
            headers: Server-owned immutable Message-ID/X-* headers.

        Returns:
            Dict with "id" (Gmail message ID) and "threadId".
        """
        try:
            validated_headers = _extra_headers(headers)
        except ValueError as exc:
            raise GmailSendError(
                "Gmail send headers are invalid", definitely_not_sent=True
            ) from exc

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
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if references:
            msg["References"] = references
        for name, value in validated_headers.items():
            msg[name] = value

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
        try:
            token = await self._get_access_token()
        except Exception as exc:
            raise GmailSendError(
                "Gmail send authentication failed", definitely_not_sent=True
            ) from exc
        client = await self._ensure_client()

        payload = {"raw": raw_b64}
        if thread_id:
            payload["threadId"] = thread_id

        try:
            response = await client.post(
                f"{GMAIL_API_BASE}/users/me/messages/send",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        except Exception as exc:
            # The request can fail after Gmail accepted the raw bytes.  A caller
            # that owns a stable Message-ID must reconcile Sent mail instead of
            # issuing another send.
            raise GmailSendError(
                "Gmail send outcome is unknown", definitely_not_sent=False
            ) from exc

        if response.status_code == 403:
            raise GmailSendError(
                "Gmail send permission denied. Re-run setup with gmail.modify scope: "
                "python scripts/setup_google_oauth.py",
                definitely_not_sent=True,
            )
        try:
            response.raise_for_status()
        except Exception as exc:
            status_code = getattr(response, "status_code", 0)
            raise GmailSendError(
                "Gmail send was rejected"
                if 400 <= status_code < 500
                else "Gmail send outcome is unknown",
                definitely_not_sent=400 <= status_code < 500,
            ) from exc

        try:
            result = response.json()
        except Exception as exc:
            raise GmailSendError(
                "Gmail send outcome is unknown", definitely_not_sent=False
            ) from exc
        logger.info(
            "Email sent via Gmail: id=%s, to=%s, subject=%s",
            result.get("id"),
            to,
            subject[:50],
        )
        return result

    async def create_draft(
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
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a Gmail draft (NOT sent) under the authenticated account.

        Same MIME shape as send() including base64 attachments. Returns the
        Gmail draft resource: {"id": draft_id, "message": {"id", "threadId"}}.
        The user can review and send the draft from their Gmail UI.
        """
        try:
            validated_headers = _extra_headers(headers)
        except ValueError as exc:
            raise GmailDraftCreateError(
                "Gmail draft headers are invalid", definitely_not_created=True
            ) from exc

        if attachments:
            msg = MIMEMultipart("mixed")
            if html:
                msg.attach(MIMEText(html, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))
        else:
            msg = MIMEText(html, "html") if html else MIMEText(body, "plain")

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
        for name, value in validated_headers.items():
            msg[name] = value

        if attachments:
            for att in attachments:
                filename = att.get("filename", "attachment")
                content_b64 = att.get("content", "")
                content_bytes = base64.b64decode(content_b64)
                part = MIMEBase("application", "octet-stream")
                part.set_payload(content_bytes)
                part.add_header("Content-Disposition", "attachment", filename=filename)
                part.add_header("Content-Transfer-Encoding", "base64")
                part.set_payload(base64.b64encode(content_bytes).decode("ascii"))
                msg.attach(part)

        raw_b64 = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")

        try:
            token = await self._get_access_token()
        except Exception as exc:
            raise GmailDraftCreateError(
                "Gmail draft authentication failed", definitely_not_created=True
            ) from exc
        client = await self._ensure_client()
        try:
            response = await client.post(
                f"{GMAIL_API_BASE}/users/me/drafts",
                json={"message": {"raw": raw_b64}},
                headers={"Authorization": f"Bearer {token}"},
            )
        except Exception:
            # A transport exception can happen after Gmail accepted the bytes.
            # Preserve it as uncertain so the caller reconciles by Message-ID
            # rather than submitting another create request.
            raise

        if response.status_code == 403:
            raise GmailDraftCreateError(
                "Gmail draft permission denied. Re-run setup with gmail.modify scope: "
                "python scripts/setup_google_oauth.py",
                definitely_not_created=True,
            )
        try:
            response.raise_for_status()
        except Exception as exc:
            status_code = getattr(response, "status_code", 0)
            raise GmailDraftCreateError(
                "Gmail draft creation was rejected"
                if 400 <= status_code < 500
                else "Gmail draft creation outcome is unknown",
                definitely_not_created=400 <= status_code < 500,
            ) from exc

        result = response.json()
        logger.info(
            "Gmail draft created: id=%s, to=%s, subject=%s",
            result.get("id"),
            to,
            subject[:50],
        )
        return result

    async def find_draft_by_rfc_message_id(
        self,
        rfc_message_id: str,
    ) -> dict[str, Any] | None:
        """Return the one Gmail draft with a stable RFC Message-ID, if present.

        Gmail's documented ``users.drafts.list`` search supports
        ``rfc822msgid:<...>`` and returns the draft resource's own id plus its
        message/thread ids.  Searching is intentionally read-only: a caller
        recovering an uncertain create may inspect this identity before deciding
        whether another draft create would be safe.
        """

        message_id = _rfc_message_id(rfc_message_id)
        token = await self._get_access_token()
        client = await self._ensure_client()
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        matches: list[dict[str, Any]] = []

        for _ in range(_MAX_GMAIL_DRAFT_LOOKUP_PAGES):
            params: dict[str, Any] = {
                "q": f"rfc822msgid:{message_id}",
                "maxResults": 100,
            }
            if page_token is not None:
                params["pageToken"] = page_token
            response = await client.get(
                f"{GMAIL_API_BASE}/users/me/drafts",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == 403:
                raise GmailDraftLookupError(
                    "Gmail draft lookup permission denied. Re-run setup with gmail.modify scope: "
                    "python scripts/setup_google_oauth.py"
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise GmailDraftLookupError("Gmail draft lookup response is invalid")
            drafts = payload.get("drafts", [])
            if not isinstance(drafts, list):
                raise GmailDraftLookupError("Gmail draft lookup response is invalid")
            for draft in drafts:
                if not isinstance(draft, dict):
                    raise GmailDraftLookupError("Gmail draft lookup response is invalid")
                draft_id = draft.get("id")
                message = draft.get("message")
                if (
                    not isinstance(draft_id, str)
                    or not draft_id.strip()
                    or not isinstance(message, dict)
                    or not isinstance(message.get("id"), str)
                    or not message["id"].strip()
                    or not isinstance(message.get("threadId"), str)
                    or not message["threadId"].strip()
                ):
                    raise GmailDraftLookupError("Gmail draft lookup response is invalid")
                matches.append(
                    {
                        "id": draft_id.strip(),
                        "message": {
                            "id": message["id"].strip(),
                            "threadId": message["threadId"].strip(),
                        },
                    }
                )
                if len(matches) > 1:
                    raise GmailDraftLookupError(
                        "Gmail draft lookup found multiple matching drafts"
                    )

            next_page = payload.get("nextPageToken")
            if next_page is None:
                return next(iter(matches), None)
            if not isinstance(next_page, str) or not next_page:
                raise GmailDraftLookupError("Gmail draft lookup response is invalid")
            if next_page in seen_page_tokens:
                raise GmailDraftLookupError("Gmail draft lookup pagination is invalid")
            seen_page_tokens.add(next_page)
            page_token = next_page

        raise GmailDraftLookupError("Gmail draft lookup exceeded its page bound")

    async def find_sent_message_by_rfc_message_id(
        self,
        rfc_message_id: str,
    ) -> dict[str, Any] | None:
        """Return one metadata-verified Sent message for a stable Message-ID.

        The Gmail list API returns only resource identifiers, so every candidate
        is fetched with ``format=metadata`` before this transport reports it.
        The caller remains responsible for comparing its business-specific
        namespaced headers; this generic transport only establishes one bounded,
        read-only mailbox result that has Gmail's ``SENT`` label.
        """

        try:
            message_id = _rfc_message_id(rfc_message_id)
        except GmailDraftLookupError as exc:
            raise GmailSentMessageLookupError(
                "Gmail sent-message RFC Message-ID is invalid"
            ) from exc
        token = await self._get_access_token()
        client = await self._ensure_client()
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        matches: list[dict[str, Any]] = []

        for _ in range(_MAX_GMAIL_SENT_LOOKUP_PAGES):
            params: dict[str, Any] = {
                "q": f"rfc822msgid:{message_id}",
                "labelIds": "SENT",
                "maxResults": 100,
            }
            if page_token is not None:
                params["pageToken"] = page_token
            response = await client.get(
                f"{GMAIL_API_BASE}/users/me/messages",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == 403:
                raise GmailSentMessageLookupError(
                    "Gmail sent-message lookup permission denied. Re-run setup with gmail.modify scope: "
                    "python scripts/setup_google_oauth.py",
                    retryable=True,
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise GmailSentMessageLookupError(
                    "Gmail sent-message lookup response is invalid"
                )
            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                raise GmailSentMessageLookupError(
                    "Gmail sent-message lookup response is invalid"
                )
            for message in messages:
                if not isinstance(message, dict):
                    raise GmailSentMessageLookupError(
                        "Gmail sent-message lookup response is invalid"
                    )
                candidate_id = _gmail_message_identifier(
                    message.get("id"), "id"
                )
                candidate_thread_id = _gmail_message_identifier(
                    message.get("threadId"), "thread ID"
                )
                metadata = await self._get_sent_message_metadata(
                    client=client,
                    token=token,
                    message_id=candidate_id,
                )
                if metadata["id"] != candidate_id or metadata["threadId"] != candidate_thread_id:
                    raise GmailSentMessageLookupError(
                        "Gmail sent-message lookup identity is inconsistent"
                    )
                matches.append(metadata)
                if len(matches) > 1:
                    raise GmailSentMessageLookupError(
                        "Gmail sent-message lookup found multiple matching messages"
                    )

            next_page = payload.get("nextPageToken")
            if next_page is None:
                return next(iter(matches), None)
            if not isinstance(next_page, str) or not next_page:
                raise GmailSentMessageLookupError(
                    "Gmail sent-message lookup pagination is invalid"
                )
            if next_page in seen_page_tokens:
                raise GmailSentMessageLookupError(
                    "Gmail sent-message lookup pagination is invalid"
                )
            seen_page_tokens.add(next_page)
            page_token = next_page

        raise GmailSentMessageLookupError(
            "Gmail sent-message lookup exceeded its page bound"
        )

    @staticmethod
    async def _get_sent_message_metadata(
        *,
        client: httpx.AsyncClient,
        token: str,
        message_id: str,
    ) -> dict[str, Any]:
        response = await client.get(
            f"{GMAIL_API_BASE}/users/me/messages/{message_id}",
            params=[
                ("format", "metadata"),
                ("metadataHeaders", "Message-ID"),
                ("metadataHeaders", "To"),
                ("metadataHeaders", "X-Atlas-Commercial-Billing-Approval"),
                ("metadataHeaders", "X-Atlas-Commercial-Billing-Invoice"),
                ("metadataHeaders", "X-Atlas-EOM-Payment-Receipt"),
            ],
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code == 403:
            raise GmailSentMessageLookupError(
                "Gmail sent-message lookup permission denied. Re-run setup with gmail.modify scope: "
                "python scripts/setup_google_oauth.py",
                retryable=True,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise GmailSentMessageLookupError("Gmail sent-message metadata is invalid")
        normalized_id = _gmail_message_identifier(payload.get("id"), "id")
        normalized_thread_id = _gmail_message_identifier(
            payload.get("threadId"), "thread ID"
        )
        labels = payload.get("labelIds")
        if (
            not isinstance(labels, list)
            or not all(isinstance(label, str) and label for label in labels)
            or "SENT" not in labels
        ):
            raise GmailSentMessageLookupError(
                "Gmail sent-message metadata is not in Sent mail"
            )
        internal_date = payload.get("internalDate")
        if not isinstance(internal_date, str) or not internal_date:
            raise GmailSentMessageLookupError(
                "Gmail sent-message metadata is missing its timestamp"
            )
        message_payload = payload.get("payload")
        if not isinstance(message_payload, dict):
            raise GmailSentMessageLookupError("Gmail sent-message metadata is invalid")
        headers = message_payload.get("headers")
        if not isinstance(headers, list):
            raise GmailSentMessageLookupError("Gmail sent-message metadata is invalid")
        normalized_headers: list[dict[str, str]] = []
        for header in headers:
            if not isinstance(header, dict):
                raise GmailSentMessageLookupError(
                    "Gmail sent-message metadata is invalid"
                )
            name = header.get("name")
            value = header.get("value")
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 128
                or "\r" in name
                or "\n" in name
                or "\x00" in name
                or not isinstance(value, str)
                or len(value) > _MAX_GMAIL_EXTRA_HEADER_VALUE_LENGTH
                or "\r" in value
                or "\n" in value
                or "\x00" in value
            ):
                raise GmailSentMessageLookupError(
                    "Gmail sent-message metadata is invalid"
                )
            normalized_headers.append({"name": name, "value": value})
        return {
            "id": normalized_id,
            "threadId": normalized_thread_id,
            "labelIds": list(labels),
            "internalDate": internal_date,
            "headers": normalized_headers,
        }

    async def modify_thread(
        self,
        thread_id: str,
        add_labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add and/or remove labels on a Gmail thread.

        Common label IDs: INBOX, UNREAD, IMPORTANT, STARRED, SENT, TRASH, SPAM.
        To mark a thread as read, remove the UNREAD label.
        Requires gmail.modify scope (or broader).
        """
        body: dict[str, list[str]] = {}
        if add_labels:
            body["addLabelIds"] = list(add_labels)
        if remove_labels:
            body["removeLabelIds"] = list(remove_labels)
        if not body:
            raise ValueError("modify_thread: provide add_labels and/or remove_labels")

        token = await self._get_access_token()
        client = await self._ensure_client()
        response = await client.post(
            f"{GMAIL_API_BASE}/users/me/threads/{thread_id}/modify",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )

        if response.status_code == 403:
            raise RuntimeError(
                "Gmail modify permission denied. Re-run setup with gmail.modify scope: "
                "python scripts/setup_google_oauth.py"
            )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "Gmail thread modified: id=%s add=%s remove=%s",
            thread_id, add_labels, remove_labels,
        )
        return result

    async def get_sent_message_id(self, gmail_message_id: str) -> str | None:
        """Fetch RFC 2822 Message-ID of a sent Gmail message."""
        token = await self._get_access_token()
        client = await self._ensure_client()
        resp = await client.get(
            f"{GMAIL_API_BASE}/users/me/messages/{gmail_message_id}",
            headers={"Authorization": f"Bearer {token}"},
            params={"format": "metadata", "metadataHeaders": "Message-ID"},
        )
        resp.raise_for_status()
        for header in resp.json().get("payload", {}).get("headers", []):
            if header.get("name", "").lower() == "message-id":
                return header.get("value", "").strip()
        return None


# Module-level singleton
_transport: GmailTransport | None = None


def get_gmail_transport() -> GmailTransport:
    """Get or create the Gmail transport singleton."""
    global _transport
    if _transport is None:
        _transport = GmailTransport()
    return _transport
