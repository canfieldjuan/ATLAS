"""
Email provider abstraction for Atlas.

Provider-agnostic interface for email send + read operations.

Three concrete providers:
  - IMAPEmailProvider    — IMAP (stdlib imaplib); works with any mail server.
  - GmailEmailProvider   — Gmail API (OAuth2); read + send. Used as fallback
                           reader when IMAP is not configured.
  - ResendEmailProvider  — Resend REST API; send-only fallback.

CompositeEmailProvider (default):
  - Reading:  IMAPEmailProvider when configured; GmailEmailProvider otherwise.
  - Sending:  Gmail preferred; Resend fallback.

IMAP configuration (env vars):
  ATLAS_EMAIL_IMAP_HOST      imap.gmail.com | outlook.office365.com | …
  ATLAS_EMAIL_IMAP_PORT      993  (SSL)
  ATLAS_EMAIL_IMAP_USERNAME  your@email.com
  ATLAS_EMAIL_IMAP_PASSWORD  app-specific-password
  ATLAS_EMAIL_IMAP_SSL       true
  ATLAS_EMAIL_IMAP_MAILBOX   INBOX

Usage:
    from atlas_brain.services.email_provider import get_email_provider

    provider = get_email_provider()
    await provider.send(to=["alice@example.com"], subject="Hi", body="Hello")
    messages = await provider.list_messages("is:unread newer_than:1d")
"""

import asyncio
import email as _email_stdlib
import imaplib
import logging
import re
from email.header import decode_header as _decode_header
from typing import Any, Optional

logger = logging.getLogger("atlas.services.email_provider")


# ---------------------------------------------------------------------------
# IMAP helpers (stdlib — no extra dependencies)
# ---------------------------------------------------------------------------

def _decode_mime_words(value: str) -> str:
    """Decode RFC 2047 encoded words (=?utf-8?b?…?=) in headers."""
    parts = []
    for raw, charset in _decode_header(value):
        if isinstance(raw, bytes):
            parts.append(raw.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(raw)
    return "".join(parts)


def _imap_search_criteria(query: str) -> str:
    """
    Convert a human-readable search query to IMAP SEARCH criteria string.

    Supports a subset of Gmail-style syntax:
        is:unread          → UNSEEN
        is:read            → SEEN
        is:starred         → FLAGGED
        from:alice@…       → FROM "alice@…"
        to:bob@…           → TO "bob@…"
        subject:foo        → SUBJECT "foo"
        newer_than:Nd      → SINCE <date N days ago>
        older_than:Nd      → BEFORE <date N days ago>
        has:attachment     → (mapped to ALL — IMAP has no native test)

    Multiple tokens are ANDed together.
    Unknown tokens are silently dropped; falls back to ALL.
    """
    from datetime import datetime, timedelta

    criteria: list[str] = []
    remaining = query.strip()

    def _since(days: int) -> str:
        d = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
        return f'SINCE "{d}"'

    def _before(days: int) -> str:
        d = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
        return f'BEFORE "{d}"'

    # Tokenise: split on whitespace but keep quoted strings together
    tokens = re.findall(r'"[^"]*"|\S+', remaining)

    for token in tokens:
        token_lower = token.lower()
        if token_lower == "is:unread":
            criteria.append("UNSEEN")
        elif token_lower == "is:read":
            criteria.append("SEEN")
        elif token_lower == "is:starred":
            criteria.append("FLAGGED")
        elif token_lower.startswith("from:"):
            addr = token[5:].strip('"')
            criteria.append(f'FROM "{addr}"')
        elif token_lower.startswith("to:"):
            addr = token[3:].strip('"')
            criteria.append(f'TO "{addr}"')
        elif token_lower.startswith("subject:"):
            subj = token[8:].strip('"')
            criteria.append(f'SUBJECT "{subj}"')
        elif token_lower.startswith("newer_than:"):
            m = re.match(r"newer_than:(\d+)d", token_lower)
            if m:
                criteria.append(_since(int(m.group(1))))
        elif token_lower.startswith("older_than:"):
            m = re.match(r"older_than:(\d+)d", token_lower)
            if m:
                criteria.append(_before(int(m.group(1))))
        elif token_lower in ("has:attachment", "in:inbox", "in:sent"):
            # Map to ALL — no direct IMAP equivalent for has:attachment
            pass
        # Unknown tokens (label:, OR, etc.) — silently skip
        else:
            logger.debug("IMAP: unsupported search token %r — ignored", token)

    return " ".join(criteria) if criteria else "ALL"


def _parse_envelope(uid: str, raw_headers: bytes) -> dict[str, Any]:
    """Parse raw RFC 822 headers into a metadata dict."""
    msg = _email_stdlib.message_from_bytes(raw_headers)
    return {
        "id": uid,
        "subject": _decode_mime_words(msg.get("Subject", "")),
        "from": _decode_mime_words(msg.get("From", "")),
        "to": _decode_mime_words(msg.get("To", "")),
        "date": msg.get("Date", ""),
        "message_id": msg.get("Message-ID", "").strip(),
        "in_reply_to": msg.get("In-Reply-To", "").strip(),
        "references": msg.get("References", "").strip(),
        "thread_id": msg.get("X-GM-THRID", ""),  # Gmail IMAP extension
        "snippet": "",
        "reply_to": _decode_mime_words(msg.get("Reply-To", "")),
        "has_unsubscribe": msg.get("List-Unsubscribe") is not None,
    }


def _extract_body(msg: _email_stdlib.message.Message) -> str:
    """Walk MIME tree and extract plain-text body."""
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if ctype == "text/plain" and "attachment" not in disp:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode(charset, errors="replace"))
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            body_parts.append(payload.decode(charset, errors="replace"))
    return "\n".join(body_parts)


# ---------------------------------------------------------------------------
# IMAPEmailProvider
# ---------------------------------------------------------------------------

class IMAPEmailProvider:
    """
    Provider-agnostic IMAP reader using Python's stdlib imaplib.

    Works with any IMAP server — Gmail, Outlook, Yahoo, custom.
    All blocking IMAP operations run in a thread executor so the
    async API stays non-blocking.

    Configuration (atlas_brain/config.py → EmailConfig → IMAP fields):
        ATLAS_EMAIL_IMAP_HOST      e.g. imap.gmail.com
        ATLAS_EMAIL_IMAP_PORT      993
        ATLAS_EMAIL_IMAP_USERNAME  your@email.com
        ATLAS_EMAIL_IMAP_PASSWORD  app-specific-password
        ATLAS_EMAIL_IMAP_SSL       true
        ATLAS_EMAIL_IMAP_MAILBOX   INBOX
    """

    def __init__(self) -> None:
        self._host: str = ""
        self._port: int = 993
        self._username: str = ""
        self._password: str = ""
        self._ssl: bool = True
        self._mailbox: str = "INBOX"
        self._loaded = False
        self._cached_conn: imaplib.IMAP4 | None = None

    def _load_config(self) -> None:
        if self._loaded:
            return
        from ..config import settings

        cfg = settings.email
        self._host = cfg.imap_host
        self._port = cfg.imap_port
        self._username = cfg.imap_username
        self._password = cfg.imap_password
        self._ssl = cfg.imap_ssl
        self._mailbox = cfg.imap_mailbox or "INBOX"
        self._loaded = True

        # Validate settings and log clear warnings rather than failing silently
        if self._host:
            if not (1 <= self._port <= 65535):
                logger.warning(
                    "IMAP: invalid port %d (must be 1-65535) — IMAP will fail",
                    self._port,
                )
            if not self._username:
                logger.warning("IMAP: host is set but ATLAS_EMAIL_IMAP_USERNAME is empty")
            if not self._password:
                logger.warning(
                    "IMAP: host is set but ATLAS_EMAIL_IMAP_PASSWORD is empty "
                    "(Gmail: use a 16-char app password from myaccount.google.com/apppasswords)"
                )

    def is_configured(self) -> bool:
        self._load_config()
        return bool(self._host and self._username and self._password)

    def _connect(self) -> imaplib.IMAP4:
        """Open and authenticate a new IMAP connection (blocking)."""
        self._load_config()
        if self._ssl:
            conn = imaplib.IMAP4_SSL(self._host, self._port)
        else:
            conn = imaplib.IMAP4(self._host, self._port)
            conn.starttls()
        conn.login(self._username, self._password)
        return conn

    def _get_conn(self) -> imaplib.IMAP4:
        """Return a cached IMAP connection, reconnecting if stale."""
        if self._cached_conn is not None:
            try:
                self._cached_conn.noop()
                return self._cached_conn
            except Exception:
                try:
                    self._cached_conn.logout()
                except Exception:
                    pass
                self._cached_conn = None
        conn = self._connect()
        self._cached_conn = conn
        return conn

    def _release_conn(self, conn: imaplib.IMAP4, *, discard: bool = False) -> None:
        """Release a connection back to cache (or discard on error)."""
        if discard:
            try:
                conn.logout()
            except Exception:
                pass
            if self._cached_conn is conn:
                self._cached_conn = None
        # Otherwise keep cached for reuse

    # -----------------------------------------------------------------------
    # Async wrappers
    # -----------------------------------------------------------------------

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _list_folders_sync(self) -> list[dict[str, Any]]:
        """List all IMAP folders/mailboxes (blocking)."""
        conn = self._get_conn()
        try:
            status, data = conn.list()
            if status != "OK" or not data:
                return []
            folders: list[dict[str, Any]] = []
            for item in data:
                if item is None:
                    continue
                raw = item.decode() if isinstance(item, bytes) else str(item)
                # IMAP LIST format: (\\Flags) "delimiter" "name"
                match = re.match(
                    r'\(([^)]*)\)\s+"([^"]+)"\s+"?([^"]+)"?', raw
                )
                if match:
                    flags_str, delimiter, name = match.groups()
                    flags = [f.strip() for f in flags_str.split() if f.strip()]
                    folders.append({
                        "name": name,
                        "delimiter": delimiter,
                        "flags": flags,
                        "selectable": "\\Noselect" not in flags,
                    })
            return folders
        except Exception:
            self._release_conn(conn, discard=True)
            raise

    def _list_messages_sync(
        self, query: str, max_results: int, mailbox: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        target = mailbox or self._mailbox
        try:
            conn.select(f'"{target}"', readonly=True)
            criteria = _imap_search_criteria(query)
            status, data = conn.uid("search", None, criteria)  # type: ignore[arg-type]
            if status != "OK" or not data or not data[0]:
                return []
            uids = data[0].decode().split()
            # Most recent first
            uids = list(reversed(uids))[:max_results]
            if not uids:
                return []
            uid_set = ",".join(uids)
            status, fetch_data = conn.uid("fetch", uid_set, "(RFC822.HEADER)")  # type: ignore[arg-type]
            if status != "OK":
                return []
            messages: list[dict[str, Any]] = []
            i = 0
            while i < len(fetch_data):
                item = fetch_data[i]
                if isinstance(item, tuple) and len(item) >= 2:
                    header_bytes = item[1]
                    # Extract UID from the fetch response descriptor
                    descriptor = item[0].decode() if isinstance(item[0], bytes) else str(item[0])
                    uid_match = re.search(r"UID (\d+)", descriptor)
                    uid = uid_match.group(1) if uid_match else str(i)
                    messages.append(_parse_envelope(uid, header_bytes))
                i += 1
            return messages
        except Exception:
            self._release_conn(conn, discard=True)
            raise

    def _get_message_sync(self, uid: str, mailbox: str | None = None) -> dict[str, Any]:
        conn = self._get_conn()
        target = mailbox or self._mailbox
        try:
            conn.select(f'"{target}"', readonly=True)
            status, data = conn.uid("fetch", uid, "(RFC822)")  # type: ignore[arg-type]
            if status != "OK" or not data or not data[0]:
                return {"error": f"Message {uid} not found"}
            item = data[0]
            if not isinstance(item, tuple) or len(item) < 2:
                return {"error": "Unexpected fetch format"}
            msg = _email_stdlib.message_from_bytes(item[1])
            meta = _parse_envelope(uid, item[1])
            meta["body_text"] = _extract_body(msg)
            return meta
        except Exception:
            self._release_conn(conn, discard=True)
            raise

    def _get_message_metadata_sync(self, uid: str, mailbox: str | None = None) -> dict[str, Any]:
        conn = self._get_conn()
        target = mailbox or self._mailbox
        try:
            conn.select(f'"{target}"', readonly=True)
            status, data = conn.uid("fetch", uid, "(RFC822.HEADER)")  # type: ignore[arg-type]
            if status != "OK" or not data or not data[0]:
                return {"error": f"Message {uid} not found"}
            item = data[0]
            if not isinstance(item, tuple) or len(item) < 2:
                return {"error": "Unexpected fetch format"}
            return _parse_envelope(uid, item[1])
        except Exception:
            self._release_conn(conn, discard=True)
            raise

    def _get_thread_sync(self, thread_id: str, mailbox: str | None = None) -> dict[str, Any]:
        """
        Fetch all messages in a thread.

        For Gmail IMAP: uses X-GM-THRID extension (thread_id = numeric string).
        For other servers: searches by References/Message-ID match using the
        thread_id as a message UID seed.
        """
        conn = self._get_conn()
        target = mailbox or self._mailbox
        try:
            conn.select(f'"{target}"', readonly=True)

            # Try Gmail's X-GM-THRID search first
            if re.match(r"^\d+$", thread_id):
                status, data = conn.uid(  # type: ignore[arg-type]
                    "search", "X-GM-THRID", thread_id
                )
                if status == "OK" and data and data[0]:
                    uids = data[0].decode().split()
                    if uids:
                        uid_set = ",".join(uids)
                        status2, fetch_data = conn.uid("fetch", uid_set, "(RFC822.HEADER)")  # type: ignore[arg-type]
                        messages: list[dict[str, Any]] = []
                        if status2 == "OK":
                            i = 0
                            while i < len(fetch_data):
                                item = fetch_data[i]
                                if isinstance(item, tuple) and len(item) >= 2:
                                    descriptor = item[0].decode() if isinstance(item[0], bytes) else str(item[0])
                                    uid_match = re.search(r"UID (\d+)", descriptor)
                                    uid = uid_match.group(1) if uid_match else str(i)
                                    messages.append(_parse_envelope(uid, item[1]))
                                i += 1
                        return {"thread_id": thread_id, "messages": messages}

            # Fallback: treat thread_id as a UID and expand via References
            root = self._get_message_sync(thread_id)
            return {"thread_id": thread_id, "messages": [root]}
        except imaplib.IMAP4.error:
            # X-GM-THRID not supported — fall back to single message
            self._release_conn(conn, discard=True)
            root = self._get_message_sync(thread_id)
            return {"thread_id": thread_id, "messages": [root]}
        except Exception:
            self._release_conn(conn, discard=True)
            raise

    # -----------------------------------------------------------------------
    # Public async interface
    # -----------------------------------------------------------------------

    async def list_folders(self) -> list[dict[str, Any]]:
        """List all IMAP folders/mailboxes."""
        return await self._run(self._list_folders_sync)

    async def list_messages(
        self, query: str = "is:unread", max_results: int = 20,
        mailbox: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._run(self._list_messages_sync, query, min(max_results, 200), mailbox)

    async def get_message(self, message_id: str, mailbox: str | None = None) -> dict[str, Any]:
        return await self._run(self._get_message_sync, message_id, mailbox)

    async def get_message_metadata(self, message_id: str, mailbox: str | None = None) -> dict[str, Any]:
        return await self._run(self._get_message_metadata_sync, message_id, mailbox)

    async def get_thread(self, thread_id: str, mailbox: str | None = None) -> dict[str, Any]:
        return await self._run(self._get_thread_sync, thread_id, mailbox)

    # IMAP is read-only in this provider — send is not supported
    async def send(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("IMAPEmailProvider does not support sending")


# ---------------------------------------------------------------------------
# GmailEmailProvider
# ---------------------------------------------------------------------------

class GmailEmailProvider:
    """
    Email provider backed by the Gmail API (OAuth2).

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
        self, query: str = "is:unread", max_results: int = 20, **kwargs: Any,
    ) -> list[dict[str, Any]]:
        client = await self._client()
        return await client.list_messages(query=query, max_results=max_results)

    async def get_message(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        client = await self._client()
        return await client.get_message_full(message_id)

    async def get_message_metadata(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        client = await self._client()
        return await client.get_message_metadata(message_id)

    async def get_thread(self, thread_id: str, **kwargs: Any) -> dict[str, Any]:
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
    Composite email provider.

    Reading:
      1. IMAPEmailProvider  (if ATLAS_EMAIL_IMAP_HOST / _USERNAME / _PASSWORD set)
         → works with any mail server — Gmail, Outlook, Yahoo, custom IMAP.
      2. GmailEmailProvider (OAuth2 API fallback when IMAP is not configured)

    Sending:
      1. GmailEmailProvider (if OAuth2 credentials available)
      2. ResendEmailProvider (API key fallback)
    """

    def __init__(self) -> None:
        self._imap = IMAPEmailProvider()
        self._gmail = GmailEmailProvider()
        self._resend = ResendEmailProvider()

    # -----------------------------------------------------------------------
    # Send
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Read  (IMAP preferred; Gmail API fallback)
    # -----------------------------------------------------------------------

    def _reader(self):
        """Return the preferred read provider (IMAP when configured; Gmail otherwise)."""
        if self._imap.is_configured():
            return self._imap
        return self._gmail

    async def _read_with_fallback(self, method: str, *args, **kwargs) -> Any:
        """
        Call `method` on the preferred reader; fall back to Gmail if IMAP raises.

        Prevents IMAP misconfiguration (bad password, wrong host, firewall) from
        crashing the MCP server.  The fallback is logged at WARNING so it's visible.
        """
        reader = self._reader()
        try:
            return await getattr(reader, method)(*args, **kwargs)
        except Exception as exc:
            if reader is self._imap:
                logger.warning(
                    "IMAP %s failed (%s) — falling back to Gmail API", method, exc
                )
                return await getattr(self._gmail, method)(*args, **kwargs)
            raise

    async def list_folders(self) -> list[dict[str, Any]]:
        """List IMAP folders. Only works when IMAP is configured."""
        if self._imap.is_configured():
            return await self._imap.list_folders()
        return [{"name": "INBOX", "delimiter": "/", "flags": [], "selectable": True}]

    async def list_messages(
        self, query: str = "is:unread", max_results: int = 20,
        mailbox: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._read_with_fallback("list_messages", query=query, max_results=max_results, mailbox=mailbox)

    async def get_message(self, message_id: str, mailbox: str | None = None) -> dict[str, Any]:
        return await self._read_with_fallback("get_message", message_id, mailbox=mailbox)

    async def get_message_metadata(self, message_id: str, mailbox: str | None = None) -> dict[str, Any]:
        return await self._read_with_fallback("get_message_metadata", message_id, mailbox=mailbox)

    async def get_thread(self, thread_id: str, mailbox: str | None = None) -> dict[str, Any]:
        return await self._read_with_fallback("get_thread", thread_id, mailbox=mailbox)


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
