"""
Gmail digest builtin task.

Fetches unread emails from Gmail API using OAuth2, extracts full body
content, deduplicates against previously processed messages, and returns
a structured summary for LLM synthesis.
"""

import asyncio
import base64
import logging
import time
from html.parser import HTMLParser
from typing import Any

import httpx

from ...config import settings
from ...services.google_oauth import get_google_token_store
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.gmail_digest")

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
TOKEN_URL = "https://oauth2.googleapis.com/token"


# ---------------------------------------------------------------------------
# HTML-to-text helpers (stdlib, no external dependency)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text extractor using stdlib HTMLParser."""

    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3",
        "h4", "h5", "h6", "blockquote", "pre", "table",
    })
    _SKIP_TAGS = frozenset({"script", "style", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS and self._parts:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Collapse runs of whitespace but preserve paragraph breaks
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = " ".join(line.split())
            cleaned.append(stripped)
        return "\n".join(cleaned).strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using stdlib parser."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        # Fallback: strip tags crudely
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


def _decode_body_data(data: str) -> str:
    """Decode Gmail base64url-encoded body data to UTF-8 text."""
    # Gmail uses URL-safe base64 without padding
    padded = data + "=" * (4 - len(data) % 4) if len(data) % 4 else data
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _extract_body_parts(payload: dict) -> tuple[str, str]:
    """
    Recursively walk a Gmail message payload to extract body text.

    Returns (plain_text, html_text). Either may be empty.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def _walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        if data:
            decoded = _decode_body_data(data)
            if mime == "text/plain":
                plain_parts.append(decoded)
            elif mime == "text/html":
                html_parts.append(decoded)

        for sub in part.get("parts", []):
            _walk(sub)

    _walk(payload)
    return "\n".join(plain_parts), "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Database helpers (dedup)
# ---------------------------------------------------------------------------

async def _get_processed_message_ids(msg_ids: list[str]) -> set[str]:
    """Check which message IDs have already been processed."""
    pool = get_db_pool()
    if not pool.is_initialized or not msg_ids:
        return set()

    try:
        rows = await pool.fetch(
            """
            SELECT gmail_message_id FROM processed_emails
            WHERE gmail_message_id = ANY($1::text[])
            """,
            msg_ids,
        )
        return {r["gmail_message_id"] for r in rows}
    except Exception as e:
        logger.warning("Dedup lookup failed (proceeding without): %s", e)
        return set()


async def _record_processed_emails(emails: list[dict[str, Any]]) -> None:
    """Record processed message IDs for future dedup. ON CONFLICT DO NOTHING."""
    pool = get_db_pool()
    if not pool.is_initialized or not emails:
        return

    try:
        async with pool.transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO processed_emails (gmail_message_id, sender, subject, category, priority, replyable)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (gmail_message_id) DO NOTHING
                """,
                [
                    (
                        e["id"],
                        e.get("from", ""),
                        e.get("subject", ""),
                        e.get("category"),
                        e.get("priority"),
                        e.get("replyable"),
                    )
                    for e in emails
                ],
            )
        logger.debug("Recorded %d processed email IDs", len(emails))
    except Exception as e:
        logger.warning("Failed to record processed emails: %s", e)


# ---------------------------------------------------------------------------
# Gmail API client
# ---------------------------------------------------------------------------

class GmailClient:
    """Lightweight Gmail API client with OAuth token refresh."""

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._token_expires: float = 0.0
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=15.0)
            return self._client

    async def close(self) -> None:
        async with self._lock:
            if self._client:
                await self._client.aclose()
                self._client = None

    async def _refresh_token(self) -> str:
        """Refresh OAuth2 access token using the Gmail refresh token."""
        async with self._lock:
            if self._access_token and time.time() < self._token_expires - 60:
                return self._access_token

            # Load credentials from token store (file first, .env fallback)
            store = get_google_token_store()
            creds = store.get_credentials("gmail")
            if not creds:
                raise RuntimeError(
                    "Gmail OAuth not configured. "
                    "Run: python scripts/setup_google_oauth.py"
                )

            if self._client is None:
                self._client = httpx.AsyncClient(timeout=15.0)
            client = self._client

            data = {
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "refresh_token": creds.refresh_token,
                "grant_type": "refresh_token",
            }

            response = await client.post(TOKEN_URL, data=data)
            if response.status_code in (400, 401):
                logger.error(
                    "Gmail refresh token rejected (HTTP %d). "
                    "Re-run: python scripts/setup_google_oauth.py",
                    response.status_code,
                )
                raise RuntimeError(
                    f"Gmail refresh token rejected (HTTP {response.status_code})"
                )
            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = time.time() + expires_in

            # Auto-persist rotated refresh token
            new_refresh = token_data.get("refresh_token")
            if new_refresh and new_refresh != creds.refresh_token:
                store.persist_refresh_token("gmail", new_refresh)

            logger.debug("Refreshed Gmail access token")
            return self._access_token

    async def _get_headers(self) -> dict[str, str]:
        token = await self._refresh_token()
        return {"Authorization": f"Bearer {token}"}

    async def list_messages(
        self, query: str, max_results: int = 20
    ) -> list[dict[str, str]]:
        """List message IDs matching a query."""
        client = await self._ensure_client()
        headers = await self._get_headers()

        response = await client.get(
            f"{GMAIL_API_BASE}/users/me/messages",
            headers=headers,
            params={"q": query, "maxResults": max_results},
        )
        response.raise_for_status()
        return response.json().get("messages", [])

    async def get_message_metadata(self, msg_id: str) -> dict[str, Any]:
        """Get message metadata (From, Subject, Date, snippet)."""
        client = await self._ensure_client()
        headers = await self._get_headers()

        response = await client.get(
            f"{GMAIL_API_BASE}/users/me/messages/{msg_id}",
            headers=headers,
            params={
                "format": "metadata",
                "metadataHeaders": ["From", "Subject", "Date"],
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract headers
        header_map: dict[str, str] = {}
        for h in data.get("payload", {}).get("headers", []):
            header_map[h["name"]] = h["value"]

        return {
            "id": data.get("id", msg_id),
            "from": header_map.get("From", ""),
            "subject": header_map.get("Subject", "(no subject)"),
            "date": header_map.get("Date", ""),
            "snippet": data.get("snippet", ""),
        }

    async def get_message_full(self, msg_id: str) -> dict[str, Any]:
        """
        Fetch a message with full body content.

        Returns enriched dict with body_text, body_html, has_unsubscribe,
        label_ids, and thread_id in addition to standard metadata fields.
        """
        client = await self._ensure_client()
        headers = await self._get_headers()

        response = await client.get(
            f"{GMAIL_API_BASE}/users/me/messages/{msg_id}",
            headers=headers,
            params={"format": "full"},
        )
        response.raise_for_status()
        data = response.json()

        payload = data.get("payload", {})

        # Extract headers
        header_map: dict[str, str] = {}
        for h in payload.get("headers", []):
            header_map[h["name"].lower()] = h["value"]

        # Extract body
        plain_text, html_text = _extract_body_parts(payload)

        # Prefer plain text; fall back to HTML-to-text.
        # Some mailers put HTML in the text/plain part -- detect and reject.
        def _looks_like_html(text: str) -> bool:
            sample = text[:500]
            return "<html" in sample.lower() or sample.count("<") > 5 or "@media" in sample

        if plain_text.strip() and not _looks_like_html(plain_text):
            body_text = plain_text.strip()
        elif html_text.strip():
            body_text = _html_to_text(html_text)
        elif plain_text.strip():
            # Plain text looked like HTML; try converting it
            body_text = _html_to_text(plain_text)
        else:
            body_text = data.get("snippet", "")

        # Truncate to configured limit
        max_chars = settings.tools.gmail_body_max_chars
        if len(body_text) > max_chars:
            body_text = body_text[:max_chars] + "..."

        # Check for List-Unsubscribe header (strong promo/newsletter signal)
        has_unsubscribe = "list-unsubscribe" in header_map

        return {
            "id": data.get("id", msg_id),
            "from": header_map.get("from", ""),
            "subject": header_map.get("subject", "(no subject)"),
            "date": header_map.get("date", ""),
            "snippet": data.get("snippet", ""),
            "body_text": body_text,
            "body_html": html_text,
            "has_unsubscribe": has_unsubscribe,
            "label_ids": data.get("labelIds", []),
            "thread_id": data.get("threadId", ""),
            "message_id": header_map.get("message-id", ""),
        }


# Module-level client (reused across invocations)
_gmail_client: GmailClient | None = None
_gmail_client_lock = asyncio.Lock()


async def _get_gmail_client() -> GmailClient:
    global _gmail_client
    async with _gmail_client_lock:
        if _gmail_client is None:
            _gmail_client = GmailClient()
        return _gmail_client


async def run(task: ScheduledTask) -> dict:
    """
    Fetch and summarize unread Gmail messages.

    Configurable via task.metadata:
        query (str): Gmail search query (default: from config)
        max_results (int): Max emails to fetch (default: from config)
    """
    cfg = settings.tools

    if not cfg.gmail_enabled:
        return {
            "query": "",
            "total_unread": 0,
            "emails": [],
            "summary": "Gmail digest is disabled. Set ATLAS_TOOLS_GMAIL_ENABLED=true.",
            "_skip_synthesis": "Gmail digest skipped -- not enabled.",
        }

    store = get_google_token_store()
    if not store.get_credentials("gmail"):
        return {
            "query": "",
            "total_unread": 0,
            "emails": [],
            "summary": "Gmail not configured. Run: python scripts/setup_google_oauth.py",
            "_skip_synthesis": "Gmail digest skipped -- not configured.",
        }

    metadata = task.metadata or {}
    query = metadata.get("query", cfg.gmail_query)
    max_results = metadata.get("max_results", cfg.gmail_max_results)

    client = await _get_gmail_client()

    try:
        messages = await client.list_messages(query=query, max_results=max_results)
    except Exception as e:
        logger.error("Failed to list Gmail messages: %s", e)
        return {
            "query": query,
            "total_unread": 0,
            "emails": [],
            "summary": f"Gmail API error: {type(e).__name__}",
        }

    total = len(messages)
    if total == 0:
        return {
            "query": query,
            "total_unread": 0,
            "emails": [],
            "summary": f"No emails matching '{query}'.",
            "_skip_synthesis": "No unread emails.",
        }

    # --- Dedup filter: skip already-processed messages ---
    all_ids = [m["id"] for m in messages]
    already_processed = await _get_processed_message_ids(all_ids)
    new_messages = [m for m in messages if m["id"] not in already_processed]

    if not new_messages:
        return {
            "query": query,
            "total_unread": total,
            "emails": [],
            "summary": f"{total} unread emails, all already processed in a previous digest.",
            "_skip_synthesis": "All emails already processed.",
        }

    logger.info(
        "Gmail digest: %d unread, %d already processed, %d new",
        total, len(already_processed), len(new_messages),
    )

    # --- Fetch full content for new messages (concurrently, batched) ---
    emails: list[dict[str, Any]] = []
    batch_size = 10
    for i in range(0, len(new_messages), batch_size):
        batch = new_messages[i : i + batch_size]
        tasks = [client.get_message_full(m["id"]) for m in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Failed to fetch message: %s", r)
            else:
                emails.append(r)

    # --- Classify emails (rule-based, no LLM) ---
    from .email_classifier import get_email_classifier

    classifier = get_email_classifier()
    emails = classifier.classify_batch(emails)

    # --- Record processed IDs before synthesis (crash-safe) ---
    await _record_processed_emails(emails)

    # --- Build result for LLM synthesis ---
    # Slim down each email to only what the LLM needs for summarization.
    # Classification is already done; the LLM just summarizes.
    SYNTHESIS_BODY_LIMIT = 500
    emails_for_llm = []
    for e in emails:
        body = e.get("body_text", "")
        if len(body) > SYNTHESIS_BODY_LIMIT:
            body = body[:SYNTHESIS_BODY_LIMIT] + "..."
        emails_for_llm.append({
            "from": e.get("from", ""),
            "subject": e.get("subject", ""),
            "date": e.get("date", ""),
            "body_text": body,
            "category": e.get("category", "other"),
            "priority": e.get("priority", "fyi"),
        })

    # Build summary
    summary_parts = [f"{len(emails)} new emails (of {total} unread)."]
    if emails:
        previews = []
        for e in emails[:5]:
            sender = e["from"].split("<")[0].strip().strip('"') or e["from"]
            previews.append(f"{sender} ({e['subject']})")
        summary_parts.append("From: " + ", ".join(previews))
        if len(emails) > 5:
            summary_parts.append(f"...and {len(emails) - 5} more.")

    result = {
        "query": query,
        "total_unread": total,
        "new_emails": len(emails),
        "already_processed": len(already_processed),
        "emails": emails_for_llm,
        "summary": " ".join(summary_parts),
    }

    logger.info("Gmail digest: %s", result["summary"])
    return result
