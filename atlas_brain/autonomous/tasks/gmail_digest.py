"""
Gmail digest builtin task.

Fetches unread emails from Gmail API using OAuth2, extracts full body
content, deduplicates against previously processed messages, and returns
a structured summary for LLM synthesis.
"""

import asyncio
import base64
import email.utils
import logging
import re
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
                INSERT INTO processed_emails (gmail_message_id, sender, subject, category, priority, replyable, contact_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
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
                        e.get("_contact_id"),
                    )
                    for e in emails
                ],
            )
        logger.debug("Recorded %d processed email IDs", len(emails))
    except Exception as e:
        logger.warning("Failed to record processed emails: %s", e)


async def _send_action_email_notifications(emails: list[dict[str, Any]]) -> None:
    """Send individual ntfy notifications for action-required emails with [Draft Reply] buttons."""
    if not settings.alerts.ntfy_enabled:
        return
    if not settings.email_draft.enabled:
        return

    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    for e in emails:
        gmail_msg_id = e.get("id", "")
        if not gmail_msg_id:
            continue

        sender = e.get("from", "unknown")
        subject = e.get("subject", "(no subject)")
        body_snippet = (e.get("body_text") or e.get("snippet") or "")[:200]
        if len(body_snippet) == 200:
            body_snippet += "..."

        sender_name = sender.split("<")[0].strip().strip('"') or sender

        # Lead emails: override sender and notification style
        is_lead = e.get("category") == "lead"
        if is_lead:
            lead_name = e.get("_lead_name", "")
            lead_email = e.get("_lead_email", "")
            if lead_name or lead_email:
                sender_name = lead_name or lead_email

        message = f"From: {sender_name}\nSubject: {subject}\n\n{body_snippet}"

        rfc_msg_id = e.get("message_id", "").strip("<>")
        view_url = (
            f"https://mail.google.com/mail/u/0/#search/rfc822msgid:{rfc_msg_id}"
            if rfc_msg_id
            else "https://mail.google.com/mail/u/0/#inbox"
        )
        actions = (
            f"http, Draft Reply, {api_url}/api/v1/email/drafts/generate/{gmail_msg_id}, method=POST, clear=true; "
            f"view, View Email, {view_url}"
        )

        if is_lead:
            ntfy_title = f"New Lead: {subject[:60]}"
            ntfy_tags = "email,star"
        else:
            ntfy_title = f"Action Required: {subject[:60]}"
            ntfy_tags = "email,warning"

        headers = {
            "Title": ntfy_title,
            "Priority": "high",
            "Tags": ntfy_tags,
            "Actions": actions,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(ntfy_url, content=message, headers=headers)
                resp.raise_for_status()
            logger.info("Action email notification sent for %s: %s", gmail_msg_id, subject[:40])
        except Exception as exc:
            logger.warning("Failed to send action email notification for %s: %s", gmail_msg_id, exc)


async def _get_email_graph_context(action_emails: list[dict[str, Any]]) -> list[str]:
    """Query the knowledge graph for historical context about action-required senders.

    Returns a list of fact strings (empty if graph is unavailable or returns nothing).
    Used to enrich the email triage synthesis: the LLM can surface patterns like
    "Cash App has sent 3 urgent payment emails this month" from graph history.
    """
    if not action_emails:
        return []

    try:
        from ...memory.rag_client import get_rag_client

        # Extract unique sender display names (skip bare email addresses)
        senders = []
        seen: set[str] = set()
        for e in action_emails[:6]:
            raw = e.get("from", "")
            name = raw.split("<")[0].strip().strip('"').strip()
            if name and name not in seen and "@" not in name:
                senders.append(name)
                seen.add(name)

        if not senders:
            # Fall back to a generic query when all senders are bare addresses
            query = "recent urgent email obligations and action items"
        else:
            query = f"emails and obligations from {', '.join(senders[:4])}"

        client = get_rag_client()
        result = await client.search(query, max_facts=6)
        facts = [s.fact for s in result.facts if s.fact]
        if facts:
            logger.debug("Email graph context: %d facts for query %r", len(facts), query)
        return facts

    except Exception as e:
        logger.warning("Email graph context fetch failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Lead (web form) processing helpers
# ---------------------------------------------------------------------------


def _parse_form_fields(body_text: str) -> dict[str, str]:
    """Extract Name/Email/Phone/Message from web form body text (Key: Value lines)."""
    fields: dict[str, str] = {}
    for match in re.finditer(
        r"^\s*(name|email|phone|message)\s*:\s*(.+)",
        body_text, re.IGNORECASE | re.MULTILINE,
    ):
        key = match.group(1).lower()
        value = match.group(2).strip()
        if value:
            fields[key] = value
    return fields


async def _process_lead_emails(emails: list[dict[str, Any]]) -> None:
    """Create CRM contacts and log interactions for lead emails.

    Only processes emails with category == "lead". For each:
    1. Parse submitter email from Reply-To, fall back to body Email: field
    2. Parse name/phone from body fields
    3. find_or_create_contact(source="web", contact_type="lead")
    4. log_interaction(type="email", summary="Web form submission: ...")
    5. Stash _contact_id / _lead_name / _lead_email on the email dict

    Fail-open: CRM errors never block the digest.
    """
    from ...services.crm_provider import get_crm_provider

    lead_emails = [e for e in emails if e.get("category") == "lead"]
    if not lead_emails:
        return

    crm = get_crm_provider()

    for e in lead_emails:
        try:
            body_text = e.get("body_text", "")
            fields = _parse_form_fields(body_text)

            # Parse submitter email: prefer Reply-To header, fall back to body field
            _, reply_to_email = email.utils.parseaddr(e.get("reply_to", ""))
            submitter_email = reply_to_email or fields.get("email", "")
            submitter_name = fields.get("name", "")
            submitter_phone = fields.get("phone")

            if not submitter_email and not submitter_name:
                logger.debug("Lead email %s: no submitter info found, skipping CRM", e.get("id"))
                continue

            # Stash lead info for ntfy enrichment
            e["_lead_name"] = submitter_name
            e["_lead_email"] = submitter_email

            contact = await crm.find_or_create_contact(
                full_name=submitter_name or submitter_email,
                email=submitter_email or None,
                phone=submitter_phone,
                source="web",
                contact_type="lead",
                tags=["web3forms"],
            )
            if not contact.get("id"):
                logger.warning("Lead CRM contact has no ID for email %s", e.get("id"))
                continue

            contact_id = str(contact["id"])
            e["_contact_id"] = contact_id

            # Log the form submission as an interaction
            subject = e.get("subject", "")
            message_preview = fields.get("message", "")[:200]
            await crm.log_interaction(
                contact_id=contact_id,
                interaction_type="email",
                summary=f"Web form submission: {subject}. {message_preview}".strip(),
            )
            logger.info("Lead CRM: contact %s linked to email %s", contact_id, e.get("id"))

        except Exception as exc:
            logger.warning("Lead CRM processing failed for email %s: %s", e.get("id"), exc)


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
            "reply_to": header_map.get("reply-to", ""),
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


async def _load_recent_processed_emails() -> list[dict[str, Any]]:
    """Load emails processed in the last 24h from the DB (used when intake is active)."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT gmail_message_id, sender, subject, category, priority,
                   replyable, contact_id, action_plan, customer_context_summary
            FROM processed_emails
            WHERE processed_at > NOW() - INTERVAL '24 hours'
            ORDER BY processed_at DESC
            LIMIT 500
            """,
        )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Failed to load recent processed emails: %s", e)
        return []


async def run(task: ScheduledTask) -> dict:
    """
    Fetch and summarize unread Gmail messages.

    When email_intake is enabled, reads from processed_emails DB (last 24h)
    instead of hitting the Gmail API (intake already polled throughout the day).

    Configurable via task.metadata:
        query (str): Gmail search query (default: from config)
        max_results (int): Max emails to fetch (default: from config)
    """
    cfg = settings.tools

    # --- Intake-aware path: build digest from DB instead of Gmail API ---
    if settings.email_intake.enabled:
        recent = await _load_recent_processed_emails()
        if not recent:
            return {
                "query": "",
                "total_unread": 0,
                "emails": [],
                "summary": "No emails processed in last 24h.",
                "_skip_synthesis": "No emails processed in last 24h",
            }

        SYNTHESIS_BODY_LIMIT = 500
        emails_for_llm = []
        for r in recent:
            emails_for_llm.append({
                "gmail_message_id": r.get("gmail_message_id", ""),
                "from": r.get("sender", ""),
                "subject": r.get("subject", ""),
                "date": "",
                "body_text": "",
                "category": r.get("category", "other"),
                "priority": r.get("priority", "fyi"),
                "replyable": r.get("replyable"),
            })

        action_count = sum(
            1 for r in recent if r.get("priority") == "action_required"
        )
        summary = f"{len(recent)} emails in last 24h ({action_count} action required)."
        logger.info("Gmail digest (from DB): %s", summary)

        return {
            "query": "processed_emails (last 24h)",
            "total_unread": len(recent),
            "new_emails": len(recent),
            "already_processed": 0,
            "emails": emails_for_llm,
            "graph_context": [],
            "summary": summary,
        }

    # --- IMAP-based fetch path (fallback when intake is disabled) ---
    from ...services.email_provider import IMAPEmailProvider

    imap = IMAPEmailProvider()
    if not imap.is_configured():
        return {
            "query": "",
            "total_unread": 0,
            "emails": [],
            "summary": "IMAP not configured. Set ATLAS_EMAIL_IMAP_HOST/USERNAME/PASSWORD.",
            "_skip_synthesis": "Gmail digest skipped -- IMAP not configured.",
        }

    metadata = task.metadata or {}
    query = metadata.get("query", cfg.gmail_query)
    max_results = metadata.get("max_results", cfg.gmail_max_results)

    try:
        messages = await imap.list_messages(query=query, max_results=max_results)
    except Exception as e:
        logger.error("Failed to list IMAP messages: %s", e)
        return {
            "query": query,
            "total_unread": 0,
            "emails": [],
            "summary": f"IMAP error: {type(e).__name__}",
        }

    total = len(messages)
    if total == 0:
        return {
            "query": query,
            "total_unread": 0,
            "emails": [],
            "summary": "No unread emails.",
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
    max_body = cfg.gmail_body_max_chars
    emails: list[dict[str, Any]] = []
    batch_size = metadata.get("batch_size", settings.autonomous.gmail_digest_batch_size)
    for i in range(0, len(new_messages), batch_size):
        batch = new_messages[i : i + batch_size]
        tasks = [imap.get_message(m["id"]) for m in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Failed to fetch message: %s", r)
            elif r.get("error"):
                logger.warning("IMAP fetch error: %s", r["error"])
            else:
                body = r.get("body_text", "")
                if len(body) > max_body:
                    r["body_text"] = body[:max_body] + "..."
                emails.append(r)

    # --- Classify emails (rule-based, no LLM) ---
    from .email_classifier import get_email_classifier

    classifier = get_email_classifier()
    emails = classifier.classify_batch(emails)

    # --- CRM processing for lead emails (web form submissions) ---
    await _process_lead_emails(emails)

    # --- Record processed IDs before synthesis (crash-safe) ---
    await _record_processed_emails(emails)

    # --- Send individual ntfy notifications for action-required replyable emails ---
    action_emails = [
        e for e in emails
        if e.get("priority") == "action_required"
        and e.get("replyable") is not False
    ]
    if action_emails:
        await _send_action_email_notifications(action_emails)

    # --- Query knowledge graph for historical context about action senders ---
    graph_context = await _get_email_graph_context(action_emails)

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
            "gmail_message_id": e.get("id", ""),
            "from": e.get("from", ""),
            "subject": e.get("subject", ""),
            "date": e.get("date", ""),
            "body_text": body,
            "category": e.get("category", "other"),
            "priority": e.get("priority", "fyi"),
            "replyable": e.get("replyable"),
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
        "graph_context": graph_context,
        "summary": " ".join(summary_parts),
    }

    logger.info("Gmail digest: %s", result["summary"])
    return result
