"""
Gmail digest builtin task.

Fetches unread emails from Gmail API using OAuth2 and returns
a structured summary.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from ...config import settings
from ...services.google_oauth import get_google_token_store
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.gmail_digest")

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
TOKEN_URL = "https://oauth2.googleapis.com/token"


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
        }

    store = get_google_token_store()
    if not store.get_credentials("gmail"):
        return {
            "query": "",
            "total_unread": 0,
            "emails": [],
            "summary": "Gmail not configured. Run: python scripts/setup_google_oauth.py",
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
        }

    # Fetch metadata for each message (concurrently, batched)
    emails = []
    batch_size = 10
    for i in range(0, len(messages), batch_size):
        batch = messages[i : i + batch_size]
        tasks = [client.get_message_metadata(m["id"]) for m in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Failed to fetch message metadata: %s", r)
            else:
                emails.append(r)

    # Build summary
    summary_parts = [f"{total} unread emails."]
    if emails:
        previews = []
        for e in emails[:5]:
            sender = e["from"].split("<")[0].strip().strip('"') or e["from"]
            previews.append(f"{sender} ({e['subject']})")
        summary_parts.append("From: " + ", ".join(previews))
        if total > 5:
            summary_parts.append(f"...and {total - 5} more.")

    result = {
        "query": query,
        "total_unread": total,
        "emails": emails,
        "summary": " ".join(summary_parts),
    }

    logger.info("Gmail digest: %s", result["summary"])
    return result
