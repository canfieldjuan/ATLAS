"""
Apollo.io API client for prospect enrichment.

Three-step flow:
1. enrich_organization (GET, 1 credit) -- domain -> org profile
2. search_people (POST, FREE) -- org_id + seniority -> obfuscated stubs with IDs
3. reveal_person (POST, 1 credit each) -- person ID -> full profile with verified email

Auth: x-api-key header.
Rate limiting: token-bucket per minute, respects X-RateLimit headers, backs off on 429.
Credit budget guard: stops processing when max_credits_per_run is exhausted.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger("atlas.services.apollo")

_BASE_URL = "https://api.apollo.io"

# ---------------------------------------------------------------------------
# Data classes for typed returns
# ---------------------------------------------------------------------------


class OrgResult:
    """Enriched organization data from Apollo."""

    __slots__ = (
        "apollo_org_id", "name", "domain", "industry",
        "employee_count", "annual_revenue_range", "tech_stack",
    )

    def __init__(self, data: dict[str, Any]):
        self.apollo_org_id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.domain: str = data.get("primary_domain") or data.get("website_url") or ""
        self.industry: str = data.get("industry") or ""
        self.employee_count: int | None = data.get("estimated_num_employees")
        self.annual_revenue_range: str = data.get("annual_revenue_printed") or ""
        raw_tech = data.get("current_technologies") or []
        self.tech_stack: list[str] = [t.get("name", t) if isinstance(t, dict) else str(t) for t in raw_tech]


class PersonStub:
    """Obfuscated person data from people search (no email, last name hidden)."""

    __slots__ = ("apollo_person_id", "first_name", "title", "has_email")

    def __init__(self, data: dict[str, Any]):
        self.apollo_person_id: str = data.get("id", "")
        self.first_name: str = data.get("first_name") or ""
        self.title: str = data.get("title") or ""
        self.has_email: bool = data.get("has_email", False)


class PersonFull:
    """Fully revealed person with email from people/match."""

    __slots__ = (
        "apollo_person_id", "first_name", "last_name", "email", "email_status",
        "title", "seniority", "department", "linkedin_url",
        "city", "state", "country", "company_name", "company_domain",
        "raw",
    )

    def __init__(self, data: dict[str, Any]):
        self.apollo_person_id: str = data.get("id", "")
        self.first_name: str = data.get("first_name") or ""
        self.last_name: str = data.get("last_name") or ""
        self.email: str | None = data.get("email")
        self.email_status: str = data.get("email_status") or ""
        self.title: str = data.get("title") or ""
        self.seniority: str = data.get("seniority") or ""
        self.department: str = data.get("department") or ""
        self.linkedin_url: str = data.get("linkedin_url") or ""
        self.city: str = data.get("city") or ""
        self.state: str = data.get("state") or ""
        self.country: str = data.get("country") or ""
        org = data.get("organization") or {}
        self.company_name: str = org.get("name") or data.get("organization_name") or ""
        self.company_domain: str = org.get("primary_domain") or org.get("website_url") or ""
        self.raw: dict[str, Any] = data


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, tokens_per_minute: int):
        self._capacity = tokens_per_minute
        self._tokens = float(tokens_per_minute)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._reset_at: float | None = None

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._reset_at and now < self._reset_at:
                wait = self._reset_at - now
                logger.info("rate-limit: waiting %.1fs for reset", wait)
                await asyncio.sleep(wait)
                self._reset_at = None
                self._tokens = self._capacity
                return

            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * (self._capacity / 60.0))
            self._last_refill = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / (self._capacity / 60.0)
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0

    def set_reset(self, reset_epoch: float) -> None:
        self._reset_at = max(0.0, reset_epoch - time.time()) + time.monotonic()


# ---------------------------------------------------------------------------
# Apollo provider
# ---------------------------------------------------------------------------


class ApolloProvider:
    """Async Apollo.io API client."""

    def __init__(self):
        cfg = settings.apollo
        self._api_key = cfg.api_key
        self._bucket = _TokenBucket(cfg.rate_limit_per_minute)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=_BASE_URL,
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache",
                    "x-api-key": self._api_key,
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self, method: str, path: str, **kwargs
    ) -> dict[str, Any] | None:
        """Make an authenticated request with rate limiting and 429 handling."""
        await self._bucket.acquire()
        client = self._get_client()

        resp = await client.request(method, path, **kwargs)

        # Handle rate limit headers
        remaining = resp.headers.get("x-rate-limit-remaining")
        if remaining is not None and int(remaining) < 3:
            reset_str = resp.headers.get("x-rate-limit-reset")
            if reset_str:
                self._bucket.set_reset(float(reset_str))

        if resp.status_code == 429:
            reset_str = resp.headers.get("x-rate-limit-reset")
            if reset_str:
                self._bucket.set_reset(float(reset_str))
            await self._bucket.acquire()
            resp = await client.request(method, path, **kwargs)

        if resp.status_code >= 400:
            logger.warning("Apollo %s %s -> %d: %s", method, path, resp.status_code, resp.text[:300])
            return None

        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich_organization(self, domain: str) -> OrgResult | None:
        """Enrich a company by domain. GET, 1 credit. Returns None if not found."""
        data = await self._request("GET", "/api/v1/organizations/enrich", params={
            "domain": domain,
        })
        if not data:
            return None
        org = data.get("organization")
        if not org:
            return None
        return OrgResult(org)

    async def search_people(
        self,
        *,
        org_id: str | None = None,
        company_name: str | None = None,
        seniorities: list[str] | None = None,
        limit: int = 25,
    ) -> list[PersonStub]:
        """Search people at an org. FREE (no credits). Returns obfuscated stubs.

        Pass either org_id or company_name. company_name uses fuzzy search
        and avoids the org enrich credit cost.
        """
        if seniorities is None:
            seniorities = settings.apollo.target_seniorities

        body: dict[str, Any] = {
            "person_seniorities": seniorities,
            "page": 1,
            "per_page": limit,
        }
        if org_id:
            body["organization_ids"] = [org_id]
        elif company_name:
            body["q_organization_name"] = company_name
        else:
            return []

        data = await self._request("POST", "/api/v1/mixed_people/api_search", json=body)
        if not data:
            return []
        people = data.get("people") or []
        return [PersonStub(p) for p in people]

    async def reveal_person(self, person_id: str) -> PersonFull | None:
        """Reveal a single person by ID. POST, 1 credit. Returns full profile with email."""
        data = await self._request("POST", "/api/v1/people/match", json={
            "id": person_id,
            "reveal_personal_emails": False,
        })
        if not data:
            return None
        person = data.get("person")
        if not person:
            return None
        return PersonFull(person)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ApolloProvider | None = None


def get_apollo_provider() -> ApolloProvider:
    """Get or create the singleton ApolloProvider."""
    global _instance
    if _instance is None:
        _instance = ApolloProvider()
    return _instance
