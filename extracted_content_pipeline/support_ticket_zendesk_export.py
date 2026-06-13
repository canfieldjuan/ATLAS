from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlencode

import httpx

from .campaign_ports import JsonDict
from .faq_macro_writeback_zendesk import ZendeskMacroCredentials


DEFAULT_ZENDESK_EXPORT_LIMIT = 50
MAX_ZENDESK_EXPORT_LIMIT = 1000
MAX_ZENDESK_COMMENT_PAGES = 100
JsonMap = Mapping[str, Any]


@dataclass(frozen=True)
class ZendeskTicketExportError(Exception):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message}: status={self.status_code}"


class ZendeskTicketExportTransport(Protocol):
    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> JsonMap:
        """Fetch one Zendesk JSON object."""


@dataclass(frozen=True)
class ZendeskHTTPTicketExportTransport:
    timeout_seconds: float = 20.0

    async def get(self, url: str, *, headers: Mapping[str, str]) -> JsonMap:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url, headers=dict(headers))
        if response.status_code >= 400:
            raise ZendeskTicketExportError(
                "zendesk_export_request_failed",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ZendeskTicketExportError("zendesk_export_envelope_invalid") from exc
        if not isinstance(payload, Mapping):
            raise ZendeskTicketExportError("zendesk_export_envelope_invalid")
        return payload


async def export_zendesk_full_thread_artifact(
    credentials: ZendeskMacroCredentials,
    *,
    limit: int = DEFAULT_ZENDESK_EXPORT_LIMIT,
    start_time: int = 0,
    transport: ZendeskTicketExportTransport | None = None,
) -> JsonDict:
    if not credentials.is_complete():
        raise ZendeskTicketExportError("zendesk_credentials_missing")
    limit = _checked_limit(limit)
    base_url = credentials.normalized_base_url()
    headers = _headers(credentials)
    client = transport or ZendeskHTTPTicketExportTransport()
    entries: list[JsonDict] = []
    next_url = _tickets_url(base_url, start_time=start_time, limit=limit)
    seen_ticket_pages: set[str] = set()
    while next_url and len(entries) < limit:
        _check_page_cycle(next_url, seen_ticket_pages, "zendesk_ticket_pagination_cycle")
        payload = await client.get(next_url, headers=headers)
        tickets = _required_sequence(payload, "tickets", "zendesk_tickets_invalid")
        if not tickets:
            break
        for raw_ticket in tickets:
            ticket = _required_mapping(raw_ticket, "zendesk_ticket_invalid")
            comments = await _comments_for_ticket(
                client,
                base_url,
                ticket_id=_ticket_id(ticket),
                headers=headers,
            )
            entries.append({
                "ticket": dict(ticket),
                "comments": comments,
            })
            if len(entries) >= limit:
                break
        next_url = _next_ticket_page(payload)
    return {"tickets": entries}


async def _comments_for_ticket(
    transport: ZendeskTicketExportTransport,
    base_url: str,
    *,
    ticket_id: str,
    headers: Mapping[str, str],
) -> list[JsonDict]:
    comments: list[JsonDict] = []
    next_url = _comments_url(base_url, ticket_id)
    seen_comment_pages: set[str] = set()
    while next_url:
        if len(seen_comment_pages) >= MAX_ZENDESK_COMMENT_PAGES:
            raise ZendeskTicketExportError("zendesk_comments_pagination_too_deep")
        _check_page_cycle(
            next_url,
            seen_comment_pages,
            "zendesk_comments_pagination_cycle",
        )
        payload = await transport.get(next_url, headers=headers)
        comments.extend(
            dict(_required_mapping(comment, "zendesk_comment_invalid"))
            for comment in _required_sequence(
                payload,
                "comments",
                "zendesk_comments_invalid",
            )
        )
        next_url = _next_page(payload)
    return comments


def _tickets_url(base_url: str, *, start_time: int, limit: int) -> str:
    query = urlencode({
        "start_time": max(0, int(start_time)),
        "page[size]": min(limit, MAX_ZENDESK_EXPORT_LIMIT),
    })
    return f"{base_url}/api/v2/incremental/tickets/cursor?{query}"


def _comments_url(base_url: str, ticket_id: str) -> str:
    return f"{base_url}/api/v2/tickets/{quote(ticket_id, safe='')}/comments"


def _headers(credentials: ZendeskMacroCredentials) -> dict[str, str]:
    return {"Accept": "application/json", "Authorization": credentials.authorization_header()}


def _checked_limit(value: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ZendeskTicketExportError("zendesk_export_limit_invalid") from exc
    if limit < 1 or limit > MAX_ZENDESK_EXPORT_LIMIT:
        raise ZendeskTicketExportError("zendesk_export_limit_invalid")
    return limit


def _required_sequence(payload: JsonMap, key: str, error: str) -> Sequence[Any]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ZendeskTicketExportError(error)
    return value


def _required_mapping(value: Any, error: str) -> JsonMap:
    if not isinstance(value, Mapping):
        raise ZendeskTicketExportError(error)
    return value


def _ticket_id(ticket: JsonMap) -> str:
    ticket_id = str(ticket.get("id") or "").strip()
    if not ticket_id:
        raise ZendeskTicketExportError("zendesk_ticket_id_missing")
    return ticket_id


def _next_ticket_page(payload: JsonMap) -> str:
    if payload.get("end_of_stream") is True:
        return ""
    after_url = _clean(payload.get("after_url"))
    if after_url:
        return after_url
    next_page = _next_page(payload)
    if next_page:
        return next_page
    if payload.get("end_of_stream") is False or _clean(payload.get("after_cursor")):
        raise ZendeskTicketExportError("zendesk_ticket_pagination_invalid")
    return ""


def _next_page(payload: JsonMap) -> str:
    next_page = str(payload.get("next_page") or "").strip()
    if next_page:
        return next_page
    links = payload.get("links")
    if isinstance(links, Mapping):
        return str(links.get("next") or "").strip()
    return ""


def _check_page_cycle(url: str, seen: set[str], error: str) -> None:
    if url in seen:
        raise ZendeskTicketExportError(error)
    seen.add(url)


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = (
    "DEFAULT_ZENDESK_EXPORT_LIMIT",
    "MAX_ZENDESK_COMMENT_PAGES",
    "MAX_ZENDESK_EXPORT_LIMIT",
    "ZendeskHTTPTicketExportTransport",
    "ZendeskTicketExportError",
    "ZendeskTicketExportTransport",
    "export_zendesk_full_thread_artifact",
)
