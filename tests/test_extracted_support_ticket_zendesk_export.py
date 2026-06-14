from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from extracted_content_pipeline.faq_macro_writeback_zendesk import ZendeskMacroCredentials
from extracted_content_pipeline.support_ticket_zendesk_export import (
    ZendeskTicketExportError,
    export_zendesk_full_thread_artifact,
)
from extracted_content_pipeline.support_ticket_zendesk_thread import (
    rows_from_zendesk_full_thread,
)


BASE_URL = "https://acme.zendesk.com"


class _Transport:
    def __init__(self, responses: Mapping[str, Mapping[str, Any]]) -> None:
        self.responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    async def get(self, url: str, *, headers: Mapping[str, str]) -> Mapping[str, Any]:
        self.calls.append({"url": url, "headers": dict(headers)})
        if url not in self.responses:
            raise AssertionError(f"unexpected url: {url}")
        return self.responses[url]


def _credentials(**overrides: str) -> ZendeskMacroCredentials:
    values = dict(
        email="agent@example.com",
        api_token="secret-token",
        subdomain="acme",
        base_url="",
    )
    values.update(overrides)
    return ZendeskMacroCredentials(**values)


def _tickets_url(limit: int) -> str:
    # The incremental cursor endpoint carries no page[size] -- the previous
    # value here masked a Zendesk 400. `limit` is accepted only so existing call
    # sites stay unchanged; the URL no longer depends on it.
    return f"{BASE_URL}/api/v2/incremental/tickets/cursor?start_time=0"


def _comments_url(ticket_id: int | str) -> str:
    return f"{BASE_URL}/api/v2/tickets/{ticket_id}/comments"


def test_tickets_url_omits_page_size_for_cursor_endpoint() -> None:
    # Regression: the incremental cursor endpoint rejects page[size]
    # (Zendesk 400 "page must be an integer"); the URL must carry only
    # start_time. This guards the live-export bug found capturing the
    # finetunelab product-proof corpus.
    from extracted_content_pipeline.support_ticket_zendesk_export import (
        _tickets_url as build_tickets_url,
    )

    url = build_tickets_url(BASE_URL, start_time=0)
    assert url == f"{BASE_URL}/api/v2/incremental/tickets/cursor?start_time=0"
    assert "page" not in url


@pytest.mark.asyncio
async def test_zendesk_export_builds_importer_artifact_and_follows_comment_pages() -> None:
    comments_101 = _comments_url(101)
    comments_101_next = f"{comments_101}?page=2"
    next_tickets = f"{BASE_URL}/api/v2/incremental/tickets/cursor?cursor=abc"
    transport = _Transport({
        _tickets_url(2): {
            "tickets": [{
                "id": 101,
                "subject": "Billing refund",
                "description": "I was billed twice.",
                "requester_id": 501,
                "status": "solved",
                "satisfaction_rating": {"score": "good"},
            }],
            "end_of_stream": False,
            "after_url": next_tickets,
        },
        comments_101: {
            "comments": [
                {"id": 1, "author_id": 900, "public": False, "plain_body": "Internal note: refund was manual."},
                {"id": 2, "author_id": 501, "public": True, "plain_body": "Can you refund the duplicate billing charge?"},
            ],
            "next_page": comments_101_next,
        },
        comments_101_next: {"comments": [
            {"id": 3, "author_id": 900, "public": True, "plain_body": "We refunded the duplicate charge to the card."}
        ]},
        next_tickets: {
            "tickets": [{
                "id": 102,
                "subject": "Export settings",
                "description": "Where do exports live?",
                "requester_id": 502,
                "status": "open",
            }],
            "end_of_stream": True,
        },
        _comments_url(102): {"comments": [
            {"id": 4, "author_id": 502, "public": True, "plain_body": "Where do I download exports?"}
        ], "links": {"next": ""}},
    })

    artifact = await export_zendesk_full_thread_artifact(
        _credentials(),
        limit=2,
        transport=transport,
    )

    assert [call["url"] for call in transport.calls] == [
        _tickets_url(2),
        comments_101,
        comments_101_next,
        next_tickets,
        _comments_url(102),
    ]
    assert all(call["headers"]["Accept"] == "application/json" for call in transport.calls)
    assert all("secret-token" not in str(call["headers"]) for call in transport.calls)
    assert "secret-token" not in str(artifact)
    assert artifact["tickets"][0]["comments"][0]["public"] is False
    result = rows_from_zendesk_full_thread(artifact)
    by_id = {row["ticket_id"]: row for row in result.rows}
    assert result.warnings == ()
    assert by_id["101"]["resolution_text"] == "We refunded the duplicate charge to the card."
    assert by_id["101"]["satisfaction_rating"] == "good"
    assert "Internal note" not in str(result.rows)
    assert "Where do I download exports?" in by_id["102"]["description"]


@pytest.mark.asyncio
async def test_zendesk_export_honors_limit_before_fetching_extra_comments() -> None:
    transport = _Transport({
        _tickets_url(1): {
            "tickets": [
                {"id": 201, "subject": "First", "description": "Question one"},
                {"id": 202, "subject": "Second", "description": "Question two"},
            ],
            "end_of_stream": False,
            "after_url": f"{BASE_URL}/api/v2/incremental/tickets/cursor?cursor=next",
        },
        _comments_url(201): {"comments": [], "next_page": ""},
    })

    artifact = await export_zendesk_full_thread_artifact(
        _credentials(),
        limit=1,
        transport=transport,
    )

    assert [entry["ticket"]["id"] for entry in artifact["tickets"]] == [201]
    assert [call["url"] for call in transport.calls] == [_tickets_url(1), _comments_url(201)]


@pytest.mark.asyncio
async def test_zendesk_export_fails_closed_on_bad_envelopes_credentials_and_limit() -> None:
    with pytest.raises(ZendeskTicketExportError, match="zendesk_tickets_invalid") as exc:
        await export_zendesk_full_thread_artifact(
            _credentials(),
            limit=1,
            transport=_Transport({_tickets_url(1): {"ticket": []}}),
        )
    assert "secret-token" not in str(exc.value)
    with pytest.raises(ZendeskTicketExportError, match="zendesk_comments_invalid"):
        await export_zendesk_full_thread_artifact(
            _credentials(),
            limit=1,
            transport=_Transport({
                _tickets_url(1): {"tickets": [{"id": 301}], "next_page": ""},
                _comments_url(301): {"comment": []},
            }),
        )
    with pytest.raises(ZendeskTicketExportError, match="zendesk_ticket_pagination_invalid"):
        await export_zendesk_full_thread_artifact(
            _credentials(),
            limit=1,
            transport=_Transport({
                _tickets_url(1): {
                    "tickets": [{"id": 302}],
                    "end_of_stream": False,
                    "after_cursor": "abc",
                },
                _comments_url(302): {"comments": []},
            }),
        )
    with pytest.raises(ZendeskTicketExportError, match="zendesk_comments_pagination_cycle"):
        await export_zendesk_full_thread_artifact(
            _credentials(),
            limit=1,
            transport=_Transport({
                _tickets_url(1): {"tickets": [{"id": 303}], "end_of_stream": True},
                _comments_url(303): {"comments": [], "next_page": _comments_url(303)},
            }),
        )
    with pytest.raises(ZendeskTicketExportError, match="zendesk_credentials_missing"):
        await export_zendesk_full_thread_artifact(
            _credentials(api_token=""),
            transport=_Transport({}),
        )
    with pytest.raises(ZendeskTicketExportError, match="zendesk_export_limit_invalid"):
        await export_zendesk_full_thread_artifact(
            _credentials(),
            limit=0,
            transport=_Transport({}),
        )
