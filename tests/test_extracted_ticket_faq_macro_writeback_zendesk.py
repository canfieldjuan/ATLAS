from __future__ import annotations

import base64
from typing import Any, Mapping

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import (
    MacroWritebackMapping,
    SupportMacroDraft,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    StaticZendeskMacroCredentialsProvider,
    ZENDESK_PLATFORM,
    ZendeskMacroCredentials,
    ZendeskMacroPublishProvider,
    ZendeskMacroTransportError,
)


class _MappingRepo:
    def __init__(
        self,
        existing: MacroWritebackMapping | None = None,
        *,
        upsert_error: Exception | None = None,
    ) -> None:
        self.existing = existing
        self.upsert_error = upsert_error
        self.get_calls: list[dict[str, Any]] = []
        self.reserve_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []

    async def get_mapping(
        self,
        *,
        platform: str,
        faq_draft_id: str,
        faq_item_id: str,
        scope: TenantScope,
    ) -> MacroWritebackMapping | None:
        self.get_calls.append({
            "platform": platform,
            "faq_draft_id": faq_draft_id,
            "faq_item_id": faq_item_id,
            "scope": scope,
        })
        return self.existing

    async def upsert_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        self.upsert_calls.append({"mapping": mapping, "scope": scope})
        if self.upsert_error is not None:
            raise self.upsert_error
        return mapping

    async def reserve_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        self.reserve_calls.append({"mapping": mapping, "scope": scope})
        return mapping


class _Transport:
    def __init__(self, *responses: Mapping[str, Any] | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers),
            "json": dict(json),
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _credentials() -> ZendeskMacroCredentials:
    return ZendeskMacroCredentials(
        subdomain="example",
        email="agent@example.com",
        api_token="secret-token",
    )


def _macro(
    *,
    draft_id: str = "11111111-1111-1111-1111-111111111111",
    item_id: str = "faq-draft-1:item-1",
    title: str = "Why was I charged twice?",
) -> SupportMacroDraft:
    return SupportMacroDraft(
        title=title,
        body="1. Open Billing.\n2. Compare pending and settled charges.",
        category="billing",
        faq_draft_id=draft_id,
        faq_item_id=item_id,
        source_ids=("ticket-1",),
    )


@pytest.mark.asyncio
async def test_zendesk_provider_creates_macro_and_persists_mapping() -> None:
    repo = _MappingRepo()
    transport = _Transport({
        "macro": {
            "id": 123,
            "url": "https://example.zendesk.com/api/v2/macros/123",
        }
    })
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert results[0].status == "published"
    assert results[0].external_id == "123"
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://example.zendesk.com/api/v2/macros"
    assert call["json"]["macro"]["title"] == "Why was I charged twice?"
    assert call["json"]["macro"]["actions"] == [
        {
            "field": "comment_value",
            "value": "1. Open Billing.\n2. Compare pending and settled charges.",
        },
        {"field": "comment_mode_is_public", "value": True},
    ]
    assert _decoded_auth(call["headers"]["Authorization"]) == (
        "agent@example.com/token:secret-token"
    )
    saved = repo.upsert_calls[0]["mapping"]
    assert saved.platform == ZENDESK_PLATFORM
    assert saved.external_id == "123"
    assert saved.external_url == "https://example.zendesk.com/api/v2/macros/123"
    assert saved.publish_status == "published"
    assert saved.metadata == {
        "title": "Why was I charged twice?",
        "category": "billing",
    }
    assert "secret-token" not in saved.as_dict()["metadata"].values()
    reserved = repo.reserve_calls[0]["mapping"]
    assert reserved.publish_status == "pending"
    assert reserved.external_id == ""


@pytest.mark.asyncio
async def test_zendesk_provider_updates_existing_macro_from_mapping() -> None:
    repo = _MappingRepo(existing=MacroWritebackMapping(
        platform=ZENDESK_PLATFORM,
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id="faq-draft-1:item-1",
        external_id="123",
    ))
    transport = _Transport({"macro": {"id": 123}})
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert results[0].status == "updated"
    assert results[0].external_id == "123"
    assert transport.calls[0]["method"] == "PUT"
    assert transport.calls[0]["url"] == "https://example.zendesk.com/api/v2/macros/123"
    assert repo.get_calls[0]["platform"] == ZENDESK_PLATFORM
    assert repo.reserve_calls == []


@pytest.mark.asyncio
async def test_zendesk_provider_fails_without_credentials_without_network_call() -> None:
    repo = _MappingRepo()
    transport = _Transport({"macro": {"id": 123}})
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(None),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert results[0].status == "failed"
    assert results[0].error == "zendesk_credentials_missing"
    assert transport.calls == []
    assert repo.get_calls == []


@pytest.mark.asyncio
async def test_zendesk_provider_reconciles_pending_mapping_before_update() -> None:
    repo = _MappingRepo(existing=MacroWritebackMapping(
        platform=ZENDESK_PLATFORM,
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id="faq-draft-1:item-1",
        external_id="",
        publish_status="pending",
        metadata={
            "title": "Why was I charged twice?",
            "category": "billing",
        },
    ))
    transport = _Transport(
        {
            "macros": [
                {
                    "id": 123,
                    "url": "https://example.zendesk.com/api/v2/macros/123",
                    "title": " Why was I charged twice? ",
                }
            ]
        },
        {"macro": {"id": 123}},
    )
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert results[0].status == "updated"
    assert results[0].external_id == "123"
    assert [call["method"] for call in transport.calls] == ["GET", "PUT"]
    assert transport.calls[0]["url"] == (
        "https://example.zendesk.com/api/v2/macros/search?"
        "query=Why+was+I+charged+twice%3F"
    )
    assert transport.calls[0]["json"] == {}
    assert transport.calls[1]["url"] == "https://example.zendesk.com/api/v2/macros/123"
    assert repo.reserve_calls == []
    assert len(repo.upsert_calls) == 2
    reconciled = repo.upsert_calls[0]["mapping"]
    saved = repo.upsert_calls[1]["mapping"]
    assert reconciled.external_id == "123"
    assert reconciled.external_url == "https://example.zendesk.com/api/v2/macros/123"
    assert saved.external_url == "https://example.zendesk.com/api/v2/macros/123"


@pytest.mark.asyncio
async def test_zendesk_provider_refuses_to_repost_when_pending_mapping_has_no_match() -> None:
    repo = _MappingRepo(existing=MacroWritebackMapping(
        platform=ZENDESK_PLATFORM,
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id="faq-draft-1:item-1",
        external_id="",
        publish_status="pending",
    ))
    transport = _Transport({"macros": []})
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert results[0].status == "failed"
    assert results[0].error == "zendesk_macro_mapping_pending_reconcile"
    assert [call["method"] for call in transport.calls] == ["GET"]
    assert repo.reserve_calls == []
    assert repo.upsert_calls == []


@pytest.mark.asyncio
async def test_zendesk_provider_rejects_pending_mapping_title_near_miss() -> None:
    repo = _MappingRepo(existing=MacroWritebackMapping(
        platform=ZENDESK_PLATFORM,
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id="faq-draft-1:item-1",
        external_id="",
        publish_status="pending",
        metadata={"title": "Why was I charged twice?"},
    ))
    transport = _Transport({
        "macros": [
            {
                "id": 999,
                "url": "https://example.zendesk.com/api/v2/macros/999",
                "title": "Why was I charged twice? old",
            }
        ]
    })
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert results[0].status == "failed"
    assert results[0].error == "zendesk_macro_mapping_pending_reconcile"
    assert [call["method"] for call in transport.calls] == ["GET"]
    assert repo.reserve_calls == []
    assert repo.upsert_calls == []


@pytest.mark.asyncio
async def test_zendesk_provider_marks_create_then_mapping_failure_without_reposting() -> None:
    repo = _MappingRepo(upsert_error=RuntimeError("database down"))
    transport = _Transport({"macro": {"id": 123}}, {"macros": []})
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    first = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))
    repo.existing = repo.reserve_calls[0]["mapping"]
    second = await provider.publish([_macro()], scope=TenantScope(account_id="acct-1"))

    assert first[0].status == "failed"
    assert first[0].external_id == "123"
    assert first[0].error == "zendesk_mapping_persist_failed"
    assert second[0].status == "failed"
    assert second[0].error == "zendesk_macro_mapping_pending_reconcile"
    assert [call["method"] for call in transport.calls] == ["POST", "GET"]


@pytest.mark.asyncio
async def test_zendesk_provider_isolates_per_item_transport_failures() -> None:
    repo = _MappingRepo()
    transport = _Transport(
        ZendeskMacroTransportError(status_code=429),
        {"macro": {"id": 456}},
    )
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(_credentials()),
        mapping_repository=repo,
        transport=transport,
    )

    results = await provider.publish(
        [
            _macro(item_id="faq-draft-1:item-1", title="First"),
            _macro(item_id="faq-draft-1:item-2", title="Second"),
        ],
        scope=TenantScope(account_id="acct-1"),
    )

    assert [result.status for result in results] == ["failed", "published"]
    assert results[0].error == "zendesk_request_failed: status=429"
    assert results[1].external_id == "456"
    assert len(transport.calls) == 2
    assert len(repo.upsert_calls) == 1
    assert len(repo.reserve_calls) == 2


def test_zendesk_credentials_repr_redacts_api_token_and_builds_base_url() -> None:
    credentials = ZendeskMacroCredentials(
        subdomain="example.zendesk.com",
        email="agent@example.com",
        api_token="secret-token",
    )

    assert credentials.normalized_base_url() == "https://example.zendesk.com"
    assert "secret-token" not in repr(credentials)
    assert _decoded_auth(credentials.authorization_header()) == (
        "agent@example.com/token:secret-token"
    )


def _decoded_auth(header: str) -> str:
    assert header.startswith("Basic ")
    return base64.b64decode(header.removeprefix("Basic ")).decode("utf-8")
