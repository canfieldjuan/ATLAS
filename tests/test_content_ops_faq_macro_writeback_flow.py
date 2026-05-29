from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

from atlas_brain._content_ops_macro_writeback import (
    ConfigZendeskMacroCredentialsProvider,
    TenantZendeskMacroCredentialsProvider,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import (
    MacroWritebackMapping,
)
from extracted_content_pipeline.faq_macro_writeback_publish import (
    FAQMacroWritebackPublishService,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZENDESK_PLATFORM,
    ZendeskMacroCredentials,
    ZendeskMacroPublishProvider,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


@dataclass(frozen=True)
class _Config:
    content_ops_zendesk_email: str = ""
    content_ops_zendesk_api_token: str = ""
    content_ops_zendesk_subdomain: str = ""
    content_ops_zendesk_base_url: str = ""


class _Pool:
    pass


class _FAQRepo:
    def __init__(self, draft: TicketFAQDraft) -> None:
        self.draft = draft
        self.get_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

    async def get_draft(
        self,
        faq_id: str,
        *,
        scope: TenantScope,
    ) -> TicketFAQDraft | None:
        self.get_calls.append({"faq_id": faq_id, "scope": scope})
        return self.draft

    async def update_status(
        self,
        faq_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        self.update_calls.append({
            "faq_id": faq_id,
            "status": status,
            "scope": scope,
        })
        return True


class _MappingRepo:
    def __init__(self) -> None:
        self.mappings: dict[tuple[str, str], MacroWritebackMapping] = {}
        self.get_calls: list[dict[str, object]] = []
        self.reserve_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []

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
        return self.mappings.get((faq_draft_id, faq_item_id))

    async def reserve_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        self.reserve_calls.append({"mapping": mapping, "scope": scope})
        key = (mapping.faq_draft_id, mapping.faq_item_id)
        existing = self.mappings.get(key)
        if existing is not None:
            return existing
        self.mappings[key] = mapping
        return mapping

    async def upsert_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        self.upsert_calls.append({"mapping": mapping, "scope": scope})
        self.mappings[(mapping.faq_draft_id, mapping.faq_item_id)] = mapping
        return mapping

    async def list_pending_mappings(
        self,
        *,
        platform: str,
        scope: TenantScope,
        limit: int,
    ) -> Sequence[MacroWritebackMapping]:
        del platform, scope, limit
        return ()


class _ZendeskTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        if method == "POST":
            return {
                "macro": {
                    "id": "macro-1",
                    "url": "https://acme.zendesk.com/api/v2/macros/macro-1",
                }
            }
        if method == "PUT":
            return {
                "macro": {
                    "id": url.rsplit("/", 1)[-1],
                    "url": url,
                }
            }
        raise AssertionError(f"unexpected Zendesk method: {method}")


@pytest.mark.asyncio
async def test_faq_macro_writeback_publish_flow_uses_tenant_credentials_and_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain import _content_ops_zendesk_credentials

    scope = TenantScope(
        account_id="11111111-1111-1111-1111-111111111111",
        user_id="user-1",
    )
    credentials = ZendeskMacroCredentials(
        email="agent@example.com",
        api_token="tenant-token",
        subdomain="acme",
    )
    lookup_calls: list[dict[str, object]] = []

    async def lookup(pool: object, *, account_id: str) -> ZendeskMacroCredentials | None:
        lookup_calls.append({"pool": pool, "account_id": account_id})
        return credentials if account_id == scope.account_id else None

    monkeypatch.setattr(_content_ops_zendesk_credentials, "lookup_zendesk_credentials", lookup)
    pool = _Pool()
    mapping_repo = _MappingRepo()
    transport = _ZendeskTransport()
    faq_repo = _FAQRepo(_approved_faq_draft())
    provider = ZendeskMacroPublishProvider(
        credentials_provider=TenantZendeskMacroCredentialsProvider(
            pool=pool,
            fallback_provider=ConfigZendeskMacroCredentialsProvider(_Config()),
        ),
        mapping_repository=mapping_repo,
        transport=transport,
    )
    service = FAQMacroWritebackPublishService(
        faq_repository=faq_repo,
        provider=provider,
    )

    first = await service.publish_faq_draft("faq-draft-1", scope=scope)
    second = await service.publish_faq_draft("faq-draft-1", scope=scope)

    assert first.ok is True
    assert first.published_count == 1
    assert first.updated_count == 0
    assert first.draft_status_updated is True
    assert second.ok is True
    assert second.published_count == 0
    assert second.updated_count == 1
    assert second.draft_status_updated is True
    assert [call["method"] for call in transport.calls] == ["POST", "PUT"]
    assert transport.calls[0]["url"] == "https://acme.zendesk.com/api/v2/macros"
    assert transport.calls[1]["url"] == "https://acme.zendesk.com/api/v2/macros/macro-1"
    assert transport.calls[0]["json"] == {
        "macro": {
            "title": "Why was I charged twice?",
            "active": True,
            "description": "FAQ macro generated from billing support tickets.",
            "actions": [
                {
                    "field": "comment_value",
                    "value": "Open Billing and compare settled charges.",
                },
                {"field": "comment_mode_is_public", "value": True},
            ],
        }
    }
    assert transport.calls[0]["headers"]["Authorization"] == credentials.authorization_header()
    assert lookup_calls == [
        {"pool": pool, "account_id": scope.account_id},
        {"pool": pool, "account_id": scope.account_id},
    ]
    assert faq_repo.update_calls == [
        {"faq_id": "faq-draft-1", "status": "published", "scope": scope},
        {"faq_id": "faq-draft-1", "status": "published", "scope": scope},
    ]
    assert mapping_repo.reserve_calls[0]["scope"] == scope
    assert mapping_repo.upsert_calls[0]["scope"] == scope
    saved_mapping = mapping_repo.mappings[("faq-draft-1", "faq-item-1")]
    assert saved_mapping.platform == ZENDESK_PLATFORM
    assert saved_mapping.external_id == "macro-1"
    assert saved_mapping.metadata["title"] == "Why was I charged twice?"


def _approved_faq_draft() -> TicketFAQDraft:
    return TicketFAQDraft(
        id="faq-draft-1",
        target_id="ticket-faq-report",
        target_mode="support_ticket_faq",
        title="Saved FAQ report",
        markdown="# Saved FAQ report",
        items=(
            {
                "faq_item_id": "faq-item-1",
                "topic": "billing",
                "question": "Why was I charged twice?",
                "resolution_text": "Open Billing and compare settled charges.",
                "answer_evidence_status": "resolution_evidence",
            },
        ),
        source_count=1,
        ticket_source_count=1,
        status="approved",
    )
