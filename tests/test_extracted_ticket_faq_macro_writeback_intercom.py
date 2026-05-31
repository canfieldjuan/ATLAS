from __future__ import annotations

from typing import Any, Mapping

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroWritebackMapping, SupportMacroDraft, build_macro_writeback_preview
from extracted_content_pipeline.faq_macro_writeback_intercom import (
    INTERCOM_PLATFORM,
    IntercomMacroCredentials,
    IntercomMacroPublishProvider,
    IntercomMacroTransportError,
    StaticIntercomMacroCredentialsProvider,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


class _Repo:
    def __init__(self, existing: MacroWritebackMapping | None = None) -> None:
        self.existing = existing
        self.get_calls: list[dict[str, Any]] = []
        self.reserve_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []

    async def get_mapping(self, **kwargs):
        self.get_calls.append(kwargs)
        return self.existing

    async def reserve_mapping(self, mapping: MacroWritebackMapping, *, scope: TenantScope):
        self.reserve_calls.append({"mapping": mapping, "scope": scope})
        return mapping

    async def upsert_mapping(self, mapping: MacroWritebackMapping, *, scope: TenantScope):
        self.upsert_calls.append({"mapping": mapping, "scope": scope})
        return mapping


class _Transport:
    def __init__(self, *responses: Mapping[str, Any] | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "json": dict(json)})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _macro(item_id: str = "draft-1:item-1", title: str = "How do I export?") -> SupportMacroDraft:
    return SupportMacroDraft(title=title, body="Open Reports, then choose Export.", category="exports", faq_draft_id="11111111-1111-1111-1111-111111111111", faq_item_id=item_id)


def _provider(repo: _Repo, transport: _Transport, credentials: IntercomMacroCredentials | None = None):
    return IntercomMacroPublishProvider(
        credentials_provider=StaticIntercomMacroCredentialsProvider(credentials or IntercomMacroCredentials(access_token="secret-token")),
        mapping_repository=repo,
        transport=transport,
    )


@pytest.mark.asyncio
async def test_intercom_provider_creates_saved_reply_and_persists_mapping() -> None:
    repo = _Repo()
    transport = _Transport({"type": "macro", "id": "macro-1"})

    result = (await _provider(repo, transport).publish([_macro()], scope=TenantScope(account_id="acct-1")))[0]

    assert result.status == "published"
    assert result.external_id == "macro-1"
    assert transport.calls[0] == {
        "method": "POST",
        "url": "https://api.intercom.io/macros",
        "headers": {
            "Authorization": "Bearer secret-token",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Intercom-Version": "Unstable",
            "Idempotency-Key": "atlas-faq-macro-writeback:acct-1:11111111-1111-1111-1111-111111111111:draft-1:item-1",
        },
        "json": {"name": "How do I export?", "body_text": "Open Reports, then choose Export.", "visible_to": "everyone", "available_on": ["inbox", "messenger"]},
    }
    assert repo.reserve_calls[0]["mapping"].publish_status == "pending"
    saved = repo.upsert_calls[0]["mapping"]
    assert saved.platform == INTERCOM_PLATFORM
    assert saved.external_id == "macro-1"
    assert "secret-token" not in saved.as_dict()["metadata"].values()


@pytest.mark.asyncio
async def test_intercom_provider_updates_existing_and_refuses_pending_mapping() -> None:
    existing = MacroWritebackMapping(platform=INTERCOM_PLATFORM, faq_draft_id="11111111-1111-1111-1111-111111111111", faq_item_id="draft-1:item-1", external_id="macro-1")
    repo = _Repo(existing)
    transport = _Transport({"type": "macro", "id": "macro-1"})

    updated = (await _provider(repo, transport).publish([_macro()], scope=TenantScope(account_id="acct-1")))[0]

    assert updated.status == "updated"
    assert transport.calls[0]["method"] == "PUT"
    assert repo.reserve_calls == []

    repo = _Repo(MacroWritebackMapping(platform=INTERCOM_PLATFORM, faq_draft_id=existing.faq_draft_id, faq_item_id=existing.faq_item_id, external_id="", publish_status="pending"))
    failed = (await _provider(repo, _Transport({"type": "macro", "id": "macro-1"})).publish([_macro()], scope=TenantScope(account_id="acct-1")))[0]
    assert failed.status == "failed"
    assert failed.error == "intercom_macro_mapping_pending_reconcile"
    assert repo.reserve_calls == []


@pytest.mark.asyncio
async def test_intercom_provider_fails_closed_for_missing_credentials_and_bad_envelope() -> None:
    repo = _Repo()
    transport = _Transport({"type": "macro", "id": "macro-1"})
    provider = IntercomMacroPublishProvider(
        credentials_provider=StaticIntercomMacroCredentialsProvider(None),
        mapping_repository=repo,
        transport=transport,
    )

    missing = (await provider.publish([_macro()], scope=TenantScope(account_id="acct-1")))[0]
    assert missing.status == "failed"
    assert missing.error == "intercom_credentials_missing"
    assert repo.get_calls == []
    assert transport.calls == []

    repo = _Repo()
    malformed = (await _provider(repo, _Transport({"type": "list", "data": []})).publish([_macro()], scope=TenantScope(account_id="acct-1")))[0]
    assert malformed.status == "failed"
    assert malformed.error == "intercom_macro_response_invalid"
    assert repo.upsert_calls == []


@pytest.mark.asyncio
async def test_intercom_provider_isolates_per_item_transport_failures() -> None:
    results = await _provider(
        _Repo(),
        _Transport(IntercomMacroTransportError(status_code=429), {"type": "macro", "id": "macro-2"}),
    ).publish([_macro(item_id="draft-1:item-1"), _macro(item_id="draft-1:item-2")], scope=TenantScope(account_id="acct-1"))

    assert [item.status for item in results] == ["failed", "published"]
    assert results[0].error == "intercom_request_failed: status=429"
    assert results[1].external_id == "macro-2"


def test_macro_writeback_double_gate_blocks_near_misses_and_allows_verified_approved() -> None:
    approved = TicketFAQDraft(
        target_id="target-1", target_mode="vendor_retention", title="FAQ", markdown="", id="11111111-1111-1111-1111-111111111111", status="approved",
        items=(
            {"question": "Allowed?", "answer_evidence_status": "resolution_evidence", "resolution_text": "Yes."},
            {"question": "Unverified?", "answer_evidence_status": "needs_review", "resolution_text": "No."},
        ),
    )
    unapproved = TicketFAQDraft(
        target_id="target-1", target_mode="vendor_retention", title="FAQ", markdown="", id="22222222-2222-2222-2222-222222222222", status="draft_needs_review",
        items=({"question": "Draft?", "answer_evidence_status": "resolution_evidence", "resolution_text": "No."},),
    )

    preview = build_macro_writeback_preview([approved, unapproved])

    assert [macro.title for macro in preview.macros] == ["Allowed?"]
    assert [item.reason for item in preview.skipped] == ["answer_not_verified", "draft_not_approved"]
