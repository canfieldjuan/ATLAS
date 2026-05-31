from __future__ import annotations

import pytest

from atlas_brain.autonomous.tasks.faq_macro_writeback_scheduled_publish import (
    PostgresScheduledFAQMacroCandidateRepository,
    StoredOnlyZendeskMacroCredentialsProvider,
    ZendeskTenant,
    run_scheduled_faq_macro_writeback,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroWritebackMapping
from extracted_content_pipeline.faq_macro_writeback_publish import FAQMacroPublishSummary
from extracted_content_pipeline.faq_macro_writeback_zendesk import ZENDESK_PLATFORM
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


def _draft(faq_id: str, *, status: str = "approved", evidence: str = "resolution_evidence") -> TicketFAQDraft:
    return TicketFAQDraft(
        id=faq_id, target_id="target-1", target_mode="vendor_retention", title="FAQ", markdown="", status=status,
        items=({"question": f"{faq_id}?", "answer_evidence_status": evidence, "resolution_text": "Use the export button."},),
    )


class _FAQRepo:
    def __init__(self, drafts) -> None:
        self.drafts = tuple(drafts)
        self.calls = []

    async def list_drafts(self, **kwargs):
        self.calls.append(kwargs)
        return self.drafts


class _MappingRepo:
    def __init__(self, published: set[str] | None = None) -> None:
        self.published = published or set()
        self.calls = []

    async def get_mapping(self, **kwargs):
        self.calls.append(kwargs)
        faq_id = str(kwargs["faq_draft_id"])
        if faq_id not in self.published:
            return None
        return MacroWritebackMapping(platform=ZENDESK_PLATFORM, faq_draft_id=faq_id, faq_item_id=str(kwargs["faq_item_id"]), external_id=f"macro-{faq_id}", publish_status="published", metadata={"title": f"{faq_id}?", "category": ""})


class _TenantRepo:
    def __init__(self, tenants, candidates) -> None:
        self.tenants = tuple(tenants)
        self.candidates = candidates

    async def list_tenants(self, *, limit: int):
        return self.tenants[:limit]

    async def list_candidate_faq_ids(self, tenant: ZendeskTenant, *, limit: int):
        return tuple(self.candidates.get(tenant.account_id, ()))[:limit]


class _PublishService:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls = []

    async def publish_faq_draft(self, faq_id: str, *, scope: TenantScope):
        self.calls.append({"faq_id": faq_id, "scope": scope})
        if faq_id in self.failures:
            raise RuntimeError("rate limited")
        return FAQMacroPublishSummary(faq_id=faq_id, found=True, draft_status="approved", publishable_count=1, published_count=1, results=({"status": "published"},))


@pytest.mark.asyncio
async def test_candidate_selection_uses_shared_gate_and_mapping_idempotency() -> None:
    faq_repo = _FAQRepo((
        _draft("allowed"),
        _draft("unverified", evidence="needs_review"),
        _draft("unapproved", status="draft_needs_review"),
        _draft("mapped"),
    ))
    mapping_repo = _MappingRepo({"mapped"})
    repo = PostgresScheduledFAQMacroCandidateRepository(pool=object(), faq_repository=faq_repo, mapping_repository=mapping_repo)

    candidates = await repo.list_candidate_faq_ids(ZendeskTenant("acct-1"), limit=10)

    assert candidates == ("allowed",)
    assert faq_repo.calls[0]["status"] == "approved"
    assert all(call["platform"] == ZENDESK_PLATFORM for call in mapping_repo.calls)


@pytest.mark.asyncio
async def test_scheduled_publish_skips_missing_credentials_and_isolates_failures() -> None:
    tenants = (ZendeskTenant("missing", has_zendesk_credentials=False), ZendeskTenant("acct-1"))
    service = _PublishService({"faq-1"})

    result = await run_scheduled_faq_macro_writeback(
        candidate_repository=_TenantRepo(tenants, {"missing": ("bad",), "acct-1": ("faq-1", "faq-2")}),
        publish_service=service,
        max_tenants=10,
        max_drafts_per_tenant=10,
    )

    assert result["tenants_skipped_no_credentials"] == 1
    assert [call["faq_id"] for call in service.calls] == ["faq-1", "faq-2"]
    assert result["tenants"][0]["status"] == "skipped_no_zendesk_credentials"
    assert result["drafts_failed"] == 1
    assert result["drafts_published_ok"] == 1
    assert result["tenants"][1]["drafts"][0]["error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_stored_only_credentials_provider_has_no_unscoped_fallback(monkeypatch) -> None:
    calls: list[str] = []

    async def lookup(pool: object, *, account_id: str):
        calls.append(account_id)
        return None

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.faq_macro_writeback_scheduled_publish.lookup_zendesk_credentials",
        lookup,
    )
    provider = StoredOnlyZendeskMacroCredentialsProvider(pool=object())

    assert await provider.credentials_for_scope(TenantScope()) is None
    assert await provider.credentials_for_scope(TenantScope(account_id="acct-1")) is None
    assert calls == ["acct-1"]
