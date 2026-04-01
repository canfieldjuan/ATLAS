from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_brain.autonomous.tasks import b2b_campaign_generation as mod


class _FakeSkillRegistry:
    def get(self, name):
        return SimpleNamespace(content=f"skill:{name}")


def test_campaign_batch_request_metadata_uses_versioned_minimal_replay_contract():
    entry = {
        "custom_id": "campaign:salesforce:email",
        "artifact_id": "artifact-1",
        "campaign_batch_id": "batch-1",
        "company_name": "Salesforce",
        "payload": {"channel": "email_cold", "target_mode": "vendor_retention", "tier": "report"},
        "best": {"vendor_name": "Salesforce", "opportunity_score": 91},
        "review_ids": ["r1", "r2"],
        "vendor_ctx": {"signal_summary": {"pain_distribution": [], "competitor_distribution": []}},
        "target": {"contact_email": "owner@example.com"},
        "followup_payload": {"channel": "email_followup", "target_mode": "vendor_retention"},
        "sequence_context": {"recipient_name": "Pat"},
        "trace_metadata": {"vendor_name": "Salesforce", "run_id": "run-1"},
        "max_tokens": 1600,
        "temperature": 0.2,
    }

    metadata = mod._campaign_batch_request_metadata(entry)

    assert metadata["replay_handler"] == "campaign_generation"
    assert metadata["channel"] == "email_cold"
    replay = metadata["replay_entry"]
    assert replay["contract_version"] == 1
    stored = replay["entry"]
    assert stored["artifact_id"] == "artifact-1"
    assert stored["company_name"] == "Salesforce"
    assert stored["payload"]["target_mode"] == "vendor_retention"
    assert stored["review_ids"] == ["r1", "r2"]
    assert stored["target"]["contact_email"] == "owner@example.com"
    assert "custom_id" not in stored
    assert "trace_metadata" not in stored
    assert "max_tokens" not in stored
    assert "temperature" not in stored


def test_normalize_campaign_batch_replay_entry_accepts_legacy_blob_and_hydrates_runtime_fields():
    metadata = {
        "replay_entry": {
            "artifact_id": "artifact-1",
            "campaign_batch_id": "batch-1",
            "company_name": "Acme",
            "payload": {"channel": "email_cold", "target_mode": "churning_company"},
            "best": {"vendor_name": "HubSpot", "opportunity_score": 88},
            "review_ids": ["r1"],
            "persona_context": {"pain_categories": [], "competitors_considering": []},
            "partner_id": "00000000-0000-0000-0000-000000000123",
            "persona": "ops",
            "trace_metadata": {"run_id": "legacy-run"},
            "max_tokens": 1400,
            "temperature": 0.15,
        },
        "_max_tokens": 1700,
        "_temperature": 0.3,
        "_trace_metadata": {"run_id": "meta-run", "vendor_name": "HubSpot"},
    }

    normalized = mod._normalize_campaign_batch_replay_entry(metadata)

    assert normalized is not None
    assert normalized["artifact_id"] == "artifact-1"
    assert normalized["company_name"] == "Acme"
    assert normalized["partner_id"] == "00000000-0000-0000-0000-000000000123"
    assert normalized["max_tokens"] == 1700
    assert normalized["temperature"] == 0.3
    assert normalized["trace_metadata"]["run_id"] == "meta-run"


def test_normalize_campaign_batch_replay_entry_rejects_invalid_contract():
    metadata = {
        "replay_entry": {
            "contract_version": 1,
            "entry": {
                "artifact_id": "artifact-1",
                "campaign_batch_id": "batch-1",
                "company_name": "Acme",
                "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
                "review_ids": ["r1"],
            },
        }
    }

    assert mod._normalize_campaign_batch_replay_entry(metadata) is None


@pytest.mark.asyncio
async def test_reconcile_batches_applies_items_and_submits_followups(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(return_value={"id": "item-1"})
    pool.fetch = AsyncMock(
        side_effect=[
            [{"id": "batch-1", "run_id": "run-1", "status": "ended"}],
            [{
                "id": "item-1",
                "batch_id": "batch-1",
                "custom_id": "campaign-1",
                "artifact_id": "artifact-1",
                "status": "batch_succeeded",
                "request_metadata": {
                    "replay_handler": "campaign_generation",
                    "replay_entry": {
                        "artifact_id": "artifact-1",
                        "company_name": "Acme",
                        "campaign_batch_id": "batch-1",
                        "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
                        "best": {"vendor_name": "HubSpot", "opportunity_score": 88},
                        "review_ids": ["r1"],
                        "vendor_ctx": {"signal_summary": {"pain_distribution": [], "competitor_distribution": []}},
                        "target": {"contact_email": "owner@example.com"},
                        "followup_payload": {"channel": "email_followup", "target_mode": "vendor_retention"},
                        "sequence_context": {"recipient_name": "Pat"},
                    },
                },
            }],
        ]
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "anthropic_batch_detached_enabled", True, raising=False)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.reconcile_anthropic_message_batch",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr("atlas_brain.services.llm_router.get_llm", lambda _purpose: SimpleNamespace(model="claude-test"))
    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _FakeSkillRegistry())

    resolve_content = AsyncMock(return_value={"subject": "Hello", "body": "Body", "cta": "CTA"})
    store_entry = AsyncMock(return_value={"custom_id": "followup-1"})
    mark_applied = AsyncMock(return_value=None)
    run_batch = AsyncMock(return_value=({}, {"submitted_items": 2}))
    monkeypatch.setattr(mod, "_resolve_campaign_batch_item_content", resolve_content)
    monkeypatch.setattr(mod, "_store_replayed_campaign_entry", store_entry)
    monkeypatch.setattr(mod, "_mark_campaign_batch_item_applied", mark_applied)
    monkeypatch.setattr(mod, "_run_campaign_batch", run_batch)

    result = await mod.reconcile_batches(SimpleNamespace())

    assert result["reconciled_batches"] == 1
    assert result["applied_items"] == 1
    assert result["submitted_followups"] == 2
    resolve_content.assert_awaited_once()
    store_entry.assert_awaited_once()
    mark_applied.assert_awaited_once_with(pool, item_id="item-1", applied_status="succeeded")
    run_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_batches_records_failed_items(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(return_value={"id": "item-1"})
    pool.fetch = AsyncMock(
        side_effect=[
            [{"id": "batch-1", "run_id": "run-1", "status": "ended"}],
            [{
                "id": "item-1",
                "batch_id": "batch-1",
                "custom_id": "campaign-1",
                "artifact_id": "artifact-1",
                "status": "batch_errored",
                "request_metadata": {
                    "replay_handler": "campaign_generation",
                    "replay_entry": {
                        "artifact_id": "artifact-1",
                        "company_name": "Acme",
                        "campaign_batch_id": "batch-1",
                        "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
                        "best": {"vendor_name": "HubSpot", "opportunity_score": 88},
                        "review_ids": ["r1"],
                        "vendor_ctx": {"signal_summary": {"pain_distribution": [], "competitor_distribution": []}},
                        "target": {"contact_email": "owner@example.com"},
                        "followup_payload": {"channel": "email_followup", "target_mode": "vendor_retention"},
                        "sequence_context": {"recipient_name": "Pat"},
                    },
                },
            }],
        ]
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "anthropic_batch_detached_enabled", True, raising=False)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.reconcile_anthropic_message_batch",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr("atlas_brain.services.llm_router.get_llm", lambda _purpose: SimpleNamespace(model="claude-test"))
    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _FakeSkillRegistry())
    monkeypatch.setattr(mod, "_resolve_campaign_batch_item_content", AsyncMock(return_value=None))
    record_failure = AsyncMock(return_value=None)
    mark_applied = AsyncMock(return_value=None)
    monkeypatch.setattr(mod, "_record_campaign_generation_failure", record_failure)
    monkeypatch.setattr(mod, "_mark_campaign_batch_item_applied", mark_applied)

    result = await mod.reconcile_batches(SimpleNamespace())

    assert result["reconciled_batches"] == 1
    assert result["failed_items"] == 1
    record_failure.assert_awaited_once()
    mark_applied.assert_awaited_once_with(pool, item_id="item-1", applied_status="failed", error="batch_errored")


@pytest.mark.asyncio
async def test_reconcile_batches_skips_items_claimed_elsewhere(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(return_value={"id": "item-1"})
    pool.fetch = AsyncMock(
        side_effect=[
            [{"id": "batch-1", "run_id": "run-1", "status": "ended"}],
            [{
                "id": "item-1",
                "batch_id": "batch-1",
                "custom_id": "campaign-1",
                "artifact_id": "artifact-1",
                "status": "batch_succeeded",
                "request_metadata": {
                    "replay_handler": "campaign_generation",
                    "replay_entry": {
                        "artifact_id": "artifact-1",
                        "company_name": "Acme",
                        "campaign_batch_id": "batch-1",
                        "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
                        "best": {"vendor_name": "HubSpot", "opportunity_score": 88},
                        "review_ids": ["r1"],
                        "vendor_ctx": {"signal_summary": {"pain_distribution": [], "competitor_distribution": []}},
                        "target": {"contact_email": "owner@example.com"},
                        "followup_payload": {"channel": "email_followup", "target_mode": "vendor_retention"},
                        "sequence_context": {"recipient_name": "Pat"},
                    },
                },
            }],
        ]
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "anthropic_batch_detached_enabled", True, raising=False)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.reconcile_anthropic_message_batch",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr("atlas_brain.services.llm_router.get_llm", lambda _purpose: SimpleNamespace(model="claude-test"))
    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _FakeSkillRegistry())
    monkeypatch.setattr(mod, "_claim_campaign_batch_item_for_apply", AsyncMock(return_value=False))
    resolve_content = AsyncMock()
    mark_applied = AsyncMock()
    monkeypatch.setattr(mod, "_resolve_campaign_batch_item_content", resolve_content)
    monkeypatch.setattr(mod, "_mark_campaign_batch_item_applied", mark_applied)

    result = await mod.reconcile_batches(SimpleNamespace(id="task-1"))

    assert result["reconciled_batches"] == 1
    assert result["applied_items"] == 0
    assert result["failed_items"] == 0
    resolve_content.assert_not_awaited()
    mark_applied.assert_not_awaited()
