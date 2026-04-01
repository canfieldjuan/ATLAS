from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_brain.autonomous.tasks import b2b_campaign_generation as mod


class _FakeSkillRegistry:
    def get(self, name):
        return SimpleNamespace(content=f"skill:{name}")


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
                        "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
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
                        "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
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
                        "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
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
