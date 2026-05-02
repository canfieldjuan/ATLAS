from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extracted_content_pipeline.autonomous.tasks import _b2b_batch_utils as batch_utils


def test_exact_stage_request_fingerprint_is_stable_and_request_sensitive() -> None:
    first = SimpleNamespace(
        namespace="campaign.first_attempt",
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        request_envelope={
            "messages": [{"role": "user", "content": "draft"}],
            "temperature": 0.2,
            "max_tokens": 1000,
        },
    )
    same_different_order = SimpleNamespace(
        namespace="campaign.first_attempt",
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        request_envelope={
            "max_tokens": 1000,
            "temperature": 0.2,
            "messages": [{"content": "draft", "role": "user"}],
        },
    )
    changed = SimpleNamespace(
        namespace="campaign.first_attempt",
        provider="openrouter",
        model="anthropic/claude-haiku-4-5",
        request_envelope=first.request_envelope,
    )

    assert batch_utils.exact_stage_request_fingerprint(first) == batch_utils.exact_stage_request_fingerprint(
        same_different_order
    )
    assert batch_utils.exact_stage_request_fingerprint(first) != batch_utils.exact_stage_request_fingerprint(changed)


def test_metadata_helpers_parse_flags_and_counts() -> None:
    metadata = {
        "global_enabled": "yes",
        "task_enabled": 1,
        "min_items": "0",
        "bad_int": "many",
    }
    task = SimpleNamespace(metadata=metadata)

    assert batch_utils.metadata_bool(metadata, ("global_enabled",), False) is True
    assert batch_utils.metadata_bool({"enabled": "off"}, ("enabled",), True) is False
    assert batch_utils.anthropic_batch_requested(
        task,
        global_default=False,
        task_default=False,
        global_keys=("global_enabled",),
        task_keys=("task_enabled",),
    ) is True
    assert batch_utils.anthropic_batch_min_items(
        task,
        default=5,
        keys=("min_items",),
        min_value=2,
    ) == 2
    assert batch_utils.metadata_int(metadata, ("bad_int",), 7, min_value=2) == 7


def test_anthropic_model_helpers_normalize_openrouter_models() -> None:
    llm = SimpleNamespace(name="openrouter", model="anthropic/claude-sonnet-4-5")

    assert batch_utils.anthropic_model_name("anthropic/claude-haiku-4-5") == "claude-haiku-4-5"
    assert batch_utils.anthropic_model_name("openai/gpt-4o") is None
    assert batch_utils.is_anthropic_llm(llm) is False
    assert batch_utils.anthropic_model_candidates(
        "anthropic/claude-sonnet-4-5",
        "claude-haiku-4-5",
        "anthropic/claude-sonnet-4-5",
        current_llm=llm,
    ) == ["claude-sonnet-4-5", "claude-haiku-4-5"]


def test_resolve_anthropic_batch_llm_preserves_matching_current_llm(monkeypatch) -> None:
    current = SimpleNamespace(name="anthropic", model="claude-sonnet-4-5")
    monkeypatch.setenv("EXTRACTED_PIPELINE_STANDALONE", "1")

    resolved = batch_utils.resolve_anthropic_batch_llm(
        current_llm=current,
        target_model_candidates=("claude-sonnet-4-5",),
    )

    assert resolved is current


def test_resolve_anthropic_batch_llm_uses_extracted_env_key_and_registry_slot(monkeypatch) -> None:
    monkeypatch.setenv("EXTRACTED_PIPELINE_STANDALONE", "1")
    monkeypatch.setenv("EXTRACTED_LLM_ANTHROPIC_API_KEY", "product-key")

    from extracted_content_pipeline import services
    from extracted_content_pipeline.pipelines import llm as llm_mod

    activation = {}

    class Registry:
        def get_slot(self, slot_name):
            activation["checked"] = slot_name
            return None

        def activate_slot(self, slot_name, provider, **kwargs):
            activation.update(
                {
                    "slot_name": slot_name,
                    "provider": provider,
                    "model": kwargs.get("model"),
                    "api_key": kwargs.get("api_key"),
                }
            )
            return SimpleNamespace(name=provider, model=kwargs.get("model"))

    monkeypatch.setattr(llm_mod, "get_pipeline_llm", lambda **_kwargs: None)
    monkeypatch.setattr(services, "llm_registry", Registry())

    resolved = batch_utils.resolve_anthropic_batch_llm(
        target_model_candidates=("anthropic/claude-sonnet-4-5",),
    )

    assert resolved.name == "anthropic"
    assert resolved.model == "claude-sonnet-4-5"
    assert activation == {
        "checked": "b2b_batch_anthropic::claude-sonnet-4-5",
        "slot_name": "b2b_batch_anthropic::claude-sonnet-4-5",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key": "product-key",
    }


def test_resolve_anthropic_batch_llm_fails_safe_without_registry_activation(monkeypatch) -> None:
    monkeypatch.setenv("EXTRACTED_PIPELINE_STANDALONE", "1")
    monkeypatch.setenv("EXTRACTED_LLM_ANTHROPIC_API_KEY", "product-key")

    from extracted_content_pipeline import services
    from extracted_content_pipeline.pipelines import llm as llm_mod

    monkeypatch.setattr(llm_mod, "get_pipeline_llm", lambda **_kwargs: None)
    monkeypatch.setattr(services, "llm_registry", SimpleNamespace(get_active=lambda: None))

    assert batch_utils.resolve_anthropic_batch_llm(
        target_model_candidates=("claude-sonnet-4-5",),
    ) is None


@pytest.mark.asyncio
async def test_reconcile_existing_batch_artifacts_returns_matching_success() -> None:
    fingerprint = "current-request"
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "artifact_id": "campaign-1",
                    "batch_id": "batch-1",
                    "status": "batch_succeeded",
                    "response_text": json.dumps({"subject": "Hello"}),
                    "error_text": "",
                    "custom_id": "custom-1",
                    "request_metadata": {"request_fingerprint": fingerprint},
                    "batch_status": "completed",
                    "provider_batch_id": "provider-1",
                }
            ]
        ),
    )

    result = await batch_utils.reconcile_existing_batch_artifacts(
        pool=pool,
        llm=None,
        task_name="campaign_generation",
        artifact_type="campaign_first_attempt",
        artifact_ids=["campaign-1"],
        expected_request_fingerprints={"campaign-1": fingerprint},
    )

    assert result == {
        "campaign-1": {
            "state": "succeeded",
            "status": "batch_succeeded",
            "cached": False,
            "response_text": json.dumps({"subject": "Hello"}),
            "error_text": None,
            "batch_id": "batch-1",
            "custom_id": "custom-1",
        }
    }


@pytest.mark.asyncio
async def test_reconcile_existing_batch_artifacts_skips_mismatched_fingerprint() -> None:
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "artifact_id": "campaign-1",
                    "batch_id": "batch-1",
                    "status": "batch_succeeded",
                    "response_text": "draft",
                    "error_text": "",
                    "custom_id": "custom-1",
                    "request_metadata": {"request_fingerprint": "stale-request"},
                    "batch_status": "completed",
                    "provider_batch_id": "provider-1",
                }
            ]
        ),
    )

    result = await batch_utils.reconcile_existing_batch_artifacts(
        pool=pool,
        llm=None,
        task_name="campaign_generation",
        artifact_type="campaign_first_attempt",
        artifact_ids=["campaign-1"],
        expected_request_fingerprints={"campaign-1": "current-request"},
    )

    assert result == {}


@pytest.mark.asyncio
async def test_reconcile_existing_batch_artifacts_returns_pending_provider_work() -> None:
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "artifact_id": "campaign-1",
                    "batch_id": "batch-1",
                    "status": "pending",
                    "response_text": "",
                    "error_text": "",
                    "custom_id": "custom-1",
                    "request_metadata": {},
                    "batch_status": "in_progress",
                    "provider_batch_id": "provider-1",
                }
            ]
        ),
    )

    result = await batch_utils.reconcile_existing_batch_artifacts(
        pool=pool,
        llm=None,
        task_name="campaign_generation",
        artifact_type="campaign_first_attempt",
        artifact_ids=["campaign-1"],
    )

    assert result["campaign-1"] == {
        "state": "pending",
        "status": "pending",
        "cached": False,
        "response_text": None,
        "error_text": None,
        "batch_id": "batch-1",
        "custom_id": "custom-1",
    }


@pytest.mark.asyncio
async def test_reconcile_existing_batch_artifacts_reconciles_in_progress_then_refetches(monkeypatch) -> None:
    monkeypatch.setenv("EXTRACTED_PIPELINE_STANDALONE", "1")

    from extracted_content_pipeline.services.b2b import anthropic_batch

    calls = []

    async def reconcile(**kwargs):
        calls.append(kwargs)

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            side_effect=[
                [
                    {
                        "artifact_id": "campaign-1",
                        "batch_id": "batch-1",
                        "status": "pending",
                        "response_text": "",
                        "error_text": "",
                        "custom_id": "custom-1",
                        "request_metadata": {},
                        "batch_status": "in_progress",
                        "provider_batch_id": "provider-1",
                    }
                ],
                [
                    {
                        "artifact_id": "campaign-1",
                        "batch_id": "batch-1",
                        "status": "fallback_succeeded",
                        "response_text": "draft",
                        "error_text": "",
                        "custom_id": "custom-1",
                        "request_metadata": {},
                        "batch_status": "completed",
                        "provider_batch_id": "provider-1",
                    }
                ],
            ]
        ),
    )
    llm = SimpleNamespace(name="anthropic", model="claude-sonnet-4-5")
    monkeypatch.setattr(anthropic_batch, "reconcile_anthropic_message_batch", reconcile)

    result = await batch_utils.reconcile_existing_batch_artifacts(
        pool=pool,
        llm=llm,
        task_name="campaign_generation",
        artifact_type="campaign_first_attempt",
        artifact_ids=["campaign-1"],
    )

    assert len(calls) == 1
    assert calls[0]["llm"] is llm
    assert calls[0]["local_batch_id"] == "batch-1"
    assert pool.fetch.await_count == 2
    assert result["campaign-1"]["state"] == "succeeded"
    assert result["campaign-1"]["status"] == "fallback_succeeded"
