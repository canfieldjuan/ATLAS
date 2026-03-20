from unittest.mock import AsyncMock

import pytest

from atlas_brain.config import settings
from atlas_brain.reasoning.graph import _node_reason, _node_synthesize, _node_triage
from atlas_brain.reasoning.reflection import run_reflection
from atlas_brain.pipelines.llm import get_pipeline_llm


def test_reasoning_prompt_exports_are_available_from_both_modules():
    from atlas_brain.reasoning import graph_prompts, prompts

    assert graph_prompts.TRIAGE_SYSTEM
    assert graph_prompts.REASONING_SYSTEM
    assert graph_prompts.SYNTHESIS_SYSTEM
    assert graph_prompts.REFLECTION_SYSTEM
    assert prompts.TRIAGE_SYSTEM == graph_prompts.TRIAGE_SYSTEM
    assert prompts.REASONING_SYSTEM == graph_prompts.REASONING_SYSTEM
    assert prompts.SYNTHESIS_SYSTEM == graph_prompts.SYNTHESIS_SYSTEM
    assert prompts.REFLECTION_SYSTEM == graph_prompts.REFLECTION_SYSTEM


@pytest.mark.asyncio
async def test_graph_triage_uses_configured_pipeline_workload(monkeypatch):
    monkeypatch.setattr(settings.reasoning, "graph_triage_workload", "openrouter")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm)
    monkeypatch.setattr(
        "atlas_brain.reasoning.graph._llm_generate",
        AsyncMock(return_value={"response": '{"priority":"high","needs_reasoning":true,"reasoning":"ok"}', "usage": {}}),
    )

    state = {"event_type": "b2b.high_intent_detected", "source": "test", "entity_type": "company", "entity_id": "Acme", "payload": {}}
    result = await _node_triage(state)

    assert calls == [{
        "workload": "openrouter",
        "auto_activate_ollama": False,
        "openrouter_model": "openai/o4-mini",
    }]
    assert result["triage_priority"] == "high"


@pytest.mark.asyncio
async def test_graph_reason_uses_configured_pipeline_workload(monkeypatch):
    monkeypatch.setattr(settings.reasoning, "graph_reasoning_workload", "openrouter")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm)
    monkeypatch.setattr(
        "atlas_brain.reasoning.graph._llm_generate",
        AsyncMock(return_value={"response": '{"connections":["pricing"],"actions":[],"rationale":"ok","should_notify":true}', "usage": {}}),
    )

    state = {"event_type": "b2b.high_intent_detected", "source": "test", "payload": {}, "b2b_churn": {}}
    result = await _node_reason(state)

    assert calls == [{
        "workload": "openrouter",
        "auto_activate_ollama": False,
        "openrouter_model": "openai/o4-mini",
    }]
    assert result["connections_found"] == ["pricing"]


@pytest.mark.asyncio
async def test_graph_synthesis_uses_configured_pipeline_workload(monkeypatch):
    monkeypatch.setattr(settings.reasoning, "graph_synthesis_workload", "openrouter")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm)
    monkeypatch.setattr(
        "atlas_brain.reasoning.graph._llm_generate",
        AsyncMock(return_value={"response": "Summary", "usage": {}}),
    )

    state = {"should_notify": True, "event_type": "b2b.high_intent_detected", "action_results": [], "rationale": "ok"}
    result = await _node_synthesize(state)

    assert calls == [{
        "workload": "openrouter",
        "auto_activate_ollama": False,
        "openrouter_model": "openai/o4-mini",
    }]
    assert result["summary"] == "Summary."


@pytest.mark.asyncio
async def test_reflection_uses_configured_pipeline_workload(monkeypatch):
    monkeypatch.setattr(settings.reasoning, "graph_reasoning_workload", "openrouter")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm)
    monkeypatch.setattr(
        "atlas_brain.reasoning.patterns.run_all_pattern_detectors",
        AsyncMock(return_value=[{"description": "signal"}]),
    )
    monkeypatch.setattr(
        "atlas_brain.reasoning.graph._llm_generate",
        AsyncMock(return_value={"response": '{"findings":[]}', "usage": {}}),
    )

    result = await run_reflection()

    assert calls == [{
        "workload": "openrouter",
        "auto_activate_ollama": False,
        "openrouter_model": "openai/o4-mini",
    }]
    assert result["findings"] == 1


def test_openrouter_workload_uses_configured_model(monkeypatch):
    monkeypatch.setattr(settings.llm, "openrouter_reasoning_model", "anthropic/claude-haiku-4-5-20251001")
    seen = []

    def _fake_try_openrouter(model=None):
        seen.append(model)
        return object()

    monkeypatch.setattr("atlas_brain.pipelines.llm._try_openrouter", _fake_try_openrouter)

    llm = get_pipeline_llm(workload="openrouter", auto_activate_ollama=False)

    assert llm is not None
    assert seen == ["anthropic/claude-haiku-4-5-20251001"]
