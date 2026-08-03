import importlib
import subprocess
import sys
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from atlas_brain.config import settings
from atlas_brain.reasoning.graph import _node_reason, _node_synthesize, _node_triage
from atlas_brain.reasoning.reflection import run_reflection


def _pipeline_llm_module():
    """Return the real pipeline LLM module, not a sibling-test stub.

    A few standalone reasoning-port tests temporarily install lightweight
    ``atlas_brain.pipelines.llm`` stubs in ``sys.modules``. In full-suite
    collection/execution order this file must not bind one of those stubs at
    module import time, because these routing tests patch internals such as
    ``_try_openrouter`` and then call the real ``get_pipeline_llm``.
    """

    module = sys.modules.get("atlas_brain.pipelines.llm")
    if module is not None and not hasattr(module, "_try_openrouter"):
        sys.modules.pop("atlas_brain.pipelines.llm", None)
        package = sys.modules.get("atlas_brain.pipelines")
        if package is not None and not getattr(package, "__file__", None):
            sys.modules.pop("atlas_brain.pipelines", None)
    return importlib.import_module("atlas_brain.pipelines.llm")


class _StaticChatService:
    def __init__(self, response: str, usage: dict | None = None) -> None:
        self.response = response
        self.usage = usage or {}
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": self.response, "usage": dict(self.usage)}


def test_pipeline_llm_helper_discards_sibling_test_stub():
    code = """
import importlib
import sys
import types

package = types.ModuleType("atlas_brain.pipelines")
package.__path__ = []
stub = types.ModuleType("atlas_brain.pipelines.llm")
stub.get_pipeline_llm = lambda **_kwargs: object()
sys.modules["atlas_brain.pipelines"] = package
sys.modules["atlas_brain.pipelines.llm"] = stub

mod = importlib.import_module("tests.test_reasoning_graph_routing")
real = mod._pipeline_llm_module()
assert real.__name__ == "atlas_brain.pipelines.llm"
assert hasattr(real, "_try_openrouter")
assert real.get_pipeline_llm.__module__ == "atlas_brain.pipelines.llm"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


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
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.reasoning, "graph_triage_workload", "triage")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []
    service = _StaticChatService(
        '{"priority":"high","needs_reasoning":true,"reasoning":"ok"}'
    )

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return service

    monkeypatch.setattr(pipeline_llm, "get_pipeline_llm", _fake_get_pipeline_llm)

    state = {"event_type": "b2b.high_intent_detected", "source": "test", "entity_type": "company", "entity_id": "Acme", "payload": {}}
    result = await _node_triage(state)

    # Triage should NOT pass graph_openrouter_model override -- uses workload default
    assert calls == [{
        "workload": "triage",
        "auto_activate_ollama": False,
        "openrouter_model": None,
    }]
    assert service.calls
    assert result["triage_priority"] == "high"


@pytest.mark.asyncio
async def test_graph_reason_uses_configured_pipeline_workload(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.reasoning, "graph_reasoning_workload", "openrouter")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []
    service = _StaticChatService(
        '{"connections":["pricing"],"actions":[],"rationale":"ok","should_notify":true}'
    )

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return service

    monkeypatch.setattr(pipeline_llm, "get_pipeline_llm", _fake_get_pipeline_llm)

    state = {"event_type": "b2b.high_intent_detected", "source": "test", "payload": {}, "b2b_churn": {}}
    result = await _node_reason(state)

    assert calls == [{
        "workload": "openrouter",
        "auto_activate_ollama": False,
        "openrouter_model": "openai/o4-mini",
    }]
    assert service.calls
    assert result["connections_found"] == ["pricing"]


@pytest.mark.asyncio
async def test_graph_synthesis_uses_configured_pipeline_workload(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.reasoning, "graph_synthesis_workload", "triage")
    monkeypatch.setattr(settings.reasoning, "graph_openrouter_model", "openai/o4-mini")
    calls = []
    service = _StaticChatService("Summary")

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return service

    monkeypatch.setattr(pipeline_llm, "get_pipeline_llm", _fake_get_pipeline_llm)

    state = {"should_notify": True, "event_type": "b2b.high_intent_detected", "action_results": [], "rationale": "ok"}
    result = await _node_synthesize(state)

    # Synthesis should NOT pass graph_openrouter_model override
    assert calls == [{
        "workload": "triage",
        "auto_activate_ollama": False,
        "openrouter_model": None,
    }]
    assert service.calls
    assert result["summary"] == "Summary."


@pytest.mark.asyncio
async def test_reflection_uses_configured_pipeline_workload(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.reasoning, "graph_synthesis_workload", "triage")
    calls = []
    service = _StaticChatService('{"findings":[]}')

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return service

    monkeypatch.setattr(pipeline_llm, "get_pipeline_llm", _fake_get_pipeline_llm)
    monkeypatch.setattr(
        "atlas_brain.reasoning.patterns.run_all_pattern_detectors",
        AsyncMock(return_value=[{"description": "signal"}]),
    )

    result = await run_reflection()

    # Reflection uses graph_synthesis_workload (triage/Haiku), no model override
    assert len(calls) == 1
    assert calls[0]["workload"] == "triage"
    assert calls[0]["auto_activate_ollama"] is False
    assert service.calls
    assert service.calls[0]["response_format"] == {"type": "json_object"}
    assert result["findings"] == 1


def test_openrouter_workload_uses_configured_model(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.llm, "openrouter_reasoning_model", "anthropic/claude-haiku-4-5-20251001")
    seen = []

    def _fake_try_openrouter(model=None):
        seen.append(model)
        return object()

    monkeypatch.setattr(pipeline_llm, "_try_openrouter", _fake_try_openrouter)

    llm = pipeline_llm.get_pipeline_llm(workload="openrouter", auto_activate_ollama=False)

    assert llm is not None
    assert seen == ["anthropic/claude-haiku-4-5-20251001"]


def test_openrouter_workload_normalizes_deprecated_gpt_oss(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.llm, "openrouter_reasoning_model", "openai/gpt-oss-120b")
    seen = []

    def _fake_try_openrouter(model=None):
        seen.append(model)
        return object()

    monkeypatch.setattr(pipeline_llm, "_try_openrouter", _fake_try_openrouter)

    llm = pipeline_llm.get_pipeline_llm(workload="openrouter", auto_activate_ollama=False)

    assert llm is not None
    assert seen == ["anthropic/claude-sonnet-4-5"]


def test_synthesis_explicit_openrouter_override_uses_settings_api_key(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    from atlas_brain.services.llm.openrouter import OpenRouterLLM

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ATLAS_B2B_CHURN_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(settings.b2b_churn, "openrouter_api_key", "test-openrouter-key")

    def _fake_load(self):
        self._loaded = True

    monkeypatch.setattr(OpenRouterLLM, "load", _fake_load)

    llm = pipeline_llm.get_pipeline_llm(
        workload="synthesis",
        openrouter_model="anthropic/claude-sonnet-4",
        auto_activate_ollama=False,
    )

    assert llm is not None
    assert llm.name == "openrouter"
    assert llm.model == "anthropic/claude-sonnet-4"


def test_synthesis_strict_openrouter_does_not_fallback(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    monkeypatch.setattr(settings.llm, "openrouter_reasoning_model", "deepseek/deepseek-v3.2")
    monkeypatch.setattr(settings.llm, "openrouter_reasoning_strict", True)
    monkeypatch.setattr(pipeline_llm, "_try_openrouter", lambda model=None: None)

    calls = []

    def _record(name):
        def _inner():
            calls.append(name)
            return object()
        return _inner

    monkeypatch.setattr("atlas_brain.services.llm_router.get_reasoning_llm", _record("reasoning"))
    monkeypatch.setattr("atlas_brain.services.llm_router.get_draft_llm", _record("draft"))
    monkeypatch.setattr("atlas_brain.services.llm_router.get_triage_llm", _record("triage"))

    llm = pipeline_llm.get_pipeline_llm(workload="synthesis", auto_activate_ollama=False)

    assert llm is None
    assert calls == []


def test_anthropic_workload_uses_anthropic_primary_without_vllm_fallback(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    calls = []

    def _fake_activate_anthropic(*, fallback_from=None):
        calls.append(fallback_from)
        return object()

    monkeypatch.setattr(pipeline_llm, "_activate_anthropic", _fake_activate_anthropic)
    monkeypatch.setattr(pipeline_llm, "_activate_vllm", lambda: pytest.fail("unexpected vLLM activation"))

    llm = pipeline_llm.get_pipeline_llm(workload="anthropic", auto_activate_ollama=False)

    assert llm is not None
    assert calls == [None]


def test_vllm_workload_uses_anthropic_as_labeled_fallback(monkeypatch):
    pipeline_llm = _pipeline_llm_module()
    calls = []

    monkeypatch.setattr(pipeline_llm, "_activate_vllm", lambda: None)

    def _fake_activate_anthropic(*, fallback_from=None):
        calls.append(fallback_from)
        return object()

    monkeypatch.setattr(pipeline_llm, "_activate_anthropic", _fake_activate_anthropic)

    llm = pipeline_llm.get_pipeline_llm(workload="vllm", auto_activate_ollama=False)

    assert llm is not None
    assert calls == ["vLLM"]
