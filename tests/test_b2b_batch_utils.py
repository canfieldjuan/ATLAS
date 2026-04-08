from types import SimpleNamespace


def test_anthropic_batch_requested_prefers_task_metadata_when_global_default_is_off():
    from atlas_brain.autonomous.tasks._b2b_batch_utils import anthropic_batch_requested

    task = SimpleNamespace(
        metadata={
            "anthropic_batch_enabled": True,
            "tenant_report_anthropic_batch_enabled": True,
        }
    )

    enabled = anthropic_batch_requested(
        task,
        global_default=False,
        task_default=False,
        task_keys=("tenant_report_anthropic_batch_enabled",),
    )

    assert enabled is True


def test_resolve_anthropic_batch_llm_preserves_matching_claude_model(monkeypatch):
    from atlas_brain.autonomous.tasks._b2b_batch_utils import resolve_anthropic_batch_llm

    class FakeAnthropicLLM:
        def __init__(self, model: str = ""):
            self.model = model
            self.name = "anthropic"

    active_llm = FakeAnthropicLLM("claude-sonnet-4-5")
    activation = {}

    monkeypatch.setattr(
        "atlas_brain.services.llm.anthropic.AnthropicLLM",
        FakeAnthropicLLM,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "atlas_brain.config.settings.llm.anthropic_api_key",
        "test-key",
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm_registry.get_slot",
        lambda slot_name: activation.update({"slot_name": slot_name}) or None,
    )
    def _activate_slot(slot_name, provider, **kwargs):
        activation.update(
            {"provider": provider, "model": kwargs.get("model"), "slot_name": slot_name}
        )
        return active_llm

    monkeypatch.setattr(
        "atlas_brain.services.llm_registry.activate_slot",
        _activate_slot,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm_registry.activate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("global activate should not run for Anthropic batch LLM resolution")
        ),
    )

    resolved = resolve_anthropic_batch_llm(
        current_llm=SimpleNamespace(
            name="openrouter",
            model="anthropic/claude-sonnet-4-5",
        ),
    )

    assert activation == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "slot_name": "b2b_batch_anthropic::claude-sonnet-4-5",
    }
    assert resolved is active_llm


def test_resolve_anthropic_batch_llm_reuses_matching_slot_without_global_activation(monkeypatch):
    from atlas_brain.autonomous.tasks._b2b_batch_utils import resolve_anthropic_batch_llm

    class FakeAnthropicLLM:
        def __init__(self, model: str = ""):
            self.model = model
            self.name = "anthropic"

    slot_llm = FakeAnthropicLLM("claude-sonnet-4-5")

    monkeypatch.setattr(
        "atlas_brain.services.llm.anthropic.AnthropicLLM",
        FakeAnthropicLLM,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "atlas_brain.config.settings.llm.anthropic_api_key",
        "test-key",
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm_registry.get_slot",
        lambda slot_name: slot_llm if slot_name == "b2b_batch_anthropic::claude-sonnet-4-5" else None,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm_registry.activate_slot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("activate_slot should not run when matching slot already exists")
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm_registry.activate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("global activate should not run for Anthropic batch LLM resolution")
        ),
    )

    resolved = resolve_anthropic_batch_llm(
        current_llm=SimpleNamespace(
            name="openrouter",
            model="anthropic/claude-sonnet-4-5",
        ),
    )

    assert resolved is slot_llm
