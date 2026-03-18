from atlas_brain.services import llm_router


def test_get_reasoning_llm_returns_existing_singleton(monkeypatch):
    fake = object()
    monkeypatch.setattr(llm_router, "_reasoning_llm", fake)

    assert llm_router.get_reasoning_llm() is fake


def test_get_reasoning_llm_lazy_initializes(monkeypatch):
    fake = object()
    monkeypatch.setattr(llm_router, "_reasoning_llm", None)
    monkeypatch.setattr(llm_router, "_reasoning_api_key", lambda: "test-key")
    monkeypatch.setattr(llm_router, "_reasoning_model", lambda: "claude-sonnet-test")
    monkeypatch.setattr(
        llm_router,
        "init_reasoning_llm",
        lambda model, api_key: fake if (model, api_key) == ("claude-sonnet-test", "test-key") else None,
    )

    assert llm_router.get_reasoning_llm() is fake


def test_get_llm_reasoning_uses_lazy_reasoning_singleton(monkeypatch):
    fake = object()
    monkeypatch.setattr(llm_router, "_reasoning_llm", None)
    monkeypatch.setattr(llm_router, "get_reasoning_llm", lambda: fake)

    assert llm_router.get_llm("reasoning") is fake
