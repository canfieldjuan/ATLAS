import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "test_call_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("test_call_pipeline", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _install_module(monkeypatch, name: str, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_init_llm_activates_configured_ollama_backend(monkeypatch):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            default_model="ollama",
            ollama_model="llama3.1",
            ollama_url="http://localhost:11434",
        )
    )
    registry = SimpleNamespace(activate=MagicMock(), get_active=MagicMock(return_value=object()))

    _install_module(monkeypatch, 'atlas_brain.config', settings=settings)
    _install_module(monkeypatch, 'atlas_brain.services', llm_registry=registry)

    _MODULE._init_llm()

    registry.activate.assert_called_once_with(
        "ollama",
        model="llama3.1",
        base_url="http://localhost:11434",
    )
    registry.get_active.assert_called_once_with()


def test_init_llm_exits_when_no_active_llm(monkeypatch):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            default_model="ollama",
            ollama_model="llama3.1",
            ollama_url="http://localhost:11434",
        )
    )
    registry = SimpleNamespace(activate=MagicMock(), get_active=MagicMock(return_value=None))

    _install_module(monkeypatch, "atlas_brain.config", settings=settings)
    _install_module(monkeypatch, "atlas_brain.services", llm_registry=registry)

    with pytest.raises(SystemExit) as exc:
        _MODULE._init_llm()

    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_init_db_returns_pool_when_initialize_fails(monkeypatch, capsys):
    pool = SimpleNamespace(is_initialized=False, initialize=AsyncMock(side_effect=RuntimeError("db down")))

    _install_module(monkeypatch, "atlas_brain.storage.database", get_db_pool=lambda: pool)

    result = await _MODULE._init_db()

    assert result is pool
    assert "DB: unavailable (db down) -- will skip DB step" in capsys.readouterr().out


def test_get_business_context_returns_none_when_context_missing(monkeypatch, capsys):
    router = SimpleNamespace(get_context=MagicMock(return_value=None))

    _install_module(monkeypatch, "atlas_brain.comms.context", get_context_router=lambda: router)

    assert _MODULE._get_business_context() is None
    assert "Context: 'effingham-maids' not found -- continuing without business context" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_main_respects_skip_db_and_skip_notify(monkeypatch, capsys):
    pool = SimpleNamespace(is_initialized=False)
    extract_mock = AsyncMock(return_value=("summary", {"customer_name": "Sarah"}, [{"type": "booked", "label": "Booked"}]))
    notify_mock = AsyncMock()
    repo_factory = MagicMock()

    monkeypatch.setattr(_MODULE, "_init_llm", lambda: None)
    monkeypatch.setattr(_MODULE, "_init_db", AsyncMock(return_value=pool))
    monkeypatch.setattr(_MODULE, "_get_business_context", lambda: None)
    monkeypatch.setenv("SKIP_DB", "1")
    monkeypatch.setenv("SKIP_NOTIFY", "1")

    _install_module(
        monkeypatch,
        "atlas_brain.comms.call_intelligence",
        _extract_call_data=extract_mock,
        _notify_call_summary=notify_mock,
    )
    _install_module(
        monkeypatch,
        "atlas_brain.storage.repositories.call_transcript",
        get_call_transcript_repo=repo_factory,
    )

    await _MODULE.main()

    extract_mock.assert_awaited_once()
    notify_mock.assert_not_awaited()
    repo_factory.assert_not_called()
    output = capsys.readouterr().out
    assert "Step 2: Skipping DB (SKIP_DB=1 or DB unavailable)" in output
    assert "Step 3: Skipping ntfy (SKIP_NOTIFY=1)" in output
