import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# main.py belongs to the standalone graphiti-wrapper service; its service deps
# (neo4j, graphiti_core) are not brain runtime deps. Skip the whole module only
# when those named optional deps are absent (e.g. a brain-only env). When they
# are present the health/readiness tests below are fully mocked (no live
# services), so they stay in the normal unit lane -- no e2e marker -- and a real
# import regression in main.py or its local modules still fails because
# exec_module is not wrapped in a broad guard.
pytest.importorskip("neo4j")
pytest.importorskip("graphiti_core")

_ROOT = Path(__file__).resolve().parents[1]
_WRAPPER_DIR = _ROOT / "graphiti-wrapper"
_MODULE_PATH = _WRAPPER_DIR / "main.py"

if str(_WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(_WRAPPER_DIR))

_SPEC = importlib.util.spec_from_file_location("graphiti_wrapper_main", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _make_settings():
    return SimpleNamespace(
        openai_api_key="test-openai-key",
        openai_base_url="https://openrouter.ai/api/v1",
        model_name="anthropic/claude-haiku-4-5",
        neo4j_uri="bolt://neo4j:7687",
        neo4j_user="neo4j",
        neo4j_password="password123",
        neo4j_database="neo4j",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
        embedder_api_key=None,
        embedder_base_url=None,
        embedder_device="cpu",
        embedder_batch_size=32,
        embedder_embedding_dim=1536,
    )


@pytest.mark.asyncio
async def test_ensure_neo4j_ready_verifies_connectivity():
    fake_driver = MagicMock()
    fake_driver.verify_connectivity = AsyncMock()
    fake_driver.close = AsyncMock()

    with patch.object(
        _MODULE.AsyncGraphDatabase,
        "driver",
        return_value=fake_driver,
    ) as driver_factory:
        await _MODULE._ensure_neo4j_ready(_make_settings())

    driver_factory.assert_called_once_with(
        "bolt://neo4j:7687",
        auth=("neo4j", "password123"),
        connection_timeout=_MODULE._NEO4J_HEALTH_CONNECTION_TIMEOUT_SECONDS,
        max_transaction_retry_time=0,
    )
    fake_driver.verify_connectivity.assert_awaited_once_with()
    fake_driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_neo4j_ready_raises_503_when_query_fails():
    fake_driver = MagicMock()
    fake_driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("bolt down"))
    fake_driver.close = AsyncMock()

    with patch.object(_MODULE.AsyncGraphDatabase, "driver", return_value=fake_driver):
        with pytest.raises(HTTPException) as exc_info:
            await _MODULE._ensure_neo4j_ready(_make_settings())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Neo4j unavailable"
    fake_driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_healthcheck_uses_neo4j_readiness_gate():
    settings = _make_settings()

    with patch.object(_MODULE, "_ensure_neo4j_ready", AsyncMock()) as readiness_mock:
        response = await _MODULE.healthcheck(settings=settings)

    readiness_mock.assert_awaited_once_with(settings)
    assert response.status == "healthy"


@pytest.mark.asyncio
async def test_health_raises_503_when_neo4j_unavailable():
    settings = _make_settings()

    with patch.object(
        _MODULE,
        "_ensure_neo4j_ready",
        AsyncMock(side_effect=HTTPException(status_code=503, detail="Neo4j unavailable")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _MODULE.health(settings=settings)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Neo4j unavailable"


@pytest.mark.asyncio
async def test_startup_event_starts_embedder_preload_in_background():
    settings = SimpleNamespace(
        embedder_preload_on_startup=True,
        embedder_preload_blocking=False,
    )
    task = MagicMock()
    task.add_done_callback = MagicMock()

    def _fake_create_task(coro):
        coro.close()
        return task

    with (
        patch.object(_MODULE, "Settings", return_value=settings),
        patch.object(_MODULE.asyncio, "create_task", side_effect=_fake_create_task) as create_task_mock,
        patch.object(_MODULE, "_preload_embedder", AsyncMock()) as preload_mock,
    ):
        await _MODULE.startup_event()

    create_task_mock.assert_called_once()
    preload_mock.assert_called_once_with(settings)
    task.add_done_callback.assert_called_once_with(_MODULE._log_embedder_preload_result)


@pytest.mark.asyncio
async def test_startup_event_can_block_on_preload_when_configured():
    settings = SimpleNamespace(
        embedder_preload_on_startup=True,
        embedder_preload_blocking=True,
    )

    with (
        patch.object(_MODULE, "Settings", return_value=settings),
        patch.object(_MODULE, "_preload_embedder", AsyncMock()) as preload_mock,
        patch.object(_MODULE.asyncio, "create_task") as create_task_mock,
    ):
        await _MODULE.startup_event()

    preload_mock.assert_awaited_once_with(settings)
    create_task_mock.assert_not_called()


def test_openrouter_base_url_uses_generic_openai_client():
    from llm_client_wrapper import RetryingOpenAIGenericClient, create_retrying_llm_client

    client = create_retrying_llm_client(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-haiku-4-5",
    )

    assert isinstance(client, RetryingOpenAIGenericClient)


def test_direct_openai_base_url_keeps_openai_specific_client():
    from llm_client_wrapper import RetryingOpenAIClient, create_retrying_llm_client

    client = create_retrying_llm_client(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )

    assert isinstance(client, RetryingOpenAIClient)


def test_atlas_neo4j_driver_clone_keeps_physical_database():
    driver = object.__new__(_MODULE.AtlasNeo4jDriver)
    driver._database = "neo4j"

    cloned = driver.clone(database="atlas-conversations")

    assert cloned is driver
    assert driver._database == "neo4j"


@pytest.mark.asyncio
async def test_graphiti_client_is_cached_and_schema_build_runs_once():
    settings = _make_settings()
    await _MODULE._close_graphiti_clients()

    fake_driver = MagicMock()
    fake_driver.build_indices_and_constraints = AsyncMock()
    fake_client = MagicMock()
    fake_client.driver = fake_driver
    fake_client.close = AsyncMock()

    with (
        patch.object(_MODULE, "create_embedder", return_value=object()) as embedder_mock,
        patch.object(_MODULE, "create_retrying_llm_client", return_value=object()) as llm_mock,
        patch.object(_MODULE, "AtlasNeo4jDriver", return_value=fake_driver) as driver_mock,
        patch.object(_MODULE, "Graphiti", return_value=fake_client) as graphiti_mock,
    ):
        first = await _MODULE._get_or_create_graphiti_client(
            settings,
            provider=settings.embedder_provider,
            base_url=settings.openai_base_url,
            model=settings.embedder_model,
            api_key=settings.openai_api_key,
        )
        second = await _MODULE._get_or_create_graphiti_client(
            settings,
            provider=settings.embedder_provider,
            base_url=settings.openai_base_url,
            model=settings.embedder_model,
            api_key=settings.openai_api_key,
        )

    assert first is fake_client
    assert second is fake_client
    embedder_mock.assert_called_once()
    llm_mock.assert_called_once()
    driver_mock.assert_called_once_with(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="password123",
        database="neo4j",
    )
    graphiti_mock.assert_called_once_with(
        graph_driver=fake_driver,
        embedder=embedder_mock.return_value,
        llm_client=llm_mock.return_value,
    )
    fake_driver.build_indices_and_constraints.assert_awaited_once_with(delete_existing=False)

    await _MODULE._close_graphiti_clients()
    fake_client.close.assert_awaited_once()
