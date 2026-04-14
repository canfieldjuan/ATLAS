import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


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
        neo4j_uri="bolt://neo4j:7687",
        neo4j_user="neo4j",
        neo4j_password="password123",
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
