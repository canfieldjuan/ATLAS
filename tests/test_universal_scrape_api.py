from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.api import universal_scrape as mod


@pytest.mark.asyncio
async def test_create_job_from_file_rejects_blank_path_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_job_from_file(mod.FromFileRequest(path="   "))

    assert exc.value.status_code == 422
    assert exc.value.detail == "path is required"


@pytest.mark.asyncio
async def test_create_job_from_file_trims_path_before_load(monkeypatch):
    pool = SimpleNamespace(is_initialized=True)
    loaded = {}

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    def _load_config_file(path):
        loaded["path"] = path
        return SimpleNamespace(targets=[1, 2])

    class Scraper:
        async def run_job(self, config):
            loaded["targets"] = len(config.targets)
            return "job-1"

    monkeypatch.setattr(mod, "load_config_file", _load_config_file)
    monkeypatch.setattr(mod, "get_universal_scraper", lambda: Scraper())
    monkeypatch.setattr(mod.settings.universal_scrape, "enabled", True, raising=False)

    result = await mod.create_job_from_file(mod.FromFileRequest(path="  /tmp/config.json  "))

    assert loaded == {"path": "/tmp/config.json", "targets": 2}
    assert result == {"job_id": "job-1", "status": "pending", "targets": 2}


@pytest.mark.asyncio
async def test_list_jobs_normalizes_blank_status_and_query_defaults(monkeypatch):
    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "status = $1" not in query
            assert args == (20, 0)
            return []

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_jobs(status="   ")

    assert result == []


@pytest.mark.asyncio
async def test_get_results_normalizes_query_defaults(monkeypatch):
    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.get_results("job-1", include_raw=False)

    assert pool.fetch.await_args.args[1:] == ("job-1", 50, 0)
    assert result == []
