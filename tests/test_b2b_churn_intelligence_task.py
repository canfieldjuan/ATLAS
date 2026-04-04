from types import SimpleNamespace
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import b2b_churn_intelligence as mod


@pytest.mark.asyncio
async def test_run_skips_when_b2b_churn_intelligence_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", False)

    task = SimpleNamespace(id=uuid4(), metadata={})

    result = await mod.run(task)

    assert result == {"_skip_synthesis": "B2B churn intelligence disabled"}


@pytest.mark.asyncio
async def test_run_allows_maintenance_override_when_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", False)

    class DummyPool:
        is_initialized = False

    monkeypatch.setattr(mod, "get_db_pool", lambda: DummyPool())

    task = SimpleNamespace(id=uuid4(), metadata={"maintenance_run": True})

    result = await mod.run(task)

    assert result == {"_skip_synthesis": "DB not ready"}
