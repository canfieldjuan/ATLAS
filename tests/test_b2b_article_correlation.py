from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks.b2b_article_correlation import run


@pytest.mark.asyncio
async def test_run_explains_incomplete_core_gap(monkeypatch):
    pool = SimpleNamespace(is_initialized=True)

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_article_correlation.settings.b2b_churn.enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_article_correlation.get_db_pool",
        lambda: pool,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.has_complete_core_run_marker",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.describe_core_run_gap",
        AsyncMock(return_value="Core churn materialization is incomplete for today"),
    )

    result = await run(SimpleNamespace())

    assert result == {"_skip_synthesis": "Core churn materialization is incomplete for today"}
