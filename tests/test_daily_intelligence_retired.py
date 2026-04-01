from types import SimpleNamespace

import pytest

from atlas_brain.autonomous.tasks import daily_intelligence


@pytest.mark.asyncio
async def test_daily_intelligence_returns_retired_skip_message():
    result = await daily_intelligence.run(SimpleNamespace())
    assert "retired" in str(result.get("_skip_synthesis") or "").lower()
