from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_temporal_engine_infer_category_uses_shared_exact_signal_adapter(monkeypatch):
    from atlas_brain.reasoning.temporal import TemporalEngine
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(
        fetchrow=AsyncMock(side_effect=AssertionError("unexpected direct fetchrow")),
    )
    read_signal = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "product_category": "CRM",
        }
    )
    monkeypatch.setattr(shared_mod, "read_vendor_signal_detail_exact", read_signal)

    engine = TemporalEngine(pool)
    category = await engine._infer_category("Zendesk")

    read_signal.assert_awaited_once_with(
        pool,
        vendor_name="Zendesk",
    )
    assert category == "CRM"

