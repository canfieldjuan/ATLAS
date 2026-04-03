from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_eligible_review_filters_uses_stable_review_timestamp():
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    sql = mod._eligible_review_filters(window_param=1, source_param=2)

    assert "COALESCE(reviewed_at, imported_at, enriched_at)" in sql
    assert "enrichment_status = 'enriched'" in sql
    assert "source = ANY($2::text[])" in sql


def test_eligible_review_filters_respects_alias_for_stable_timestamp():
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    sql = mod._eligible_review_filters(window_param=3, source_param=4, alias="r")

    assert "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)" in sql
    assert "r.source = ANY($4::text[])" in sql


@pytest.mark.asyncio
async def test_fetch_vendor_provenance_uses_stable_review_window(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            side_effect=[
                [{"vendor_name": "Acme", "source": "g2", "cnt": 2}],
                [{
                    "vendor_name": "Acme",
                    "sample_ids": ["r1", "r2"],
                    "window_start": "2026-03-01T00:00:00+00:00",
                    "window_end": "2026-03-18T00:00:00+00:00",
                }],
            ],
        ),
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "intelligence_source_allowlist",
        "g2,reddit",
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "deprecated_review_sources",
        "",
        raising=False,
    )

    result = await mod._fetch_vendor_provenance(pool, 90)

    assert result["Acme"]["source_distribution"] == {"g2": 2}
    sample_sql = pool.fetch.await_args_list[1].args[0]
    assert "MIN(COALESCE(reviewed_at, imported_at, enriched_at)) AS window_start" in sample_sql
    assert "MAX(COALESCE(reviewed_at, imported_at, enriched_at)) AS window_end" in sample_sql
