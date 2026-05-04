"""Postgres-backed campaign analytics refresh runner."""

from __future__ import annotations

from typing import Any

from .campaign_analytics import (
    CampaignAnalyticsRefreshResult,
    CampaignAnalyticsRefreshService,
)
from .campaign_postgres import PostgresCampaignAuditSink, PostgresCampaignRepository


async def refresh_campaign_analytics_from_postgres(
    pool: Any,
) -> CampaignAnalyticsRefreshResult:
    """Refresh campaign analytics materialized views through Postgres ports."""

    service = CampaignAnalyticsRefreshService(
        campaigns=PostgresCampaignRepository(pool),
        audit=PostgresCampaignAuditSink(pool),
    )
    return await service.refresh()


__all__ = [
    "refresh_campaign_analytics_from_postgres",
]
