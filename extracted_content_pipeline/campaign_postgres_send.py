"""Postgres-backed send runner for queued campaign emails."""

from __future__ import annotations

from typing import Any

from .campaign_ports import CampaignSender
from .campaign_postgres import (
    PostgresCampaignAuditSink,
    PostgresCampaignRepository,
    PostgresSuppressionRepository,
)
from .campaign_send import CampaignSendConfig, CampaignSendService, CampaignSendSummary
from .campaign_suppression import CampaignSuppressionService


async def send_due_campaigns_from_postgres(
    pool: Any,
    *,
    sender: CampaignSender,
    config: CampaignSendConfig | None = None,
    limit: int | None = None,
) -> CampaignSendSummary:
    """Send queued campaigns from Postgres through an injected sender."""
    effective_limit = int(
        limit
        if limit is not None
        else (config.limit if config is not None else CampaignSendConfig().limit)
    )
    if effective_limit <= 0:
        return CampaignSendSummary()
    service = CampaignSendService(
        campaigns=PostgresCampaignRepository(pool),
        suppression=CampaignSuppressionService(PostgresSuppressionRepository(pool)),
        sender=sender,
        audit=PostgresCampaignAuditSink(pool),
        config=config,
    )
    return await service.send_due(limit=effective_limit)
