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
    service = CampaignSendService(
        campaigns=PostgresCampaignRepository(pool),
        suppression=CampaignSuppressionService(PostgresSuppressionRepository(pool)),
        sender=sender,
        audit=PostgresCampaignAuditSink(pool),
        config=config,
    )
    return await service.send_due(limit=limit)
