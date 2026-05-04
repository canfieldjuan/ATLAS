"""Postgres-backed webhook ingestion runner for campaign ESP events."""

from __future__ import annotations

from typing import Any, Mapping

from .campaign_postgres import (
    PostgresCampaignAuditSink,
    PostgresCampaignRepository,
    PostgresSuppressionRepository,
)
from .campaign_suppression import CampaignSuppressionService
from .campaign_webhooks import (
    CampaignWebhookIngestionConfig,
    CampaignWebhookIngestionResult,
    CampaignWebhookIngestionService,
    ResendWebhookConfig,
    ResendWebhookVerifier,
)


async def ingest_resend_webhook_from_postgres(
    pool: Any,
    *,
    body: bytes,
    headers: Mapping[str, str],
    signing_secret: str = "",
    verify_signatures: bool = True,
    config: CampaignWebhookIngestionConfig | None = None,
) -> CampaignWebhookIngestionResult:
    """Verify a Resend webhook and ingest it through Postgres campaign ports."""
    if verify_signatures and not str(signing_secret or "").strip():
        raise ValueError(
            "signing_secret is required when signature verification is enabled"
        )

    service = CampaignWebhookIngestionService(
        verifier=ResendWebhookVerifier(
            ResendWebhookConfig(
                signing_secret=signing_secret,
                verify_signatures=verify_signatures,
            )
        ),
        campaigns=PostgresCampaignRepository(pool),
        suppression=CampaignSuppressionService(PostgresSuppressionRepository(pool)),
        audit=PostgresCampaignAuditSink(pool),
        config=config,
    )
    return await service.ingest(body=body, headers=headers)
