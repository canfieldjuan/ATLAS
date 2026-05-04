"""FastAPI router factory for extracted campaign webhook ingestion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    HTMLResponse = None
    HTTPException = None
    Query = None
    Request = Any
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_postgres import PostgresSuppressionRepository
from ..campaign_postgres_webhooks import ingest_resend_webhook_from_postgres
from ..campaign_suppression import CampaignSuppressionService
from ..campaign_webhooks import (
    CampaignWebhookIngestionConfig,
    WebhookPayloadError,
    WebhookVerificationError,
)


PoolProvider = Callable[[], Any | Awaitable[Any]]
SigningSecretProvider = Callable[[], str | Awaitable[str]]


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create campaign webhook routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


@dataclass(frozen=True)
class CampaignWebhookApiConfig:
    """Host-owned API defaults for campaign webhook routes."""

    prefix: str = "/webhooks"
    tags: tuple[str, ...] = ("webhooks",)
    verify_signatures: bool = True
    record_unknown_events: bool = False
    soft_bounce_suppression_days: int = 30


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _clean_required_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field_name} is required")
    return text


async def _resolve_pool(pool_provider: PoolProvider) -> Any:
    pool = await _maybe_await(pool_provider())
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    if getattr(pool, "is_initialized", True) is False:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return pool


async def _resolve_signing_secret(
    signing_secret_provider: SigningSecretProvider | None,
) -> str:
    if signing_secret_provider is None:
        return ""
    return str(await _maybe_await(signing_secret_provider()) or "")


def create_campaign_webhook_router(
    *,
    pool_provider: PoolProvider,
    signing_secret_provider: SigningSecretProvider | None = None,
    config: CampaignWebhookApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted campaign webhook routes without Atlas globals."""
    _require_fastapi()
    resolved_config = config or CampaignWebhookApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("/unsubscribe", response_class=HTMLResponse)
    async def unsubscribe(
        email: str = Query(..., description="Email to unsubscribe"),
    ) -> HTMLResponse:
        cleaned_email = _clean_required_text(email, "email")
        pool = await _resolve_pool(pool_provider)
        suppression = CampaignSuppressionService(PostgresSuppressionRepository(pool))
        await suppression.add_suppression(
            email=cleaned_email,
            reason="unsubscribe",
            source="recipient",
        )
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2>You have been unsubscribed</h2>"
            "<p>You will no longer receive campaign emails from us.</p>"
            "</body></html>"
        )

    @router.post("/campaign-email")
    async def campaign_email_webhook(
        request: Request,
        provider: str = Query("resend", description="Webhook provider name."),
    ) -> dict[str, Any]:
        provider_name = str(provider or "").strip().lower()
        if provider_name != "resend":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported webhook provider: {provider}",
            )

        signing_secret = await _resolve_signing_secret(signing_secret_provider)
        if resolved_config.verify_signatures and not signing_secret.strip():
            raise HTTPException(
                status_code=500,
                detail="Webhook signing secret is not configured",
            )

        pool = await _resolve_pool(pool_provider)
        try:
            result = await ingest_resend_webhook_from_postgres(
                pool,
                body=await request.body(),
                headers=dict(request.headers.items()),
                signing_secret=signing_secret,
                verify_signatures=resolved_config.verify_signatures,
                config=CampaignWebhookIngestionConfig(
                    soft_bounce_suppression_days=(
                        resolved_config.soft_bounce_suppression_days
                    ),
                    record_unknown_events=resolved_config.record_unknown_events,
                ),
            )
        except WebhookVerificationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except WebhookPayloadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return result.as_dict()

    return router
