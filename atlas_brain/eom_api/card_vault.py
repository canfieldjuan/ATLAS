"""EOM card-vault API and separately signed Stripe webhook."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..services.eom_card_vault import (
    EOMCardVaultError,
    EOMCardVaultProvider,
    EOMCardVaultProviderError,
    EOMCardVaultService,
    EOMCardVaultSignatureError,
    StripeEOMCardVaultProvider,
)
from ..services.eom_terms_acceptance import (
    EOMTermsAcceptanceError,
    authenticate_eom_terms_token,
)
from .funnel_auth import (
    EOMCardVaultConfig,
    EOMCardVaultProviderConfig,
    EOMPublicOnboardingConfig,
    require_eom_card_vault_config,
    require_eom_card_vault_provider_config,
    require_eom_funnel_api,
    require_eom_public_onboarding_config,
)

router = APIRouter(
    prefix="/card-vault",
    tags=["eom-card-vault"],
    dependencies=[Depends(require_eom_funnel_api)],
)
webhook_router = APIRouter(tags=["eom-card-vault"])
_MAX_WEBHOOK_BYTES = 256 * 1024


class EOMCardVaultSessionRequest(BaseModel):
    """Opaque Terms bearer forwarded by the authenticated Tracker."""

    model_config = ConfigDict(extra="forbid")
    token: object


class EOMCardVaultSessionResponse(BaseModel):
    """Closed redirect projection; provider identifiers remain server-only."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enrollment_id: UUID = Field(alias="enrollmentId")
    contact_id: UUID = Field(alias="contactId")
    candidate_id: UUID = Field(alias="candidateId")
    status: Literal["pending", "ready"]
    checkout_url: str | None = Field(default=None, alias="checkoutUrl")
    checkout_expires_at: datetime | None = Field(
        default=None, alias="checkoutExpiresAt"
    )
    provider_confirmed_at: datetime | None = Field(
        default=None, alias="providerConfirmedAt"
    )
    idempotent: bool

    @model_validator(mode="after")
    def _state_matches_projection(self) -> "EOMCardVaultSessionResponse":
        pending = (
            self.checkout_url is not None
            and self.checkout_url.startswith("https://")
            and self.checkout_expires_at is not None
            and self.provider_confirmed_at is None
        )
        ready = (
            self.checkout_url is None
            and self.checkout_expires_at is None
            and self.provider_confirmed_at is not None
        )
        if (self.status == "pending" and not pending) or (
            self.status == "ready" and not ready
        ):
            raise ValueError("card-vault session state and projection do not match")
        return self


class EOMCardVaultReadinessResponse(BaseModel):
    """Card-only readiness; this is not an overall onboarding completion flag."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contact_id: UUID = Field(alias="contactId")
    audience: Literal["residential", "commercial"]
    card_required: bool = Field(alias="cardRequired")
    card_ready: bool = Field(alias="cardReady")
    reason: Literal[
        "not_required",
        "terms_not_ready",
        "first_clean_not_confirmed",
        "not_started",
        "pending",
        "ready",
    ]
    candidate_id: UUID | None = Field(default=None, alias="candidateId")
    enrollment_id: UUID | None = Field(default=None, alias="enrollmentId")
    provider_confirmed_at: datetime | None = Field(
        default=None, alias="providerConfirmedAt"
    )


class EOMCardVaultWebhookResponse(BaseModel):
    """Minimal acknowledgement for one verified Stripe event."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: str = Field(alias="eventId")
    enrollment_id: UUID | None = Field(default=None, alias="enrollmentId")
    status: Literal["ignored", "ready"]
    idempotent: bool


def _provider_dependency(
    config: EOMCardVaultProviderConfig = Depends(
        require_eom_card_vault_provider_config
    ),
) -> EOMCardVaultProvider:
    return StripeEOMCardVaultProvider(
        secret_key=config.secret_key,
        webhook_secret=config.webhook_secret,
        timeout_seconds=config.request_timeout_seconds,
    )


def _service_dependency(
    request: Request,
    provider: EOMCardVaultProvider = Depends(_provider_dependency),
) -> EOMCardVaultService:
    pool_factory = getattr(request.app.state, "eom_funnel_card_vault_pool", None)
    if callable(pool_factory):
        pool = pool_factory()
    else:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
    return EOMCardVaultService(pool=pool, provider=provider)


def _card_vault_error(exc: EOMCardVaultError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": str(exc)},
    )


@router.post(
    "/public/session",
    response_model=EOMCardVaultSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_eom_card_vault_session(
    payload: EOMCardVaultSessionRequest,
    response: Response,
    public_onboarding: EOMPublicOnboardingConfig = Depends(
        require_eom_public_onboarding_config
    ),
    config: EOMCardVaultConfig = Depends(require_eom_card_vault_config),
    service: EOMCardVaultService = Depends(_service_dependency),
) -> EOMCardVaultSessionResponse:
    """Return a hosted setup URL only after token-bound Atlas eligibility."""

    try:
        token = authenticate_eom_terms_token(
            token=payload.token,
            secret=public_onboarding.hmac_secret,
            previous_secret=public_onboarding.previous_hmac_secret,
        )
        result = await service.start_session(
            token=token,
            public_base_url=config.public_base_url,
        )
    except EOMTermsAcceptanceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except EOMCardVaultError as exc:
        raise _card_vault_error(exc) from exc
    response.status_code = (
        status.HTTP_200_OK if bool(result["idempotent"]) else status.HTTP_201_CREATED
    )
    return EOMCardVaultSessionResponse.model_validate(result)


@router.get(
    "/readiness/{contact_id}",
    response_model=EOMCardVaultReadinessResponse,
)
async def get_eom_card_vault_readiness(
    contact_id: UUID,
    service: EOMCardVaultService = Depends(_service_dependency),
) -> EOMCardVaultReadinessResponse:
    """Read provider-confirmed card state without mutating onboarding."""

    try:
        result = await service.get_readiness(contact_id=contact_id)
    except EOMCardVaultError as exc:
        raise _card_vault_error(exc) from exc
    return EOMCardVaultReadinessResponse.model_validate(result)


@webhook_router.post(
    "/webhooks/eom-card-vault",
    response_model=EOMCardVaultWebhookResponse,
)
async def receive_eom_card_vault_webhook(
    request: Request,
    stripe_signature: Annotated[
        str,
        Header(alias="Stripe-Signature", max_length=2048),
    ],
    provider: EOMCardVaultProvider = Depends(_provider_dependency),
    service: EOMCardVaultService = Depends(_service_dependency),
) -> EOMCardVaultWebhookResponse:
    """Verify the raw Stripe body before any database read or write."""

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_WEBHOOK_BYTES:
            raise HTTPException(status_code=413, detail="Stripe event is too large")
        body.extend(chunk)
    if not body or not stripe_signature.strip():
        raise HTTPException(status_code=400, detail="Stripe signature is required")
    try:
        event = provider.construct_event(bytes(body), stripe_signature)
        result = await service.confirm_checkout_session(event=event)
    except EOMCardVaultSignatureError as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature") from exc
    except EOMCardVaultProviderError as exc:
        raise HTTPException(
            status_code=503, detail="Stripe event is unavailable"
        ) from exc
    except EOMCardVaultError as exc:
        raise _card_vault_error(exc) from exc
    return EOMCardVaultWebhookResponse.model_validate(result)
