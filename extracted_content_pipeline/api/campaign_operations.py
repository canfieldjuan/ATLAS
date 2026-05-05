"""FastAPI router factory for hosted campaign operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any

try:
    from fastapi import APIRouter, Body, HTTPException
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_generation import CampaignGenerationConfig
from ..campaign_ports import CampaignSender, LLMClient, SkillStore, TenantScope
from ..campaign_postgres_generation import generate_campaign_drafts_from_postgres
from ..campaign_postgres_analytics import refresh_campaign_analytics_from_postgres
from ..campaign_postgres_send import send_due_campaigns_from_postgres
from ..campaign_postgres_sequence_progression import (
    progress_campaign_sequences_from_postgres,
)
from ..campaign_send import CampaignSendConfig
from ..campaign_sequence_progression import CampaignSequenceProgressionConfig


PoolProvider = Callable[[], Any | Awaitable[Any]]
SenderProvider = Callable[[], CampaignSender | Awaitable[CampaignSender]]
LLMProvider = Callable[[], LLMClient | Awaitable[LLMClient]]
SkillsProvider = Callable[[], SkillStore | Awaitable[SkillStore]]
ReasoningProvider = Callable[[], Any | Awaitable[Any]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]

logger = logging.getLogger(__name__)
_ANALYTICS_ERROR_SUMMARY = "Campaign analytics refresh failed."


@dataclass(frozen=True)
class CampaignOperationsApiConfig:
    """Host-owned API defaults for campaign operation routes."""

    prefix: str = "/campaigns/operations"
    tags: tuple[str, ...] = ("campaign-operations",)
    default_send_limit: int = 20
    max_send_limit: int = 200
    send_default_from_email: str = ""
    send_default_reply_to: str | None = None
    send_unsubscribe_base_url: str = ""
    send_unsubscribe_token_secret: str = ""
    send_company_address: str = ""
    default_sequence_limit: int = 20
    max_sequence_limit: int = 200
    default_sequence_max_steps: int = 5
    max_sequence_steps: int = 20
    sequence_from_email: str = ""
    sequence_onboarding_product_name: str = ""
    sequence_temperature: float = 0.7
    default_generation_limit: int = 20
    max_generation_limit: int = 200
    generation_target_mode: str = "vendor_retention"
    generation_channel: str = "email"
    generation_channels: tuple[str, ...] = ()
    generation_skill_name: str = "digest/b2b_campaign_generation"
    generation_max_tokens: int = 1200
    generation_temperature: float = 0.4
    generation_include_source_opportunity: bool = True
    generation_opportunity_table: str = "campaign_opportunities"
    generation_vendor_targets_table: str = "vendor_targets"

    def __post_init__(self) -> None:
        _validate_default_limit(
            self.default_send_limit,
            self.max_send_limit,
            default_name="default_send_limit",
            max_name="max_send_limit",
        )
        _validate_default_limit(
            self.default_sequence_limit,
            self.max_sequence_limit,
            default_name="default_sequence_limit",
            max_name="max_sequence_limit",
        )
        _validate_default_limit(
            self.default_sequence_max_steps,
            self.max_sequence_steps,
            default_name="default_sequence_max_steps",
            max_name="max_sequence_steps",
        )
        _validate_default_limit(
            self.default_generation_limit,
            self.max_generation_limit,
            default_name="default_generation_limit",
            max_name="max_generation_limit",
        )
        if not _clean(self.generation_target_mode):
            raise ValueError("generation_target_mode is required")
        if not _clean(self.generation_channel):
            raise ValueError("generation_channel is required")
        if not _clean(self.generation_skill_name):
            raise ValueError("generation_skill_name is required")
        if self.generation_max_tokens <= 0:
            raise ValueError("generation_max_tokens must be positive")


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create campaign operation API routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _resolve_pool(pool_provider: PoolProvider) -> Any:
    pool = await _maybe_await(pool_provider())
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    if getattr(pool, "is_initialized", True) is False:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return pool


async def _resolve_sender(
    sender_provider: SenderProvider | None,
) -> CampaignSender:
    if sender_provider is None:
        raise HTTPException(status_code=503, detail="Campaign sender unavailable")
    sender = await _maybe_await(sender_provider())
    if sender is None:
        raise HTTPException(status_code=503, detail="Campaign sender unavailable")
    return sender


async def _resolve_optional(provider: Callable[[], Any | Awaitable[Any]] | None) -> Any:
    if provider is None:
        return None
    return await _maybe_await(provider())


async def _resolve_scope(
    scope_provider: ScopeProvider | None,
) -> TenantScope | Mapping[str, Any] | None:
    if scope_provider is None:
        return None
    scope = await _maybe_await(scope_provider())
    if scope is None or isinstance(scope, (TenantScope, Mapping)):
        return scope
    raise HTTPException(status_code=500, detail="Invalid tenant scope")


def _validate_default_limit(
    default: int,
    max_value: int,
    *,
    default_name: str,
    max_name: str,
) -> None:
    if default <= 0:
        raise ValueError(f"{default_name} must be positive")
    if max_value <= 0:
        raise ValueError(f"{max_name} must be positive")
    if default > max_value:
        raise ValueError(f"{default_name} must be less than or equal to {max_name}")


def _payload_limit(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    *,
    max_value: int,
) -> int:
    raw_value = payload.get(key)
    if isinstance(raw_value, bool) or isinstance(raw_value, float):
        raise HTTPException(status_code=400, detail=f"{key} must be an integer")
    try:
        value = default if raw_value is None else int(raw_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{key} must be an integer") from exc
    if value <= 0:
        raise HTTPException(status_code=400, detail=f"{key} must be greater than 0")
    if value > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be less than or equal to {max_value}",
        )
    return value


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _scope_account_id(scope: TenantScope | Mapping[str, Any] | None) -> str | None:
    if isinstance(scope, TenantScope):
        return scope.account_id
    if isinstance(scope, Mapping):
        return _clean(scope.get("account_id")) or None
    return None


def _generation_scope(
    scope: TenantScope | Mapping[str, Any] | None,
    payload: Mapping[str, Any],
) -> TenantScope | Mapping[str, Any] | None:
    scoped_account_id = _scope_account_id(scope)
    payload_account_id = _clean(payload.get("account_id")) or None
    if scoped_account_id and payload_account_id and payload_account_id != scoped_account_id:
        raise HTTPException(status_code=403, detail="account_id does not match scope")
    if scope is not None:
        return scope
    user_id = _clean(payload.get("user_id")) or None
    if payload_account_id or user_id:
        return {"account_id": payload_account_id, "user_id": user_id}
    return None


def _payload_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any] | None:
    raw_value = payload.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, Mapping):
        return raw_value
    raise HTTPException(status_code=400, detail=f"{key} must be an object")


def _payload_channels(
    payload: Mapping[str, Any],
    default: Sequence[str],
) -> tuple[str, ...]:
    raw_value = payload.get("channels")
    if raw_value is None:
        return tuple(default)
    if isinstance(raw_value, str):
        candidates: Sequence[Any] = raw_value.split(",")
    elif isinstance(raw_value, Sequence) and not isinstance(
        raw_value,
        (bytes, bytearray),
    ):
        candidates = raw_value
    else:
        raise HTTPException(status_code=400, detail="channels must be a list or string")
    return tuple(item for item in (_clean(value) for value in candidates) if item)


def _send_config(config: CampaignOperationsApiConfig, *, limit: int) -> CampaignSendConfig:
    return CampaignSendConfig(
        default_from_email=config.send_default_from_email,
        default_reply_to=config.send_default_reply_to,
        unsubscribe_base_url=config.send_unsubscribe_base_url,
        unsubscribe_token_secret=config.send_unsubscribe_token_secret,
        company_address=config.send_company_address,
        limit=limit,
    )


def _sequence_config(
    config: CampaignOperationsApiConfig,
    *,
    limit: int,
    max_steps: int,
) -> CampaignSequenceProgressionConfig:
    from_email = _clean(config.sequence_from_email)
    if not from_email:
        raise HTTPException(
            status_code=503,
            detail="Campaign sequence from_email is not configured",
        )
    return CampaignSequenceProgressionConfig(
        batch_limit=limit,
        max_steps=max_steps,
        from_email=from_email,
        onboarding_product_name=config.sequence_onboarding_product_name,
        temperature=float(config.sequence_temperature),
    )


def _generation_config(
    config: CampaignOperationsApiConfig,
    *,
    channel: str,
    channels: tuple[str, ...],
    limit: int,
) -> CampaignGenerationConfig:
    return CampaignGenerationConfig(
        skill_name=config.generation_skill_name,
        channel=channel,
        channels=channels,
        limit=limit,
        max_tokens=config.generation_max_tokens,
        temperature=float(config.generation_temperature),
        include_source_opportunity=config.generation_include_source_opportunity,
    )


def _public_analytics_result(result: Any) -> dict[str, Any]:
    data = result.as_dict()
    if data.get("error"):
        logger.warning("Campaign analytics refresh failed")
        data["error"] = _ANALYTICS_ERROR_SUMMARY
    return data


def create_campaign_operations_router(
    *,
    pool_provider: PoolProvider,
    sender_provider: SenderProvider | None = None,
    scope_provider: ScopeProvider | None = None,
    llm_provider: LLMProvider | None = None,
    skills_provider: SkillsProvider | None = None,
    reasoning_context_provider: ReasoningProvider | None = None,
    config: CampaignOperationsApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted campaign operation routes."""
    _require_fastapi()
    resolved_config = config or CampaignOperationsApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.post("/drafts/generate")
    async def generate_drafts(
        payload: dict[str, Any] | None = Body(None),
    ) -> dict[str, Any]:
        resolved_payload = payload or {}
        limit = _payload_limit(
            resolved_payload,
            "limit",
            resolved_config.default_generation_limit,
            max_value=resolved_config.max_generation_limit,
        )
        target_mode = (
            _clean(resolved_payload.get("target_mode"))
            or resolved_config.generation_target_mode
        )
        channel = _clean(resolved_payload.get("channel")) or resolved_config.generation_channel
        filters = _payload_mapping(resolved_payload, "filters")
        channels = _payload_channels(
            resolved_payload,
            resolved_config.generation_channels,
        )
        scope = _generation_scope(
            await _resolve_scope(scope_provider),
            resolved_payload,
        )
        pool = await _resolve_pool(pool_provider)
        llm = await _resolve_optional(llm_provider)
        skills = await _resolve_optional(skills_provider)
        reasoning_context = await _resolve_optional(reasoning_context_provider)
        try:
            result = await generate_campaign_drafts_from_postgres(
                pool,
                scope=scope,
                target_mode=target_mode,
                channel=channel,
                channels=channels,
                limit=limit,
                filters=filters,
                llm=llm,
                skills=skills,
                reasoning_context=reasoning_context,
                config=_generation_config(
                    resolved_config,
                    channel=channel,
                    channels=channels,
                    limit=limit,
                ),
                opportunity_table=resolved_config.generation_opportunity_table,
                vendor_targets_table=resolved_config.generation_vendor_targets_table,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.as_dict()

    @router.post("/send/queued")
    async def send_queued(
        payload: dict[str, Any] | None = Body(None),
    ) -> dict[str, Any]:
        resolved_payload = payload or {}
        limit = _payload_limit(
            resolved_payload,
            "limit",
            resolved_config.default_send_limit,
            max_value=resolved_config.max_send_limit,
        )
        pool = await _resolve_pool(pool_provider)
        sender = await _resolve_sender(sender_provider)
        result = await send_due_campaigns_from_postgres(
            pool,
            sender=sender,
            config=_send_config(resolved_config, limit=limit),
            limit=limit,
        )
        return result.as_dict()

    @router.post("/sequences/progress")
    async def progress_sequences(
        payload: dict[str, Any] | None = Body(None),
    ) -> dict[str, Any]:
        resolved_payload = payload or {}
        limit = _payload_limit(
            resolved_payload,
            "limit",
            resolved_config.default_sequence_limit,
            max_value=resolved_config.max_sequence_limit,
        )
        max_steps = _payload_limit(
            resolved_payload,
            "max_steps",
            resolved_config.default_sequence_max_steps,
            max_value=resolved_config.max_sequence_steps,
        )
        sequence_config = _sequence_config(
            resolved_config,
            limit=limit,
            max_steps=max_steps,
        )
        pool = await _resolve_pool(pool_provider)
        llm = await _resolve_optional(llm_provider)
        skills = await _resolve_optional(skills_provider)
        result = await progress_campaign_sequences_from_postgres(
            pool,
            llm=llm,
            skills=skills,
            config=sequence_config,
        )
        return result.as_dict()

    @router.post("/analytics/refresh")
    async def refresh_analytics(
        payload: dict[str, Any] | None = Body(None),
    ) -> dict[str, Any]:
        _ = payload
        pool = await _resolve_pool(pool_provider)
        result = await refresh_campaign_analytics_from_postgres(pool)
        return _public_analytics_result(result)

    return router


__all__ = [
    "CampaignOperationsApiConfig",
    "create_campaign_operations_router",
]
