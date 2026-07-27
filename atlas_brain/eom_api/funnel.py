"""Private endpoint used by the EOM office estimate-approval command."""

from __future__ import annotations

import re
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..services.crm_provider import get_crm_provider
from ..services.eom_lead_conversion import (
    EOMCustomerHandoff,
    EOMLeadConversionError,
    finalize_eom_customer_handoff,
)
from .funnel_auth import require_eom_funnel_actor, require_eom_funnel_api

router = APIRouter(prefix="/eom-funnel", tags=["eom-funnel"])

_APPROVAL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_MAX_SIGNED_BIGINT = 2**63 - 1


class EOMCustomerHandoffRequest(BaseModel):
    """Tracker-owned customer/site IDs; never operational estimate details."""

    model_config = ConfigDict(extra="forbid")
    contact_id: UUID
    tracker_customer_id: Annotated[
        int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)
    ]
    tracker_site_id: Annotated[
        int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)
    ]


def _crm_dependency() -> Any:
    return get_crm_provider()


def _approval_key_dependency(
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> str:
    key = idempotency_key.strip()
    if not _APPROVAL_KEY_PATTERN.fullmatch(key):
        raise HTTPException(
            status_code=422,
            detail=(
                "Idempotency-Key must be 16-128 characters and contain only "
                "letters, numbers, dot, underscore, colon, or hyphen"
            ),
        )
    return key


@router.post(
    "/customer-handoffs",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def create_customer_handoff(
    payload: EOMCustomerHandoffRequest,
    approval_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Finalize exactly one tracker-created Customer/Site against an EOM lead."""
    try:
        result = await finalize_eom_customer_handoff(
            crm,
            EOMCustomerHandoff(
                contact_id=str(payload.contact_id),
                tracker_customer_id=payload.tracker_customer_id,
                tracker_site_id=payload.tracker_site_id,
                approval_key=approval_key,
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK if bool(result.get("idempotent")) else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )
