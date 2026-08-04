"""Private endpoint used by the EOM office estimate-approval command."""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..services.eom_estimate_booking import (
    EOMEstimateBooking,
    EOMEstimateBookingError,
    schedule_eom_estimate_booking,
)
from ..services.eom_lead_conversion import (
    EOMCustomerHandoff,
    EOMLeadConversionError,
    finalize_eom_customer_handoff,
)
from ..services.crm_provider import get_crm_provider
from .funnel_auth import require_eom_funnel_actor, require_eom_funnel_api

router = APIRouter(prefix="/eom-funnel", tags=["eom-funnel"])

_APPROVAL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_LEAD_REVIEW_CURSOR_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,512}$")
# RFC 3339 date-time shape ('T'/'t' separator only; the space relaxation is
# not RFC 3339). The offset stays optional here so a naive date-time still
# reaches the window validator's dedicated timezone error.
_RFC3339_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:[Zz]|[+-]\d{2}:\d{2})?$"
)
_MAX_SIGNED_BIGINT = 2**63 - 1
_DEFAULT_LEAD_REVIEW_LIMIT = 100
_MAX_LEAD_REVIEW_LIMIT = 200


class EOMCustomerHandoffRequest(BaseModel):
    """Tracker-owned customer/site IDs; never operational estimate details."""

    model_config = ConfigDict(extra="forbid")
    contact_id: UUID
    tracker_customer_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]
    tracker_site_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]


class EOMEstimateBookingRequest(BaseModel):
    """The office-selected estimate appointment window for one EOM lead."""

    model_config = ConfigDict(extra="forbid")

    scheduled_start: datetime
    scheduled_end: datetime
    calendar_id: Annotated[
        str | None, Field(default=None, min_length=1, max_length=256)
    ]
    notes: Annotated[str | None, Field(default=None, max_length=1000)]

    @field_validator("scheduled_start", "scheduled_end", mode="before")
    @classmethod
    def _require_datetime_strings(cls, value: Any) -> Any:
        # Pydantic's lax mode coerces JSON numbers AND digit-only strings
        # (epoch seconds, e.g. "3600") into UTC-aware datetimes, which would
        # pass the timezone/ordering checks as a 1970 appointment. Only
        # strings with RFC 3339 date-time syntax are valid at this boundary.
        if not isinstance(value, str) or not _RFC3339_DATETIME_PATTERN.fullmatch(
            value
        ):
            raise ValueError("must be an RFC 3339 date-time string")
        return value

    @model_validator(mode="after")
    def _validate_window(self) -> "EOMEstimateBookingRequest":
        if self.scheduled_start.tzinfo is None:
            raise ValueError("scheduled_start must include a timezone")
        if self.scheduled_end.tzinfo is None:
            raise ValueError("scheduled_end must include a timezone")
        if self.scheduled_end <= self.scheduled_start:
            raise ValueError("scheduled_end must be after scheduled_start")
        return self


class EOMLeadReviewItem(BaseModel):
    """The only CRM identity data the office-review queue may expose."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contact_id: UUID = Field(serialization_alias="contactId")
    full_name: str = Field(serialization_alias="fullName")
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    source: str | None = None
    lead_stage: str = Field(serialization_alias="leadStage")
    created_at: datetime = Field(serialization_alias="createdAt")


class EOMLeadReviewResponse(BaseModel):
    """Closed response envelope for the tracker-owned office queue."""

    model_config = ConfigDict(extra="forbid")

    leads: list[EOMLeadReviewItem]
    limit: Annotated[int, Field(ge=1, le=_MAX_LEAD_REVIEW_LIMIT)]
    cursor: str | None = None
    has_more: bool = Field(serialization_alias="hasMore")
    next_cursor: str | None = Field(
        default=None,
        serialization_alias="nextCursor",
    )


def _crm_dependency(request: Request) -> Any:
    provider_factory = getattr(request.app.state, "eom_funnel_crm_provider", None)
    if callable(provider_factory):
        return provider_factory()
    return get_crm_provider()


def _calendar_dependency() -> Any:
    from ..tools.calendar import calendar_tool

    return calendar_tool


def _encode_lead_review_cursor(*, created_at: datetime, contact_id: UUID) -> str:
    payload = f"{created_at.isoformat()}|{contact_id}"
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_lead_review_cursor(cursor: str | None) -> dict[str, object] | None:
    if cursor is None:
        return None
    token = cursor.strip()
    if not _LEAD_REVIEW_CURSOR_PATTERN.fullmatch(token):
        raise HTTPException(status_code=422, detail="Invalid lead review cursor")
    padding = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode((token + padding).encode("ascii"))
        created_at_text, contact_id_text = raw.decode("utf-8").split("|", 1)
        created_at = datetime.fromisoformat(created_at_text)
        contact_id = UUID(contact_id_text)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        raise HTTPException(
            status_code=422, detail="Invalid lead review cursor"
        ) from None
    if created_at.tzinfo is None:
        raise HTTPException(status_code=422, detail="Invalid lead review cursor")
    return {"created_at": created_at, "contact_id": contact_id}


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


@router.get(
    "/leads",
    response_model=EOMLeadReviewResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_eom_lead_review_items(
    limit: Annotated[
        int,
        Query(ge=1, le=_MAX_LEAD_REVIEW_LIMIT),
    ] = _DEFAULT_LEAD_REVIEW_LIMIT,
    cursor: Annotated[str | None, Query(min_length=16, max_length=512)] = None,
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> EOMLeadReviewResponse:
    """List active EOM lead records that still need office review.

    The tracker keeps the service bearer and the browser never calls this
    route directly. Reading this projection does not alter CRM lifecycle,
    interactions, or customer-handoff state.
    """
    decoded_cursor = _decode_lead_review_cursor(cursor)
    rows = await crm.list_eom_new_lead_review_items(
        limit=limit + 1,
        cursor_created_at=(
            decoded_cursor["created_at"] if decoded_cursor is not None else None
        ),
        cursor_contact_id=(
            decoded_cursor["contact_id"] if decoded_cursor is not None else None
        ),
    )
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page_rows:
        last_row = EOMLeadReviewItem.model_validate(page_rows[-1])
        next_cursor = _encode_lead_review_cursor(
            created_at=last_row.created_at,
            contact_id=last_row.contact_id,
        )
    return EOMLeadReviewResponse(
        leads=[EOMLeadReviewItem.model_validate(row) for row in page_rows],
        limit=limit,
        cursor=cursor,
        has_more=has_more,
        next_cursor=next_cursor,
    )


@router.post(
    "/leads/{contact_id}/estimate-bookings",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def create_estimate_booking(
    contact_id: UUID,
    payload: EOMEstimateBookingRequest,
    booking_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
    calendar: Any = Depends(_calendar_dependency),
) -> JSONResponse:
    """Book an estimate appointment without converting the lead to a customer."""
    try:
        result = await schedule_eom_estimate_booking(
            crm,
            calendar,
            EOMEstimateBooking(
                contact_id=str(contact_id),
                scheduled_start=payload.scheduled_start,
                scheduled_end=payload.scheduled_end,
                calendar_id=payload.calendar_id,
                notes=payload.notes,
                booking_key=booking_key,
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
        )
    except EOMEstimateBookingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )


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
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )
