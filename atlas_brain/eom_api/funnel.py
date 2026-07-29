"""Private endpoint used by the EOM office estimate-approval command."""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime, timedelta
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..services.crm_provider import get_crm_provider
from ..services.eom_lead_booking import (
    EOMLeadBookingError,
    EOMLeadBookingService,
    EstimateBookingCommand,
)
from ..services.eom_lead_conversion import (
    EOMCustomerHandoff,
    EOMLeadConversionError,
    finalize_eom_customer_handoff,
)
from .funnel_auth import require_eom_funnel_actor, require_eom_funnel_api

router = APIRouter(prefix="/eom-funnel", tags=["eom-funnel"])

_APPROVAL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_LEAD_REVIEW_CURSOR_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,512}$")
_MAX_SIGNED_BIGINT = 2**63 - 1
_DEFAULT_LEAD_REVIEW_LIMIT = 100
_MAX_LEAD_REVIEW_LIMIT = 200


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


class EOMLeadReviewItem(BaseModel):
    """The only CRM identity data the office-review queue may expose."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contact_id: UUID = Field(serialization_alias="contactId")
    full_name: str = Field(serialization_alias="fullName")
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    source: str | None = None
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


class EOMEstimateBookingRequest(BaseModel):
    """One first-estimate booking command from the office proxy."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    start_time: datetime = Field(alias="startTime")
    duration_minutes: int = Field(default=60, ge=15, le=240, alias="durationMinutes")
    service_type: str = Field(
        default="estimate",
        min_length=1,
        max_length=128,
        alias="serviceType",
    )
    location: str | None = Field(default=None, max_length=1000)
    notes: str = Field(default="", max_length=4000)

    @field_validator("start_time")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("startTime must include a timezone offset")
        return value

    @model_validator(mode="after")
    def _require_representable_end_time(self) -> "EOMEstimateBookingRequest":
        try:
            self.start_time + timedelta(minutes=self.duration_minutes)
        except OverflowError as exc:
            raise ValueError(
                "startTime plus durationMinutes must be representable"
            ) from exc
        return self

    @field_validator("service_type", "location", "notes", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str) and "\x00" in value:
            raise ValueError("Text fields cannot contain NUL characters")
        return value.strip() if isinstance(value, str) else value


class EOMEstimateBookingResponse(BaseModel):
    """Closed response envelope for one estimate booking operation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    success: bool = True
    operation_id: UUID = Field(serialization_alias="operationId")
    appointment_id: UUID | None = Field(
        default=None,
        serialization_alias="appointmentId",
    )
    calendar_event_id: str = Field(serialization_alias="calendarEventId")
    status: str
    idempotent: bool


def _crm_dependency() -> Any:
    return get_crm_provider()


def _booking_service_dependency() -> EOMLeadBookingService:
    return EOMLeadBookingService()


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
        raise HTTPException(status_code=422, detail="Invalid lead review cursor") from None
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


def _booking_key_dependency(
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
    """List active EOM lead records that remain reachable for office review.

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
    dependencies=[Depends(require_eom_funnel_api)],
)
async def create_estimate_booking(
    contact_id: UUID,
    payload: EOMEstimateBookingRequest,
    booking_key: str = Depends(_booking_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    service: EOMLeadBookingService = Depends(_booking_service_dependency),
) -> JSONResponse:
    """Create or replay exactly one first estimate booking for an EOM lead."""
    command = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key=booking_key,
        actor_id=int(actor["id"]),
        actor_name=str(actor["name"]),
        start_time=payload.start_time,
        duration_minutes=payload.duration_minutes,
        service_type=payload.service_type,
        location=payload.location or None,
        notes=payload.notes,
    )
    try:
        result = await service.book_estimate(command)
    except EOMLeadBookingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    response = EOMEstimateBookingResponse.model_validate(
        {"success": True, **result.to_dict()}
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if result.idempotent else status.HTTP_201_CREATED,
        content=response.model_dump(mode="json", by_alias=True),
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
            status.HTTP_200_OK if bool(result.get("idempotent")) else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )
