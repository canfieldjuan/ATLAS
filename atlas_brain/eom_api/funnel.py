"""Private endpoint used by the EOM office estimate-approval command."""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..services.eom_estimate_booking import (
    EOMEstimateBooking,
    EOMEstimateBookingError,
    EOMFirstCleanBooking,
    schedule_eom_estimate_booking,
    schedule_eom_first_clean_booking,
)
from ..services.eom_lead_conversion import (
    EOMCustomerHandoff,
    EOMLeadConversionError,
    EOMLeadLost,
    EOMLeadReopen,
    finalize_eom_customer_handoff,
    mark_eom_lead_lost,
    reopen_eom_lead,
)
from ..services.eom_onboarding_drafts import (
    EOMOnboardingDraftApproval,
    EOMOnboardingDraftError,
    approve_and_send_eom_onboarding_draft,
    record_operator_confirmed_send_evidence,
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
# Same conservative shape the public intake boundary accepts
# (atlas_brain/api/leads.py), so an office-corrected recipient can never be
# stricter or looser than an intake-submitted one.
_RECIPIENT_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ONBOARDING_DRAFT_STATUSES = ("pending", "sending", "sent", "revoked")


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


class EOMLeadLostRequest(BaseModel):
    """The office's disposition for a lead that will not convert."""

    model_config = ConfigDict(extra="forbid")

    reason_code: Literal[
        "spam", "no_response", "declined_after_estimate", "price", "other"
    ]
    note: Annotated[str | None, Field(default=None, max_length=1000)] = None

    @field_validator("note", mode="before")
    @classmethod
    def _blank_note_is_none(cls, value: Any) -> Any:
        # An all-whitespace note carries no signal; store NULL instead so the
        # reason code stays the single structured field.
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


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


class EOMOnboardingDraftEditRequest(BaseModel):
    """Office edits to a still-pending onboarding draft."""

    model_config = ConfigDict(extra="forbid")

    subject: Annotated[str | None, Field(default=None, min_length=1, max_length=500)]
    body: Annotated[str | None, Field(default=None, min_length=1, max_length=20000)]
    # Same 254-character bound as the public intake email field: an
    # office-corrected address must never be looser than an intake one,
    # or the edit clears the blocker only for the transport to reject it
    # after the claim.
    recipient_email: Annotated[
        str | None, Field(default=None, min_length=3, max_length=254)
    ]

    @field_validator("subject", "body", mode="after")
    @classmethod
    def _reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("recipient_email", mode="after")
    @classmethod
    def _validate_recipient(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not _RECIPIENT_EMAIL_PATTERN.fullmatch(candidate):
            raise ValueError("must be a valid email address")
        return candidate

    @model_validator(mode="after")
    def _require_one_field(self) -> "EOMOnboardingDraftEditRequest":
        if self.subject is None and self.body is None and self.recipient_email is None:
            raise ValueError("at least one editable field is required")
        return self


class EOMOnboardingDraftItem(BaseModel):
    """The only draft data the office approval queue may expose."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    draft_id: UUID = Field(serialization_alias="draftId")
    contact_id: UUID = Field(serialization_alias="contactId")
    full_name: str = Field(serialization_alias="fullName")
    recipient_email: str | None = Field(
        default=None, serialization_alias="recipientEmail"
    )
    blocker: str | None = None
    subject: str
    body: str
    status: str
    created_at: datetime = Field(serialization_alias="createdAt")
    claimed_at: datetime | None = Field(
        default=None, serialization_alias="claimedAt"
    )
    sent_at: datetime | None = Field(default=None, serialization_alias="sentAt")
    revoked_at: datetime | None = Field(
        default=None, serialization_alias="revokedAt"
    )
    approved_by_name: str | None = Field(
        default=None, serialization_alias="approvedByName"
    )


class EOMOnboardingDraftListResponse(BaseModel):
    """Closed response envelope for the office draft-approval queue."""

    model_config = ConfigDict(extra="forbid")

    drafts: list[EOMOnboardingDraftItem]
    status: Literal["pending", "sending", "sent", "revoked"]
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


def _onboarding_sender_dependency() -> Any:
    """Test seam for the direct Resend sender; None means the real one."""
    return None


def _onboarding_email_history_dependency() -> Any:
    """Test seam for the sent-email history writer; None means the real one."""
    return None


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
    "/leads/{contact_id}/first-clean-bookings",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def create_first_clean_booking(
    contact_id: UUID,
    payload: EOMEstimateBookingRequest,
    booking_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
    calendar: Any = Depends(_calendar_dependency),
) -> JSONResponse:
    """Book the first cleaning: the lead becomes won and an onboarding
    email draft is enqueued for office approval. Nothing sends here."""
    try:
        result = await schedule_eom_first_clean_booking(
            crm,
            calendar,
            EOMFirstCleanBooking(
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


@router.post(
    "/leads/{contact_id}/lost",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def mark_lead_lost(
    contact_id: UUID,
    payload: EOMLeadLostRequest,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Disposition a lead that will not convert; it leaves the review queue.

    Reversible via the reopen endpoint. Records a reason on the lifecycle
    ledger. No calendar or customer/site side effect."""
    try:
        result = await mark_eom_lead_lost(
            crm,
            EOMLeadLost(
                contact_id=str(contact_id),
                reason_code=payload.reason_code,
                note=payload.note,
                operation_key=operation_key,
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _draft_action_response(result)


@router.post(
    "/leads/{contact_id}/reopen",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def reopen_lead(
    contact_id: UUID,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Return a previously-lost lead to its pre-loss active stage."""
    try:
        result = await reopen_eom_lead(
            crm,
            EOMLeadReopen(
                contact_id=str(contact_id),
                operation_key=operation_key,
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _draft_action_response(result)


def _draft_action_response(result: dict[str, Any]) -> JSONResponse:
    """201 on a fresh transition, 200 on an idempotent replay."""
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )


@router.get(
    "/onboarding-drafts",
    response_model=EOMOnboardingDraftListResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_eom_onboarding_drafts(
    draft_status: Annotated[
        Literal["pending", "sending", "sent", "revoked"],
        Query(alias="status"),
    ] = "pending",
    limit: Annotated[
        int,
        Query(ge=1, le=_MAX_LEAD_REVIEW_LIMIT),
    ] = _DEFAULT_LEAD_REVIEW_LIMIT,
    cursor: Annotated[str | None, Query(min_length=16, max_length=512)] = None,
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> EOMOnboardingDraftListResponse:
    """List onboarding drafts for office review; default view is the queue.

    Reading this projection alters nothing: drafts advance only through the
    explicit edit/approve/revoke/confirm commands below.
    """
    decoded_cursor = _decode_lead_review_cursor(cursor)
    rows = await crm.list_eom_onboarding_drafts(
        status=draft_status,
        limit=limit + 1,
        cursor_created_at=(
            decoded_cursor["created_at"] if decoded_cursor is not None else None
        ),
        cursor_draft_id=(
            decoded_cursor["contact_id"] if decoded_cursor is not None else None
        ),
    )
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page_rows:
        last_row = EOMOnboardingDraftItem.model_validate(page_rows[-1])
        next_cursor = _encode_lead_review_cursor(
            created_at=last_row.created_at,
            contact_id=last_row.draft_id,
        )
    return EOMOnboardingDraftListResponse(
        drafts=[EOMOnboardingDraftItem.model_validate(row) for row in page_rows],
        status=draft_status,
        limit=limit,
        cursor=cursor,
        has_more=has_more,
        next_cursor=next_cursor,
    )


@router.patch(
    "/onboarding-drafts/{draft_id}",
    dependencies=[Depends(require_eom_funnel_api)],
)
async def edit_eom_onboarding_draft(
    draft_id: UUID,
    payload: EOMOnboardingDraftEditRequest,
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Edit a still-pending draft; setting a recipient clears no_email."""
    try:
        result = await crm.update_eom_onboarding_draft(
            draft_id=str(draft_id),
            subject=payload.subject,
            body=payload.body,
            recipient_email=payload.recipient_email,
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, **result},
    )


@router.post(
    "/onboarding-drafts/{draft_id}/approve-send",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def approve_and_send_onboarding_draft(
    draft_id: UUID,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
    sender: Any = Depends(_onboarding_sender_dependency),
) -> JSONResponse:
    """Claim the pending draft, send it, then confirm delivery.

    The draft id plus the migration-360 status machine is the idempotency
    mechanism for this action, so no Idempotency-Key header is taken: an
    already-sent draft replays 200 without a second transport call, and a
    concurrent approval loses the atomic claim.
    """
    try:
        result = await approve_and_send_eom_onboarding_draft(
            crm,
            EOMOnboardingDraftApproval(
                draft_id=str(draft_id),
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
            sender=sender,
        )
    except EOMOnboardingDraftError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _draft_action_response(result)


@router.post(
    "/onboarding-drafts/{draft_id}/revoke",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def revoke_onboarding_draft(
    draft_id: UUID,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Revoke a pending draft, or reconcile a stuck 'sending' one."""
    try:
        result = await crm.revoke_eom_onboarding_draft(draft_id=str(draft_id))
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if not bool(result.get("idempotent")):
        await _log_draft_reconciliation(
            crm,
            result,
            f"employee:{actor['id']}:{actor['name']} revoked onboarding "
            f"draft {result['draft_id']}",
        )
    return _draft_action_response(result)


@router.post(
    "/onboarding-drafts/{draft_id}/confirm-sent",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def confirm_onboarding_draft_sent(
    draft_id: UUID,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
    email_history: Any = Depends(_onboarding_email_history_dependency),
) -> JSONResponse:
    """Operator reconciliation: mark a stale 'sending' draft as delivered.

    Only for migration 360 step 4, after verifying the send in the
    transport log (query Resend by the draft-id idempotency key). The
    stale requirement keeps an operator from recording a still-in-flight
    send whose outcome the transport has not yet reported.
    """
    try:
        result = await crm.confirm_eom_onboarding_draft_sent(
            draft_id=str(draft_id), require_stale=True
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if not bool(result.get("idempotent")):
        await _log_draft_reconciliation(
            crm,
            result,
            f"employee:{actor['id']}:{actor['name']} confirmed onboarding "
            f"draft {result['draft_id']} as sent after transport-log "
            "reconciliation",
        )
        # The delivery happened; without this the crash-recovery path
        # would leave the customer's sent-email history permanently
        # missing the row the normal approve path records.
        await record_operator_confirmed_send_evidence(
            crm, result, email_history=email_history
        )
    return _draft_action_response(result)


async def _log_draft_reconciliation(
    crm: Any, result: dict[str, Any], summary: str
) -> None:
    """Actor provenance for revoke/confirm; never flips the outcome."""
    try:
        log_interaction = getattr(crm, "log_interaction", None)
        if callable(log_interaction):
            await log_interaction(str(result["contact_id"]), "note", summary)
    except Exception:  # pragma: no cover - warning-only evidence path
        import logging

        logging.getLogger("atlas.eom_api.funnel").warning(
            "Draft reconciliation interaction log failed for draft %s",
            result.get("draft_id"),
            exc_info=True,
        )
