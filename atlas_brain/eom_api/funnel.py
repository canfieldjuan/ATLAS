"""Private endpoint used by the EOM office estimate-approval command."""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from typing import Annotated, Any, Literal, Mapping
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
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
    EOMContactArchive,
    EOMContactRestore,
    EOMCustomerHandoff,
    EOMLeadConversionError,
    EOMLeadLost,
    EOMLeadReopen,
    archive_eom_contact,
    finalize_eom_customer_handoff,
    reopen_eom_lead,
    restore_eom_contact,
)
from ..services.eom_won_lead_loss import mark_eom_lead_lost_with_won_teardown
from ..services.eom_crm_mutations import (
    EOM_CUSTOMER_TYPES,
    EOM_OPERATOR_CONTACT_TYPES,
    EOM_OPERATOR_EDIT_BLOCK_REASONS,
    EOMOperatorContactMutation,
    EOMOperatorContactMutationError,
    mutate_eom_operator_contact,
)
from ..services.eom_onboarding_drafts import (
    EOMOnboardingDraftApproval,
    EOMOnboardingDraftError,
    approve_and_send_eom_onboarding_draft,
    record_operator_confirmed_send_evidence,
)
from ..services.eom_missed_call_recovery import (
    EOMMissedCallRecoveryError,
    EOMMissedCallRecoveryService,
)
from ..services.eom_first_clean_completion import (
    EOMFirstCleanCompletionError,
    EOMFirstCleanCompletionService,
)
from ..services.eom_terms_authority import (
    EOMTermsAuthority,
    EOMTermsAuthorityError,
)
from ..services.eom_public_onboarding_tokens import (
    AuthenticatedEOMPublicOnboardingToken,
    EOMPublicOnboardingTokenError,
    authenticate_eom_public_onboarding_token,
    eom_public_onboarding_hmac_key_fingerprint,
)
from ..services.crm_provider import get_crm_provider
from .config import funnel_settings
from .funnel_auth import (
    EOMPublicOnboardingConfig,
    get_eom_funnel_api_config,
    require_eom_funnel_actor,
    require_eom_funnel_api,
    require_eom_public_onboarding_config,
)

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
# The exact query-parameter names the contact directory accepts. Unknown names
# are rejected rather than tolerated: the directory is new and has exactly one
# caller, and a typoed filter silently ignored would return the unfiltered
# directory while looking filtered.
_CONTACT_DIRECTORY_QUERY_PARAMS = frozenset(
    {"limit", "cursor", "search", "kind", "lifecycle"}
)
_MAX_CONTACT_DIRECTORY_SEARCH_LENGTH = 120
# The status axis the directory may select over. Closed on purpose: admission
# is one value per page, so an active view can never leak an archived row and
# the archived view is a real server-side read, not a client-side filter.
_CONTACT_DIRECTORY_LIFECYCLES = ("active", "archived")
# DERIVED from the canonical operator-mutation kind set, not re-enumerated:
# if the write boundary ever admits another contact kind, the directory must
# widen with it in the same commit, or that kind's records become write-only
# -- the exact defect this slice exists to close (website #240).
_CONTACT_DIRECTORY_KINDS = ("all", *EOM_OPERATOR_CONTACT_TYPES)
# Ids ride in the query string, so the cap is a URL-length budget rather than a
# database one: 100 ids costs roughly 4.8 KB of `contact_id=<uuid>&`, comfortably
# inside the 8 KB request line every proxy in front of this accepts. Callers with
# more links to check page through them.
_MAX_KNOWN_CONTACT_IDS = 100
# Same conservative shape the public intake boundary accepts
# (atlas_brain/api/leads.py), so an office-corrected recipient can never be
# stricter or looser than an intake-submitted one.
_RECIPIENT_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ONBOARDING_DRAFT_STATUSES = ("pending", "sending", "sent", "revoked")


def _route_surrogates_to_safe_text(value: Any) -> Any:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            return "\x00"
        return value
    if isinstance(value, Mapping):
        return {
            _route_surrogates_to_safe_text(key): _route_surrogates_to_safe_text(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_route_surrogates_to_safe_text(item) for item in value]
    return value


class EOMCustomerHandoffRequest(BaseModel):
    """Tracker-owned customer/site IDs; never operational estimate details."""

    model_config = ConfigDict(extra="forbid")
    contact_id: UUID
    tracker_customer_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]
    tracker_site_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]


class EOMFirstCleanCompletionRequest(BaseModel):
    """One tracker-owned report that a first residential service completed."""

    model_config = ConfigDict(extra="forbid")

    tracker_customer_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]
    tracker_site_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]
    tracker_service_kind: Literal["job", "planned_visit"]
    tracker_service_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]
    completed_at: datetime

    @field_validator("completed_at", mode="before")
    @classmethod
    def _require_completed_at_string(cls, value: Any) -> Any:
        if not isinstance(value, str) or not _RFC3339_DATETIME_PATTERN.fullmatch(value):
            raise ValueError("must be an RFC 3339 date-time string")
        return value

    @model_validator(mode="after")
    def _require_completed_at_timezone(self) -> "EOMFirstCleanCompletionRequest":
        if self.completed_at.tzinfo is None:
            raise ValueError("completed_at must include a timezone")
        return self


class EOMPostCleanOnboardingCandidateItem(BaseModel):
    """Non-sendable candidate derived from actual first-clean evidence."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_id: UUID = Field(serialization_alias="candidateId")
    completion_receipt_id: UUID = Field(serialization_alias="completionReceiptId")
    contact_id: UUID = Field(serialization_alias="contactId")
    handoff_id: UUID = Field(serialization_alias="handoffId")
    status: Literal["pending"]
    full_name: str = Field(serialization_alias="fullName")
    recipient_email: str | None = Field(serialization_alias="recipientEmail")
    blocker: Literal["inactive_customer", "not_residential", "no_email"] | None
    tracker_service_kind: Literal["job", "planned_visit"] = Field(
        serialization_alias="trackerServiceKind"
    )
    tracker_service_id: int = Field(serialization_alias="trackerServiceId")
    completed_at: datetime = Field(serialization_alias="completedAt")
    created_at: datetime = Field(serialization_alias="createdAt")


class EOMPostCleanOnboardingCandidateResponse(BaseModel):
    """Bounded Atlas-owned queue for the future CRM consumer."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[EOMPostCleanOnboardingCandidateItem]
    limit: Annotated[int, Field(ge=1, le=_MAX_LEAD_REVIEW_LIMIT)]
    cursor: str | None = None
    has_more: bool = Field(serialization_alias="hasMore")
    next_cursor: str | None = Field(default=None, serialization_alias="nextCursor")


class EOMTermsVersionCreateRequest(BaseModel):
    """One exact bilingual residential/commercial Terms release candidate."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version_label: object = Field(alias="versionLabel")
    material_change: object = Field(alias="materialChange")
    documents: object


class EOMTermsVersionResponse(BaseModel):
    """Closed private projection of one Atlas Terms version."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version_id: UUID = Field(alias="versionId")
    version_label: str = Field(alias="versionLabel")
    status: Literal["draft", "published"]
    material_change: bool = Field(alias="materialChange")
    documents: dict[str, Any]
    content_hash: str = Field(alias="contentHash")
    created_by_id: int = Field(alias="createdById")
    created_by_name: str = Field(alias="createdByName")
    created_at: datetime = Field(alias="createdAt")
    published_by_id: int | None = Field(default=None, alias="publishedById")
    published_by_name: str | None = Field(default=None, alias="publishedByName")
    published_at: datetime | None = Field(default=None, alias="publishedAt")
    idempotent: bool


class EOMPublicOnboardingSessionRequest(BaseModel):
    """Opaque bearer supplied by the tracker after the Website reads a fragment."""

    model_config = ConfigDict(extra="forbid")

    # Deliberately accept the JSON shape here and hand every supplied value to
    # the canonical parser. It is the one bearer-admission choke point, so a
    # non-string or oversized value receives the same unavailable result before
    # any CRM access rather than becoming a second token validator in Pydantic.
    token: object


class EOMPublicOnboardingFinalizeRequest(EOMPublicOnboardingSessionRequest):
    """Tracker-owned local IDs for the one-time Atlas finalizer."""

    tracker_customer_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]
    tracker_site_id: Annotated[int, Field(strict=True, gt=0, le=_MAX_SIGNED_BIGINT)]


class EOMPublicOnboardingRecoveryRequest(EOMCustomerHandoffRequest):
    """Stored IDs for staff recovery when raw-bearer redemption cannot finish."""

    token_id: UUID


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
        if not isinstance(value, str) or not _RFC3339_DATETIME_PATTERN.fullmatch(value):
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


class EOMMissedCallRecoveryCancelRequest(BaseModel):
    """One verified reason an operator must stop future recovery mail."""

    model_config = ConfigDict(extra="forbid")

    reason: Literal["callback_recorded", "response_recorded", "opt_out", "manual"]


class EOMOperatorContactRequest(BaseModel):
    """Authenticated operator-authored EOM contact create/update request."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contact_id: UUID | None = Field(default=None, alias="contactId")
    contact_type: Literal["lead", "customer"] | None = Field(
        default=None,
        alias="contactType",
    )
    full_name: Annotated[
        str | None,
        Field(default=None, min_length=1, max_length=256, alias="fullName"),
    ]
    email: Annotated[str | None, Field(default=None, max_length=256)]
    phone: Annotated[str | None, Field(default=None, max_length=64)]
    address: Annotated[str | None, Field(default=None, max_length=2000)]
    city: Annotated[str | None, Field(default=None, max_length=128)]
    state: Annotated[str | None, Field(default=None, max_length=64)]
    zip: Annotated[str | None, Field(default=None, max_length=16)]
    notes: Annotated[str | None, Field(default=None, max_length=4000)]
    # Bounded here, but the admitted VALUES are decided once in
    # eom_crm_mutations.EOM_CUSTOMER_TYPES, which is bound to the
    # chk_contacts_customer_type CHECK. A Literal here would be a third copy of
    # that set and would reject the capitalised 'Residential'/'Commercial' the
    # tracker actually stores, before the case-folding normalizer ever runs.
    customer_type: Annotated[
        str | None,
        Field(default=None, max_length=32, alias="customerType"),
    ]
    source_channel: Annotated[
        str,
        Field(min_length=1, max_length=64, alias="sourceChannel"),
    ]
    source_ref: Annotated[
        str,
        Field(min_length=1, max_length=220, alias="sourceRef"),
    ]

    @model_validator(mode="before")
    @classmethod
    def _sanitize_request_surrogates(cls, value: Any) -> Any:
        return _route_surrogates_to_safe_text(value)

    @field_validator(
        "full_name",
        "email",
        "phone",
        "address",
        "city",
        "state",
        "zip",
        "notes",
        "source_channel",
        "source_ref",
        mode="before",
    )
    @classmethod
    def _route_surrogates_to_domain_validation(cls, value: Any) -> Any:
        if isinstance(value, str) and any(
            0xD800 <= ord(char) <= 0xDFFF for char in value
        ):
            return "\x00"
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
    # What this deployment actually serves, so a caller can disable a control
    # instead of shipping a button that 404s. Website (Vercel) and tracker
    # (Render) auto-deploy from main; Atlas deploys by hand, so callers
    # routinely run ahead of it. See ATLAS #2275 and website #112.
    capabilities: list[str] = Field(default_factory=list)
    # Names alone are a presentation convenience. The Tracker derives the
    # public-onboarding controls from these registered route signatures so it
    # cannot accidentally treat a copied capability spelling as deployment
    # evidence.
    capability_routes: list["EOMFunnelCapabilityRoute"] = Field(
        default_factory=list,
        serialization_alias="capabilityRoutes",
    )


class EOMMissedCallRecoveryStatusItem(BaseModel):
    """Non-PII status that the tracker may render on an existing lead card."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contact_id: UUID = Field(alias="contactId")
    sequence_id: UUID = Field(alias="sequenceId")
    state: Literal[
        "active",
        "blocked_configuration",
        "completed",
        "cancelled",
        "failed",
        "recovery_required",
    ]
    blocked_reason: str | None = Field(default=None, alias="blockedReason")
    cancellation_reason: str | None = Field(default=None, alias="cancellationReason")
    next_step_number: int | None = Field(default=None, alias="nextStepNumber")
    next_follow_up_at: datetime | None = Field(default=None, alias="nextFollowUpAt")
    last_event: str | None = Field(default=None, alias="lastEvent")
    last_reason: str | None = Field(default=None, alias="lastReason")
    created_at: datetime = Field(alias="createdAt")
    terminal_at: datetime | None = Field(default=None, alias="terminalAt")


class EOMMissedCallRecoveryStatusResponse(BaseModel):
    """Bounded recovery-status batch for the current CRM lead page."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sequences: list[EOMMissedCallRecoveryStatusItem]
    checked: int
    limit: Annotated[int, Field(ge=1, le=_MAX_KNOWN_CONTACT_IDS)]


class EOMFunnelCapabilityRoute(BaseModel):
    """One registered method/path signature behind an advertised capability."""

    model_config = ConfigDict(extra="forbid")

    method: str
    path: str


class EOMKnownContactsResponse(BaseModel):
    """Which submitted ids name a live EOM contact, its type, and its source version.

    Still not an identity read: no name, email, phone or address is disclosed.
    A caller holding a stored contact id is asking whether its link resolves
    and what kind of account it points at, and both answers are available
    without any of that.

    ``customerType`` was added deliberately rather than incidentally. This
    route was introduced id-only, so widening it is a disclosure decision, not
    a formatting one. It is included because: the value is not personal data,
    it is a classification the operator themselves set; the credential is
    already EOM-scoped, so no cross-tenant information is exposed; and the
    alternative is a mirror that can never self-correct, which is the concrete
    defect this closes (ATLAS #2357).

    ``knownContactIds`` keeps its exact prior shape. The tracker's link audit
    already consumes it, and changing a list of ids into a list of objects
    would break that consumer for no gain -- so the types arrive alongside it
    as a parallel mapping instead.
    """

    model_config = ConfigDict(extra="forbid")

    known_contact_ids: list[UUID] = Field(serialization_alias="knownContactIds")
    # Keyed by the same ids, so a caller can answer "does it resolve" and
    # "what is it" from one response. Only ever contains ids that appear in
    # known_contact_ids above.
    customer_types: dict[str, str] = Field(
        default_factory=dict, serialization_alias="customerTypes"
    )
    # Database-owned and monotonic for this contact's customer_type evidence.
    # Keep it parallel so legacy fields retain their shapes.
    customer_type_revisions: dict[str, int] = Field(
        default_factory=dict, serialization_alias="customerTypeRevisions"
    )
    checked: int
    limit: Annotated[int, Field(ge=1, le=_MAX_KNOWN_CONTACT_IDS)]


class EOMContactDirectoryItem(BaseModel):
    """The only CRM identity data the operator contact directory may expose.

    A closed projection over canonical contact columns: no metadata, notes,
    tags, receipts, or interaction history. The pipeline read's latest-intake
    email/phone overlay stays unique to the review queue -- the directory's
    job is discoverability of the canonical record, so it reads the record.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contact_id: UUID = Field(serialization_alias="contactId")
    full_name: str = Field(serialization_alias="fullName")
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    contact_type: str = Field(serialization_alias="contactType")
    customer_type: str = Field(serialization_alias="customerType")
    lead_stage: str | None = Field(default=None, serialization_alias="leadStage")
    # The WHERE clause pins admission to the one status the requested
    # lifecycle selects; this Literal plus the route's page-homogeneity check
    # make the response a second, independent enforcement of the same
    # invariant, so neither lifecycle view can silently leak the other's rows.
    status: Literal["active", "archived"]
    source: str | None = None
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime | None = Field(default=None, serialization_alias="updatedAt")
    editable: bool
    edit_block_reason: str | None = Field(serialization_alias="editBlockedReason")

    @field_validator("edit_block_reason")
    @classmethod
    def _edit_block_reason_is_closed(cls, value: str | None) -> str | None:
        if value is not None and value not in EOM_OPERATOR_EDIT_BLOCK_REASONS:
            raise ValueError("edit_block_reason must be a supported reason")
        return value

    @model_validator(mode="after")
    def _editability_matches_reason(self) -> "EOMContactDirectoryItem":
        if self.editable != (self.edit_block_reason is None):
            raise ValueError("editable must match edit_block_reason")
        return self

    @field_validator("contact_type")
    @classmethod
    def _contact_type_is_directory_kind(cls, value: str) -> str:
        # Validated against the operator boundary's own set rather than a
        # local Literal, so the two cannot drift apart.
        if value not in EOM_OPERATOR_CONTACT_TYPES:
            raise ValueError("contact_type must be a directory contact kind")
        return value

    @field_validator("customer_type")
    @classmethod
    def _customer_type_is_admitted(cls, value: str) -> str:
        # Same single-source rule: EOM_CUSTOMER_TYPES is bound to the
        # chk_contacts_customer_type CHECK (migration 366).
        if value not in EOM_CUSTOMER_TYPES:
            raise ValueError("customer_type must be an admitted account type")
        return value


class EOMContactDirectoryResponse(BaseModel):
    """Closed response envelope for the operator contact-directory read."""

    model_config = ConfigDict(extra="forbid")

    contacts: list[EOMContactDirectoryItem]
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
    claimed_at: datetime | None = Field(default=None, serialization_alias="claimedAt")
    sent_at: datetime | None = Field(default=None, serialization_alias="sentAt")
    revoked_at: datetime | None = Field(default=None, serialization_alias="revokedAt")
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


class EOMPublicOnboardingIssuedLinkItem(BaseModel):
    """Closed office projection of one currently usable onboarding link.

    The durable token row is the lifecycle authority.  The opaque token ID and
    raw bearer are deliberately absent: office revocation already addresses the
    record through its draft ID, so neither value has a browser use here.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    draft_id: UUID = Field(serialization_alias="draftId")
    contact_id: UUID = Field(serialization_alias="contactId")
    full_name: str = Field(serialization_alias="fullName")
    recipient_email: str | None = Field(
        default=None, serialization_alias="recipientEmail"
    )
    status: Literal["issued"]
    issued_at: datetime = Field(serialization_alias="issuedAt")


class EOMPublicOnboardingIssuedLinkListResponse(BaseModel):
    """Bounded current-state list for the office follow-up queue."""

    model_config = ConfigDict(extra="forbid")

    links: list[EOMPublicOnboardingIssuedLinkItem]
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


def _missed_call_recovery_dependency(request: Request) -> EOMMissedCallRecoveryService:
    """Bind recovery state to the same authoritative pool as this API profile.

    The full Atlas application uses its global canonical pool.  The slim EOM
    profile injects its explicitly validated funnel pool on app state, so this
    route never silently splits lead lifecycle state across databases.
    """

    pool_factory = getattr(
        request.app.state, "eom_funnel_missed_call_recovery_pool", None
    )
    if callable(pool_factory):
        pool = pool_factory()
    else:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
    return EOMMissedCallRecoveryService(pool=pool, config=funnel_settings)


def _first_clean_completion_dependency(
    request: Request,
) -> EOMFirstCleanCompletionService:
    """Bind completion evidence to the same canonical funnel database."""

    pool_factory = getattr(
        request.app.state, "eom_funnel_first_clean_completion_pool", None
    )
    if callable(pool_factory):
        pool = pool_factory()
    else:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
    return EOMFirstCleanCompletionService(pool=pool)


def _terms_authority_dependency(request: Request) -> EOMTermsAuthority:
    """Bind Terms state to the same canonical funnel database."""

    pool_factory = getattr(request.app.state, "eom_funnel_terms_pool", None)
    if callable(pool_factory):
        pool = pool_factory()
    else:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
    return EOMTermsAuthority(pool=pool)


def _authenticated_public_onboarding_token(
    token: object,
    public_onboarding: EOMPublicOnboardingConfig,
) -> AuthenticatedEOMPublicOnboardingToken:
    """Authenticate and bind a raw bearer before it reaches the CRM provider."""

    try:
        return authenticate_eom_public_onboarding_token(
            token=token,
            secret=public_onboarding.hmac_secret,
            previous_secret=public_onboarding.previous_hmac_secret,
        )
    except EOMPublicOnboardingTokenError as exc:
        # The same result as an unknown/revoked durable token avoids telling a
        # caller whether its grammar or MAC was the rejected component.
        raise HTTPException(
            status_code=404,
            detail="Public onboarding link is unavailable",
        ) from exc


def _accepted_public_onboarding_signing_key_fingerprints(
    public_onboarding: EOMPublicOnboardingConfig,
) -> tuple[str, ...]:
    """Return the only durable key identities that can authenticate now."""

    secrets = (public_onboarding.hmac_secret, public_onboarding.previous_hmac_secret)
    return tuple(
        dict.fromkeys(
            eom_public_onboarding_hmac_key_fingerprint(secret=secret)
            for secret in secrets
            if secret is not None
        )
    )


def _public_onboarding_session_content(result: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist the token-bound projection the tracker may bridge onward."""

    state = str(result["status"])
    content: dict[str, Any] = {"success": True, "status": state}
    if state == "ready":
        for field in (
            "full_name",
            "email",
            "phone",
            "address",
            "city",
            "state",
            "zip",
            "customer_type",
        ):
            content[field] = result[field]
        return content
    if state == "completed":
        for field in ("tracker_customer_id", "tracker_site_id", "idempotent"):
            content[field] = result[field]
        return content
    raise RuntimeError("CRM provider returned an invalid public onboarding session")


def _public_onboarding_tracker_context_content(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Add durable Atlas IDs only to the tracker-only context projection."""

    content = _public_onboarding_session_content(result)
    content.update(
        {
            "token_id": result["token_id"],
            "draft_id": result["draft_id"],
            "contact_id": result["contact_id"],
        }
    )
    return content


def _public_onboarding_finalize_content(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return only completion evidence the tracker can reconcile locally."""

    if str(result["status"]) != "completed":
        raise RuntimeError(
            "CRM provider returned an invalid public onboarding completion"
        )
    return {
        "success": True,
        "status": "completed",
        "tracker_customer_id": result["tracker_customer_id"],
        "tracker_site_id": result["tracker_site_id"],
        "idempotent": result["idempotent"],
    }


def _public_onboarding_recovery_content(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return only completion evidence after an actor-audited recovery."""

    return _public_onboarding_finalize_content(result)


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


# Capability name -> the (method, path) that must be registered for this
# deployment to advertise it.
#
# Names are an explicit ENUMERATION -- a route cannot join by existing or by
# being named a certain way. Membership in the advertised set is DERIVED from
# the router's registered routes, so the manifest cannot claim a capability
# this build does not serve: that claim is the exact failure this slice exists
# to prevent, and a hand-maintained list would drift the moment a route moved.
#
# Out-of-set default is OMIT: an unregistered route is simply absent, and a
# caller that treats absence as "disable the control" fails closed.
_CAPABILITY_ROUTES: dict[str, tuple[str, str]] = {
    "lead.estimate_booking": (
        "POST",
        "/eom-funnel/leads/{contact_id}/estimate-bookings",
    ),
    "lead.first_clean_booking": (
        "POST",
        "/eom-funnel/leads/{contact_id}/first-clean-bookings",
    ),
    "lead.customer_handoff": ("POST", "/eom-funnel/customer-handoffs"),
    "customer.first_clean_completion.record": (
        "POST",
        "/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions",
    ),
    # Same completion route, but a semantic capability: this build atomically
    # creates/reuses a distinct non-sendable post-clean candidate.
    "customer.post_clean_onboarding_candidate.create": (
        "POST",
        "/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions",
    ),
    "customer.post_clean_onboarding_candidate.list": (
        "GET",
        "/eom-funnel/post-clean-onboarding-candidates",
    ),
    "contact.operator_mutation": ("POST", "/eom-funnel/operator-contacts"),
    # Same route as contact.operator_mutation ON PURPOSE: this name versions
    # the route's SEMANTICS, not its existence. A build advertises it only by
    # shipping this dict entry, which lands together with the audited
    # null-clear contract (present-null clears email/phone, cleared_fields in
    # the lifecycle event). A caller must not infer null semantics from the
    # mutation route existing -- an older build serves the same route without
    # them -- so the tracker gates field-clearing on THIS name + route pair.
    "contact.field_clear": ("POST", "/eom-funnel/operator-contacts"),
    "lead.lost": ("POST", "/eom-funnel/leads/{contact_id}/lost"),
    "lead.reopen": ("POST", "/eom-funnel/leads/{contact_id}/reopen"),
    # A separate, narrow seam rather than an inferred lead-stage: the office
    # must record an actual unanswered call before Atlas creates its durable
    # recovery outbox. The status read is batch-only and contains no recipient
    # or template data, so the active CRM card can render it without owning
    # sequence state in the browser.
    "lead.missed_call_attempt.record": (
        "POST",
        "/eom-funnel/leads/{contact_id}/missed-call-attempts",
    ),
    "lead.missed_call_recovery.status": (
        "GET",
        "/eom-funnel/missed-call-recovery-status",
    ),
    "lead.missed_call_recovery.resume": (
        "POST",
        "/eom-funnel/leads/{contact_id}/missed-call-recovery/resume",
    ),
    "lead.missed_call_recovery.cancel": (
        "POST",
        "/eom-funnel/leads/{contact_id}/missed-call-recovery/cancel",
    ),
    "onboarding.draft.list": ("GET", "/eom-funnel/onboarding-drafts"),
    "onboarding.draft.edit": ("PATCH", "/eom-funnel/onboarding-drafts/{draft_id}"),
    "onboarding.draft.approve_send": (
        "POST",
        "/eom-funnel/onboarding-drafts/{draft_id}/approve-send",
    ),
    "onboarding.draft.revoke": (
        "POST",
        "/eom-funnel/onboarding-drafts/{draft_id}/revoke",
    ),
    "onboarding.draft.confirm_sent": (
        "POST",
        "/eom-funnel/onboarding-drafts/{draft_id}/confirm-sent",
    ),
    "onboarding.public_link.list": (
        "GET",
        "/eom-funnel/public-onboarding/issued-links",
    ),
    "onboarding.public_link.revoke": (
        "POST",
        "/eom-funnel/onboarding-drafts/{draft_id}/revoke-link",
    ),
    "onboarding.public_handoff.recover": (
        "POST",
        "/eom-funnel/public-onboarding/recover",
    ),
    "contact.link_verification": ("GET", "/eom-funnel/known-contacts"),
    "contact.directory": ("GET", "/eom-funnel/contact-directory"),
    # Same route as contact.directory: this name proves the response includes
    # the per-contact editability verdict and closed reason code. Older builds
    # cannot advertise it, so downstream consumers can fail closed on a stale
    # deployment instead of treating route reachability as semantic parity.
    "contact.directory.editability": ("GET", "/eom-funnel/contact-directory"),
    "contact.archive": ("POST", "/eom-funnel/contacts/{contact_id}/archive"),
    "contact.restore": ("POST", "/eom-funnel/contacts/{contact_id}/restore"),
    # Same registered route as contact.directory, deliberately: the name
    # asserts THIS build's directory understands the closed `lifecycle`
    # filter. An older build never carries this map entry, so it can never
    # advertise the name -- and its unknown-parameter 422 rejects the filter
    # outright, so a stale deployment fails closed twice over.
    "contact.directory.archived": ("GET", "/eom-funnel/contact-directory"),
}

_served_capabilities_cache: tuple[str, ...] | None = None


def served_capabilities() -> tuple[str, ...]:
    """Capability names this build serves, derived from registered routes.

    Computed on first call rather than at import: the routes are registered by
    decorators further down this module, so an import-time constant would read
    a partially-populated router and silently under-report.
    """
    global _served_capabilities_cache
    if _served_capabilities_cache is None:
        registered = {
            (method, route.path)
            for route in router.routes
            for method in (getattr(route, "methods", None) or ())
        }
        _served_capabilities_cache = tuple(
            sorted(
                name
                for name, signature in _CAPABILITY_ROUTES.items()
                if signature in registered
            )
        )
    return _served_capabilities_cache


def served_capability_routes() -> tuple[tuple[str, str], ...]:
    """Registered signatures for the same derived capability set.

    Keep this projection mechanically tied to ``served_capabilities``: callers
    may use the names for existing generic controls, but new cross-service
    controls must derive their proof from the registered method/path pair.
    """

    return tuple(_CAPABILITY_ROUTES[name] for name in served_capabilities())


def _operator_contact_fields(payload: EOMOperatorContactRequest) -> dict[str, Any]:
    model_to_contact = {
        "full_name": "full_name",
        "email": "email",
        "phone": "phone",
        "address": "address",
        "city": "city",
        "state": "state",
        "zip": "zip",
        "notes": "notes",
        "customer_type": "customer_type",
    }
    return {
        contact_field: getattr(payload, model_field)
        for model_field, contact_field in model_to_contact.items()
        if model_field in payload.model_fields_set
    }


def _contact_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _operator_contact_item(contact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contactId": str(contact["id"]),
        "fullName": contact.get("full_name"),
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "address": contact.get("address"),
        "city": contact.get("city"),
        "state": contact.get("state"),
        "zip": contact.get("zip"),
        "notes": contact.get("notes"),
        "contactType": contact.get("contact_type"),
        "customerType": contact.get("customer_type"),
        "leadStage": contact.get("lead_stage"),
        "status": contact.get("status"),
        "source": contact.get("source"),
        "sourceRef": contact.get("source_ref"),
        "createdAt": _contact_datetime(contact.get("created_at")),
        "updatedAt": _contact_datetime(contact.get("updated_at")),
    }


@router.post(
    "/operator-contacts",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def mutate_operator_contact(
    payload: EOMOperatorContactRequest,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Create or edit one EOM contact through the operator mutation boundary."""
    try:
        result = await mutate_eom_operator_contact(
            crm,
            EOMOperatorContactMutation.from_raw(
                operation_key=operation_key,
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
                source_channel=payload.source_channel,
                source_ref=payload.source_ref,
                fields=_operator_contact_fields(payload),
                contact_id=str(payload.contact_id) if payload.contact_id else None,
                contact_type=payload.contact_type,
            ),
        )
    except EOMOperatorContactMutationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={
            "success": True,
            "contactId": result["contact_id"],
            "operation": result["operation"],
            "idempotent": bool(result.get("idempotent")),
            "contact": _operator_contact_item(result["contact"]),
        },
    )


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
        capabilities=list(served_capabilities()),
        capability_routes=[
            EOMFunnelCapabilityRoute(method=method, path=path)
            for method, path in served_capability_routes()
        ],
    )


@router.post(
    "/leads/{contact_id}/missed-call-attempts",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def record_eom_missed_call_attempt(
    contact_id: UUID,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    recovery: EOMMissedCallRecoveryService = Depends(_missed_call_recovery_dependency),
) -> JSONResponse:
    """Record a real office no-answer; Atlas alone decides whether mail starts."""

    try:
        await recovery.require_schema_ready()
        result = await recovery.record_no_answer(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=int(actor["id"]),
            actor_name=str(actor["name"]),
        )
    except EOMMissedCallRecoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )


@router.get(
    "/missed-call-recovery-status",
    response_model=EOMMissedCallRecoveryStatusResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_eom_missed_call_recovery_status(
    contact_id: Annotated[
        list[UUID],
        Query(min_length=1, max_length=_MAX_KNOWN_CONTACT_IDS),
    ],
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    recovery: EOMMissedCallRecoveryService = Depends(_missed_call_recovery_dependency),
) -> EOMMissedCallRecoveryStatusResponse:
    """Return bounded status for lead cards already visible to the office."""

    requested = list(dict.fromkeys(contact_id))
    try:
        await recovery.require_schema_ready()
        rows = await recovery.statuses(contact_ids=requested)
    except EOMMissedCallRecoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return EOMMissedCallRecoveryStatusResponse(
        sequences=[EOMMissedCallRecoveryStatusItem.model_validate(row) for row in rows],
        checked=len(requested),
        limit=_MAX_KNOWN_CONTACT_IDS,
    )


@router.post(
    "/leads/{contact_id}/missed-call-recovery/resume",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def resume_eom_missed_call_recovery(
    contact_id: UUID,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    recovery: EOMMissedCallRecoveryService = Depends(_missed_call_recovery_dependency),
) -> JSONResponse:
    """Explicitly resume one previously configuration-blocked sequence."""

    try:
        await recovery.require_schema_ready()
        result = await recovery.resume_blocked_sequence(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=int(actor["id"]),
            actor_name=str(actor["name"]),
        )
    except EOMMissedCallRecoveryError as exc:
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
    "/leads/{contact_id}/missed-call-recovery/cancel",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def cancel_eom_missed_call_recovery(
    contact_id: UUID,
    payload: EOMMissedCallRecoveryCancelRequest,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    recovery: EOMMissedCallRecoveryService = Depends(_missed_call_recovery_dependency),
) -> JSONResponse:
    """Persist a verified stop condition without altering lead history."""

    try:
        await recovery.require_schema_ready()
        result = await recovery.cancel_sequence(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=int(actor["id"]),
            actor_name=str(actor["name"]),
            reason=payload.reason,
        )
    except EOMMissedCallRecoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )


@router.get(
    "/known-contacts",
    response_model=EOMKnownContactsResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_known_eom_contacts(
    contact_id: Annotated[
        list[UUID],
        Query(min_length=1, max_length=_MAX_KNOWN_CONTACT_IDS),
    ],
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> EOMKnownContactsResponse:
    """Report which submitted contact ids still name a live EOM contact.

    A system holding its own copy of a contact id -- the tracker's
    ``customers.atlas_contact_id`` -- cannot tell a good link from a dangling
    one on its own, so a link to a contact that no longer exists stays silent
    until someone opens the record. This answers that question and nothing
    else: an id comes back only when it names an ``effingham_maids`` contact.
    Lifecycle is not part of the answer -- an archived or lost contact is still
    a link that resolves.

    An id that exists under a different tenant is reported the same way as one
    that does not exist at all. The distinction would be more useful to the
    caller and is deliberately withheld: this credential is scoped to EOM, and
    confirming the existence of another tenant's contact would make this route
    a cross-tenant existence oracle. Either answer means the same thing to the
    caller anyway -- the link does not point at an EOM contact.

    Reading this projection alters nothing.
    """
    requested = list(dict.fromkeys(contact_id))
    known = await crm.list_known_eom_contact_ids(contact_ids=requested)
    # Answer in terms of what was asked rather than echoing the provider's rows:
    # an id the caller never submitted must never appear in the response, or the
    # route would report a verdict the caller cannot attribute to a link it holds.
    # The same filter governs the types, so neither field can carry an id the
    # other does not.
    by_id = {UUID(str(row["id"])): row for row in known}
    known_ids = [value for value in requested if value in by_id]
    customer_types = {
        str(value): str(by_id[value].get("customer_type") or "unknown")
        for value in known_ids
    }
    customer_type_revisions = {
        str(value): int(by_id[value]["customer_type_revision"]) for value in known_ids
    }
    return EOMKnownContactsResponse(
        known_contact_ids=known_ids,
        customer_types=customer_types,
        customer_type_revisions=customer_type_revisions,
        checked=len(requested),
        limit=_MAX_KNOWN_CONTACT_IDS,
    )


def _reject_unknown_contact_directory_filters(request: Request) -> None:
    """422 on unrecognized query-parameter names instead of tolerating them.

    FastAPI's default tolerance is left in place on the pipeline read, whose
    callers predate this slice. The directory has exactly one caller (the
    tracker proxy), so it is held to the exact filter set.
    """
    unknown = sorted(set(request.query_params.keys()) - _CONTACT_DIRECTORY_QUERY_PARAMS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown contact directory filter: {', '.join(unknown)}",
        )


@router.get(
    "/contact-directory",
    response_model=EOMContactDirectoryResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_eom_contact_directory(
    request: Request,
    limit: Annotated[
        int,
        Query(ge=1, le=_MAX_LEAD_REVIEW_LIMIT),
    ] = _DEFAULT_LEAD_REVIEW_LIMIT,
    cursor: Annotated[str | None, Query(min_length=16, max_length=512)] = None,
    search: Annotated[
        str | None,
        Query(min_length=1, max_length=_MAX_CONTACT_DIRECTORY_SEARCH_LENGTH),
    ] = None,
    # A plain string validated against the DERIVED kind tuple below, rather
    # than a Literal that would re-enumerate the canonical set a third time.
    kind: Annotated[str, Query(min_length=1, max_length=32)] = "all",
    lifecycle: Annotated[str, Query(min_length=1, max_length=32)] = "active",
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> EOMContactDirectoryResponse:
    """List active EOM lead and customer contacts for operator discovery.

    The discovery boundary the operator mutation never had: the pipeline read
    admits only stage-active leads, so a customer created or matched through
    ``/operator-contacts`` was otherwise unreachable from every portal surface
    (website #240). This read is separate from ``/leads`` on purpose -- the
    pipeline projection keeps its exact contract, and the directory stays a
    directory rather than growing review-queue semantics.

    The tracker keeps the service bearer and the browser never calls this
    route directly. Reading this projection alters no CRM state.
    """
    _reject_unknown_contact_directory_filters(request)
    if kind not in _CONTACT_DIRECTORY_KINDS:
        raise HTTPException(status_code=422, detail="kind is not supported")
    if lifecycle not in _CONTACT_DIRECTORY_LIFECYCLES:
        raise HTTPException(status_code=422, detail="lifecycle is not supported")
    normalized_search = search.strip() if search is not None else None
    if search is not None and not normalized_search:
        raise HTTPException(status_code=422, detail="search must not be blank")
    # NUL and lone surrogates cannot reach an asyncpg text parameter (they
    # would 500 mid-query instead of failing closed here). Routed through the
    # module's one surrogate choke point so this is not a second copy of the
    # database-invalid character class.
    if normalized_search is not None and "\x00" in _route_surrogates_to_safe_text(
        normalized_search
    ):
        raise HTTPException(status_code=422, detail="search must be valid text")
    decoded_cursor = _decode_lead_review_cursor(cursor)
    rows = await crm.list_eom_contact_directory(
        limit=limit + 1,
        kind=kind,
        lifecycle=lifecycle,
        search=normalized_search,
        cursor_created_at=(
            decoded_cursor["created_at"] if decoded_cursor is not None else None
        ),
        cursor_contact_id=(
            decoded_cursor["contact_id"] if decoded_cursor is not None else None
        ),
    )
    # Page homogeneity is the second enforcement behind the widened status
    # Literal: every admitted row must carry exactly the requested lifecycle.
    # A mixed page is a provider admission bug, and serving it would leak one
    # view's rows into the other, so it fails closed instead of rendering.
    for row in rows:
        if str(row.get("status")) != lifecycle:
            raise HTTPException(
                status_code=500,
                detail="contact directory page violated lifecycle admission",
            )
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page_rows:
        last_row = EOMContactDirectoryItem.model_validate(page_rows[-1])
        next_cursor = _encode_lead_review_cursor(
            created_at=last_row.created_at,
            contact_id=last_row.contact_id,
        )
    return EOMContactDirectoryResponse(
        contacts=[EOMContactDirectoryItem.model_validate(row) for row in page_rows],
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
    "/customer-handoffs/{contact_id}/first-clean-completions",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def record_eom_first_clean_completion(
    contact_id: UUID,
    payload: EOMFirstCleanCompletionRequest,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    completion: EOMFirstCleanCompletionService = Depends(
        _first_clean_completion_dependency
    ),
) -> JSONResponse:
    """Record actual first-clean evidence and its non-sendable candidate."""

    try:
        await completion.require_schema_ready()
        result = await completion.record_completion(
            contact_id=contact_id,
            tracker_customer_id=payload.tracker_customer_id,
            tracker_site_id=payload.tracker_site_id,
            tracker_service_kind=payload.tracker_service_kind,
            tracker_service_id=payload.tracker_service_id,
            completed_at=payload.completed_at,
            operation_key=operation_key,
            actor_id=int(actor["id"]),
            actor_name=str(actor["name"]),
        )
    except EOMFirstCleanCompletionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content={"success": True, **result},
    )


@router.get(
    "/post-clean-onboarding-candidates",
    response_model=EOMPostCleanOnboardingCandidateResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_eom_post_clean_onboarding_candidates(
    limit: Annotated[
        int,
        Query(ge=1, le=_MAX_LEAD_REVIEW_LIMIT),
    ] = _DEFAULT_LEAD_REVIEW_LIMIT,
    cursor: Annotated[str | None, Query(min_length=16, max_length=512)] = None,
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    completion: EOMFirstCleanCompletionService = Depends(
        _first_clean_completion_dependency
    ),
) -> EOMPostCleanOnboardingCandidateResponse:
    """List pending candidates; reading never creates customer side effects."""

    decoded_cursor = _decode_lead_review_cursor(cursor)
    try:
        rows = await completion.list_candidates(
            limit=limit + 1,
            cursor_created_at=(
                decoded_cursor["created_at"] if decoded_cursor is not None else None
            ),
            cursor_candidate_id=(
                decoded_cursor["contact_id"] if decoded_cursor is not None else None
            ),
        )
    except EOMFirstCleanCompletionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page_rows:
        last_row = EOMPostCleanOnboardingCandidateItem.model_validate(page_rows[-1])
        next_cursor = _encode_lead_review_cursor(
            created_at=last_row.created_at,
            contact_id=last_row.candidate_id,
        )
    return EOMPostCleanOnboardingCandidateResponse(
        candidates=[
            EOMPostCleanOnboardingCandidateItem.model_validate(row)
            for row in page_rows
        ],
        limit=limit,
        cursor=cursor,
        has_more=has_more,
        next_cursor=next_cursor,
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
    calendar: Any = Depends(_calendar_dependency),
) -> JSONResponse:
    """Disposition a lead that will not convert; it leaves the review queue.

    Pre-won leads retain the reversible direct disposition. A won lead first
    cancels its persisted first-clean Calendar event and revokes its unsent
    onboarding draft; it is never reported lost after an uncertain cancellation.
    """
    try:
        result = await mark_eom_lead_lost_with_won_teardown(
            crm,
            calendar,
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


@router.post(
    "/contacts/{contact_id}/archive",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def archive_contact(
    contact_id: UUID,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Reversibly park a contact out of the active directory.

    A status-axis soft archive -- never a delete. Both directory kinds are
    admitted; an active won-stage lead is refused toward the lost flow, whose
    Calendar teardown owns that transition. The response echoes the contact's
    identity and resulting status so the caller can validate the target.
    """
    try:
        result = await archive_eom_contact(
            crm,
            EOMContactArchive(
                contact_id=str(contact_id),
                operation_key=operation_key,
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _draft_action_response(result)


@router.post(
    "/contacts/{contact_id}/restore",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def restore_contact(
    contact_id: UUID,
    operation_key: str = Depends(_approval_key_dependency),
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Return one archived contact to the active directory, exactly as it was.

    Archive never touches the stage axis, so restore is a pure status flip
    with the same receipt discipline. Duplicate-identity ambiguity after a
    restore is not re-checked here by design: contact info is not identity
    (#105/#107), no uniqueness constraint can be violated, and the operator
    mutation boundary already 409s ambiguous matches where they matter.
    """
    try:
        result = await restore_eom_contact(
            crm,
            EOMContactRestore(
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


@router.post(
    "/public-onboarding/session",
    dependencies=[Depends(require_eom_funnel_api)],
)
async def get_eom_public_onboarding_session(
    payload: EOMPublicOnboardingSessionRequest,
    public_onboarding: EOMPublicOnboardingConfig = Depends(
        require_eom_public_onboarding_config
    ),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Resolve a valid public link for the tracker, never for a browser directly."""

    authenticated_token = _authenticated_public_onboarding_token(
        payload.token, public_onboarding
    )
    try:
        result = await crm.get_eom_public_onboarding_session(
            token_id=str(authenticated_token.token_id),
            signing_key_fingerprint=authenticated_token.signing_key_fingerprint,
        )
    except EOMLeadConversionError as exc:
        # Durable invalid/revoked/contact-state outcomes deliberately have the
        # same external text as a malformed bearer.
        if exc.status_code in (404, 409):
            raise HTTPException(
                status_code=404,
                detail="Public onboarding link is unavailable",
            ) from exc
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_public_onboarding_session_content(result),
    )


@router.post(
    "/public-onboarding/tracker-context",
    dependencies=[Depends(require_eom_funnel_api)],
)
async def get_eom_public_onboarding_tracker_context(
    payload: EOMPublicOnboardingSessionRequest,
    public_onboarding: EOMPublicOnboardingConfig = Depends(
        require_eom_public_onboarding_config
    ),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Resolve private token context for the Tracker, never for a browser."""

    authenticated_token = _authenticated_public_onboarding_token(
        payload.token, public_onboarding
    )
    try:
        result = await crm.get_eom_public_onboarding_tracker_context(
            token_id=str(authenticated_token.token_id),
            signing_key_fingerprint=authenticated_token.signing_key_fingerprint,
        )
    except EOMLeadConversionError as exc:
        if exc.status_code in (404, 409):
            raise HTTPException(
                status_code=404,
                detail="Public onboarding link is unavailable",
            ) from exc
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_public_onboarding_tracker_context_content(result),
    )


@router.post(
    "/public-onboarding/finalize",
    dependencies=[Depends(require_eom_funnel_api)],
)
async def finalize_eom_public_onboarding(
    payload: EOMPublicOnboardingFinalizeRequest,
    public_onboarding: EOMPublicOnboardingConfig = Depends(
        require_eom_public_onboarding_config
    ),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Redeem one public bearer after the tracker has made local records."""

    authenticated_token = _authenticated_public_onboarding_token(
        payload.token, public_onboarding
    )
    try:
        result = await crm.complete_eom_public_onboarding(
            token_id=str(authenticated_token.token_id),
            signing_key_fingerprint=authenticated_token.signing_key_fingerprint,
            tracker_customer_id=payload.tracker_customer_id,
            tracker_site_id=payload.tracker_site_id,
        )
    except EOMLeadConversionError as exc:
        if exc.status_code in (404, 409):
            raise HTTPException(
                status_code=404,
                detail="Public onboarding link is unavailable",
            ) from exc
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content=_public_onboarding_finalize_content(result),
    )


@router.post(
    "/public-onboarding/recover",
    dependencies=[Depends(require_eom_funnel_api)],
)
async def recover_eom_public_onboarding(
    payload: EOMPublicOnboardingRecoveryRequest,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Complete a durable Tracker reservation after an ambiguous finalization."""

    try:
        result = await crm.recover_eom_public_onboarding(
            token_id=str(payload.token_id),
            contact_id=str(payload.contact_id),
            tracker_customer_id=payload.tracker_customer_id,
            tracker_site_id=payload.tracker_site_id,
            actor_id=int(actor["id"]),
            actor_name=str(actor["name"]),
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if bool(result.get("idempotent"))
            else status.HTTP_201_CREATED
        ),
        content=_public_onboarding_recovery_content(result),
    )


@router.get(
    "/public-onboarding/issued-links",
    response_model=EOMPublicOnboardingIssuedLinkListResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def list_eom_public_onboarding_issued_links(
    limit: Annotated[
        int,
        Query(ge=1, le=_MAX_LEAD_REVIEW_LIMIT),
    ] = _DEFAULT_LEAD_REVIEW_LIMIT,
    cursor: Annotated[str | None, Query(min_length=16, max_length=512)] = None,
    _actor: dict[str, object] = Depends(require_eom_funnel_actor),
    public_onboarding: EOMPublicOnboardingConfig = Depends(
        require_eom_public_onboarding_config
    ),
    crm: Any = Depends(_crm_dependency),
) -> EOMPublicOnboardingIssuedLinkListResponse:
    """List only durable tokens that remain issued for office follow-up.

    A sent onboarding draft is not sufficient evidence here: the customer may
    have redeemed its token, an operator may have revoked it, or its signing
    key may no longer be accepted. This projection reads that live authority
    and alters no handoff, delivery, or token state.
    """

    decoded_cursor = _decode_lead_review_cursor(cursor)
    try:
        rows = await crm.list_eom_public_onboarding_issued_links(
            accepted_signing_key_fingerprints=(
                _accepted_public_onboarding_signing_key_fingerprints(public_onboarding)
            ),
            limit=limit + 1,
            cursor_issued_at=(
                decoded_cursor["created_at"] if decoded_cursor is not None else None
            ),
            cursor_draft_id=(
                decoded_cursor["contact_id"] if decoded_cursor is not None else None
            ),
        )
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page_rows:
        last_row = EOMPublicOnboardingIssuedLinkItem.model_validate(page_rows[-1])
        next_cursor = _encode_lead_review_cursor(
            created_at=last_row.issued_at,
            contact_id=last_row.draft_id,
        )
    return EOMPublicOnboardingIssuedLinkListResponse(
        links=[
            EOMPublicOnboardingIssuedLinkItem.model_validate(row) for row in page_rows
        ],
        limit=limit,
        cursor=cursor,
        has_more=has_more,
        next_cursor=next_cursor,
    )


def _terms_error(exc: EOMTermsAuthorityError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": str(exc)},
    )


@router.post(
    "/terms/versions",
    response_model=EOMTermsVersionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def create_eom_terms_version(
    payload: EOMTermsVersionCreateRequest,
    response: Response,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    authority: EOMTermsAuthority = Depends(_terms_authority_dependency),
) -> EOMTermsVersionResponse:
    """Store one reviewed-but-unpublished Terms snapshot."""

    try:
        result = await authority.create_draft(
            version_label=payload.version_label,
            material_change=payload.material_change,
            documents=payload.documents,
            actor_id=actor["id"],
            actor_name=actor["name"],
        )
    except EOMTermsAuthorityError as exc:
        raise _terms_error(exc) from exc
    response.status_code = (
        status.HTTP_200_OK
        if bool(result["idempotent"])
        else status.HTTP_201_CREATED
    )
    return EOMTermsVersionResponse.model_validate(result)


@router.post(
    "/terms/versions/{version_id}/publish",
    response_model=EOMTermsVersionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def publish_eom_terms_version(
    version_id: UUID,
    response: Response,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    authority: EOMTermsAuthority = Depends(_terms_authority_dependency),
) -> EOMTermsVersionResponse:
    """Publish an immutable snapshot and select it as current."""

    try:
        result = await authority.publish(
            version_id=version_id,
            actor_id=actor["id"],
            actor_name=actor["name"],
        )
    except EOMTermsAuthorityError as exc:
        raise _terms_error(exc) from exc
    response.status_code = (
        status.HTTP_200_OK
        if bool(result["idempotent"])
        else status.HTTP_201_CREATED
    )
    return EOMTermsVersionResponse.model_validate(result)


@router.get(
    "/terms/current",
    response_model=EOMTermsVersionResponse,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def get_current_eom_terms_version(
    authority: EOMTermsAuthority = Depends(_terms_authority_dependency),
) -> EOMTermsVersionResponse:
    """Return the exact current published snapshot without changing state."""

    try:
        result = await authority.get_current()
    except EOMTermsAuthorityError as exc:
        raise _terms_error(exc) from exc
    return EOMTermsVersionResponse.model_validate(result)


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
    config: Any = Depends(get_eom_funnel_api_config),
    crm: Any = Depends(_crm_dependency),
    sender: Any = Depends(_onboarding_sender_dependency),
    email_history: Any = Depends(_onboarding_email_history_dependency),
) -> JSONResponse:
    """Claim the pending draft, send it, then confirm delivery.

    The draft id plus the migration-360 status machine is the idempotency
    mechanism for this action, so no Idempotency-Key header is taken: an
    already-sent draft replays 200 without a second transport call, and a
    concurrent approval loses the atomic claim.
    """
    public_onboarding = (
        require_eom_public_onboarding_config(config)
        if config.public_onboarding_issuance_is_enabled
        else None
    )
    try:
        result = await approve_and_send_eom_onboarding_draft(
            crm,
            EOMOnboardingDraftApproval(
                draft_id=str(draft_id),
                actor_id=int(actor["id"]),
                actor_name=str(actor["name"]),
            ),
            sender=sender,
            email_history=email_history,
            public_onboarding_base_url=(
                public_onboarding.base_url if public_onboarding is not None else None
            ),
            public_onboarding_hmac_secret=(
                public_onboarding.hmac_secret if public_onboarding is not None else None
            ),
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
    "/onboarding-drafts/{draft_id}/revoke-link",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eom_funnel_api)],
)
async def revoke_eom_public_onboarding_link(
    draft_id: UUID,
    actor: dict[str, object] = Depends(require_eom_funnel_actor),
    crm: Any = Depends(_crm_dependency),
) -> JSONResponse:
    """Invalidate an issued customer link without changing sent-email evidence.

    This recovery command intentionally remains available if public issuance is
    later disabled. Otherwise an existing issued token would still fence office
    handoff while staff had no private way to revoke it. It never mints or
    resolves a bearer, and it still requires the normal service credential plus
    an office actor.
    """

    try:
        result = await crm.revoke_eom_public_onboarding_token(draft_id=str(draft_id))
    except EOMLeadConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if not bool(result.get("idempotent")):
        await _log_draft_reconciliation(
            crm,
            {
                "draft_id": str(draft_id),
                "contact_id": result["contact_id"],
            },
            f"employee:{actor['id']}:{actor['name']} revoked public onboarding "
            f"link for draft {draft_id}",
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
