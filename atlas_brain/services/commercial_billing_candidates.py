"""Pure commercial billing-candidate generation for the EOM operator flow.

This module is intentionally separate from ``monthly_invoice_generation``.
That task performs invoice, PDF, service-marker, CRM, notification, and
optional email writes even in its legacy review mode.  A candidate preview must
be safe to regenerate before an operator explicitly approves any of those
effects, so this service owns only source reads and deterministic projection.
"""

from __future__ import annotations

import hashlib
import json
import re
from calendar import monthrange
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable, Mapping, Optional, Protocol
from uuid import UUID

import asyncpg

from .crm_provider import EOM_BILLING_DELIVERY_METHODS
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError


_CENT = Decimal("0.01")
_TAX_RATE_QUANTUM = Decimal("0.0001")
_MAX_CURRENCY = Decimal("9999999999.99")
_PERIOD_PATTERN = re.compile(r"^(?P<year>[0-9]{4})-(?P<month>0[1-9]|1[0-2])$")
_FINGERPRINT_VERSION = 1
_CANDIDATE_CONTRACT_VERSION = 1

RATE_LABEL_PER_VISIT = "Per Visit"
RATE_LABEL_PER_MONTH = "Per Month"
RATE_LABEL_PER_HOUR = "Per Hour"
SUPPORTED_RATE_LABELS = frozenset(
    {RATE_LABEL_PER_VISIT, RATE_LABEL_PER_MONTH, RATE_LABEL_PER_HOUR}
)

BILLING_CANDIDATE_BLOCKER_CODES = frozenset(
    {
        "ambiguous_calendar_service_match",
        "customer_not_commercial",
        "invalid_rate",
        "invalid_rate_label",
        "invalid_tax_rate",
        "missing_billing_delivery_preference",
        "missing_billing_email",
        "missing_calendar_service_evidence",
        "missing_canonical_customer",
        "missing_hours",
        "no_invoice_delivery_preference",
        "source_evidence_invalid",
        "zero_or_invalid_total",
    }
)

_SOURCE_UNAVAILABLE_ERRORS = (
    DatabaseOperationError,
    DatabaseUnavailableError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
    asyncpg.UndefinedTableError,
    asyncpg.UndefinedColumnError,
    asyncpg.InvalidSchemaNameError,
    asyncpg.InsufficientPrivilegeError,
    asyncpg.InvalidAuthorizationSpecificationError,
    OSError,
    TimeoutError,
)


class CommercialBillingCandidatesError(Exception):
    """Base error for the read-only commercial candidate contract."""

    code = "commercial_billing_candidates_error"


class CommercialBillingCandidatesValidationError(CommercialBillingCandidatesError):
    """The requested candidate period is not a calendar month."""

    code = "invalid_billing_period"


class CommercialBillingCandidatesUnavailableError(CommercialBillingCandidatesError):
    """A source of billing evidence cannot be read safely."""

    code = "commercial_billing_candidates_unavailable"


class _CustomerServiceRepository(Protocol):
    async def list_active(self, auto_invoice_only: bool = False) -> list[dict]: ...


class _CalendarProvider(Protocol):
    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> list[Any]: ...


class _CRMProvider(Protocol):
    async def get_eom_payment_customer(
        self,
        contact_id: UUID,
    ) -> dict[str, Any] | None: ...

    async def get_billing_recipient(self, contact_id: UUID) -> dict[str, Any]: ...

    async def get_eom_billing_delivery_preference(
        self,
        contact_id: UUID,
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class _BillingPeriod:
    label: str
    start_date: date
    end_date: date
    calendar_start: datetime
    calendar_end: datetime


@dataclass(frozen=True)
class _SourceService:
    service_id: str
    contact_id: UUID | None
    service_name: str
    rate_label: str
    rate_value: Any
    tax_rate_value: Any
    calendar_keyword: str | None
    service_calendar_id: str | None
    source_index: int


@dataclass(frozen=True)
class _SourceEvent:
    event_id: str
    summary: str
    source_date: date
    start: str
    end: str | None
    calendar_id: str | None
    location: str | None
    status: str
    all_day: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "allDay": self.all_day,
            "calendarId": self.calendar_id,
            "end": self.end,
            "eventId": self.event_id,
            "location": self.location,
            "sourceDate": self.source_date.isoformat(),
            "start": self.start,
            "status": self.status,
            "summary": self.summary,
        }


def parse_billing_period(value: str) -> _BillingPeriod:
    """Parse a strict YYYY-MM period before any provider read."""

    if not isinstance(value, str):
        raise CommercialBillingCandidatesValidationError(
            "billing_period must use YYYY-MM"
        )
    match = _PERIOD_PATTERN.fullmatch(value)
    if match is None:
        raise CommercialBillingCandidatesValidationError(
            "billing_period must use YYYY-MM"
        )
    year = int(match.group("year"))
    month = int(match.group("month"))
    try:
        start_date = date(year, month, 1)
    except ValueError as exc:
        raise CommercialBillingCandidatesValidationError(
            "billing_period must name a supported calendar month"
        ) from exc
    last_day = monthrange(year, month)[1]
    end_date = date(year, month, last_day)
    # Retain the legacy task's 30-hour fetch window. Event inclusion itself is
    # decided by the event's calendar date below, not by this request bound.
    calendar_start = datetime(year, month, 1, tzinfo=timezone.utc)
    calendar_end = datetime(year, month, last_day, tzinfo=timezone.utc) + timedelta(
        hours=30
    )
    return _BillingPeriod(
        label=value,
        start_date=start_date,
        end_date=end_date,
        calendar_start=calendar_start,
        calendar_end=calendar_end,
    )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_service(raw: Mapping[str, Any], source_index: int) -> _SourceService:
    raw_service_id = _string_or_none(raw.get("id"))
    service_id = raw_service_id or f"source-index:{source_index}"
    contact_id: UUID | None
    try:
        contact_raw = raw.get("contact_id")
        contact_id = UUID(str(contact_raw)) if contact_raw is not None else None
    except (TypeError, ValueError, AttributeError):
        contact_id = None
    return _SourceService(
        service_id=service_id,
        contact_id=contact_id,
        service_name=_string_or_none(raw.get("service_name")) or "Unnamed service",
        rate_label=_string_or_none(raw.get("rate_label")) or "",
        rate_value=raw.get("rate"),
        tax_rate_value=raw.get("tax_rate", 0),
        calendar_keyword=_string_or_none(raw.get("calendar_keyword")),
        service_calendar_id=_string_or_none(raw.get("calendar_id")),
        source_index=source_index,
    )


def _currency_cents(value: Any) -> int | None:
    """Return a positive, persisted-money-compatible integer-cent value."""

    try:
        raw = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not raw.is_finite():
        return None
    try:
        normalized = raw.quantize(_CENT)
    except InvalidOperation:
        return None
    if raw != normalized or normalized <= 0 or normalized > _MAX_CURRENCY:
        return None
    return int(normalized * 100)


def _tax_rate_basis_points(value: Any) -> int | None:
    """Normalize a nonnegative NUMERIC(5,4) fractional rate to basis points."""

    try:
        raw = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not raw.is_finite():
        return None
    try:
        normalized = raw.quantize(_TAX_RATE_QUANTUM)
    except InvalidOperation:
        return None
    if raw != normalized or normalized < 0 or normalized > Decimal("1"):
        return None
    return int(normalized * 10_000)


def _event_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise ValueError("calendar event start must be a date or datetime")


def _iso_datetime_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise ValueError("calendar event time must be a date or datetime")


def _source_event(raw: Any, *, selected_calendar_id: str | None) -> _SourceEvent:
    source_date = _event_date(getattr(raw, "start", None))
    event_id = _string_or_none(getattr(raw, "uid", None))
    summary = _string_or_none(getattr(raw, "summary", None))
    if event_id is None or summary is None:
        raise ValueError("calendar event must have a UID and summary")
    status = _string_or_none(getattr(raw, "status", None)) or ""
    return _SourceEvent(
        event_id=event_id,
        summary=summary,
        source_date=source_date,
        start=_iso_datetime_or_none(getattr(raw, "start", None)) or "",
        end=_iso_datetime_or_none(getattr(raw, "end", None)),
        calendar_id=(
            _string_or_none(getattr(raw, "calendar_id", None))
            or selected_calendar_id
        ),
        location=_string_or_none(getattr(raw, "location", None)),
        status=status,
        all_day=bool(getattr(raw, "all_day", False)),
    )


def _blocker(
    code: str,
    message: str,
    *,
    service_id: str | None = None,
    event_ids: list[str] | None = None,
) -> dict[str, Any]:
    if code not in BILLING_CANDIDATE_BLOCKER_CODES:
        code = "source_evidence_invalid"
        message = "Billing source evidence could not be normalized safely."
    return {
        "code": code,
        "eventIds": sorted(set(event_ids or [])),
        "message": message,
        "serviceId": service_id,
    }


def _append_blocker(
    blockers: list[dict[str, Any]],
    seen: set[tuple[str, str | None, tuple[str, ...]]],
    blocker: dict[str, Any],
) -> None:
    key = (
        str(blocker["code"]),
        blocker["serviceId"],
        tuple(blocker["eventIds"]),
    )
    if key not in seen:
        seen.add(key)
        blockers.append(blocker)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _amount_with_tax(subtotal_cents: int, tax_rate_basis_points: int) -> tuple[int, int]:
    tax_cents = int(
        (Decimal(subtotal_cents) * Decimal(tax_rate_basis_points) / Decimal(10_000))
        .quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    return tax_cents, subtotal_cents + tax_cents


def _line_item(
    *,
    service: _SourceService,
    source_date: date,
    quantity: int | None,
    quantity_unit: str,
    rate_cents: int | None,
    event_ids: list[str],
    locations: list[str],
) -> dict[str, Any]:
    amount_cents = (
        quantity * rate_cents
        if quantity is not None and rate_cents is not None
        else None
    )
    return {
        "amountCents": amount_cents,
        "description": service.service_name,
        "eventIds": sorted(set(event_ids)),
        "locations": sorted(set(locations)),
        "quantity": quantity,
        "quantityUnit": quantity_unit,
        "rateCents": rate_cents,
        "serviceId": service.service_id,
        "sourceDate": source_date.isoformat(),
    }


class CommercialBillingCandidateService:
    """Project current commercial billing evidence without any financial write."""

    def __init__(
        self,
        *,
        customer_service_repository: _CustomerServiceRepository,
        calendar_provider_loader: Callable[[], _CalendarProvider],
        crm_provider_loader: Callable[[], _CRMProvider],
        calendar_id: str | None,
    ) -> None:
        self._customer_service_repository = customer_service_repository
        self._calendar_provider_loader = calendar_provider_loader
        self._crm_provider_loader = crm_provider_loader
        self._calendar_id = _string_or_none(calendar_id)

    async def preview(self, *, billing_period: str) -> dict[str, Any]:
        """Return a deterministic, non-persisted preview for one month."""

        period = parse_billing_period(billing_period)
        services = await self._list_active_services()
        if not services:
            return {
                "billingPeriod": period.label,
                "calendarId": self._calendar_id,
                "candidates": [],
                "contractVersion": _CANDIDATE_CONTRACT_VERSION,
                "summary": {"blockedCandidateCount": 0, "candidateCount": 0},
            }

        events, malformed_event_count = await self._list_calendar_events(period)
        return await self._build_preview(
            period=period,
            services=services,
            events=events,
            malformed_event_count=malformed_event_count,
        )

    async def _list_active_services(self) -> list[_SourceService]:
        try:
            rows = await self._customer_service_repository.list_active(
                auto_invoice_only=True
            )
        except _SOURCE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingCandidatesUnavailableError(
                "Commercial service evidence is unavailable"
            ) from exc
        if not isinstance(rows, list):
            raise CommercialBillingCandidatesUnavailableError(
                "Commercial service evidence is unavailable"
            )
        services: list[_SourceService] = []
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                services.append(
                    _SourceService(
                        service_id=f"source-index:{index}",
                        contact_id=None,
                        service_name="Unnamed service",
                        rate_label="",
                        rate_value=None,
                        tax_rate_value=None,
                        calendar_keyword=None,
                        service_calendar_id=None,
                        source_index=index,
                    )
                )
                continue
            services.append(_source_service(raw, index))
        return sorted(services, key=lambda item: (item.service_id, item.source_index))

    async def _list_calendar_events(
        self,
        period: _BillingPeriod,
    ) -> tuple[list[_SourceEvent], int]:
        try:
            provider = self._calendar_provider_loader()
            rows = await provider.list_events(
                period.calendar_start,
                period.calendar_end,
                calendar_id=self._calendar_id,
            )
        except (RuntimeError, *_SOURCE_UNAVAILABLE_ERRORS) as exc:
            raise CommercialBillingCandidatesUnavailableError(
                "Commercial calendar evidence is unavailable"
            ) from exc
        if not isinstance(rows, list):
            raise CommercialBillingCandidatesUnavailableError(
                "Commercial calendar evidence is unavailable"
            )
        events: list[_SourceEvent] = []
        malformed_event_count = 0
        for raw in rows:
            if _string_or_none(getattr(raw, "status", None)) != "confirmed":
                continue
            try:
                event = _source_event(raw, selected_calendar_id=self._calendar_id)
            except (TypeError, ValueError, AttributeError):
                # A malformed calendar row cannot be safely assigned. The
                # candidate builder adds a visible blocker to each bundle.
                malformed_event_count += 1
                continue
            if period.start_date <= event.source_date <= period.end_date:
                events.append(event)
        return (
            sorted(
                events,
                key=lambda item: (
                    item.source_date.isoformat(),
                    item.start,
                    item.event_id,
                    item.summary,
                ),
            ),
            malformed_event_count,
        )

    async def _build_preview(
        self,
        *,
        period: _BillingPeriod,
        services: list[_SourceService],
        events: list[_SourceEvent],
        malformed_event_count: int,
    ) -> dict[str, Any]:
        assignments, matches, collisions = self._assign_events(events, services)
        bundles: dict[str, list[_SourceService]] = defaultdict(list)
        for service in services:
            bundle_key = (
                str(service.contact_id)
                if service.contact_id is not None
                else f"unlinked:{service.service_id}"
            )
            bundles[bundle_key].append(service)

        crm: _CRMProvider | None = None
        candidates: list[dict[str, Any]] = []
        for bundle_key in sorted(bundles):
            bundle_services = sorted(
                bundles[bundle_key], key=lambda item: (item.service_id, item.source_index)
            )
            contact_id = bundle_services[0].contact_id
            customer: dict[str, Any] | None = None
            recipient: dict[str, Any] | None = None
            delivery_preference: dict[str, Any] | None = None
            if contact_id is not None:
                if crm is None:
                    crm = self._load_crm_provider()
                (
                    customer,
                    recipient,
                    delivery_preference,
                ) = await self._load_customer_evidence(crm, contact_id)
            candidate = self._build_candidate(
                period=period,
                contact_id=contact_id,
                services=bundle_services,
                customer=customer,
                recipient=recipient,
                delivery_preference=delivery_preference,
                assignments=assignments,
                matches=matches,
                collisions=collisions,
                malformed_event_count=malformed_event_count,
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                item["customer"]["contactId"] or "",
                item["candidateKey"],
            )
        )
        return {
            "billingPeriod": period.label,
            "calendarId": self._calendar_id,
            "candidates": candidates,
            "contractVersion": _CANDIDATE_CONTRACT_VERSION,
            "summary": {
                "blockedCandidateCount": sum(
                    1 for candidate in candidates if candidate["blockers"]
                ),
                "candidateCount": len(candidates),
            },
        }

    def _load_crm_provider(self) -> _CRMProvider:
        try:
            return self._crm_provider_loader()
        except (RuntimeError, *_SOURCE_UNAVAILABLE_ERRORS) as exc:
            raise CommercialBillingCandidatesUnavailableError(
                "Canonical customer evidence is unavailable"
            ) from exc

    async def _load_customer_evidence(
        self,
        crm: _CRMProvider,
        contact_id: UUID,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        try:
            customer = await crm.get_eom_payment_customer(contact_id)
            delivery_preference = (
                await crm.get_eom_billing_delivery_preference(contact_id)
                if customer is not None
                else None
            )
            delivery_method = _string_or_none(
                (delivery_preference or {}).get("deliveryMethod")
            )
            recipient = None
            if customer is not None and delivery_method in {
                None,
                "gmail_pdf",
            }:
                # Preserve today's recipient evidence for unconfigured
                # customers while avoiding an irrelevant Gmail dependency for
                # explicit manual-Square and receipt-only policies.
                recipient = await crm.get_billing_recipient(contact_id)
        except (RuntimeError, *_SOURCE_UNAVAILABLE_ERRORS) as exc:
            raise CommercialBillingCandidatesUnavailableError(
                "Canonical customer evidence is unavailable"
            ) from exc
        return customer, recipient, delivery_preference

    @staticmethod
    def _assign_events(
        events: list[_SourceEvent],
        services: list[_SourceService],
    ) -> tuple[
        dict[str, list[_SourceEvent]],
        dict[str, list[_SourceEvent]],
        dict[str, list[dict[str, Any]]],
    ]:
        assignments: dict[str, list[_SourceEvent]] = defaultdict(list)
        matches: dict[str, list[_SourceEvent]] = defaultdict(list)
        collisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            summary = event.summary.casefold()
            matchers = [
                service
                for service in services
                if service.calendar_keyword
                and service.calendar_keyword.casefold() in summary
            ]
            if not matchers:
                continue
            matchers.sort(
                key=lambda service: (
                    -len(service.calendar_keyword or ""),
                    (service.calendar_keyword or "").casefold(),
                    service.service_id,
                )
            )
            for service in matchers:
                matches[service.service_id].append(event)
            assignments[matchers[0].service_id].append(event)
            if len(matchers) > 1:
                collision = {
                    "eventId": event.event_id,
                    "matchedServiceIds": [service.service_id for service in matchers],
                    "resolution": (
                        "alphabetical_tiebreak"
                        if len(matchers[0].calendar_keyword or "")
                        == len(matchers[1].calendar_keyword or "")
                        else "longest_keyword"
                    ),
                    "selectedServiceId": matchers[0].service_id,
                }
                for service in matchers:
                    collisions[service.service_id].append(collision)
        return assignments, matches, collisions

    def _build_candidate(
        self,
        *,
        period: _BillingPeriod,
        contact_id: UUID | None,
        services: list[_SourceService],
        customer: dict[str, Any] | None,
        recipient: dict[str, Any] | None,
        delivery_preference: dict[str, Any] | None,
        assignments: Mapping[str, list[_SourceEvent]],
        matches: Mapping[str, list[_SourceEvent]],
        collisions: Mapping[str, list[dict[str, Any]]],
        malformed_event_count: int,
    ) -> dict[str, Any]:
        contact_text = str(contact_id) if contact_id is not None else None
        candidate_key_contact = contact_text or f"unlinked:{services[0].service_id}"
        blockers: list[dict[str, Any]] = []
        seen_blockers: set[tuple[str, str | None, tuple[str, ...]]] = set()
        incomplete_total = False

        if malformed_event_count:
            _append_blocker(
                blockers,
                seen_blockers,
                _blocker(
                    "source_evidence_invalid",
                    "One or more confirmed calendar events could not be normalized safely.",
                ),
            )
            incomplete_total = True

        customer_type = _string_or_none((customer or {}).get("customer_type"))
        customer_view = {
            "contactId": contact_text,
            "customerType": customer_type or "unknown",
            "displayName": _string_or_none((customer or {}).get("customer_name")),
        }
        recipient_eligible = bool((recipient or {}).get("eligible"))
        recipient_view = {
            "contactId": contact_text,
            "displayName": (
                _string_or_none((recipient or {}).get("displayName"))
                if recipient_eligible
                else None
            ),
            "email": (
                _string_or_none((recipient or {}).get("email"))
                if recipient_eligible
                else None
            ),
        }
        delivery_method = _string_or_none(
            (delivery_preference or {}).get("deliveryMethod")
        )

        if customer is None:
            _append_blocker(
                blockers,
                seen_blockers,
                _blocker(
                    "missing_canonical_customer",
                    "The service agreement has no active canonical EOM customer.",
                ),
            )
        elif customer_type != "commercial":
            _append_blocker(
                blockers,
                seen_blockers,
                _blocker(
                    "customer_not_commercial",
                    "The canonical customer is not classified as commercial.",
                ),
            )
        if (
            customer is not None
            and delivery_method in {None, "gmail_pdf"}
            and not recipient_eligible
        ):
            recipient_reason = _string_or_none((recipient or {}).get("reason"))
            if recipient_reason == "no_email":
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "missing_billing_email",
                        "The canonical commercial customer has no valid billing email.",
                    ),
                )
            else:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "source_evidence_invalid",
                        "Canonical customer and billing-recipient evidence disagree.",
                    ),
                )
                incomplete_total = True
        if customer_type == "commercial":
            if delivery_method is None:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "missing_billing_delivery_preference",
                        "No explicit billing delivery preference is recorded.",
                    ),
                )
            elif delivery_method not in EOM_BILLING_DELIVERY_METHODS:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "source_evidence_invalid",
                        "The recorded billing delivery preference is not supported.",
                    ),
                )
            elif delivery_method == "no_invoice_residential_receipt":
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "no_invoice_delivery_preference",
                        "The recorded delivery preference does not permit a commercial invoice.",
                    ),
                )

        line_items: list[dict[str, Any]] = []
        service_views: list[dict[str, Any]] = []
        source_events: dict[str, _SourceEvent] = {}
        rate_basis_points: list[int] = []
        subtotal_cents = 0

        for service in services:
            rate_cents = _currency_cents(service.rate_value)
            tax_basis_points = _tax_rate_basis_points(service.tax_rate_value)
            service_views.append(
                {
                    "calendarId": service.service_calendar_id,
                    "calendarKeyword": service.calendar_keyword,
                    "rateCents": rate_cents,
                    "rateLabel": service.rate_label or None,
                    "serviceId": service.service_id,
                    "serviceName": service.service_name,
                    "taxRateBasisPoints": tax_basis_points,
                }
            )
            matched_events = matches.get(service.service_id, [])
            assigned_events = assignments.get(service.service_id, [])
            for event in matched_events:
                source_events[event.event_id] = event
            for collision in collisions.get(service.service_id, []):
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "ambiguous_calendar_service_match",
                        "A calendar event matches more than one service agreement.",
                        service_id=service.service_id,
                        event_ids=[collision["eventId"]],
                    ),
                )
                incomplete_total = True
            if service.contact_id is None:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "missing_canonical_customer",
                        "The service agreement is missing a canonical customer link.",
                        service_id=service.service_id,
                    ),
                )
                incomplete_total = True
            if service.rate_label not in SUPPORTED_RATE_LABELS:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "invalid_rate_label",
                        "The service rate label is not supported for billing.",
                        service_id=service.service_id,
                    ),
                )
                incomplete_total = True
                continue
            if rate_cents is None:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "invalid_rate",
                        "The service rate must be a positive cent-precise amount.",
                        service_id=service.service_id,
                    ),
                )
                incomplete_total = True
                continue
            if tax_basis_points is None:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "invalid_tax_rate",
                        "The service tax rate must be a finite nonnegative rate.",
                        service_id=service.service_id,
                    ),
                )
                incomplete_total = True
            else:
                rate_basis_points.append(tax_basis_points)

            if service.rate_label == RATE_LABEL_PER_MONTH:
                item = _line_item(
                    service=service,
                    source_date=period.start_date,
                    quantity=1,
                    quantity_unit="month",
                    rate_cents=rate_cents,
                    event_ids=[],
                    locations=[],
                )
                line_items.append(item)
                subtotal_cents += item["amountCents"] or 0
                continue

            if not matched_events:
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "missing_calendar_service_evidence",
                        "No confirmed calendar event matches this service in the billing period.",
                        service_id=service.service_id,
                    ),
                )
                incomplete_total = True
                continue

            if service.rate_label == RATE_LABEL_PER_HOUR:
                by_date: dict[date, list[_SourceEvent]] = defaultdict(list)
                for event in assigned_events:
                    by_date[event.source_date].append(event)
                for source_date, dated_events in sorted(by_date.items()):
                    line_items.append(
                        _line_item(
                            service=service,
                            source_date=source_date,
                            quantity=None,
                            quantity_unit="hour",
                            rate_cents=rate_cents,
                            event_ids=[event.event_id for event in dated_events],
                            locations=[
                                event.location
                                for event in dated_events
                                if event.location is not None
                            ],
                        )
                    )
                _append_blocker(
                    blockers,
                    seen_blockers,
                    _blocker(
                        "missing_hours",
                        "Hours must be supplied before this service can be billed.",
                        service_id=service.service_id,
                        event_ids=[event.event_id for event in matched_events],
                    ),
                )
                incomplete_total = True
                continue

            by_date: dict[date, list[_SourceEvent]] = defaultdict(list)
            for event in assigned_events:
                by_date[event.source_date].append(event)
            for source_date, dated_events in sorted(by_date.items()):
                item = _line_item(
                    service=service,
                    source_date=source_date,
                    quantity=len(dated_events),
                    quantity_unit="visit",
                    rate_cents=rate_cents,
                    event_ids=[event.event_id for event in dated_events],
                    locations=[
                        event.location
                        for event in dated_events
                        if event.location is not None
                    ],
                )
                line_items.append(item)
                subtotal_cents += item["amountCents"] or 0

        line_items.sort(
            key=lambda item: (
                item["sourceDate"],
                item["serviceId"],
                item["eventIds"],
            )
        )
        service_views.sort(key=lambda item: item["serviceId"])
        source_event_views = [
            source_events[event_id].as_dict()
            for event_id in sorted(
                source_events,
                key=lambda event_id: (
                    source_events[event_id].source_date.isoformat(),
                    source_events[event_id].start,
                    event_id,
                ),
            )
        ]

        tax_rate_basis_points = max(rate_basis_points) if rate_basis_points else 0
        tax_cents: int | None
        total_cents: int | None
        if incomplete_total:
            tax_cents = None
            total_cents = None
        else:
            tax_cents, total_cents = _amount_with_tax(
                subtotal_cents, tax_rate_basis_points
            )
        if total_cents is None or total_cents <= 0:
            _append_blocker(
                blockers,
                seen_blockers,
                _blocker(
                    "zero_or_invalid_total",
                    "The candidate total is zero or cannot be calculated from current evidence.",
                ),
            )
        blockers.sort(
            key=lambda blocker: (
                blocker["code"],
                blocker["serviceId"] or "",
                blocker["eventIds"],
            )
        )
        candidate = {
            "billingPeriod": period.label,
            "blockers": blockers,
            "calendarId": self._calendar_id,
            "candidateKey": f"commercial-billing:{candidate_key_contact}:{period.label}",
            "customer": customer_view,
            "deliveryMethod": delivery_method,
            "fingerprintVersion": _FINGERPRINT_VERSION,
            "lineItems": line_items,
            "recipient": recipient_view,
            "services": service_views,
            "sourceEvents": source_event_views,
            "subtotalCents": subtotal_cents,
            "taxCents": tax_cents,
            "taxRateBasisPoints": tax_rate_basis_points,
            "totalCents": total_cents,
        }
        candidate["sourceFingerprint"] = _fingerprint(candidate)
        return candidate


def get_commercial_billing_candidate_service() -> CommercialBillingCandidateService:
    """Construct the production read-only candidate service lazily per request."""

    from ..config import settings
    from ..storage.repositories.customer_service import get_customer_service_repo
    from .calendar_provider import get_calendar_provider
    from .crm_provider import get_crm_provider

    return CommercialBillingCandidateService(
        customer_service_repository=get_customer_service_repo(),
        calendar_provider_loader=get_calendar_provider,
        crm_provider_loader=get_crm_provider,
        calendar_id=settings.invoicing.auto_invoice_calendar_id,
    )


__all__ = [
    "BILLING_CANDIDATE_BLOCKER_CODES",
    "CommercialBillingCandidateService",
    "CommercialBillingCandidatesError",
    "CommercialBillingCandidatesUnavailableError",
    "CommercialBillingCandidatesValidationError",
    "SUPPORTED_RATE_LABELS",
    "get_commercial_billing_candidate_service",
    "parse_billing_period",
]
