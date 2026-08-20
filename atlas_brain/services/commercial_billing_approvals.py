"""Explicit, exact-cent approval of one retained commercial billing candidate.

This is intentionally separate from the pure preview/run service and the
legacy monthly task.  It creates only a draft ATLAS invoice plus durable
approval evidence; it never creates a PDF, Gmail draft, email, CRM event, or
service-invoiced marker.
"""

from __future__ import annotations

import hashlib
import json
import re
from calendar import month_name
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable, Mapping, Optional
from uuid import UUID, uuid4

import asyncpg

from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError
from ..storage.repositories.invoice import recurring_invoice_dedup_schema_ready
from .commercial_billing_candidates import (
    CommercialBillingCandidateService,
    CommercialBillingCandidatesUnavailableError,
    CommercialBillingCandidatesValidationError,
    get_commercial_billing_candidate_service,
)
from .commercial_billing_runs import (
    lock_commercial_billing_candidate_identity,
    lock_commercial_billing_run_candidate,
)
from .commercial_billing_candidate_overrides import (
    CommercialBillingCandidateOverrideValidationError,
    MAX_INVOICE_LINE_DESCRIPTION_LENGTH,
    override_review_fingerprint,
)
from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID


_APPROVAL_SOURCE = "eom_admin"
_INVOICE_SOURCE = "eom_commercial_billing"
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_CANDIDATE_KEY_LENGTH = 512
_MAX_INVOICE_CENTS = 999_999_999_999
_DATABASE_UNAVAILABLE_ERRORS = (
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
)


class CommercialBillingApprovalError(Exception):
    code = "commercial_billing_approval_error"


class CommercialBillingApprovalValidationError(CommercialBillingApprovalError):
    code = "invalid_commercial_billing_approval"


class CommercialBillingApprovalNotFoundError(CommercialBillingApprovalError):
    code = "commercial_billing_candidate_not_found"


class CommercialBillingApprovalConflictError(CommercialBillingApprovalError):
    code = "commercial_billing_approval_idempotency_conflict"


class CommercialBillingApprovalStaleError(CommercialBillingApprovalError):
    code = "stale_commercial_billing_candidate"


class CommercialBillingApprovalUnavailableError(CommercialBillingApprovalError):
    code = "commercial_billing_approvals_unavailable"


@dataclass(frozen=True)
class _InvoiceDraft:
    billing_period: date
    contact_id: UUID
    customer_email: str | None
    customer_name: str
    due_date: date
    invoice_for: str
    issue_date: date
    line_items: list[dict[str, Any]]
    metadata: dict[str, Any]
    source_ref: str
    subtotal: Decimal
    tax_amount: Decimal
    tax_rate: Decimal
    total: Decimal


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CommercialBillingApprovalValidationError(
            "Commercial billing candidate evidence is invalid"
        ) from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _verified_candidate_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Return the retained candidate fingerprint only when it covers its evidence."""

    declared = snapshot.get("sourceFingerprint")
    if not isinstance(declared, str) or _FINGERPRINT.fullmatch(declared) is None:
        raise CommercialBillingApprovalUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        )
    evidence = dict(snapshot)
    evidence.pop("sourceFingerprint", None)
    try:
        calculated = _fingerprint(evidence)
    except CommercialBillingApprovalValidationError as exc:
        raise CommercialBillingApprovalUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        ) from exc
    if calculated != declared:
        raise CommercialBillingApprovalUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        )
    return declared


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _timestamp(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _money(cents: int) -> Decimal:
    return Decimal(cents) / Decimal(100)


def _money_text(cents: int) -> str:
    return f"{_money(cents):.2f}"


def _cents(value: Any) -> int:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CommercialBillingApprovalUnavailableError(
            "Approved invoice money is invalid"
        ) from exc
    cents = decimal * Decimal(100)
    if not cents.is_finite() or cents != cents.to_integral_value():
        raise CommercialBillingApprovalUnavailableError("Approved invoice money is invalid")
    return int(cents)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing snapshot evidence is invalid"
            ) from exc
    if not isinstance(value, Mapping):
        raise CommercialBillingApprovalUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        )
    return dict(value)


def _required_text(value: Any, field: str, *, limit: int = 256) -> str:
    text = _text(value)
    if text is None or not text.strip() or len(text.strip()) > limit:
        raise CommercialBillingApprovalValidationError(
            f"Commercial billing candidate {field} is invalid"
        )
    return text.strip()


def _validate_key(value: str, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingApprovalValidationError(f"{field} is required")
    key = value.strip()
    if not key or len(key) > limit:
        raise CommercialBillingApprovalValidationError(
            f"{field} must contain 1 to {limit} characters"
        )
    return key


def _source_ref(candidate_key: str, review_fingerprint: str) -> str:
    digest = hashlib.sha256(f"{candidate_key}:{review_fingerprint}".encode()).hexdigest()
    return f"eom-commercial-billing:{digest}"


def _invoice_draft(
    snapshot: Mapping[str, Any],
    *,
    billing_run_id: UUID,
    expected_candidate_key: str,
    expected_source_fingerprint: str,
    expected_review_fingerprint: str,
    actor: str,
    due_days: int,
    issue_date: date,
) -> _InvoiceDraft:
    if snapshot.get("blockers") != []:
        raise CommercialBillingApprovalValidationError(
            "Blocked commercial billing candidates cannot be approved"
        )
    if snapshot.get("sourceFingerprint") != expected_source_fingerprint:
        raise CommercialBillingApprovalConflictError(
            "The submitted source fingerprint does not match the reviewed candidate"
        )
    candidate_key = _validate_key(
        snapshot.get("candidateKey"), field="Candidate key", limit=_MAX_CANDIDATE_KEY_LENGTH
    )
    if candidate_key != expected_candidate_key:
        raise CommercialBillingApprovalUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        )
    customer = snapshot.get("customer")
    if not isinstance(customer, Mapping) or customer.get("customerType") != "commercial":
        raise CommercialBillingApprovalValidationError(
            "Only canonical commercial customers can receive a commercial invoice"
        )
    try:
        contact_id = UUID(_required_text(customer.get("contactId"), "customer contact"))
    except ValueError as exc:
        raise CommercialBillingApprovalValidationError(
            "Commercial billing candidate customer contact is invalid"
        ) from exc
    customer_name = _required_text(customer.get("displayName"), "customer name")
    delivery_method = snapshot.get("deliveryMethod")
    if delivery_method not in {"gmail_pdf", "manual_square"}:
        raise CommercialBillingApprovalValidationError(
            "Commercial billing candidate delivery method cannot create an invoice"
        )
    customer_email = None
    if delivery_method == "gmail_pdf":
        recipient = snapshot.get("recipient")
        if not isinstance(recipient, Mapping):
            raise CommercialBillingApprovalValidationError(
                "Commercial billing candidate billing recipient is invalid"
            )
        customer_email = _required_text(recipient.get("email"), "billing email")
    if isinstance(due_days, bool) or not isinstance(due_days, int) or not 1 <= due_days <= 365:
        raise CommercialBillingApprovalUnavailableError("Configured invoice due days are invalid")

    period_text = _required_text(snapshot.get("billingPeriod"), "billing period", limit=7)
    try:
        period = date.fromisoformat(f"{period_text}-01")
    except ValueError as exc:
        raise CommercialBillingApprovalValidationError(
            "Commercial billing candidate period is invalid"
        ) from exc
    lines = snapshot.get("lineItems")
    if not isinstance(lines, list) or not lines:
        raise CommercialBillingApprovalValidationError(
            "Commercial billing candidate has no billable line items"
        )
    line_items: list[dict[str, Any]] = []
    subtotal_cents = 0
    names: list[str] = []
    for line in lines:
        if not isinstance(line, Mapping):
            raise CommercialBillingApprovalValidationError(
                "Commercial billing line item is invalid"
            )
        amount_cents = line.get("amountCents")
        description = _required_text(
            line.get("description"),
            "line-item description",
            limit=MAX_INVOICE_LINE_DESCRIPTION_LENGTH,
        )
        source_date = _required_text(line.get("sourceDate"), "line-item source date", limit=10)
        try:
            date.fromisoformat(source_date)
        except ValueError as exc:
            raise CommercialBillingApprovalValidationError(
                "Commercial billing line-item source date is invalid"
            ) from exc
        if line.get("kind") == "adjustment":
            if (
                isinstance(amount_cents, bool)
                or not isinstance(amount_cents, int)
                or amount_cents == 0
                or abs(amount_cents) > _MAX_INVOICE_CENTS
            ):
                raise CommercialBillingApprovalValidationError(
                    "Commercial billing adjustment cents are invalid"
                )
            subtotal_cents += amount_cents
            if description not in names:
                names.append(description)
            line_items.append(
                {
                    "amount": _money_text(amount_cents),
                    "date": source_date,
                    "description": description,
                    "quantity": 1,
                    "unit_price": _money_text(amount_cents),
                }
            )
            continue
        quantity = line.get("quantity")
        rate_cents = line.get("rateCents")
        quantity_unit = line.get("quantityUnit")
        if quantity_unit == "hour":
            minutes = line.get("quantityMinutes")
            if (
                isinstance(minutes, bool)
                or not isinstance(minutes, int)
                or minutes <= 0
                or isinstance(rate_cents, bool)
                or not isinstance(rate_cents, int)
                or rate_cents <= 0
                or isinstance(amount_cents, bool)
                or not isinstance(amount_cents, int)
                or amount_cents
                != int(
                    (Decimal(rate_cents) * Decimal(minutes) / Decimal(60)).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP
                    )
                )
                or amount_cents > _MAX_INVOICE_CENTS
            ):
                raise CommercialBillingApprovalValidationError(
                    "Commercial billing hourly line-item cents are invalid"
                )
            quantity_for_invoice = f"{(Decimal(minutes) / Decimal(60)):.4f}".rstrip("0").rstrip(".")
        else:
            quantity_for_invoice = quantity
            if (
                isinstance(quantity, bool)
                or not isinstance(quantity, int)
                or quantity <= 0
                or isinstance(rate_cents, bool)
                or not isinstance(rate_cents, int)
                or rate_cents <= 0
                or isinstance(amount_cents, bool)
                or not isinstance(amount_cents, int)
                or amount_cents != quantity * rate_cents
                or amount_cents > _MAX_INVOICE_CENTS
            ):
                raise CommercialBillingApprovalValidationError(
                    "Commercial billing line-item cents are invalid"
                )
        subtotal_cents += amount_cents
        if description not in names:
            names.append(description)
        line_items.append(
            {
                "amount": _money_text(amount_cents),
                "date": source_date,
                "description": description,
                "quantity": quantity_for_invoice,
                "unit_price": _money_text(rate_cents),
            }
        )
    subtotal = snapshot.get("subtotalCents")
    tax = snapshot.get("taxCents")
    total = snapshot.get("totalCents")
    basis_points = snapshot.get("taxRateBasisPoints")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in (subtotal, tax, total, basis_points))
        or subtotal != subtotal_cents
        or subtotal > _MAX_INVOICE_CENTS
        or tax < 0
        or total != subtotal + tax
        or not 0 < total <= _MAX_INVOICE_CENTS
        or not 0 <= basis_points <= 10_000
        or tax != int(
            (Decimal(subtotal) * Decimal(basis_points) / Decimal(10_000)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
    ):
        raise CommercialBillingApprovalValidationError(
            "Commercial billing candidate totals are invalid"
        )
    return _InvoiceDraft(
        billing_period=period,
        contact_id=contact_id,
        customer_email=customer_email,
        customer_name=customer_name,
        due_date=issue_date + timedelta(days=due_days),
        invoice_for=f"{', '.join(names)} - {month_name[period.month]} {period.year}",
        issue_date=issue_date,
        line_items=line_items,
        metadata={
            "approvedBy": actor,
            "candidateKey": candidate_key,
            "commercialBillingExactLineAmounts": True,
            "commercialBillingRunId": str(billing_run_id),
            "deliveryMethod": delivery_method,
            "reviewFingerprint": expected_review_fingerprint,
            "sourceFingerprint": expected_source_fingerprint,
        },
        source_ref=_source_ref(candidate_key, expected_review_fingerprint),
        subtotal=_money(subtotal),
        tax_amount=_money(tax),
        tax_rate=Decimal(basis_points) / Decimal(10_000),
        total=_money(total),
    )


class CommercialBillingApprovalService:
    """Own one approved-candidate -> draft-invoice transaction."""

    def __init__(
        self,
        *,
        pool: Optional[DatabasePool] = None,
        candidate_service_loader: Callable[[], CommercialBillingCandidateService] = (
            get_commercial_billing_candidate_service
        ),
        due_days_loader: Callable[[], int] | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._configured_pool = pool
        self._candidate_service_loader = candidate_service_loader
        self._due_days_loader = due_days_loader or self._configured_due_days
        self._today = today

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing database unavailable"
            )
        return pool

    @staticmethod
    def _configured_due_days() -> int:
        from ..config import settings

        return settings.invoicing.auto_invoice_due_days

    async def approve(
        self,
        *,
        billing_run_id: UUID,
        candidate_key: str,
        expected_source_fingerprint: str,
        idempotency_key: str,
        actor: str,
        expected_review_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(billing_run_id, UUID):
            raise CommercialBillingApprovalValidationError("Billing run id is invalid")
        key = _validate_key(idempotency_key, field="Idempotency key", limit=_MAX_IDEMPOTENCY_KEY_LENGTH)
        selected = _validate_key(candidate_key, field="Candidate key", limit=_MAX_CANDIDATE_KEY_LENGTH)
        if (
            not isinstance(expected_source_fingerprint, str)
            or _FINGERPRINT.fullmatch(expected_source_fingerprint) is None
        ):
            raise CommercialBillingApprovalValidationError("Source fingerprint is invalid")
        if expected_review_fingerprint is None:
            submitted_review_fingerprint = expected_source_fingerprint
        elif isinstance(expected_review_fingerprint, str) and _FINGERPRINT.fullmatch(
            expected_review_fingerprint
        ) is not None:
            submitted_review_fingerprint = expected_review_fingerprint
        else:
            raise CommercialBillingApprovalValidationError("Review fingerprint is invalid")
        approved_by = _required_text(actor, "authenticated actor", limit=128)
        request_fingerprint = _fingerprint(
            {
                "billingRunId": str(billing_run_id),
                "candidateKey": selected,
                "reviewFingerprint": submitted_review_fingerprint,
                "sourceFingerprint": expected_source_fingerprint,
            }
        )
        legacy_request_fingerprint = (
            _fingerprint(
                {
                    "billingRunId": str(billing_run_id),
                    "candidateKey": selected,
                    "sourceFingerprint": expected_source_fingerprint,
                }
            )
            if submitted_review_fingerprint == expected_source_fingerprint
            else None
        )
        try:
            async with self.pool.transaction() as conn:
                await self._lock(conn, f"operation:{key}")
                existing = await self._find_by_idempotency(conn, key)
                if existing is not None:
                    self._assert_request(
                        existing, request_fingerprint, legacy_request_fingerprint
                    )
                    return {"approval": self._view(existing), "replayed": True}

            stored = await self._stored_candidate(billing_run_id, selected)
            source_snapshot = _json_object(stored["snapshot"])
            if (
                stored["source_fingerprint"] != _verified_candidate_fingerprint(source_snapshot)
                or source_snapshot.get("billingPeriod") != stored["billing_period"]
            ):
                raise CommercialBillingApprovalUnavailableError(
                    "Commercial billing snapshot evidence is invalid"
                )
            snapshot, actual_review_fingerprint = self._effective_snapshot(
                billing_run_id, stored, source_snapshot
            )
            if actual_review_fingerprint != submitted_review_fingerprint:
                raise CommercialBillingApprovalConflictError(
                    "Commercial billing candidate review identity changed before approval"
                )
            issue_date = self._today()
            if isinstance(issue_date, datetime) or not isinstance(issue_date, date):
                raise CommercialBillingApprovalUnavailableError("Invoice issue date is invalid")
            draft = _invoice_draft(
                snapshot,
                billing_run_id=billing_run_id,
                expected_candidate_key=selected,
                expected_source_fingerprint=expected_source_fingerprint,
                expected_review_fingerprint=submitted_review_fingerprint,
                actor=approved_by,
                due_days=self._due_days_loader(),
                issue_date=issue_date,
            )
            await self._assert_current(stored["billing_period"], selected, expected_source_fingerprint)

            async with self.pool.transaction() as conn:
                await self._lock(conn, f"operation:{key}")
                existing = await self._find_by_idempotency(conn, key)
                if existing is not None:
                    self._assert_request(
                        existing, request_fingerprint, legacy_request_fingerprint
                    )
                    return {"approval": self._view(existing), "replayed": True}
                await lock_commercial_billing_candidate_identity(
                    conn,
                    candidate_key=selected,
                    source_fingerprint=expected_source_fingerprint,
                )
                locked_candidate = await lock_commercial_billing_run_candidate(
                    conn,
                    billing_run_id=billing_run_id,
                    candidate_key=selected,
                )
                if locked_candidate is None:
                    raise CommercialBillingApprovalNotFoundError(
                        "Commercial billing candidate not found"
                    )
                if locked_candidate["source_fingerprint"] != expected_source_fingerprint:
                    raise CommercialBillingApprovalConflictError(
                        "Commercial billing candidate fingerprint changed before approval"
                    )
                locked_source_snapshot = _json_object(locked_candidate["snapshot"])
                if (
                    locked_candidate["source_fingerprint"]
                    != _verified_candidate_fingerprint(locked_source_snapshot)
                ):
                    raise CommercialBillingApprovalUnavailableError(
                        "Commercial billing snapshot evidence is invalid"
                    )
                locked_override = await self._latest_override(
                    conn,
                    billing_run_id=billing_run_id,
                    candidate_key=selected,
                    source_fingerprint=expected_source_fingerprint,
                )
                _, locked_review_fingerprint = self._effective_snapshot(
                    billing_run_id,
                    {**dict(locked_candidate), **({"override": locked_override} if locked_override else {})},
                    locked_source_snapshot,
                )
                if locked_review_fingerprint != submitted_review_fingerprint:
                    raise CommercialBillingApprovalConflictError(
                        "Commercial billing candidate review identity changed before approval"
                    )
                review_decision = await self._latest_review_decision(
                    conn,
                    candidate_key=selected,
                    source_fingerprint=expected_source_fingerprint,
                    review_fingerprint=submitted_review_fingerprint,
                )
                if (
                    submitted_review_fingerprint != expected_source_fingerprint
                    and (review_decision is None or review_decision["decision"] != "included")
                ):
                    raise CommercialBillingApprovalConflictError(
                        "Commercial billing candidate override requires an explicit include decision"
                    )
                if review_decision is not None and review_decision["decision"] == "excluded":
                    raise CommercialBillingApprovalConflictError(
                        "Commercial billing candidate is excluded; include it before approval"
                    )
                existing = await self._find_by_candidate(conn, selected, expected_source_fingerprint)
                if existing is not None:
                    return {"approval": self._view(existing), "replayed": True}
                if not await recurring_invoice_dedup_schema_ready(conn):
                    raise CommercialBillingApprovalUnavailableError(
                        "Recurring invoice dedup schema is unavailable"
                    )
                conflicting = await self._find_recurring_period_conflict(
                    conn,
                    contact_id=draft.contact_id,
                    billing_period=f"{draft.billing_period:%Y-%m}",
                )
                if conflicting is not None:
                    raise CommercialBillingApprovalConflictError(
                        f"A recurring invoice already exists for this contact and billing "
                        f"period (source={conflicting['source']}, invoice={conflicting['invoice_number']})"
                    )
                invoice = await self._insert_invoice(conn, draft)
                if invoice is None:
                    raise CommercialBillingApprovalConflictError(
                        "Invoice source reference already exists without approval evidence"
                    )
                await conn.fetchrow(
                    """
                    INSERT INTO commercial_billing_candidate_approvals (
                        id, billing_run_id, candidate_key, source_fingerprint, review_fingerprint, source,
                        idempotency_key, request_fingerprint, invoice_id, state,
                        approved_by, approved_at, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'invoice_created', $10, $11, $11, $11)
                    RETURNING id
                    """,
                    uuid4(), billing_run_id, selected, expected_source_fingerprint,
                    submitted_review_fingerprint, _APPROVAL_SOURCE, key,
                    request_fingerprint, invoice["id"], approved_by, datetime.now(timezone.utc),
                )
                created = await self._find_by_idempotency(conn, key)
                if created is None:
                    raise CommercialBillingApprovalUnavailableError(
                        "Commercial billing approval could not be reconciled"
                    )
                return {"approval": self._view(created), "replayed": False}
        except CommercialBillingApprovalError:
            raise
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise CommercialBillingApprovalConflictError(
                "Commercial billing approval could not be reconciled"
            ) from exc
        except (CommercialBillingCandidatesUnavailableError, CommercialBillingCandidatesValidationError) as exc:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing candidate evidence is unavailable"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def _stored_candidate(self, billing_run_id: UUID, candidate_key: str) -> Any:
        row = await self.pool.fetchrow(
            """
            SELECT run.billing_period, candidate.source_fingerprint, candidate.snapshot,
                   override.id AS override_id,
                   override.revision AS override_revision,
                   override.review_fingerprint AS override_review_fingerprint,
                   override.effective_snapshot AS override_effective_snapshot
            FROM commercial_billing_runs AS run
            JOIN commercial_billing_run_candidates AS candidate
              ON candidate.billing_run_id = run.id
            LEFT JOIN LATERAL (
                SELECT id, revision, review_fingerprint, effective_snapshot
                FROM commercial_billing_candidate_overrides AS candidate_override
                WHERE candidate_override.billing_run_id = candidate.billing_run_id
                  AND candidate_override.candidate_key = candidate.candidate_key
                  AND candidate_override.source_fingerprint = candidate.source_fingerprint
                ORDER BY candidate_override.revision DESC
                LIMIT 1
            ) AS override ON TRUE
            WHERE run.id = $1 AND candidate.candidate_key = $2
            """,
            billing_run_id, candidate_key,
        )
        if row is None:
            raise CommercialBillingApprovalNotFoundError("Commercial billing candidate not found")
        return row

    @staticmethod
    def _effective_snapshot(
        billing_run_id: UUID, stored: Any, source_snapshot: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(billing_run_id, UUID):
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing run identity is invalid"
            )
        source_fingerprint = _row_value(stored, "source_fingerprint")
        if not isinstance(source_fingerprint, str):
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing snapshot evidence is invalid"
            )
        override = _row_value(stored, "override")
        if override is None and _row_value(stored, "override_id") is not None:
            override = {
                "revision": _row_value(stored, "override_revision"),
                "review_fingerprint": _row_value(stored, "override_review_fingerprint"),
                "effective_snapshot": _row_value(stored, "override_effective_snapshot"),
            }
        if override is None:
            return dict(source_snapshot), source_fingerprint
        effective = _json_object(_row_value(override, "effective_snapshot"))
        if effective.get("sourceFingerprint") != source_fingerprint:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing override evidence is invalid"
            )
        try:
            expected = override_review_fingerprint(
                billing_run_id,
                source_fingerprint,
                _row_value(override, "revision"),
                effective,
            )
        except CommercialBillingCandidateOverrideValidationError as exc:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing override evidence is invalid"
            ) from exc
        if _row_value(override, "review_fingerprint") != expected:
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing override evidence is invalid"
            )
        return effective, expected

    @staticmethod
    async def _latest_override(
        conn: Any,
        *,
        billing_run_id: UUID,
        candidate_key: str,
        source_fingerprint: str,
    ) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT id, revision, review_fingerprint, effective_snapshot
            FROM commercial_billing_candidate_overrides
            WHERE billing_run_id = $1
              AND candidate_key = $2
              AND source_fingerprint = $3
            ORDER BY revision DESC
            LIMIT 1
            """,
            billing_run_id,
            candidate_key,
            source_fingerprint,
        )

    async def _assert_current(self, period: str, candidate_key: str, fingerprint: str) -> None:
        preview = await self._candidate_service_loader().preview(billing_period=period)
        candidates = preview.get("candidates") if isinstance(preview, Mapping) else None
        if not isinstance(candidates, list):
            raise CommercialBillingApprovalUnavailableError(
                "Commercial billing candidate evidence is invalid"
            )
        current = next(
            (item for item in candidates if isinstance(item, Mapping) and item.get("candidateKey") == candidate_key),
            None,
        )
        if current is None or _verified_candidate_fingerprint(current) != fingerprint:
            raise CommercialBillingApprovalStaleError(
                "Commercial billing candidate changed; regenerate and review it before approval"
            )

    @staticmethod
    async def _latest_review_decision(
        conn: Any,
        *,
        candidate_key: str,
        source_fingerprint: str,
        review_fingerprint: str,
    ) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT decision
            FROM commercial_billing_candidate_review_decisions
            WHERE candidate_key = $1
              AND source_fingerprint = $2
              AND review_fingerprint = $3
            ORDER BY revision DESC
            LIMIT 1
            """,
            candidate_key,
            source_fingerprint,
            review_fingerprint,
        )

    @staticmethod
    async def _lock(conn: Any, scope: str) -> None:
        await conn.fetchval("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"commercial-billing-approval:{scope}")

    @staticmethod
    async def _find_by_idempotency(conn: Any, key: str) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT a.id AS approval_id, a.billing_run_id, a.candidate_key,
                   a.source_fingerprint, a.review_fingerprint, a.request_fingerprint, a.state,
                   a.approved_by, a.approved_at, i.id AS invoice_id,
                   i.invoice_number, i.status AS invoice_status, i.total_amount,
                   i.issue_date, i.due_date, i.source_ref
            FROM commercial_billing_candidate_approvals AS a
            JOIN invoices AS i ON i.id = a.invoice_id
            WHERE a.source = $1 AND a.idempotency_key = $2
            """,
            _APPROVAL_SOURCE, key,
        )

    @staticmethod
    async def _find_by_candidate(conn: Any, candidate_key: str, fingerprint: str) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT a.id AS approval_id, a.billing_run_id, a.candidate_key,
                   a.source_fingerprint, a.review_fingerprint, a.request_fingerprint, a.state,
                   a.approved_by, a.approved_at, i.id AS invoice_id,
                   i.invoice_number, i.status AS invoice_status, i.total_amount,
                   i.issue_date, i.due_date, i.source_ref
            FROM commercial_billing_candidate_approvals AS a
            JOIN invoices AS i ON i.id = a.invoice_id
            WHERE a.candidate_key = $1 AND a.source_fingerprint = $2
            """,
            candidate_key, fingerprint,
        )

    @staticmethod
    async def _find_recurring_period_conflict(
        conn: Any, *, contact_id: UUID, billing_period: str
    ) -> Any | None:
        """Non-void recurring invoice for this contact/period from either
        recurring writer, or a synthetic hit if that contact/period is
        quarantined (an ambiguous historical collision -- see migration 385's
        Backfill 2/2 and invoices_billing_period_reservations). See migration
        385 / ATLAS #2363: this is the app-level pre-check ahead of
        idx_invoices_recurring_contact_period_source, which is the
        authoritative DB-enforced guarantee for every UNAMBIGUOUS period -- a
        quarantined period has no row claiming that index slot by design, so
        this pre-check is that period's only guard."""
        return await conn.fetchrow(
            """
            SELECT source, invoice_number FROM invoices
            WHERE contact_id = $1 AND billing_period = $2
              AND source IN ('monthly_auto', 'eom_commercial_billing')
              AND status <> 'void'
            UNION ALL
            SELECT
                'quarantined_collision' AS source,
                'historical billing_period collision for this contact+period -- see invoices.metadata.billing_period_backfill_collision' AS invoice_number
            FROM invoices_billing_period_reservations
            WHERE contact_id = $1 AND billing_period = $2
            LIMIT 1
            """,
            contact_id, billing_period,
        )

    @staticmethod
    def _assert_request(
        row: Any, request_fingerprint: str, legacy_request_fingerprint: str | None = None
    ) -> None:
        if row["request_fingerprint"] not in {
            request_fingerprint,
            legacy_request_fingerprint,
        }:
            raise CommercialBillingApprovalConflictError(
                "Idempotency key was already used with a different commercial billing candidate"
            )

    @staticmethod
    async def _insert_invoice(conn: Any, draft: _InvoiceDraft) -> Any | None:
        return await conn.fetchrow(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, customer_email,
                line_items, subtotal, tax_rate, tax_amount, discount_amount,
                total_amount, issue_date, due_date, status, source, source_ref,
                business_context_id, notes, metadata, invoice_for, created_at, updated_at,
                billing_period
            )
            VALUES (
                $1, 'INV-' || to_char($2::date, 'YYYY-Mon') || '-' || lpad(nextval('invoice_number_seq')::text, 4, '0'),
                $3, $4, $5, $6::jsonb, $7, $8, $9, 0, $10, $11, $12, 'draft',
                $13, $14, $15, $16, $17::jsonb, $18, $19, $19, to_char($2::date, 'YYYY-MM')
            )
            ON CONFLICT (source, source_ref)
                WHERE source = 'eom_commercial_billing' AND source_ref IS NOT NULL
                DO NOTHING
            RETURNING id
            """,
            uuid4(), draft.billing_period, draft.contact_id, draft.customer_name,
            draft.customer_email, _canonical_json(draft.line_items), draft.subtotal,
            draft.tax_rate, draft.tax_amount, draft.total, draft.issue_date,
            draft.due_date, _INVOICE_SOURCE, draft.source_ref, EOM_BUSINESS_CONTEXT_ID,
            f"Approved commercial billing candidate for {draft.billing_period:%Y-%m}.",
            _canonical_json(draft.metadata), draft.invoice_for, datetime.now(timezone.utc),
        )

    @staticmethod
    def _view(row: Any) -> dict[str, Any]:
        return {
            "approvedAt": _timestamp(row["approved_at"]),
            "approvedBy": row["approved_by"],
            "billingRunId": str(row["billing_run_id"]),
            "candidateKey": row["candidate_key"],
            "id": str(row["approval_id"]),
            "invoice": {
                "dueDate": str(row["due_date"]),
                "id": str(row["invoice_id"]),
                "invoiceNumber": row["invoice_number"],
                "issueDate": str(row["issue_date"]),
                "sourceRef": row["source_ref"],
                "status": row["invoice_status"],
                "totalCents": _cents(row["total_amount"]),
            },
            "sourceFingerprint": row["source_fingerprint"],
            "reviewFingerprint": _row_value(
                row, "review_fingerprint", row["source_fingerprint"]
            ),
            "state": row["state"],
        }


def get_commercial_billing_approval_service() -> CommercialBillingApprovalService:
    return CommercialBillingApprovalService()


__all__ = [
    "CommercialBillingApprovalConflictError",
    "CommercialBillingApprovalError",
    "CommercialBillingApprovalNotFoundError",
    "CommercialBillingApprovalService",
    "CommercialBillingApprovalStaleError",
    "CommercialBillingApprovalUnavailableError",
    "CommercialBillingApprovalValidationError",
    "get_commercial_billing_approval_service",
]
