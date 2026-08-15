"""Durable manual-Square reference and explicit sent-state provider boundary.

Approved commercial candidates already create their ATLAS invoice before this
service runs.  This service only records the operator's external Square
reference and, in a separate explicit operation, marks that exact draft invoice
sent via Square.  It intentionally has no Square API, Gmail, PDF, email,
payment, or service-marker dependency.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional
from uuid import UUID, uuid4

import asyncpg

from ..services.eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID
from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError

_DELIVERY_METHOD = "manual_square"
_INVOICE_SOURCE = "eom_commercial_billing"
_OPERATION_SOURCE = "eom_admin"
_MAX_ACTOR_LENGTH = 128
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_REFERENCE_LENGTH = 256
_MANUAL_STATES = frozenset({"reference_recorded", "sent_via_square"})
_OPERATION_KINDS = frozenset({"record_reference", "mark_sent"})
_POST_SENT_INVOICE_STATUSES = frozenset({"sent", "partial", "overdue", "paid", "void"})
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

_CONTEXT_SELECT = """
    SELECT approval.id AS approval_id,
           approval.invoice_id AS approval_invoice_id,
           approval.state AS approval_state,
           invoice.id AS invoice_id,
           invoice.invoice_number,
           invoice.customer_name,
           invoice.total_amount,
           invoice.issue_date,
           invoice.due_date,
           invoice.status AS invoice_status,
           invoice.sent_at AS invoice_sent_at,
           invoice.sent_via AS invoice_sent_via,
           invoice.source AS invoice_source,
           invoice.source_ref AS invoice_source_ref,
           invoice.business_context_id AS invoice_business_context_id,
           invoice.metadata AS invoice_metadata
    FROM commercial_billing_candidate_approvals AS approval
    JOIN invoices AS invoice ON invoice.id = approval.invoice_id
"""

_RECORD_SELECT = """
    SELECT id AS manual_square_invoice_id,
           approval_id,
           invoice_id,
           state,
           square_invoice_reference,
           reference_recorded_by,
           reference_recorded_at,
           sent_via_square_by,
           sent_via_square_at
    FROM commercial_billing_manual_square_invoices
"""


class CommercialBillingManualSquareInvoiceError(Exception):
    code = "commercial_billing_manual_square_invoice_error"


class CommercialBillingManualSquareInvoiceValidationError(
    CommercialBillingManualSquareInvoiceError
):
    code = "invalid_commercial_billing_manual_square_invoice"


class CommercialBillingManualSquareInvoiceNotFoundError(
    CommercialBillingManualSquareInvoiceError
):
    code = "commercial_billing_manual_square_invoice_not_found"


class CommercialBillingManualSquareInvoiceConflictError(
    CommercialBillingManualSquareInvoiceError
):
    code = "commercial_billing_manual_square_invoice_conflict"


class CommercialBillingManualSquareInvoiceUnavailableError(
    CommercialBillingManualSquareInvoiceError
):
    code = "commercial_billing_manual_square_invoice_unavailable"


def _request_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingManualSquareInvoiceValidationError(
            f"{field} is required"
        )
    text = value.strip()
    if not text or len(text) > limit or "\r" in text or "\n" in text or "\x00" in text:
        raise CommercialBillingManualSquareInvoiceValidationError(
            f"{field} must contain 1 to {limit} safe characters"
        )
    return text


def _stored_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingManualSquareInvoiceConflictError(
            f"Commercial billing manual Square {field} is invalid"
        )
    text = value.strip()
    if not text or len(text) > limit or "\r" in text or "\n" in text or "\x00" in text:
        raise CommercialBillingManualSquareInvoiceConflictError(
            f"Commercial billing manual Square {field} is invalid"
        )
    return text


def _request_uuid(value: Any, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CommercialBillingManualSquareInvoiceValidationError(
            f"{field} is invalid"
        ) from exc


def _stored_uuid(value: Any, field: str) -> UUID:
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CommercialBillingManualSquareInvoiceConflictError(
            f"Commercial billing manual Square {field} is invalid"
        ) from exc


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CommercialBillingManualSquareInvoiceConflictError(
                f"Commercial billing manual Square {field} is invalid"
            ) from exc
    if not isinstance(value, Mapping):
        raise CommercialBillingManualSquareInvoiceConflictError(
            f"Commercial billing manual Square {field} is invalid"
        )
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CommercialBillingManualSquareInvoiceConflictError(
            f"Commercial billing manual Square {field} is invalid"
        )
    return value


def _optional_timestamp(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    return _timestamp(value, field)


def _date_text(value: Any, field: str) -> str:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise CommercialBillingManualSquareInvoiceConflictError(
            f"Commercial billing manual Square {field} is invalid"
        )
    return value.isoformat()


def _cents(value: Any) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise CommercialBillingManualSquareInvoiceConflictError(
            "Commercial billing manual Square invoice amount is invalid"
        )
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CommercialBillingManualSquareInvoiceConflictError(
            "Commercial billing manual Square invoice amount is invalid"
        ) from exc
    cents = amount * Decimal(100)
    if not cents.is_finite() or cents != cents.to_integral_value() or cents <= 0:
        raise CommercialBillingManualSquareInvoiceConflictError(
            "Commercial billing manual Square invoice amount is invalid"
        )
    return int(cents)


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommercialBillingManualSquareInvoiceValidationError(
            "Manual Square request is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _manual_state(value: Any) -> str:
    state = _stored_text(value, "state", limit=32)
    if state not in _MANUAL_STATES:
        raise CommercialBillingManualSquareInvoiceConflictError(
            "Commercial billing manual Square state is invalid"
        )
    return state


def _operation_kind(value: Any) -> str:
    kind = _stored_text(value, "operation kind", limit=32)
    if kind not in _OPERATION_KINDS:
        raise CommercialBillingManualSquareInvoiceConflictError(
            "Commercial billing manual Square operation kind is invalid"
        )
    return kind


class CommercialBillingManualSquareInvoiceService:
    """Own the durable reference and explicit Square sent-state transition."""

    def __init__(
        self,
        *,
        pool: Optional[DatabasePool] = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._configured_pool = pool
        self._now = now

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            )
        return pool

    async def list_needs_square_invoices(
        self, *, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise CommercialBillingManualSquareInvoiceValidationError(
                "Limit must be between 1 and 100"
            )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise CommercialBillingManualSquareInvoiceValidationError(
                "Offset must be zero or greater"
            )
        try:
            rows = await self.pool.fetch(
                """
                SELECT approval.id AS approval_id,
                       approval.invoice_id AS approval_invoice_id,
                       approval.state AS approval_state,
                       invoice.id AS invoice_id,
                       invoice.invoice_number,
                       invoice.customer_name,
                       invoice.total_amount,
                       invoice.issue_date,
                       invoice.due_date,
                       invoice.status AS invoice_status,
                       invoice.sent_at AS invoice_sent_at,
                       invoice.sent_via AS invoice_sent_via,
                       invoice.source AS invoice_source,
                       invoice.source_ref AS invoice_source_ref,
                       invoice.business_context_id AS invoice_business_context_id,
                       invoice.metadata AS invoice_metadata,
                       record.id AS manual_square_invoice_id,
                       record.approval_id AS record_approval_id,
                       record.invoice_id AS record_invoice_id,
                       record.state AS record_state,
                       record.square_invoice_reference,
                       record.reference_recorded_by,
                       record.reference_recorded_at,
                       record.sent_via_square_by,
                       record.sent_via_square_at,
                       COUNT(*) OVER() AS total_count
                FROM commercial_billing_candidate_approvals AS approval
                JOIN invoices AS invoice ON invoice.id = approval.invoice_id
                LEFT JOIN commercial_billing_manual_square_invoices AS record
                  ON record.approval_id = approval.id
                WHERE approval.state = 'invoice_created'
                  AND invoice.source = $1
                  AND invoice.business_context_id = $2
                  AND invoice.metadata ->> 'deliveryMethod' = $3
                  AND (record.id IS NULL OR record.state <> 'sent_via_square')
                ORDER BY invoice.issue_date DESC, invoice.invoice_number ASC, approval.id ASC
                LIMIT $4 OFFSET $5
                """,
                _INVOICE_SOURCE,
                EOM_BUSINESS_CONTEXT_ID,
                _DELIVERY_METHOD,
                limit,
                offset,
            )
            items = [
                self._view(
                    self._context(dict(row)),
                    self._record_from_queue_row(dict(row)),
                )
                for row in rows
            ]
            total = int(next(iter(rows))["total_count"]) if rows else 0
            return {
                "items": items,
                "limit": limit,
                "offset": offset,
                "total": total,
            }
        except CommercialBillingManualSquareInvoiceError:
            raise
        except asyncpg.PostgresError as exc:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            ) from exc

    async def record_reference(
        self,
        *,
        approval_id: UUID,
        square_invoice_reference: str,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        approved = _request_uuid(approval_id, "Approval id")
        reference = _request_text(
            square_invoice_reference,
            "Square invoice reference",
            limit=_MAX_REFERENCE_LENGTH,
        )
        key = _request_text(
            idempotency_key, "Idempotency key", limit=_MAX_IDEMPOTENCY_KEY_LENGTH
        )
        requested_by = _request_text(
            actor, "Authenticated actor", limit=_MAX_ACTOR_LENGTH
        )
        request_fingerprint = _fingerprint(
            {
                "approvalId": str(approved),
                "operation": "record_reference",
                "squareInvoiceReference": reference,
            }
        )
        try:
            async with self.pool.transaction() as conn:
                await self._lock(conn, f"operation:{key}")
                existing = await self._find_operation_by_key(conn, key)
                if existing is not None:
                    self._assert_operation(
                        existing,
                        approval_id=approved,
                        request_fingerprint=request_fingerprint,
                        operation_kind="record_reference",
                    )
                    return await self._replayed_result(
                        conn, approval_id=approved, reused=True
                    )

                await self._lock(conn, f"approval:{approved}")
                context = await self._find_context(
                    conn, approval_id=approved, for_update=True
                )
                if context is None:
                    raise CommercialBillingManualSquareInvoiceNotFoundError(
                        "Commercial billing approval not found"
                    )
                checked = self._context(context)
                record = await self._find_record(
                    conn, approval_id=approved, for_update=True
                )
                reused = False
                if record is None:
                    self._assert_referenceable(checked)
                    record = await self._insert_record(
                        conn,
                        context=checked,
                        reference=reference,
                        actor=requested_by,
                        now=self._now_timestamp(),
                    )
                else:
                    self._assert_record_matches_context(record, checked)
                    if (
                        _stored_text(
                            record.get("square_invoice_reference"),
                            "Square invoice reference",
                            limit=_MAX_REFERENCE_LENGTH,
                        )
                        != reference
                    ):
                        raise CommercialBillingManualSquareInvoiceConflictError(
                            "A different Square invoice reference is already recorded"
                        )
                    reused = True

                await self._insert_operation(
                    conn,
                    manual_square_invoice_id=_stored_uuid(
                        record.get("manual_square_invoice_id"),
                        "manual Square invoice id",
                    ),
                    idempotency_key=key,
                    request_fingerprint=request_fingerprint,
                    operation_kind="record_reference",
                    actor=requested_by,
                    now=self._now_timestamp(),
                )
                return {
                    "manualSquareInvoice": self._view(checked, record),
                    "replayed": False,
                    "reused": reused,
                }
        except CommercialBillingManualSquareInvoiceError:
            raise
        except asyncpg.UniqueViolationError as exc:
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing manual Square operation could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            ) from exc

    async def mark_sent(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        approved = _request_uuid(approval_id, "Approval id")
        key = _request_text(
            idempotency_key, "Idempotency key", limit=_MAX_IDEMPOTENCY_KEY_LENGTH
        )
        requested_by = _request_text(
            actor, "Authenticated actor", limit=_MAX_ACTOR_LENGTH
        )
        request_fingerprint = _fingerprint(
            {"approvalId": str(approved), "operation": "mark_sent"}
        )
        try:
            async with self.pool.transaction() as conn:
                await self._lock(conn, f"operation:{key}")
                existing = await self._find_operation_by_key(conn, key)
                if existing is not None:
                    self._assert_operation(
                        existing,
                        approval_id=approved,
                        request_fingerprint=request_fingerprint,
                        operation_kind="mark_sent",
                    )
                    return await self._replayed_result(
                        conn, approval_id=approved, reused=True
                    )

                await self._lock(conn, f"approval:{approved}")
                context = await self._find_context(
                    conn, approval_id=approved, for_update=True
                )
                if context is None:
                    raise CommercialBillingManualSquareInvoiceNotFoundError(
                        "Commercial billing approval not found"
                    )
                checked = self._context(context)
                record = await self._find_record(
                    conn, approval_id=approved, for_update=True
                )
                if record is None:
                    raise CommercialBillingManualSquareInvoiceConflictError(
                        "Record the Square invoice reference before marking it sent"
                    )
                self._assert_record_matches_context(record, checked)
                state = _manual_state(record.get("state"))
                reused = state == "sent_via_square"
                if reused:
                    self._assert_square_sent_lifecycle(checked)
                else:
                    if state != "reference_recorded":
                        raise CommercialBillingManualSquareInvoiceConflictError(
                            "Commercial billing manual Square state is invalid"
                        )
                    self._assert_referenceable(checked)
                    now = self._now_timestamp()
                    updated_invoice = await conn.fetchrow(
                        """
                        UPDATE invoices
                           SET status = 'sent', sent_at = $2, sent_via = 'square',
                               updated_at = $2
                         WHERE id = $1
                           AND status = 'draft'
                           AND sent_at IS NULL
                           AND sent_via IS NULL
                        RETURNING id
                        """,
                        checked["invoice_id"],
                        now,
                    )
                    if updated_invoice is None:
                        raise CommercialBillingManualSquareInvoiceConflictError(
                            "Commercial billing invoice changed outside manual Square delivery"
                        )
                    updated_record = await conn.fetchrow(
                        """
                        UPDATE commercial_billing_manual_square_invoices
                           SET state = 'sent_via_square',
                               sent_via_square_by = $2,
                               sent_via_square_at = $3,
                               updated_at = $3
                         WHERE id = $1 AND state = 'reference_recorded'
                        RETURNING id AS manual_square_invoice_id,
                                  approval_id, invoice_id, state,
                                  square_invoice_reference,
                                  reference_recorded_by,
                                  reference_recorded_at,
                                  sent_via_square_by,
                                  sent_via_square_at
                        """,
                        _stored_uuid(
                            record.get("manual_square_invoice_id"),
                            "manual Square invoice id",
                        ),
                        requested_by,
                        now,
                    )
                    if updated_record is None:
                        raise CommercialBillingManualSquareInvoiceConflictError(
                            "Commercial billing manual Square state changed concurrently"
                        )
                    record = dict(updated_record)
                    context = await self._find_context(
                        conn, approval_id=approved, for_update=False
                    )
                    if context is None:
                        raise CommercialBillingManualSquareInvoiceUnavailableError(
                            "Commercial billing approval is unavailable"
                        )
                    checked = self._context(context)
                    self._assert_square_sent_lifecycle(checked)

                await self._insert_operation(
                    conn,
                    manual_square_invoice_id=_stored_uuid(
                        record.get("manual_square_invoice_id"),
                        "manual Square invoice id",
                    ),
                    idempotency_key=key,
                    request_fingerprint=request_fingerprint,
                    operation_kind="mark_sent",
                    actor=requested_by,
                    now=self._now_timestamp(),
                )
                return {
                    "manualSquareInvoice": self._view(checked, record),
                    "replayed": False,
                    "reused": reused,
                }
        except CommercialBillingManualSquareInvoiceError:
            raise
        except asyncpg.UniqueViolationError as exc:
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing manual Square operation could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square database unavailable"
            ) from exc

    def _now_timestamp(self) -> datetime:
        now = self._now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square clock is invalid"
            )
        return now

    @staticmethod
    async def _lock(conn: Any, scope: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-manual-square:{scope}",
        )

    @staticmethod
    async def _find_context(
        conn: Any, *, approval_id: UUID, for_update: bool
    ) -> dict[str, Any] | None:
        query = _CONTEXT_SELECT + " WHERE approval.id = $1"
        if for_update:
            query += " FOR UPDATE OF approval, invoice"
        row = await conn.fetchrow(query, approval_id)
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_record(
        conn: Any, *, approval_id: UUID, for_update: bool
    ) -> dict[str, Any] | None:
        query = _RECORD_SELECT + " WHERE approval_id = $1"
        if for_update:
            query += " FOR UPDATE"
        row = await conn.fetchrow(query, approval_id)
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_operation_by_key(
        conn: Any, idempotency_key: str
    ) -> dict[str, Any] | None:
        row = await conn.fetchrow(
            """
            SELECT operation.id AS operation_id,
                   operation.manual_square_invoice_id,
                   operation.request_fingerprint,
                   operation.operation_kind,
                   record.approval_id
            FROM commercial_billing_manual_square_invoice_operations AS operation
            JOIN commercial_billing_manual_square_invoices AS record
              ON record.id = operation.manual_square_invoice_id
            WHERE operation.source = $1 AND operation.idempotency_key = $2
            """,
            _OPERATION_SOURCE,
            idempotency_key,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _insert_record(
        conn: Any,
        *,
        context: Mapping[str, Any],
        reference: str,
        actor: str,
        now: datetime,
    ) -> dict[str, Any]:
        row = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_manual_square_invoices (
                id, approval_id, invoice_id, state, square_invoice_reference,
                reference_recorded_by, reference_recorded_at, created_at, updated_at
            )
            VALUES ($1, $2, $3, 'reference_recorded', $4, $5, $6, $6, $6)
            RETURNING id AS manual_square_invoice_id,
                      approval_id, invoice_id, state,
                      square_invoice_reference, reference_recorded_by,
                      reference_recorded_at, sent_via_square_by,
                      sent_via_square_at
            """,
            uuid4(),
            context["approval_id"],
            context["invoice_id"],
            reference,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square reference could not be recorded"
            )
        return dict(row)

    @staticmethod
    async def _insert_operation(
        conn: Any,
        *,
        manual_square_invoice_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        operation_kind: str,
        actor: str,
        now: datetime,
    ) -> None:
        row = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_manual_square_invoice_operations (
                id, manual_square_invoice_id, source, idempotency_key,
                request_fingerprint, operation_kind, requested_by,
                requested_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
            RETURNING id
            """,
            uuid4(),
            manual_square_invoice_id,
            _OPERATION_SOURCE,
            idempotency_key,
            request_fingerprint,
            operation_kind,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square operation could not be recorded"
            )

    async def _replayed_result(
        self, conn: Any, *, approval_id: UUID, reused: bool
    ) -> dict[str, Any]:
        context = await self._find_context(
            conn, approval_id=approval_id, for_update=False
        )
        record = await self._find_record(
            conn, approval_id=approval_id, for_update=False
        )
        if context is None or record is None:
            raise CommercialBillingManualSquareInvoiceUnavailableError(
                "Commercial billing manual Square operation is unavailable"
            )
        checked = self._context(context)
        self._assert_record_matches_context(record, checked)
        return {
            "manualSquareInvoice": self._view(checked, record),
            "replayed": True,
            "reused": reused,
        }

    @staticmethod
    def _assert_operation(
        operation: Mapping[str, Any],
        *,
        approval_id: UUID,
        request_fingerprint: str,
        operation_kind: str,
    ) -> None:
        if (
            operation.get("request_fingerprint") != request_fingerprint
            or _stored_uuid(operation.get("approval_id"), "operation approval id")
            != approval_id
            or _operation_kind(operation.get("operation_kind")) != operation_kind
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Idempotency key was already used with a different manual Square request"
            )

    @staticmethod
    def _context(row: Mapping[str, Any]) -> dict[str, Any]:
        approval_id = _stored_uuid(row.get("approval_id"), "approval id")
        invoice_id = _stored_uuid(row.get("invoice_id"), "invoice id")
        if (
            _stored_uuid(row.get("approval_invoice_id"), "approval invoice id")
            != invoice_id
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing approval no longer matches its invoice"
            )
        if (
            _stored_text(row.get("approval_state"), "approval state", limit=32)
            != "invoice_created"
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing approval is not ready for manual Square delivery"
            )
        if (
            _stored_text(row.get("invoice_source"), "invoice source", limit=32)
            != _INVOICE_SOURCE
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing invoice is not eligible for manual Square delivery"
            )
        if row.get("invoice_business_context_id") != EOM_BUSINESS_CONTEXT_ID:
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing invoice is not eligible for manual Square delivery"
            )
        metadata = _mapping(row.get("invoice_metadata"), "invoice metadata")
        if metadata.get("deliveryMethod") != _DELIVERY_METHOD:
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing invoice is not configured for manual Square delivery"
            )
        sent_via = row.get("invoice_sent_via")
        return {
            "approval_id": approval_id,
            "invoice_id": invoice_id,
            "invoice_number": _stored_text(
                row.get("invoice_number"), "invoice number", limit=32
            ),
            "customer_name": _stored_text(
                row.get("customer_name"), "customer name", limit=256
            ),
            "total_cents": _cents(row.get("total_amount")),
            "issue_date": _date_text(row.get("issue_date"), "invoice issue date"),
            "due_date": _date_text(row.get("due_date"), "invoice due date"),
            "invoice_status": _stored_text(
                row.get("invoice_status"), "invoice status", limit=32
            ),
            "invoice_sent_at": _optional_timestamp(
                row.get("invoice_sent_at"), "invoice sent timestamp"
            ),
            "invoice_sent_via": (
                None
                if sent_via is None
                else _stored_text(sent_via, "invoice sent via", limit=32)
            ),
            "invoice_source_ref": row.get("invoice_source_ref"),
        }

    @staticmethod
    def _record_from_queue_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
        if row.get("manual_square_invoice_id") is None:
            return None
        return {
            "manual_square_invoice_id": row.get("manual_square_invoice_id"),
            "approval_id": row.get("record_approval_id"),
            "invoice_id": row.get("record_invoice_id"),
            "state": row.get("record_state"),
            "square_invoice_reference": row.get("square_invoice_reference"),
            "reference_recorded_by": row.get("reference_recorded_by"),
            "reference_recorded_at": row.get("reference_recorded_at"),
            "sent_via_square_by": row.get("sent_via_square_by"),
            "sent_via_square_at": row.get("sent_via_square_at"),
        }

    @staticmethod
    def _assert_record_matches_context(
        record: Mapping[str, Any], context: Mapping[str, Any]
    ) -> None:
        if (
            _stored_uuid(record.get("approval_id"), "manual Square approval id")
            != context["approval_id"]
            or _stored_uuid(record.get("invoice_id"), "manual Square invoice id")
            != context["invoice_id"]
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing manual Square record no longer matches its invoice"
            )
        _manual_state(record.get("state"))
        _stored_text(
            record.get("square_invoice_reference"),
            "Square invoice reference",
            limit=_MAX_REFERENCE_LENGTH,
        )
        _stored_text(
            record.get("reference_recorded_by"),
            "reference recorded actor",
            limit=_MAX_ACTOR_LENGTH,
        )
        _timestamp(record.get("reference_recorded_at"), "reference recorded timestamp")

    @staticmethod
    def _assert_referenceable(context: Mapping[str, Any]) -> None:
        if (
            context["invoice_status"] != "draft"
            or context["invoice_sent_at"] is not None
            or context["invoice_sent_via"] is not None
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Commercial billing invoice changed outside manual Square delivery"
            )

    @staticmethod
    def _assert_square_sent_lifecycle(context: Mapping[str, Any]) -> None:
        if (
            context["invoice_status"] not in _POST_SENT_INVOICE_STATUSES
            or context["invoice_sent_via"] != "square"
            or context["invoice_sent_at"] is None
        ):
            raise CommercialBillingManualSquareInvoiceConflictError(
                "Manual Square sent evidence no longer matches invoice lifecycle"
            )

    @staticmethod
    def _view(
        context: Mapping[str, Any], record: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        state = "needs_square_invoice"
        manual_square_invoice_id = None
        reference = None
        reference_recorded_by = None
        reference_recorded_at = None
        sent_via_square_by = None
        sent_via_square_at = None
        if record is None:
            if (
                context["invoice_status"] != "draft"
                or context["invoice_sent_at"] is not None
                or context["invoice_sent_via"] is not None
            ):
                state = "lifecycle_conflict"
        else:
            CommercialBillingManualSquareInvoiceService._assert_record_matches_context(
                record, context
            )
            manual_square_invoice_id = str(
                _stored_uuid(
                    record.get("manual_square_invoice_id"), "manual Square invoice id"
                )
            )
            reference = _stored_text(
                record.get("square_invoice_reference"),
                "Square invoice reference",
                limit=_MAX_REFERENCE_LENGTH,
            )
            reference_recorded_by = _stored_text(
                record.get("reference_recorded_by"),
                "reference recorded actor",
                limit=_MAX_ACTOR_LENGTH,
            )
            reference_recorded_at = _timestamp(
                record.get("reference_recorded_at"), "reference recorded timestamp"
            ).isoformat()
            durable_state = _manual_state(record.get("state"))
            if durable_state == "reference_recorded":
                if (
                    context["invoice_status"] == "draft"
                    and context["invoice_sent_at"] is None
                    and context["invoice_sent_via"] is None
                ):
                    state = "reference_recorded"
                else:
                    state = "lifecycle_conflict"
            else:
                CommercialBillingManualSquareInvoiceService._assert_square_sent_lifecycle(
                    context
                )
                state = "sent_via_square"
                sent_via_square_by = _stored_text(
                    record.get("sent_via_square_by"),
                    "Square sent actor",
                    limit=_MAX_ACTOR_LENGTH,
                )
                sent_via_square_at = _timestamp(
                    record.get("sent_via_square_at"), "Square sent timestamp"
                ).isoformat()
        return {
            "approvalId": str(context["approval_id"]),
            "id": manual_square_invoice_id,
            "invoice": {
                "dueDate": context["due_date"],
                "id": str(context["invoice_id"]),
                "invoiceNumber": context["invoice_number"],
                "issueDate": context["issue_date"],
                "sentAt": (
                    context["invoice_sent_at"].isoformat()
                    if context["invoice_sent_at"] is not None
                    else None
                ),
                "sentVia": context["invoice_sent_via"],
                "sourceRef": context["invoice_source_ref"],
                "status": context["invoice_status"],
                "totalCents": context["total_cents"],
            },
            "customerName": context["customer_name"],
            "referenceRecordedAt": reference_recorded_at,
            "referenceRecordedBy": reference_recorded_by,
            "sentViaSquareAt": sent_via_square_at,
            "sentViaSquareBy": sent_via_square_by,
            "squareInvoiceReference": reference,
            "state": state,
        }


def get_commercial_billing_manual_square_invoice_service() -> (
    CommercialBillingManualSquareInvoiceService
):
    return CommercialBillingManualSquareInvoiceService()


__all__ = [
    "CommercialBillingManualSquareInvoiceConflictError",
    "CommercialBillingManualSquareInvoiceError",
    "CommercialBillingManualSquareInvoiceNotFoundError",
    "CommercialBillingManualSquareInvoiceService",
    "CommercialBillingManualSquareInvoiceUnavailableError",
    "CommercialBillingManualSquareInvoiceValidationError",
    "get_commercial_billing_manual_square_invoice_service",
]
