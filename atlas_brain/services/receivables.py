"""Atomic receivables ledger for customer receipts and invoice allocations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

from ..templates.email.payment_receipt import render_residential_payment_receipt
from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseUnavailableError

ACTIVE_PAYMENT_STATUSES = ("legacy", "received", "deposited", "cleared")
OPEN_INVOICE_STATUSES = ("sent", "partial", "overdue")
API_PAYMENT_METHODS = ("check", "ach", "square")
EOM_CUSTOMER_TYPES = ("residential", "commercial", "unknown")
RESIDENTIAL_CUSTOMER_TYPE = "residential"
_CENT = Decimal("0.01")
_MAX_DATABASE_MONEY = Decimal("9999999999.99")
_LEDGER_ALLOCATION_HISTORY_LIMIT = 100
_RECEIVABLES_REQUIRED_COLUMNS = {
    "customer_payments": (
        "id",
        "contact_id",
        "payer_name",
        "total_amount",
        "payment_method",
        "reference",
        "received_date",
        "check_date",
        "received_through",
        "status",
        "source",
        "idempotency_key",
        "request_fingerprint",
        "notes",
        "recorded_by",
        "deposited_at",
        "cleared_at",
        "returned_at",
        "return_reason",
        "voided_at",
        "void_reason",
        "metadata",
        "created_at",
        "updated_at",
    ),
    "invoice_payments": (
        "payment_id",
        "reversed_at",
        "reversed_by",
        "reversal_reason",
    ),
    "payment_deposit_batches": (
        "id",
        "deposit_date",
        "bank_reference",
        "status",
        "idempotency_key",
        "request_fingerprint",
        "clear_idempotency_key",
        "clear_request_fingerprint",
        "created_by",
        "cleared_at",
        "cleared_by",
        "voided_at",
        "voided_by",
        "void_reason",
        "metadata",
        "created_at",
        "updated_at",
    ),
    "payment_deposit_items": (
        "batch_id",
        "payment_id",
        "created_at",
    ),
    "payment_events": (
        "id",
        "payment_id",
        "event_type",
        "previous_status",
        "new_status",
        "effective_date",
        "actor",
        "reason",
        "idempotency_key",
        "request_fingerprint",
        "metadata",
        "created_at",
    ),
}
_RECEIVABLES_REQUIRED_INDEXES = (
    (
        "customer_payments",
        "customer_payments_pkey",
        True,
        ("id",),
        None,
        "p",
    ),
    (
        "customer_payments",
        "idx_customer_payments_idempotency",
        True,
        ("source", "idempotency_key"),
        "idempotency_key IS NOT NULL",
        None,
    ),
    (
        "customer_payments",
        "idx_customer_payments_contact_received",
        False,
        ("contact_id", "received_date", "created_at"),
        None,
        None,
    ),
    (
        "customer_payments",
        "idx_customer_payments_status_received",
        False,
        ("status", "received_date"),
        None,
        None,
    ),
    (
        "invoice_payments",
        "idx_invoice_payments_payment_id",
        False,
        ("payment_id",),
        None,
        None,
    ),
    (
        "invoice_payments",
        "idx_invoice_payments_active_payment_invoice",
        True,
        ("payment_id", "invoice_id"),
        "payment_id IS NOT NULL AND reversed_at IS NULL",
        None,
    ),
    (
        "payment_deposit_batches",
        "payment_deposit_batches_pkey",
        True,
        ("id",),
        None,
        "p",
    ),
    (
        "payment_deposit_batches",
        "idx_payment_deposit_batches_idempotency",
        True,
        ("idempotency_key",),
        "idempotency_key IS NOT NULL",
        None,
    ),
    (
        "payment_deposit_batches",
        "idx_payment_deposit_batches_clear_idempotency",
        True,
        ("clear_idempotency_key",),
        "clear_idempotency_key IS NOT NULL",
        None,
    ),
    (
        "payment_deposit_batches",
        "idx_payment_deposit_batches_status_date",
        False,
        ("status", "deposit_date"),
        None,
        None,
    ),
    (
        "payment_deposit_items",
        "payment_deposit_items_pkey",
        True,
        ("batch_id", "payment_id"),
        None,
        "p",
    ),
    (
        "payment_deposit_items",
        "payment_deposit_items_payment_id_key",
        True,
        ("payment_id",),
        None,
        "u",
    ),
    (
        "payment_events",
        "payment_events_pkey",
        True,
        ("id",),
        None,
        "p",
    ),
    (
        "payment_events",
        "idx_payment_events_idempotency",
        True,
        ("payment_id", "idempotency_key"),
        "idempotency_key IS NOT NULL",
        None,
    ),
    (
        "payment_events",
        "idx_payment_events_payment_created",
        False,
        ("payment_id", "created_at"),
        None,
        None,
    ),
    (
        "payment_events",
        "idx_payment_events_key_lookup",
        False,
        ("idempotency_key",),
        "idempotency_key IS NOT NULL",
        None,
    ),
)

# Receipt delivery is a capability layered onto the existing ledger.  It is
# deliberately NOT part of the legacy receivables readiness set: full Atlas
# and MCP callers must remain able to use the established payment lifecycle
# while an EOM-only outbox migration rolls out.  The receipt-aware full and
# slim EOM routes require this additive set explicitly.
_RECEIPT_DELIVERY_REQUIRED_COLUMNS = {
    "payment_receipt_deliveries": (
        "id",
        "payment_id",
        "contact_id",
        "receipt_number",
        "recipient_email",
        "delivery_status",
        "skip_reason",
        "subject",
        "body",
        "created_at",
        "updated_at",
    ),
}
_RECEIPT_DELIVERY_REQUIRED_INDEXES = (
    (
        "payment_receipt_deliveries",
        "payment_receipt_deliveries_pkey",
        True,
        ("id",),
        None,
        "p",
    ),
    (
        "payment_receipt_deliveries",
        "payment_receipt_deliveries_payment_id_key",
        True,
        ("payment_id",),
        None,
        "u",
    ),
    (
        "payment_receipt_deliveries",
        "payment_receipt_deliveries_receipt_number_key",
        True,
        ("receipt_number",),
        None,
        "u",
    ),
    (
        "payment_receipt_deliveries",
        "idx_payment_receipt_deliveries_contact_created",
        False,
        ("contact_id", "created_at"),
        None,
        None,
    ),
    (
        "payment_receipt_deliveries",
        "idx_payment_receipt_deliveries_status_created",
        False,
        ("delivery_status", "created_at"),
        None,
        None,
    ),
)


@dataclass(frozen=True)
class PaymentCreationOutcome:
    """Internal payment write result; public transports emit only ``payment``."""

    payment: dict[str, Any]
    replayed: bool


@dataclass(frozen=True)
class PaymentReceiptRecipient:
    """Canonical customer snapshot used only by receipt-aware EOM routes."""

    contact_id: UUID
    customer_name: str
    customer_type: str
    recipient_email: Optional[str]


class ReceivablesError(Exception):
    """Base error with a stable machine-readable code."""

    code = "receivables_error"


class ReceivablesValidationError(ReceivablesError):
    code = "validation_error"


class ReceivablesNotFoundError(ReceivablesError):
    code = "not_found"


class ReceivablesConflictError(ReceivablesError):
    code = "conflict"


class ReceivablesSchemaUnavailableError(ReceivablesError):
    code = "schema_unavailable"


class ReceivablesReceiptContextRequiredError(ReceivablesError):
    """An EOM route could not prove a new payment's canonical customer."""

    code = "canonical_customer_unavailable"


def money(value: Any) -> Decimal:
    """Normalize a value to finite, two-decimal currency."""
    try:
        raw = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ReceivablesValidationError("Amount must be valid currency") from exc
    if not raw.is_finite():
        raise ReceivablesValidationError("Amount must be finite")
    try:
        normalized = raw.quantize(_CENT)
    except InvalidOperation as exc:
        raise ReceivablesValidationError("Amount must be valid currency") from exc
    if raw != normalized:
        raise ReceivablesValidationError("Amount must use cent precision")
    if abs(normalized) > _MAX_DATABASE_MONEY:
        raise ReceivablesValidationError("Amount exceeds the supported currency range")
    return normalized


def cents(value: Any) -> int:
    """Convert a database currency value to integer cents."""
    return int(money(value) * 100)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime, UUID)):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def request_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_index_predicate(predicate: Optional[str]) -> str:
    """Normalize simple catalog predicates without hiding semantic tokens."""
    return re.sub(r"[\s()]+", "", predicate or "").lower()


def _serialize_row(row: Any) -> dict[str, Any]:
    if not row:
        return {}
    result = dict(row)
    for key, value in list(result.items()):
        if isinstance(value, UUID):
            result[key] = str(value)
        elif isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, str) and key == "metadata":
            try:
                result[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                result[key] = {}
    return result


class ReceivablesService:
    """Owns receipt, allocation, return, and deposit lifecycle transactions."""

    def __init__(self, pool: Optional[DatabasePool] = None) -> None:
        self._configured_pool = pool

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("receivables ledger")
        return pool

    async def is_ready(self, conn: Any | None = None) -> bool:
        """Return whether the established receivables ledger is usable.

        This remains the compatibility readiness probe for full Atlas and MCP
        callers.  Receipt delivery is an EOM-only additive capability and is
        checked by :meth:`is_receipt_delivery_ready` instead.
        """
        return await self._schema_objects_ready(
            conn,
            required_columns=_RECEIVABLES_REQUIRED_COLUMNS,
            required_indexes=_RECEIVABLES_REQUIRED_INDEXES,
        )

    async def is_receipt_delivery_ready(self, conn: Any | None = None) -> bool:
        """Return whether the ledger and residential receipt outbox are usable."""
        return (
            await self.is_ready(conn)
            and await self._schema_objects_ready(
                conn,
                required_columns=_RECEIPT_DELIVERY_REQUIRED_COLUMNS,
                required_indexes=_RECEIPT_DELIVERY_REQUIRED_INDEXES,
            )
        )

    async def _schema_objects_ready(
        self,
        conn: Any | None = None,
        *,
        required_columns: dict[str, tuple[str, ...]],
        required_indexes: tuple[tuple[Any, ...], ...],
    ) -> bool:
        """Probe one closed schema contract without broadening another one."""
        executor = conn if conn is not None else self.pool
        required = [
            (table_name, column_name)
            for table_name, columns in required_columns.items()
            for column_name in columns
        ]
        columns_ready = bool(
            await executor.fetchval(
                """
                SELECT NOT EXISTS (
                    SELECT 1
                    FROM unnest($1::text[], $2::text[])
                        AS required(table_name, column_name)
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns AS actual
                        WHERE actual.table_schema = current_schema()
                          AND actual.table_name = required.table_name
                          AND actual.column_name = required.column_name
                    )
                )
                """,
                [table_name for table_name, _column_name in required],
                [column_name for _table_name, column_name in required],
            )
        )
        if not columns_ready:
            return False

        index_rows = await executor.fetch(
            """
            SELECT
                table_class.relname AS table_name,
                index_class.relname AS index_name,
                index_state.indisunique AS is_unique,
                index_state.indisvalid AS is_valid,
                index_state.indisready AS is_ready,
                ARRAY(
                    SELECT attribute.attname
                    FROM unnest(index_state.indkey::smallint[])
                        WITH ORDINALITY AS key(attnum, position)
                    JOIN pg_attribute AS attribute
                      ON attribute.attrelid = table_class.oid
                     AND attribute.attnum = key.attnum
                    WHERE key.position <= index_state.indnkeyatts
                    ORDER BY key.position
                ) AS key_columns,
                pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate,
                constraint_state.contype::text AS constraint_type
            FROM pg_index AS index_state
            JOIN pg_class AS table_class
              ON table_class.oid = index_state.indrelid
            JOIN pg_namespace AS table_namespace
              ON table_namespace.oid = table_class.relnamespace
            JOIN pg_class AS index_class
              ON index_class.oid = index_state.indexrelid
            LEFT JOIN pg_constraint AS constraint_state
              ON constraint_state.conrelid = index_state.indrelid
             AND constraint_state.conindid = index_state.indexrelid
            WHERE table_namespace.nspname = current_schema()
              AND index_class.relname = ANY($1::text[])
            """,
            [
                index_name
                for _table_name, index_name, *_rest in required_indexes
            ],
        )
        actual_indexes = {
            row["index_name"]: (
                row["table_name"],
                bool(row["is_unique"]),
                tuple(row["key_columns"]),
                _normalize_index_predicate(row["predicate"]),
                row["constraint_type"],
                bool(row["is_valid"]),
                bool(row["is_ready"]),
            )
            for row in index_rows
        }
        return all(
            actual_indexes.get(index_name)
            == (
                table_name,
                is_unique,
                key_columns,
                _normalize_index_predicate(predicate),
                constraint_type,
                True,
                True,
            )
            for (
                table_name,
                index_name,
                is_unique,
                key_columns,
                predicate,
                constraint_type,
            ) in required_indexes
        )

    async def list_open_invoices(
        self,
        *,
        contact_id: Optional[UUID] = None,
        search: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """List allocatable invoices in deterministic oldest-due-first order."""
        conditions = [
            "contact_id IS NOT NULL",
            "status = ANY($1::varchar[])",
            "amount_due > 0",
        ]
        args: list[Any] = [list(OPEN_INVOICE_STATUSES)]
        if contact_id:
            args.append(contact_id)
            conditions.append(f"contact_id = ${len(args)}")
        if search and search.strip():
            args.append(f"%{search.strip()}%")
            conditions.append(
                f"(customer_name ILIKE ${len(args)} "
                f"OR invoice_number ILIKE ${len(args)})"
            )
        args.append(max(1, min(limit, 1000)))
        rows = await self.pool.fetch(
            f"""
            SELECT id, invoice_number, contact_id, customer_name, issue_date,
                   due_date, status, total_amount, amount_paid, amount_due
            FROM invoices
            WHERE {' AND '.join(conditions)}
            ORDER BY due_date, issue_date, customer_name, invoice_number
            LIMIT ${len(args)}
            """,
            *args,
        )
        return [self._invoice_view(row) for row in rows]

    async def suggest_allocations(
        self, *, contact_id: UUID, total_amount: Decimal
    ) -> list[dict[str, Any]]:
        """Suggest oldest-first allocations without writing any state."""
        remaining = money(total_amount)
        if remaining <= 0:
            raise ReceivablesValidationError("Payment total must be positive")
        suggestions: list[dict[str, Any]] = []
        for invoice in await self.list_open_invoices(contact_id=contact_id):
            if remaining <= 0:
                break
            allocation = min(remaining, money(invoice["amount_due"]))
            suggestions.append(
                {
                    "invoice_id": invoice["id"],
                    "invoice_number": invoice["invoice_number"],
                    "amount_cents": cents(allocation),
                }
            )
            remaining -= allocation
        return suggestions

    async def create_payment(
        self,
        *,
        contact_id: Optional[UUID],
        payer_name: str,
        total_amount: Decimal,
        payment_method: str,
        received_date: date,
        check_date: Optional[date] = None,
        received_through: Optional[str] = None,
        allocations: list[dict[str, Any]],
        idempotency_key: str,
        reference: Optional[str] = None,
        notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        source: str = "eom_admin",
        metadata: Optional[dict[str, Any]] = None,
        enforce_api_methods: bool = True,
        allow_unapplied: bool = False,
        unapplied_contact_context_id: Optional[str] = None,
        receipt_recipient: Optional[PaymentReceiptRecipient] = None,
        require_receipt_recipient: bool = False,
    ) -> dict[str, Any]:
        """Create one receipt and all of its allocations atomically."""
        outcome = await self.create_payment_with_outcome(
            contact_id=contact_id,
            payer_name=payer_name,
            total_amount=total_amount,
            payment_method=payment_method,
            received_date=received_date,
            check_date=check_date,
            received_through=received_through,
            allocations=allocations,
            idempotency_key=idempotency_key,
            reference=reference,
            notes=notes,
            recorded_by=recorded_by,
            source=source,
            metadata=metadata,
            enforce_api_methods=enforce_api_methods,
            allow_unapplied=allow_unapplied,
            unapplied_contact_context_id=unapplied_contact_context_id,
            receipt_recipient=receipt_recipient,
            require_receipt_recipient=require_receipt_recipient,
        )
        return outcome.payment

    async def create_payment_with_outcome(
        self,
        *,
        contact_id: Optional[UUID],
        payer_name: str,
        total_amount: Decimal,
        payment_method: str,
        received_date: date,
        check_date: Optional[date] = None,
        received_through: Optional[str] = None,
        allocations: list[dict[str, Any]],
        idempotency_key: str,
        reference: Optional[str] = None,
        notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        source: str = "eom_admin",
        metadata: Optional[dict[str, Any]] = None,
        enforce_api_methods: bool = True,
        allow_unapplied: bool = False,
        unapplied_contact_context_id: Optional[str] = None,
        receipt_recipient: Optional[PaymentReceiptRecipient] = None,
        require_receipt_recipient: bool = False,
    ) -> PaymentCreationOutcome:
        """Create a receipt and report whether this call replayed its first write."""
        total = money(total_amount)
        if total <= 0:
            raise ReceivablesValidationError("Payment total must be positive")
        method = payment_method.strip().lower()
        if enforce_api_methods and method not in API_PAYMENT_METHODS:
            raise ReceivablesValidationError(
                f"Payment method must be one of: {', '.join(API_PAYMENT_METHODS)}"
            )
        if not method:
            raise ReceivablesValidationError("Payment method is required")
        payer = payer_name.strip()
        if not payer:
            raise ReceivablesValidationError("Payer name is required")
        key = idempotency_key.strip()
        if not key or len(key) > 128:
            raise ReceivablesValidationError(
                "Idempotency key must contain 1 to 128 characters"
            )
        if allow_unapplied and not unapplied_contact_context_id:
            raise ReceivablesValidationError(
                "Unapplied payments require a canonical customer context"
            )
        self._assert_receipt_recipient_invariant(
            contact_id=contact_id,
            receipt_recipient=receipt_recipient,
        )
        is_residential_receipt = bool(
            receipt_recipient is not None
            and receipt_recipient.customer_type == RESIDENTIAL_CUSTOMER_TYPE
        )

        normalized = self._normalize_allocations(
            allocations, allow_empty=allow_unapplied
        )
        allocated_total = sum((item["amount"] for item in normalized), Decimal("0"))
        if allocated_total > total:
            raise ReceivablesValidationError(
                "Allocated amount cannot exceed the payment total"
            )
        initial_status = "received" if method == "check" else "cleared"
        raw_received_through = received_through or ""
        if len(raw_received_through) > 128:
            raise ReceivablesValidationError(
                "Received through must contain at most 128 characters"
            )
        normalized_received_through = raw_received_through.strip() or None
        has_check_metadata = (
            check_date is not None or normalized_received_through is not None
        )
        if has_check_metadata and method != "check":
            raise ReceivablesValidationError(
                "Check metadata requires a check payment method"
            )
        payload = {
            "contact_id": contact_id,
            "payer_name": payer,
            "total_amount": total,
            "payment_method": method,
            "received_date": received_date,
            "allocations": normalized,
            "reference": (reference or "").strip() or None,
            "notes": (notes or "").strip() or None,
        }
        if has_check_metadata:
            if not await self.is_ready():
                raise ReceivablesSchemaUnavailableError(
                    "Receivables schema unavailable for check metadata"
                )
            payload.update(
                {
                    "check_date": check_date,
                    "received_through": normalized_received_through,
                }
            )
        fingerprint = request_fingerprint(payload)

        async with self.pool.transaction() as conn:
            await self._lock_operation_key(conn, "payment-event", key)
            await self._lock_operation_key(
                conn, "payment-create", f"{source}:{key}"
            )
            existing = await conn.fetchrow(
                """
                SELECT id, request_fingerprint
                FROM customer_payments
                WHERE source = $1 AND idempotency_key = $2
                FOR UPDATE
                """,
                source,
                key,
            )
            if existing:
                self._assert_idempotent(existing, fingerprint)
                return PaymentCreationOutcome(
                    payment=await self._payment_view(
                        conn,
                        existing["id"],
                        include_receipt_delivery=(
                            await self._has_replay_receipt_delivery(
                                conn,
                                payment_id=existing["id"],
                                require_receipt_recipient=require_receipt_recipient,
                            )
                        ),
                    ),
                    replayed=True,
                )
            if require_receipt_recipient and receipt_recipient is None:
                # This check follows the first idempotency lookup on purpose:
                # an unchanged retry must recover its original payment even if
                # the canonical CRM contact has been edited or is unavailable
                # after the original commit.  A new payment still fails before
                # any financial row is inserted when an EOM route lacks an
                # authoritative customer snapshot.
                raise ReceivablesReceiptContextRequiredError(
                    "Canonical customer data is required for a new EOM payment"
                )
            if is_residential_receipt and not await self.is_receipt_delivery_ready(
                conn
            ):
                raise ReceivablesSchemaUnavailableError(
                    "Receivables schema unavailable for payment receipt delivery"
                )
            await self._assert_event_key_available(
                conn, key=key, fingerprint=fingerprint
            )

            invoices = await self._lock_and_validate_invoices(
                conn,
                contact_id=contact_id,
                allocations=normalized,
                unapplied_contact_context_id=unapplied_contact_context_id,
            )
            # A same-key create can arrive while the first transaction is still
            # holding these invoice locks. Reconcile again after the wait, before
            # validating the now-reduced balances, so a true retry returns the
            # original receipt instead of an over-allocation error.
            existing = await conn.fetchrow(
                """
                SELECT id, request_fingerprint
                FROM customer_payments
                WHERE source = $1 AND idempotency_key = $2
                FOR UPDATE
                """,
                source,
                key,
            )
            if existing:
                self._assert_idempotent(existing, fingerprint)
                return PaymentCreationOutcome(
                    payment=await self._payment_view(
                        conn,
                        existing["id"],
                        include_receipt_delivery=(
                            await self._has_replay_receipt_delivery(
                                conn,
                                payment_id=existing["id"],
                                require_receipt_recipient=require_receipt_recipient,
                            )
                        ),
                    ),
                    replayed=True,
                )

            invoice_by_id = {str(row["id"]): row for row in invoices}
            for allocation in normalized:
                invoice = invoice_by_id[str(allocation["invoice_id"])]
                if allocation["amount"] > money(invoice["amount_due"]):
                    raise ReceivablesValidationError(
                        f"Allocation exceeds balance for {invoice['invoice_number']}"
                    )

            payment_id = uuid4()
            if has_check_metadata:
                payment_row = await conn.fetchrow(
                    """
                    INSERT INTO customer_payments (
                        id, contact_id, payer_name, total_amount, payment_method,
                        reference, received_date, check_date, received_through,
                        status, source, idempotency_key, request_fingerprint,
                        notes, recorded_by, cleared_at, metadata, created_at,
                        updated_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                        $13, $14, $15, CASE WHEN $10::varchar = 'cleared'
                            THEN $16::timestamptz ELSE NULL END,
                        $17::jsonb, $16, $16
                    )
                    ON CONFLICT (source, idempotency_key)
                        WHERE idempotency_key IS NOT NULL
                    DO NOTHING
                    RETURNING id
                    """,
                    payment_id,
                    contact_id,
                    payer,
                    total,
                    method,
                    payload["reference"],
                    received_date,
                    payload["check_date"],
                    payload["received_through"],
                    initial_status,
                    source,
                    key,
                    fingerprint,
                    payload["notes"],
                    recorded_by,
                    datetime.now(timezone.utc),
                    json.dumps(metadata or {}),
                )
            else:
                payment_row = await conn.fetchrow(
                    """
                    INSERT INTO customer_payments (
                        id, contact_id, payer_name, total_amount, payment_method,
                        reference, received_date, status, source, idempotency_key,
                        request_fingerprint, notes, recorded_by, cleared_at,
                        metadata, created_at, updated_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                        $13, CASE WHEN $8::varchar = 'cleared'
                            THEN $14::timestamptz ELSE NULL END,
                        $15::jsonb, $14, $14
                    )
                    ON CONFLICT (source, idempotency_key)
                        WHERE idempotency_key IS NOT NULL
                    DO NOTHING
                    RETURNING id
                    """,
                    payment_id,
                    contact_id,
                    payer,
                    total,
                    method,
                    payload["reference"],
                    received_date,
                    initial_status,
                    source,
                    key,
                    fingerprint,
                    payload["notes"],
                    recorded_by,
                    datetime.now(timezone.utc),
                    json.dumps(metadata or {}),
                )
            if not payment_row:
                existing = await conn.fetchrow(
                    """
                    SELECT id, request_fingerprint FROM customer_payments
                    WHERE source = $1 AND idempotency_key = $2
                    """,
                    source,
                    key,
                )
                if not existing:
                    raise ReceivablesConflictError(
                        "Payment idempotency conflict could not be reconciled"
                    )
                self._assert_idempotent(existing, fingerprint)
                return PaymentCreationOutcome(
                    payment=await self._payment_view(
                        conn,
                        existing["id"],
                        include_receipt_delivery=(
                            await self._has_replay_receipt_delivery(
                                conn,
                                payment_id=existing["id"],
                                require_receipt_recipient=require_receipt_recipient,
                            )
                        ),
                    ),
                    replayed=True,
                )

            now = datetime.now(timezone.utc)
            if (
                receipt_recipient is not None
                and receipt_recipient.customer_type == RESIDENTIAL_CUSTOMER_TYPE
            ):
                await self._enqueue_residential_payment_receipt(
                    conn,
                    payment_id=payment_id,
                    receipt_recipient=receipt_recipient,
                    payer_name=payer,
                    total_amount=total,
                    payment_method=method,
                    reference=payload["reference"],
                    received_date=received_date,
                    created_at=now,
                )
            for allocation in normalized:
                await conn.execute(
                    """
                    INSERT INTO invoice_payments (
                        id, invoice_id, payment_id, amount, payment_date,
                        payment_method, reference, notes, recorded_by,
                        created_at, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                    """,
                    uuid4(),
                    allocation["invoice_id"],
                    payment_id,
                    allocation["amount"],
                    received_date,
                    method,
                    payload["reference"],
                    payload["notes"],
                    recorded_by,
                    now,
                    json.dumps(metadata or {}),
                )
            await self._insert_event(
                conn,
                payment_id=payment_id,
                event_type="payment_recorded",
                previous_status=None,
                new_status=initial_status,
                effective_date=received_date,
                actor=recorded_by,
                idempotency_key=key,
                fingerprint=fingerprint,
                metadata={"allocated_amount": str(allocated_total)},
            )
            await self._recalculate_invoices(
                conn, [item["invoice_id"] for item in normalized]
            )
            return PaymentCreationOutcome(
                payment=await self._payment_view(
                    conn,
                    payment_id,
                    include_receipt_delivery=is_residential_receipt,
                ),
                replayed=False,
            )

    async def list_payments(
        self,
        *,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        args: list[Any] = []
        if status:
            args.append(status)
            conditions.append(f"cp.status = ${len(args)}")
        if search and search.strip():
            args.append(f"%{search.strip()}%")
            conditions.append(
                f"(cp.payer_name ILIKE ${len(args)} "
                f"OR cp.reference ILIKE ${len(args)})"
            )
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        args.extend([max(1, min(limit, 500)), max(0, offset)])
        rows = await self.pool.fetch(
            f"""
            SELECT cp.*, pdi.batch_id
            FROM customer_payments cp
            LEFT JOIN payment_deposit_items pdi ON pdi.payment_id = cp.id
            {where}
            ORDER BY cp.received_date DESC, cp.created_at DESC
            LIMIT ${len(args) - 1} OFFSET ${len(args)}
            """,
            *args,
        )
        if not rows:
            return []
        return await self._payment_views_for_rows(self.pool, rows)

    async def list_customer_ledger(
        self,
        *,
        contact_id: UUID,
        payment_status: Optional[str] = None,
        payment_method: Optional[str] = None,
        search: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return one bounded, receipt-aware ledger page for a customer.

        Entries are a current financial snapshot, not a reconstructed historical
        balance.  Invoices do not yet have an immutable lifecycle-event stream,
        so the response names current invoice and unapplied-payment balances
        explicitly instead of inferring a running balance from mutable rows.
        """
        if from_date is not None and to_date is not None and from_date > to_date:
            raise ReceivablesValidationError(
                "Ledger start date cannot be after end date"
            )

        normalized_search = (search or "").strip() or None
        if payment_status is not None and not payment_status.strip():
            raise ReceivablesValidationError("Payment status cannot be blank")
        if payment_method is not None and not payment_method.strip():
            raise ReceivablesValidationError("Payment method cannot be blank")
        normalized_status = payment_status.strip() if payment_status is not None else None
        normalized_method = payment_method.strip() if payment_method is not None else None
        page_size = max(1, min(limit, 200))
        page_offset = max(0, offset)
        search_pattern = f"%{normalized_search}%" if normalized_search else None

        async with self.pool.transaction() as conn:
            await conn.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            if not await self.is_receipt_delivery_ready(conn):
                raise ReceivablesSchemaUnavailableError(
                    "Receivables schema unavailable for customer ledger"
                )

            entry_rows = await conn.fetch(
                """
                WITH ledger_entries AS (
                    SELECT 'invoice'::text AS entry_type,
                           i.id AS entry_id,
                           i.created_at,
                           i.issue_date AS occurred_date
                    FROM invoices i
                    WHERE i.contact_id = $1
                      AND ($2::date IS NULL OR i.issue_date >= $2)
                      AND ($3::date IS NULL OR i.issue_date <= $3)
                      AND ($5::varchar IS NULL AND $6::varchar IS NULL)
                      AND (
                          $4::text IS NULL
                          OR i.invoice_number ILIKE $4
                          OR i.customer_name ILIKE $4
                      )

                    UNION ALL

                    SELECT 'payment'::text AS entry_type,
                           cp.id AS entry_id,
                           cp.created_at,
                           cp.received_date AS occurred_date
                    FROM customer_payments cp
                    WHERE cp.contact_id = $1
                      AND ($2::date IS NULL OR cp.received_date >= $2)
                      AND ($3::date IS NULL OR cp.received_date <= $3)
                      AND ($5::varchar IS NULL OR cp.status = $5)
                      AND ($6::varchar IS NULL OR cp.payment_method = $6)
                      AND (
                          $4::text IS NULL
                          OR cp.payer_name ILIKE $4
                          OR cp.reference ILIKE $4
                          OR EXISTS (
                              SELECT 1
                              FROM payment_receipt_deliveries prd
                              WHERE prd.payment_id = cp.id
                                AND prd.receipt_number ILIKE $4
                          )
                          OR EXISTS (
                              SELECT 1
                              FROM invoice_payments ip
                              JOIN invoices i ON i.id = ip.invoice_id
                              WHERE ip.payment_id = cp.id
                                AND (
                                    i.invoice_number ILIKE $4
                                    OR i.customer_name ILIKE $4
                                )
                          )
                      )
                )
                SELECT entry_type, entry_id, created_at, occurred_date
                FROM ledger_entries
                ORDER BY occurred_date DESC NULLS LAST,
                         created_at DESC,
                         entry_type DESC,
                         entry_id DESC
                LIMIT $7 OFFSET $8
                """,
                contact_id,
                from_date,
                to_date,
                search_pattern,
                normalized_status,
                normalized_method,
                page_size + 1,
                page_offset,
            )
            page_rows = entry_rows[:page_size]
            payment_ids = [
                row["entry_id"] for row in page_rows if row["entry_type"] == "payment"
            ]
            invoice_ids = [
                row["entry_id"] for row in page_rows if row["entry_type"] == "invoice"
            ]
            payment_rows = (
                await conn.fetch(
                    """
                SELECT cp.*, pdi.batch_id
                FROM customer_payments cp
                LEFT JOIN payment_deposit_items pdi ON pdi.payment_id = cp.id
                WHERE cp.id = ANY($1::uuid[])
                """,
                    payment_ids,
                )
                if payment_ids
                else []
            )
            payment_views = {
                view["id"]: view
                for view in await self._payment_views_for_rows(
                    conn,
                    payment_rows,
                    include_receipt_delivery=True,
                    allocation_history_limit=_LEDGER_ALLOCATION_HISTORY_LIMIT,
                )
            }
            invoice_rows = (
                await conn.fetch(
                    """
                SELECT id, invoice_number, contact_id, customer_name, issue_date,
                       due_date, status, total_amount, amount_paid, amount_due,
                       sent_at, paid_at, voided_at, void_reason, source, source_ref,
                       created_at, updated_at
                FROM invoices
                WHERE id = ANY($1::uuid[])
                """,
                    invoice_ids,
                )
                if invoice_ids
                else []
            )
            invoice_views = {
                view["id"]: view for view in map(self._invoice_view, invoice_rows)
            }
            balance_row = await conn.fetchrow(
                """
                WITH customer_payments_in_scope AS (
                    SELECT cp.id, cp.total_amount, cp.status
                    FROM customer_payments cp
                    WHERE cp.contact_id = $1
                ),
                applied_by_payment AS (
                    SELECT ip.payment_id, COALESCE(SUM(ip.amount), 0) AS allocated_amount
                    FROM invoice_payments ip
                    JOIN customer_payments_in_scope cp ON cp.id = ip.payment_id
                    WHERE ip.reversed_at IS NULL
                    GROUP BY ip.payment_id
                ),
                invoice_balance AS (
                    SELECT COALESCE(SUM(i.amount_due), 0) AS open_invoice_balance
                    FROM invoices i
                    WHERE i.contact_id = $1
                      AND i.status NOT IN ('draft', 'void')
                ),
                unapplied_balance AS (
                    SELECT COALESCE(
                        SUM(cp.total_amount - COALESCE(abp.allocated_amount, 0)),
                        0
                    ) AS unapplied_payment_balance
                    FROM customer_payments_in_scope cp
                    LEFT JOIN applied_by_payment abp ON abp.payment_id = cp.id
                    WHERE cp.status = ANY($2::varchar[])
                )
                SELECT invoice_balance.open_invoice_balance,
                       unapplied_balance.unapplied_payment_balance
                FROM invoice_balance
                CROSS JOIN unapplied_balance
                """,
                contact_id,
                list(ACTIVE_PAYMENT_STATUSES),
            )

        entries: list[dict[str, Any]] = []
        for row in page_rows:
            entry_id = str(row["entry_id"])
            entry_type = row["entry_type"]
            entries.append(
                {
                    "entry_type": entry_type,
                    "entry_id": entry_id,
                    "created_at": row["created_at"],
                    "occurred_date": row["occurred_date"],
                    entry_type: (
                        payment_views[entry_id]
                        if entry_type == "payment"
                        else invoice_views[entry_id]
                    ),
                }
            )

        return {
            "contact_id": str(contact_id),
            "entries": entries,
            "next_offset": (
                page_offset + page_size if len(entry_rows) > page_size else None
            ),
            "balances": {
                "open_invoice_balance_cents": cents(
                    balance_row["open_invoice_balance"]
                ),
                "unapplied_payment_balance_cents": cents(
                    balance_row["unapplied_payment_balance"]
                ),
            },
        }

    async def adjust_allocations(
        self,
        *,
        payment_id: UUID,
        allocations: list[dict[str, Any]],
        reason: str,
        actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Replace active allocations while retaining reversed audit rows."""
        normalized = self._normalize_allocations(allocations)
        why = reason.strip()
        if not why:
            raise ReceivablesValidationError("Adjustment reason is required")
        key = idempotency_key.strip()
        if not key:
            raise ReceivablesValidationError("Idempotency key is required")
        fingerprint = request_fingerprint(
            {"payment_id": payment_id, "allocations": normalized, "reason": why}
        )

        async with self.pool.transaction() as conn:
            await self._lock_operation_key(conn, "payment-event", key)
            payment = await conn.fetchrow(
                "SELECT * FROM customer_payments WHERE id = $1 FOR UPDATE",
                payment_id,
            )
            if not payment:
                raise ReceivablesNotFoundError("Payment not found")
            existing_event = await self._event_for_key(conn, key)
            if existing_event:
                self._assert_idempotent(existing_event, fingerprint)
                return await self._payment_view(conn, payment_id)
            if payment["status"] not in ACTIVE_PAYMENT_STATUSES:
                raise ReceivablesConflictError(
                    f"Cannot adjust a {payment['status']} payment"
                )

            old_rows = await conn.fetch(
                """
                SELECT invoice_id, amount FROM invoice_payments
                WHERE payment_id = $1 AND reversed_at IS NULL
                FOR UPDATE
                """,
                payment_id,
            )
            current = {str(row["invoice_id"]): money(row["amount"]) for row in old_rows}
            invoices = await self._lock_and_validate_invoices(
                conn,
                contact_id=payment["contact_id"],
                allocations=normalized,
                current_allocations=current,
                additional_invoice_ids=[row["invoice_id"] for row in old_rows],
            )
            invoice_by_id = {str(row["id"]): row for row in invoices}
            for allocation in normalized:
                available = money(
                    invoice_by_id[str(allocation["invoice_id"])]["amount_due"]
                ) + current.get(str(allocation["invoice_id"]), Decimal("0"))
                if allocation["amount"] > available:
                    raise ReceivablesValidationError(
                        "Replacement allocation exceeds an invoice balance"
                    )
            if sum((item["amount"] for item in normalized), Decimal("0")) > money(
                payment["total_amount"]
            ):
                raise ReceivablesValidationError(
                    "Allocated amount cannot exceed the payment total"
                )

            now = datetime.now(timezone.utc)
            await conn.execute(
                """
                UPDATE invoice_payments
                SET reversed_at = $2, reversed_by = $3, reversal_reason = $4
                WHERE payment_id = $1 AND reversed_at IS NULL
                """,
                payment_id,
                now,
                actor,
                why,
            )
            for allocation in normalized:
                await conn.execute(
                    """
                    INSERT INTO invoice_payments (
                        id, invoice_id, payment_id, amount, payment_date,
                        payment_method, reference, notes, recorded_by,
                        created_at, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, '{}'::jsonb)
                    """,
                    uuid4(),
                    allocation["invoice_id"],
                    payment_id,
                    allocation["amount"],
                    payment["received_date"],
                    payment["payment_method"],
                    payment["reference"],
                    f"Allocation adjustment: {why}",
                    actor,
                    now,
                )
            await self._insert_event(
                conn,
                payment_id=payment_id,
                event_type="allocations_adjusted",
                previous_status=payment["status"],
                new_status=payment["status"],
                effective_date=date.today(),
                actor=actor,
                reason=why,
                idempotency_key=key,
                fingerprint=fingerprint,
                metadata={
                    "previous": [dict(row) for row in old_rows],
                    "replacement": normalized,
                },
            )
            affected = {row["invoice_id"] for row in old_rows}
            affected.update(item["invoice_id"] for item in normalized)
            await self._recalculate_invoices(conn, affected)
            return await self._payment_view(conn, payment_id)

    async def return_payment(
        self,
        *,
        payment_id: UUID,
        reason: str,
        actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return await self._deactivate_payment(
            payment_id=payment_id,
            target_status="returned",
            reason=reason,
            actor=actor,
            idempotency_key=idempotency_key,
        )

    async def void_payment(
        self,
        *,
        payment_id: UUID,
        reason: str,
        actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return await self._deactivate_payment(
            payment_id=payment_id,
            target_status="voided",
            reason=reason,
            actor=actor,
            idempotency_key=idempotency_key,
        )

    async def _deactivate_payment(
        self,
        *,
        payment_id: UUID,
        target_status: str,
        reason: str,
        actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        why = reason.strip()
        if not why:
            raise ReceivablesValidationError("Reason is required")
        key = idempotency_key.strip()
        if not key:
            raise ReceivablesValidationError("Idempotency key is required")
        fingerprint = request_fingerprint(
            {"payment_id": payment_id, "status": target_status, "reason": why}
        )
        timestamp_column = "returned_at" if target_status == "returned" else "voided_at"
        reason_column = (
            "return_reason" if target_status == "returned" else "void_reason"
        )

        async with self.pool.transaction() as conn:
            await self._lock_operation_key(conn, "payment-event", key)
            payment = await conn.fetchrow(
                "SELECT * FROM customer_payments WHERE id = $1 FOR UPDATE",
                payment_id,
            )
            if not payment:
                raise ReceivablesNotFoundError("Payment not found")
            event = await self._event_for_key(conn, key)
            if event:
                self._assert_idempotent(event, fingerprint)
                if payment["status"] == target_status:
                    return await self._payment_view(conn, payment_id)
                raise ReceivablesConflictError(
                    "Idempotency key already belongs to another payment event"
                )
            if payment["status"] == target_status:
                raise ReceivablesConflictError(f"Payment is already {target_status}")
            if payment["status"] not in ACTIVE_PAYMENT_STATUSES:
                action = "return" if target_status == "returned" else "void"
                raise ReceivablesConflictError(
                    f"Cannot {action} a {payment['status']} payment"
                )
            if target_status == "voided":
                deposit_batch_id = await conn.fetchval(
                    """
                    SELECT batch_id FROM payment_deposit_items
                    WHERE payment_id = $1
                    """,
                    payment_id,
                )
                if deposit_batch_id:
                    raise ReceivablesConflictError(
                        "A deposited payment cannot be voided; record it as returned"
                    )
            now = datetime.now(timezone.utc)
            await conn.execute(
                f"""
                UPDATE customer_payments
                SET status = $2, {timestamp_column} = $3, {reason_column} = $4,
                    updated_at = $3
                WHERE id = $1
                """,
                payment_id,
                target_status,
                now,
                why,
            )
            await self._insert_event(
                conn,
                payment_id=payment_id,
                event_type=f"payment_{target_status}",
                previous_status=payment["status"],
                new_status=target_status,
                effective_date=date.today(),
                actor=actor,
                reason=why,
                idempotency_key=key,
                fingerprint=fingerprint,
            )
            rows = await conn.fetch(
                """
                SELECT invoice_id FROM invoice_payments
                WHERE payment_id = $1 AND reversed_at IS NULL
                """,
                payment_id,
            )
            await self._recalculate_invoices(conn, [row["invoice_id"] for row in rows])
            return await self._payment_view(conn, payment_id)

    async def create_deposit_batch(
        self,
        *,
        payment_ids: list[UUID],
        deposit_date: date,
        bank_reference: Optional[str],
        actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        ids = sorted(set(payment_ids), key=str)
        if not ids:
            raise ReceivablesValidationError("Select at least one payment")
        if len(ids) != len(payment_ids):
            raise ReceivablesValidationError("A payment may only appear once")
        key = idempotency_key.strip()
        if not key:
            raise ReceivablesValidationError("Idempotency key is required")
        reference = (bank_reference or "").strip() or None
        fingerprint = request_fingerprint(
            {
                "payment_ids": ids,
                "deposit_date": deposit_date,
                "bank_reference": reference,
            }
        )

        async with self.pool.transaction() as conn:
            await self._lock_operation_key(conn, "payment-event", key)
            await self._lock_operation_key(conn, "deposit-create", key)
            existing = await conn.fetchrow(
                """
                SELECT id, request_fingerprint FROM payment_deposit_batches
                WHERE idempotency_key = $1 FOR UPDATE
                """,
                key,
            )
            if existing:
                self._assert_idempotent(existing, fingerprint)
                return await self._batch_view(conn, existing["id"])
            payments = await conn.fetch(
                """
                SELECT id, status, payment_method FROM customer_payments
                WHERE id = ANY($1::uuid[])
                ORDER BY id FOR UPDATE
                """,
                ids,
            )
            # A concurrent same-key retry waits on these payment locks. Recheck
            # the batch key after that wait so the retry returns the committed
            # batch instead of treating its now-deposited checks as a conflict.
            existing = await conn.fetchrow(
                """
                SELECT id, request_fingerprint FROM payment_deposit_batches
                WHERE idempotency_key = $1 FOR UPDATE
                """,
                key,
            )
            if existing:
                self._assert_idempotent(existing, fingerprint)
                return await self._batch_view(conn, existing["id"])
            if len(payments) != len(ids):
                raise ReceivablesNotFoundError("One or more payments were not found")
            await self._assert_event_key_available(
                conn, key=key, fingerprint=fingerprint
            )
            for payment in payments:
                if (
                    payment["status"] != "received"
                    or payment["payment_method"] != "check"
                ):
                    raise ReceivablesConflictError(
                        "Only received check payments can be deposited"
                    )
            already_batched = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM payment_deposit_items
                    WHERE payment_id = ANY($1::uuid[])
                )
                """,
                ids,
            )
            if already_batched:
                raise ReceivablesConflictError(
                    "A payment is already in a deposit batch"
                )

            batch_id = uuid4()
            now = datetime.now(timezone.utc)
            await conn.execute(
                """
                INSERT INTO payment_deposit_batches (
                    id, deposit_date, bank_reference, status, idempotency_key,
                    request_fingerprint, created_by, created_at, updated_at
                )
                VALUES ($1, $2, $3, 'deposited', $4, $5, $6, $7, $7)
                """,
                batch_id,
                deposit_date,
                reference,
                key,
                fingerprint,
                actor,
                now,
            )
            for payment_id in ids:
                await conn.execute(
                    """
                    INSERT INTO payment_deposit_items (batch_id, payment_id)
                    VALUES ($1, $2)
                    """,
                    batch_id,
                    payment_id,
                )
                await conn.execute(
                    """
                    UPDATE customer_payments
                    SET status = 'deposited', deposited_at = $2, updated_at = $2
                    WHERE id = $1
                    """,
                    payment_id,
                    now,
                )
                await self._insert_event(
                    conn,
                    payment_id=payment_id,
                    event_type="payment_deposited",
                    previous_status="received",
                    new_status="deposited",
                    effective_date=deposit_date,
                    actor=actor,
                    idempotency_key=key,
                    fingerprint=fingerprint,
                    metadata={"batch_id": batch_id},
                )
            return await self._batch_view(conn, batch_id)

    async def clear_deposit_batch(
        self,
        *,
        batch_id: UUID,
        actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        key = idempotency_key.strip()
        if not key:
            raise ReceivablesValidationError("Idempotency key is required")
        fingerprint = request_fingerprint({"batch_id": batch_id, "action": "clear"})
        async with self.pool.transaction() as conn:
            await self._lock_operation_key(conn, "payment-event", key)
            await self._lock_operation_key(conn, "deposit-clear", key)
            existing_clear = await conn.fetchrow(
                """
                SELECT id, clear_request_fingerprint AS request_fingerprint
                FROM payment_deposit_batches
                WHERE clear_idempotency_key = $1
                FOR UPDATE
                """,
                key,
            )
            if existing_clear:
                self._assert_idempotent(existing_clear, fingerprint)
                return await self._batch_view(conn, existing_clear["id"])
            batch = await conn.fetchrow(
                "SELECT * FROM payment_deposit_batches WHERE id = $1 FOR UPDATE",
                batch_id,
            )
            if not batch:
                raise ReceivablesNotFoundError("Deposit batch not found")
            if batch["status"] == "cleared":
                if batch["clear_idempotency_key"] != key:
                    raise ReceivablesConflictError("Deposit batch is already cleared")
                self._assert_idempotent(
                    {"request_fingerprint": batch["clear_request_fingerprint"]},
                    fingerprint,
                )
                return await self._batch_view(conn, batch_id)
            if batch["status"] != "deposited":
                raise ReceivablesConflictError(
                    f"Cannot clear a {batch['status']} deposit batch"
                )
            payments = await conn.fetch(
                """
                SELECT cp.id, cp.status
                FROM customer_payments cp
                JOIN payment_deposit_items pdi ON pdi.payment_id = cp.id
                WHERE pdi.batch_id = $1
                ORDER BY cp.id FOR UPDATE
                """,
                batch_id,
            )
            if not payments:
                raise ReceivablesConflictError("Deposit batch has no payments")
            if any(payment["status"] != "deposited" for payment in payments):
                raise ReceivablesConflictError(
                    "Every payment in the batch must still be deposited before clearing"
                )
            await self._assert_event_key_available(
                conn,
                key=key,
                fingerprint=fingerprint,
            )
            now = datetime.now(timezone.utc)
            await conn.execute(
                """
                UPDATE payment_deposit_batches
                SET status = 'cleared', cleared_at = $2, cleared_by = $3,
                    clear_idempotency_key = $4, clear_request_fingerprint = $5,
                    updated_at = $2
                WHERE id = $1
                """,
                batch_id,
                now,
                actor,
                key,
                fingerprint,
            )
            for payment in payments:
                await conn.execute(
                    """
                    UPDATE customer_payments
                    SET status = 'cleared', cleared_at = $2, updated_at = $2
                    WHERE id = $1
                    """,
                    payment["id"],
                    now,
                )
                await self._insert_event(
                    conn,
                    payment_id=payment["id"],
                    event_type="payment_cleared",
                    previous_status="deposited",
                    new_status="cleared",
                    effective_date=date.today(),
                    actor=actor,
                    idempotency_key=key,
                    fingerprint=fingerprint,
                    metadata={"batch_id": batch_id},
                )
            return await self._batch_view(conn, batch_id)

    async def list_deposit_batches(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            SELECT b.*, COUNT(i.payment_id) AS payment_count,
                   COALESCE(SUM(p.total_amount), 0) AS total_amount
            FROM payment_deposit_batches b
            LEFT JOIN payment_deposit_items i ON i.batch_id = b.id
            LEFT JOIN customer_payments p ON p.id = i.payment_id
            GROUP BY b.id
            ORDER BY b.deposit_date DESC, b.created_at DESC
            LIMIT $1
            """,
            max(1, min(limit, 500)),
        )
        return [self._batch_summary(row) for row in rows]

    async def recalculate_invoice(self, invoice_id: UUID) -> None:
        """Compatibility entry point for legacy repository callers."""
        async with self.pool.transaction() as conn:
            await self._recalculate_invoices(conn, [invoice_id])

    @staticmethod
    def _assert_receipt_recipient_invariant(
        *,
        contact_id: Optional[UUID],
        receipt_recipient: Optional[PaymentReceiptRecipient],
    ) -> None:
        """Reject malformed internal receipt context before a new write.

        The public HTTP body never carries these fields: EOM routes resolve
        them from their canonical CRM provider.  This defensive check keeps a
        future direct caller from attaching one customer's receipt metadata to
        another customer's payment.
        """
        if receipt_recipient is None:
            return
        if not isinstance(receipt_recipient, PaymentReceiptRecipient):
            raise ReceivablesValidationError("Receipt customer context is invalid")
        if contact_id is None or receipt_recipient.contact_id != contact_id:
            raise ReceivablesValidationError(
                "Receipt customer context must match the payment customer"
            )
        if receipt_recipient.customer_type not in EOM_CUSTOMER_TYPES:
            raise ReceivablesValidationError("Receipt customer type is invalid")
        if not receipt_recipient.customer_name.strip():
            raise ReceivablesValidationError("Receipt customer name is required")
        if receipt_recipient.recipient_email is not None:
            from .eom_crm_mutations import normalize_contact_email

            if (
                normalize_contact_email(receipt_recipient.recipient_email)
                != receipt_recipient.recipient_email
            ):
                raise ReceivablesValidationError("Receipt email is invalid")

    @staticmethod
    async def _enqueue_residential_payment_receipt(
        conn: Any,
        *,
        payment_id: UUID,
        receipt_recipient: PaymentReceiptRecipient,
        payer_name: str,
        total_amount: Decimal,
        payment_method: str,
        reference: Optional[str],
        received_date: date,
        created_at: datetime,
    ) -> None:
        """Persist one non-sending receipt record in the payment transaction."""
        receipt_number, subject, body = render_residential_payment_receipt(
            payment_id=payment_id,
            customer_name=receipt_recipient.customer_name.strip(),
            payer_name=payer_name,
            total_amount=total_amount,
            payment_method=payment_method,
            reference=reference,
            received_date=received_date,
        )
        recipient_email = receipt_recipient.recipient_email
        status = "pending" if recipient_email else "skipped"
        skip_reason = None if recipient_email else "no_email"
        await conn.execute(
            """
            INSERT INTO payment_receipt_deliveries (
                id, payment_id, contact_id, receipt_number, recipient_email,
                delivery_status, skip_reason, subject, body, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10)
            """,
            uuid4(),
            payment_id,
            receipt_recipient.contact_id,
            receipt_number,
            recipient_email,
            status,
            skip_reason,
            subject,
            body,
            created_at,
        )

    @staticmethod
    def _normalize_allocations(
        allocations: list[dict[str, Any]],
        *,
        allow_empty: bool = False,
    ) -> list[dict[str, Any]]:
        if not allocations:
            if allow_empty and allocations == []:
                return []
            raise ReceivablesValidationError(
                "At least one invoice allocation is required"
            )
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for allocation in allocations:
            try:
                invoice_id = UUID(str(allocation["invoice_id"]))
                amount = money(allocation["amount"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReceivablesValidationError(
                    "Each allocation requires a valid invoice_id and amount"
                ) from exc
            if amount <= 0:
                raise ReceivablesValidationError("Allocation amounts must be positive")
            if str(invoice_id) in seen:
                raise ReceivablesValidationError(
                    "An invoice may only appear once in a payment"
                )
            seen.add(str(invoice_id))
            normalized.append({"invoice_id": invoice_id, "amount": amount})
        return sorted(normalized, key=lambda item: str(item["invoice_id"]))

    async def _lock_and_validate_invoices(
        self,
        conn: Any,
        *,
        contact_id: Optional[UUID],
        allocations: list[dict[str, Any]],
        current_allocations: Optional[dict[str, Decimal]] = None,
        additional_invoice_ids: Optional[Iterable[UUID]] = None,
        unapplied_contact_context_id: Optional[str] = None,
    ) -> list[Any]:
        allocated_ids = {item["invoice_id"] for item in allocations}
        invoice_ids = sorted(allocated_ids.union(additional_invoice_ids or []), key=str)
        if not invoice_ids:
            if contact_id is None:
                raise ReceivablesValidationError(
                    "A customer is required when a payment has no invoice allocations"
                )
            contact = await conn.fetchrow(
                """
                SELECT id
                FROM contacts
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'customer'
                  AND status = 'active'
                FOR SHARE
                """,
                contact_id,
                unapplied_contact_context_id,
            )
            if not contact:
                raise ReceivablesNotFoundError("Customer not found")
            return []
        rows = await conn.fetch(
            """
            SELECT id, invoice_number, contact_id, status, amount_due
            FROM invoices
            WHERE id = ANY($1::uuid[])
            ORDER BY id FOR UPDATE
            """,
            invoice_ids,
        )
        if len(rows) != len(invoice_ids):
            raise ReceivablesNotFoundError("One or more invoices were not found")
        current_allocations = current_allocations or {}
        for row in rows:
            if contact_id is None:
                same_customer = len(invoice_ids) == 1 and row["contact_id"] is None
            else:
                same_customer = (
                    row["contact_id"] is not None
                    and str(row["contact_id"]) == str(contact_id)
                )
            if not same_customer:
                raise ReceivablesValidationError(
                    "All allocated invoices must belong to the same customer"
                )
            if row["id"] not in allocated_ids:
                continue
            eligible = row["status"] in OPEN_INVOICE_STATUSES
            if row["status"] == "paid" and str(row["id"]) in current_allocations:
                eligible = True
            if not eligible:
                raise ReceivablesConflictError(
                    f"Invoice {row['invoice_number']} is not open for payment"
                )
        return rows

    @staticmethod
    def _assert_idempotent(row: Any, fingerprint: str) -> None:
        if row["request_fingerprint"] != fingerprint:
            raise ReceivablesConflictError(
                "Idempotency key was already used for a different request"
            )

    @staticmethod
    async def _event_for_key(conn: Any, key: str) -> Optional[Any]:
        return await conn.fetchrow(
            """
            SELECT payment_id, request_fingerprint
            FROM payment_events
            WHERE idempotency_key = $1
            ORDER BY payment_id
            LIMIT 1
            """,
            key,
        )

    async def _assert_event_key_available(
        self,
        conn: Any,
        *,
        key: str,
        fingerprint: str,
    ) -> None:
        event = await self._event_for_key(conn, key)
        if not event:
            return
        self._assert_idempotent(event, fingerprint)
        # Matching endpoint retries reconcile through their payment/batch parent
        # before reaching this guard. An orphaned same-fingerprint event is not
        # safe evidence that the requested state transition completed.
        raise ReceivablesConflictError(
            "Idempotency key already belongs to a payment event"
        )

    @staticmethod
    async def _insert_event(
        conn: Any,
        *,
        payment_id: UUID,
        event_type: str,
        previous_status: Optional[str],
        new_status: Optional[str],
        effective_date: Optional[date],
        actor: Optional[str],
        idempotency_key: str,
        fingerprint: str,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO payment_events (
                id, payment_id, event_type, previous_status, new_status,
                effective_date, actor, reason, idempotency_key,
                request_fingerprint, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
            """,
            uuid4(),
            payment_id,
            event_type,
            previous_status,
            new_status,
            effective_date,
            actor,
            reason,
            idempotency_key,
            fingerprint,
            json.dumps(_jsonable(metadata or {})),
        )

    @staticmethod
    async def _lock_operation_key(conn: Any, scope: str, key: str) -> None:
        """Serialize equal idempotency keys before checking their persisted row."""
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"{scope}:{key}",
        )

    @staticmethod
    async def _recalculate_invoices(conn: Any, invoice_ids: Iterable[UUID]) -> None:
        ids = sorted(set(invoice_ids), key=str)
        if not ids:
            return
        # Serialize every balance recomputation on the invoice rows themselves.
        # In particular, a return/void must take this lock before its aggregate
        # snapshot so it cannot overwrite a concurrently-created allocation with
        # a stale amount_paid value after waiting on the UPDATE row lock.
        await conn.fetch(
            """
            SELECT id
            FROM invoices
            WHERE id = ANY($1::uuid[])
            ORDER BY id FOR UPDATE
            """,
            ids,
        )
        await conn.execute(
            """
            WITH totals AS (
                SELECT i.id,
                       COALESCE(SUM(ip.amount) FILTER (
                           WHERE ip.reversed_at IS NULL
                             AND (
                                 ip.payment_id IS NULL
                                 OR cp.status = ANY($2::varchar[])
                             )
                       ), 0) AS paid
                FROM invoices i
                LEFT JOIN invoice_payments ip ON ip.invoice_id = i.id
                LEFT JOIN customer_payments cp ON cp.id = ip.payment_id
                WHERE i.id = ANY($1::uuid[])
                GROUP BY i.id
            )
            UPDATE invoices i
            SET amount_paid = totals.paid,
                status = CASE
                    WHEN i.status IN ('draft', 'void') THEN i.status
                    WHEN i.total_amount - totals.paid <= 0 THEN 'paid'
                    WHEN i.due_date < $3 THEN 'overdue'
                    WHEN totals.paid > 0 THEN 'partial'
                    ELSE 'sent'
                END,
                paid_at = CASE
                    WHEN i.total_amount - totals.paid <= 0
                        THEN COALESCE(i.paid_at, $4)
                    ELSE NULL
                END,
                updated_at = $4
            FROM totals
            WHERE i.id = totals.id
            """,
            ids,
            list(ACTIVE_PAYMENT_STATUSES),
            date.today(),
            datetime.now(timezone.utc),
        )

    async def _payment_view(
        self,
        conn: Any,
        payment_id: UUID,
        *,
        include_receipt_delivery: bool = False,
    ) -> dict[str, Any]:
        row = await conn.fetchrow(
            """
            SELECT cp.*, pdi.batch_id
            FROM customer_payments cp
            LEFT JOIN payment_deposit_items pdi ON pdi.payment_id = cp.id
            WHERE cp.id = $1
            """,
            payment_id,
        )
        if not row:
            raise ReceivablesNotFoundError("Payment not found")
        allocations = await conn.fetch(
            """
            SELECT ip.payment_id, ip.id, ip.invoice_id, i.invoice_number,
                   ip.amount, ip.payment_date, ip.payment_method, ip.reference,
                   ip.notes, ip.recorded_by, ip.created_at, ip.metadata,
                   ip.reversed_at, ip.reversal_reason
            FROM invoice_payments ip
            JOIN invoices i ON i.id = ip.invoice_id
            WHERE ip.payment_id = $1
            ORDER BY i.due_date, i.invoice_number, ip.created_at
            """,
            payment_id,
        )
        receipt_delivery = None
        if include_receipt_delivery:
            receipt_delivery = await conn.fetchrow(
                """
                SELECT receipt_number, recipient_email,
                       delivery_status AS status, skip_reason
                FROM payment_receipt_deliveries
                WHERE payment_id = $1
                """,
                payment_id,
            )
        return self._compose_payment(
            row,
            allocations,
            receipt_delivery=receipt_delivery,
            include_receipt_delivery=include_receipt_delivery,
        )

    async def _payment_views_for_rows(
        self,
        executor: Any,
        rows: list[Any],
        *,
        include_receipt_delivery: bool = False,
        allocation_history_limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Hydrate payment list rows without changing their requested order."""
        if not rows:
            return []
        payment_ids = [row["id"] for row in rows]
        if allocation_history_limit is None:
            allocation_rows = await executor.fetch(
                """
                SELECT ip.payment_id, ip.id, ip.invoice_id, i.invoice_number,
                       ip.amount, ip.payment_date, ip.payment_method, ip.reference,
                       ip.notes, ip.recorded_by, ip.created_at, ip.metadata,
                       ip.reversed_at, ip.reversal_reason
                FROM invoice_payments ip
                JOIN invoices i ON i.id = ip.invoice_id
                WHERE ip.payment_id = ANY($1::uuid[])
                ORDER BY i.due_date, i.invoice_number, ip.created_at
                """,
                payment_ids,
            )
        else:
            if allocation_history_limit < 1:
                raise ValueError("allocation_history_limit must be positive")
            allocation_rows = await executor.fetch(
                """
                WITH allocation_totals AS (
                    SELECT ip.payment_id,
                           COALESCE(SUM(ip.amount) FILTER (
                               WHERE ip.reversed_at IS NULL
                           ), 0) AS active_allocated_amount,
                           COUNT(*) AS allocation_history_count
                    FROM invoice_payments ip
                    WHERE ip.payment_id = ANY($1::uuid[])
                    GROUP BY ip.payment_id
                ),
                ranked_allocations AS (
                    SELECT ip.payment_id, ip.id, ip.invoice_id, i.invoice_number,
                           ip.amount, ip.payment_date, ip.payment_method,
                           ip.reference, ip.notes, ip.recorded_by, ip.created_at,
                           ip.metadata, ip.reversed_at, ip.reversal_reason,
                           ROW_NUMBER() OVER (
                               PARTITION BY ip.payment_id
                               ORDER BY (ip.reversed_at IS NULL) DESC,
                                        i.due_date,
                                        i.invoice_number,
                                        ip.created_at,
                                        ip.id
                           ) AS allocation_rank
                    FROM invoice_payments ip
                    JOIN invoices i ON i.id = ip.invoice_id
                    WHERE ip.payment_id = ANY($1::uuid[])
                )
                SELECT ranked_allocations.payment_id,
                       ranked_allocations.id,
                       ranked_allocations.invoice_id,
                       ranked_allocations.invoice_number,
                       ranked_allocations.amount,
                       ranked_allocations.payment_date,
                       ranked_allocations.payment_method,
                       ranked_allocations.reference,
                       ranked_allocations.notes,
                       ranked_allocations.recorded_by,
                       ranked_allocations.created_at,
                       ranked_allocations.metadata,
                       ranked_allocations.reversed_at,
                       ranked_allocations.reversal_reason,
                       allocation_totals.active_allocated_amount,
                       allocation_totals.allocation_history_count
                FROM ranked_allocations
                JOIN allocation_totals
                  ON allocation_totals.payment_id = ranked_allocations.payment_id
                WHERE ranked_allocations.allocation_rank <= $2
                ORDER BY ranked_allocations.payment_id, ranked_allocations.allocation_rank
                """,
                payment_ids,
                allocation_history_limit,
            )
        allocations_by_payment: dict[str, list[Any]] = {
            str(payment_id): [] for payment_id in payment_ids
        }
        allocation_summaries_by_payment: dict[str, dict[str, Any]] = {}
        for allocation in allocation_rows:
            payment_id = str(allocation["payment_id"])
            allocations_by_payment[payment_id].append(allocation)
            if allocation_history_limit is not None:
                allocation_summaries_by_payment[payment_id] = {
                    "active_allocated_amount": allocation["active_allocated_amount"],
                    "allocation_history_count": allocation["allocation_history_count"],
                }

        receipt_by_payment: dict[str, dict[str, Any]] = {}
        if include_receipt_delivery:
            receipt_rows = await executor.fetch(
                """
                SELECT payment_id, receipt_number, recipient_email,
                       delivery_status AS status, skip_reason
                FROM payment_receipt_deliveries
                WHERE payment_id = ANY($1::uuid[])
                """,
                payment_ids,
            )
            receipt_by_payment = {
                str(receipt["payment_id"]): {
                    "receipt_number": receipt["receipt_number"],
                    "recipient_email": receipt["recipient_email"],
                    "status": receipt["status"],
                    "skip_reason": receipt["skip_reason"],
                }
                for receipt in receipt_rows
            }

        payment_views: list[dict[str, Any]] = []
        for row in rows:
            payment_id = str(row["id"])
            allocation_summary = allocation_summaries_by_payment.get(payment_id)
            payment_views.append(
                self._compose_payment(
                    row,
                    allocations_by_payment.get(payment_id, []),
                    receipt_delivery=receipt_by_payment.get(payment_id),
                    include_receipt_delivery=include_receipt_delivery,
                    active_allocated_amount=(
                        allocation_summary["active_allocated_amount"]
                        if allocation_summary is not None
                        else Decimal("0")
                        if allocation_history_limit is not None
                        else None
                    ),
                    allocation_history_count=(
                        int(allocation_summary["allocation_history_count"])
                        if allocation_summary is not None
                        else 0
                        if allocation_history_limit is not None
                        else None
                    ),
                )
            )
        return payment_views

    async def _has_replay_receipt_delivery(
        self,
        conn: Any,
        *,
        payment_id: UUID,
        require_receipt_recipient: bool,
    ) -> bool:
        """Read a receipt projection only when an EOM replay can prove it exists.

        The original receipt is immutable payment evidence, whereas the CRM
        customer type is mutable.  An unchanged retry must therefore derive
        its projection from the committed outbox row.  The held transaction
        connection performs the readiness/catalog probe so this path never
        tries to acquire a second pool connection.  A pre-369 ledger simply
        returns no projection without querying its missing outbox table.
        """
        if not require_receipt_recipient:
            return False
        if not await self.is_receipt_delivery_ready(conn):
            return False
        return bool(
            await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM payment_receipt_deliveries
                    WHERE payment_id = $1
                )
                """,
                payment_id,
            )
        )

    @staticmethod
    def _compose_payment(
        row: Any,
        allocations: list[Any],
        *,
        receipt_delivery: Optional[Any] = None,
        include_receipt_delivery: bool = False,
        active_allocated_amount: Optional[Decimal] = None,
        allocation_history_count: Optional[int] = None,
    ) -> dict[str, Any]:
        result = _serialize_row(row)
        if include_receipt_delivery:
            result["receipt_delivery"] = (
                _serialize_row(receipt_delivery)
                if receipt_delivery is not None
                else None
            )
        allocation_views = []
        for allocation in allocations:
            view = _serialize_row(allocation)
            view.pop("active_allocated_amount", None)
            view.pop("allocation_history_count", None)
            allocation_views.append(view)
        active_allocated = Decimal("0")
        is_active = row["status"] in ACTIVE_PAYMENT_STATUSES
        if is_active and active_allocated_amount is not None:
            active_allocated = money(active_allocated_amount)
        elif is_active:
            active_allocated = sum(
                (
                    money(item["amount"])
                    for item in allocations
                    if item["reversed_at"] is None
                ),
                Decimal("0"),
            )
        result["total_amount_cents"] = cents(row["total_amount"])
        result["allocated_amount_cents"] = cents(active_allocated)
        available_unapplied = (
            money(row["total_amount"]) - active_allocated
            if is_active
            else Decimal("0")
        )
        result["unapplied_amount_cents"] = cents(available_unapplied)
        for view in allocation_views:
            view["amount_cents"] = cents(view["amount"])
        result["allocations"] = allocation_views
        if allocation_history_count is not None:
            result["allocation_history_count"] = allocation_history_count
            result["allocations_truncated"] = (
                allocation_history_count > len(allocation_views)
            )
        return result

    @staticmethod
    def _invoice_view(row: Any) -> dict[str, Any]:
        result = _serialize_row(row)
        for field in ("total_amount", "amount_paid", "amount_due"):
            result[f"{field}_cents"] = cents(row[field])
        return result

    async def _batch_view(self, conn: Any, batch_id: UUID) -> dict[str, Any]:
        row = await conn.fetchrow(
            """
            SELECT b.*, COUNT(i.payment_id) AS payment_count,
                   COALESCE(SUM(p.total_amount), 0) AS total_amount
            FROM payment_deposit_batches b
            LEFT JOIN payment_deposit_items i ON i.batch_id = b.id
            LEFT JOIN customer_payments p ON p.id = i.payment_id
            WHERE b.id = $1
            GROUP BY b.id
            """,
            batch_id,
        )
        if not row:
            raise ReceivablesNotFoundError("Deposit batch not found")
        result = self._batch_summary(row)
        payments = await conn.fetch(
            """
            SELECT p.id, p.payer_name, p.reference, p.total_amount, p.status
            FROM customer_payments p
            JOIN payment_deposit_items i ON i.payment_id = p.id
            WHERE i.batch_id = $1
            ORDER BY p.payer_name, p.received_date, p.id
            """,
            batch_id,
        )
        result["payments"] = [
            {
                **_serialize_row(payment),
                "total_amount_cents": cents(payment["total_amount"]),
            }
            for payment in payments
        ]
        return result

    @staticmethod
    def _batch_summary(row: Any) -> dict[str, Any]:
        result = _serialize_row(row)
        result["total_amount_cents"] = cents(row["total_amount"])
        return result


_receivables_service: Optional[ReceivablesService] = None


def get_receivables_service() -> ReceivablesService:
    global _receivables_service
    if _receivables_service is None:
        _receivables_service = ReceivablesService()
    return _receivables_service
