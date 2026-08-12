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

from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseUnavailableError

ACTIVE_PAYMENT_STATUSES = ("legacy", "received", "deposited", "cleared")
OPEN_INVOICE_STATUSES = ("sent", "partial", "overdue")
API_PAYMENT_METHODS = ("check", "ach", "square")
_CENT = Decimal("0.01")
_MAX_DATABASE_MONEY = Decimal("9999999999.99")
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


@dataclass(frozen=True)
class PaymentCreationOutcome:
    """Internal payment write result; public transports emit only ``payment``."""

    payment: dict[str, Any]
    replayed: bool


class ReceivablesError(Exception):
    """Base error with a stable machine-readable code."""

    code = "receivables_error"


class ReceivablesValidationError(ReceivablesError):
    code = "validation_error"


class ReceivablesNotFoundError(ReceivablesError):
    code = "not_found"


class ReceivablesConflictError(ReceivablesError):
    code = "conflict"


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

    async def is_ready(self) -> bool:
        """Return whether every required ledger column and index is usable."""
        required = [
            (table_name, column_name)
            for table_name, columns in _RECEIVABLES_REQUIRED_COLUMNS.items()
            for column_name in columns
        ]
        columns_ready = bool(
            await self.pool.fetchval(
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

        index_rows = await self.pool.fetch(
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
                for _table_name, index_name, *_rest in _RECEIVABLES_REQUIRED_INDEXES
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
            ) in _RECEIVABLES_REQUIRED_INDEXES
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
        allocations: list[dict[str, Any]],
        idempotency_key: str,
        check_date: Optional[date] = None,
        received_through: Optional[str] = None,
        reference: Optional[str] = None,
        notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        source: str = "eom_admin",
        metadata: Optional[dict[str, Any]] = None,
        enforce_api_methods: bool = True,
        allow_unapplied: bool = False,
        unapplied_contact_context_id: Optional[str] = None,
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
        allocations: list[dict[str, Any]],
        idempotency_key: str,
        check_date: Optional[date] = None,
        received_through: Optional[str] = None,
        reference: Optional[str] = None,
        notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        source: str = "eom_admin",
        metadata: Optional[dict[str, Any]] = None,
        enforce_api_methods: bool = True,
        allow_unapplied: bool = False,
        unapplied_contact_context_id: Optional[str] = None,
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

        normalized = self._normalize_allocations(
            allocations, allow_empty=allow_unapplied
        )
        allocated_total = sum((item["amount"] for item in normalized), Decimal("0"))
        if allocated_total > total:
            raise ReceivablesValidationError(
                "Allocated amount cannot exceed the payment total"
            )
        initial_status = "received" if method == "check" else "cleared"
        payload = {
            "contact_id": contact_id,
            "payer_name": payer,
            "total_amount": total,
            "payment_method": method,
            "received_date": received_date,
            "check_date": check_date,
            "received_through": (received_through or "").strip() or None,
            "allocations": normalized,
            "reference": (reference or "").strip() or None,
            "notes": (notes or "").strip() or None,
        }
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
                    payment=await self._payment_view(conn, existing["id"]),
                    replayed=True,
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
                    payment=await self._payment_view(conn, existing["id"]),
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
            payment_row = await conn.fetchrow(
                """
                INSERT INTO customer_payments (
                    id, contact_id, payer_name, total_amount, payment_method,
                    reference, received_date, check_date, received_through,
                    status, source, idempotency_key, request_fingerprint, notes,
                    recorded_by, cleared_at, metadata, created_at, updated_at
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
                    payment=await self._payment_view(conn, existing["id"]),
                    replayed=True,
                )

            now = datetime.now(timezone.utc)
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
                payment=await self._payment_view(conn, payment_id),
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
        payment_ids = [row["id"] for row in rows]
        allocation_rows = await self.pool.fetch(
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
        grouped: dict[str, list[Any]] = {str(item): [] for item in payment_ids}
        for allocation in allocation_rows:
            grouped[str(allocation["payment_id"])].append(allocation)
        return [
            self._compose_payment(row, grouped.get(str(row["id"]), [])) for row in rows
        ]

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

    async def _payment_view(self, conn: Any, payment_id: UUID) -> dict[str, Any]:
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
        return self._compose_payment(row, allocations)

    @staticmethod
    def _compose_payment(row: Any, allocations: list[Any]) -> dict[str, Any]:
        result = _serialize_row(row)
        allocation_views = [_serialize_row(item) for item in allocations]
        active_allocated = Decimal("0")
        is_active = row["status"] in ACTIVE_PAYMENT_STATUSES
        if is_active:
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
