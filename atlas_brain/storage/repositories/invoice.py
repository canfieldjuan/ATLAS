"""
Invoice repository for billing and payment tracking.

Provides CRUD operations for invoices and payments stored in PostgreSQL.
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.invoice")


# Only the commercial-billing approval writer sets this marker.  Its line
# amounts are derived from retained whole-minute evidence, so a later
# non-financial draft edit must preserve those exact cents rather than multiply
# a rounded display quantity by its unit price again.
_COMMERCIAL_BILLING_EXACT_LINE_AMOUNTS_MARKER = "commercialBillingExactLineAmounts"
_RECURRING_INVOICE_DEDUP_INDEX = "idx_invoices_recurring_contact_period_source"
_RECURRING_INVOICE_DEDUP_CONSTRAINTS = (
    "invoices_billing_period_check",
    "invoices_recurring_billing_period_required_check",
)


async def recurring_invoice_dedup_schema_ready(conn: Any) -> bool:
    """Return whether recurring invoice writers can safely rely on period dedup.

    This is deliberately separate from receivables/payment readiness. Check,
    ACH, Square, ad-hoc invoice, and EOM funnel surfaces do not create
    ``monthly_auto`` or ``eom_commercial_billing`` invoices and must not be
    blocked by this writer-only schema requirement.
    """
    columns_ready = bool(
        await conn.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM (
                    VALUES
                        ('invoices', 'billing_period'),
                        ('invoices', 'billing_period_legacy_null'),
                        ('invoices_billing_period_reservations', 'contact_id'),
                        ('invoices_billing_period_reservations', 'billing_period')
                ) AS required(table_name, column_name)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns AS actual
                    WHERE actual.table_schema = current_schema()
                      AND actual.table_name = required.table_name
                      AND actual.column_name = required.column_name
                )
            )
            """
        )
    )
    if not columns_ready:
        return False

    constraints_ready = bool(
        await conn.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM unnest($1::text[]) AS required(conname)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint AS actual
                    JOIN pg_class AS table_class
                      ON table_class.oid = actual.conrelid
                    JOIN pg_namespace AS table_namespace
                      ON table_namespace.oid = table_class.relnamespace
                    WHERE table_namespace.nspname = current_schema()
                      AND actual.conname = required.conname
                      AND table_class.relname = 'invoices'
                )
            )
            """,
            list(_RECURRING_INVOICE_DEDUP_CONSTRAINTS),
        )
    )
    if not constraints_ready:
        return False

    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_index AS index_state
                JOIN pg_class AS table_class
                  ON table_class.oid = index_state.indrelid
                JOIN pg_namespace AS table_namespace
                  ON table_namespace.oid = table_class.relnamespace
                JOIN pg_class AS index_class
                  ON index_class.oid = index_state.indexrelid
                WHERE table_namespace.nspname = current_schema()
                  AND table_class.relname = 'invoices'
                  AND index_class.relname = $1
                  AND index_state.indisunique
                  AND index_state.indisvalid
                  AND index_state.indisready
            )
            """,
            _RECURRING_INVOICE_DEDUP_INDEX,
        )
    )


@dataclass(frozen=True)
class InvoicePaymentRecordOutcome:
    """Internal singular-payment result with atomic replay classification."""

    payment: dict[str, Any]
    replayed: bool


def _line_items_are_billable(items: list[dict]) -> bool:
    """Return True when every line item has positive quantity and unit price."""
    if not items:
        return False
    try:
        return all(
            Decimal(str(item.get("quantity", 0))) > 0
            and Decimal(str(item.get("unit_price", 0))) > 0
            for item in items
        )
    except Exception:
        return False


def _line_items_with_amounts(items: list[dict], *, overwrite: bool) -> list[dict]:
    """Return line items with calculated amounts, optionally replacing stale values."""
    normalized: list[dict] = []
    for item in items:
        line = dict(item)
        if overwrite or "amount" not in line:
            line["amount"] = float(
                Decimal(str(line.get("quantity", 1))) * Decimal(str(line.get("unit_price", 0)))
            )
        normalized.append(line)
    return normalized


def _line_items_subtotal(
    items: list[dict], *, prefer_recorded_amounts: bool
) -> Decimal:
    """Calculate a draft subtotal without losing trusted committed cents.

    A notes-only or due-date-only edit does not replace the line-item evidence.
    A commercial-billing approval explicitly marks its persisted amounts as
    authoritative because an hourly line can display rounded hours while its
    approved amount was calculated from whole minutes.  Generic invoices do
    not receive that authority: their caller-supplied ``amount`` field remains
    display data and their subtotal is recalculated from quantity and price.
    Replacing line items always retains the existing recalculation behavior.
    """

    subtotal = Decimal("0")
    for item in items:
        amount: Decimal | None = None
        if prefer_recorded_amounts and "amount" in item:
            try:
                recorded = Decimal(str(item["amount"]))
            except (InvalidOperation, TypeError, ValueError):
                recorded = None
            if (
                recorded is not None
                and recorded.is_finite()
                and recorded == recorded.quantize(Decimal("0.01"))
            ):
                amount = recorded
        if amount is None:
            amount = Decimal(str(item.get("quantity", 1))) * Decimal(
                str(item.get("unit_price", 0))
            )
        subtotal += amount
    return subtotal


def _has_trusted_commercial_billing_line_amounts(invoice: dict[str, Any]) -> bool:
    """Return whether approved commercial evidence may retain line amounts.

    The generic repository and MCP create paths accept line-item dictionaries
    from callers, including an optional ``amount`` field.  Those fields cannot
    become authoritative merely because a later edit omitted ``line_items``.
    The approval writer is the one controlled producer that records exact-cent
    amounts from retained billing evidence and marks that fact explicitly.
    """

    metadata = invoice.get("metadata")
    return (
        invoice.get("source") == "eom_commercial_billing"
        and isinstance(metadata, dict)
        and metadata.get(_COMMERCIAL_BILLING_EXACT_LINE_AMOUNTS_MARKER) is True
    )


class InvoiceRepository:
    """Repository for invoice and payment storage and retrieval."""

    def __init__(
        self,
        *,
        pool: Any = None,
        receivables_service: Any = None,
    ) -> None:
        self._pool_override = pool
        self._receivables_service_override = receivables_service

    def _get_pool(self) -> Any:
        if self._pool_override is not None:
            return self._pool_override
        return get_db_pool()

    def _get_receivables_service(self) -> Any:
        if self._receivables_service_override is not None:
            return self._receivables_service_override
        from ...services.receivables import get_receivables_service

        return get_receivables_service()

    async def recurring_dedup_ready(self) -> bool:
        """Return whether recurring-source invoice writes are safely fenced."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("recurring invoice dedup readiness")
        return await recurring_invoice_dedup_schema_ready(pool)

    # -- Invoice CRUD ----------------------------------------------

    async def create(
        self,
        customer_name: str,
        due_date: date,
        line_items: list[dict],
        contact_id: Optional[UUID] = None,
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None,
        customer_address: Optional[str] = None,
        tax_rate: float = 0.0,
        discount_amount: float = 0.0,
        invoice_for: Optional[str] = None,
        contact_name: Optional[str] = None,
        issue_date: Optional[date] = None,
        source: str = "manual",
        source_ref: Optional[str] = None,
        appointment_id: Optional[UUID] = None,
        business_context_id: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
        billing_period: Optional[date] = None,
    ) -> dict:
        """Create a new invoice with auto-generated invoice number.

        billing_period: the date whose YYYY-Mon is embedded in the invoice
        number. Defaults to issue_date if not given. Set this to the first
        day of the period when invoicing past work (e.g. April 1 for an
        April-billing invoice issued in early May).

        Also persisted verbatim (as YYYY-MM) to invoices.billing_period for
        cross-pipeline recurring-invoice dedup (migration 385). Leave unset
        for ad-hoc/non-recurring invoices (e.g. the MCP create_invoice tool)
        so they never participate in the recurring-source uniqueness
        constraint.
        """
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("create invoice")

        invoice_id = uuid4()
        now = datetime.now(timezone.utc)
        issue = issue_date or date.today()

        # Calculate amounts from line items
        subtotal = sum(
            Decimal(str(item.get("quantity", 1))) * Decimal(str(item.get("unit_price", 0)))
            for item in line_items
        )
        tax_amt = subtotal * Decimal(str(tax_rate))
        total = subtotal + tax_amt - Decimal(str(discount_amount))

        # Ensure each line item has an amount field
        for item in line_items:
            if "amount" not in item:
                item["amount"] = float(
                    Decimal(str(item.get("quantity", 1))) * Decimal(str(item.get("unit_price", 0)))
                )

        try:
            args = (
                invoice_id,
                contact_id,
                customer_name,
                customer_email,
                customer_phone,
                customer_address,
                json.dumps(line_items),
                float(subtotal),
                float(tax_rate),
                float(tax_amt),
                float(discount_amount),
                float(total),
                issue,
                due_date,
                source,
                source_ref,
                appointment_id,
                business_context_id,
                "INV",  # $19 - prefix
                notes,
                json.dumps(metadata or {}),
                invoice_for,
                contact_name,
                now,
            )
            if billing_period is None:
                row = await pool.fetchrow(
                    """
                    INSERT INTO invoices (
                        id, invoice_number,
                        contact_id, customer_name, customer_email, customer_phone, customer_address,
                        line_items, subtotal, tax_rate, tax_amount, discount_amount, total_amount,
                        issue_date, due_date, status, source, source_ref, appointment_id,
                        business_context_id, notes, metadata, invoice_for, contact_name,
                        created_at, updated_at
                    )
                    VALUES (
                        $1,
                        (SELECT $19 || '-' || to_char($13::date, 'YYYY-Mon') || '-' || lpad(nextval('invoice_number_seq')::text, 4, '0')),
                        $2, $3, $4, $5, $6,
                        $7::jsonb, $8, $9, $10, $11, $12,
                        $13, $14, 'draft', $15, $16, $17,
                        $18, $20, $21::jsonb, $22, $23,
                        $24, $24
                    )
                    RETURNING *
                    """,
                    *args,
                )
            else:
                row = await pool.fetchrow(
                    """
                    INSERT INTO invoices (
                        id, invoice_number,
                        contact_id, customer_name, customer_email, customer_phone, customer_address,
                        line_items, subtotal, tax_rate, tax_amount, discount_amount, total_amount,
                        issue_date, due_date, status, source, source_ref, appointment_id,
                        business_context_id, notes, metadata, invoice_for, contact_name,
                        created_at, updated_at, billing_period
                    )
                    VALUES (
                        $1,
                        (SELECT $19 || '-' || to_char($25::date, 'YYYY-Mon') || '-' || lpad(nextval('invoice_number_seq')::text, 4, '0')),
                        $2, $3, $4, $5, $6,
                        $7::jsonb, $8, $9, $10, $11, $12,
                        $13, $14, 'draft', $15, $16, $17,
                        $18, $20, $21::jsonb, $22, $23,
                        $24, $24, to_char($25::date, 'YYYY-MM')
                    )
                    RETURNING *
                    """,
                    *args,
                    billing_period,
                )
            if row:
                logger.info("Created invoice %s number=%s total=%.2f", invoice_id, row["invoice_number"], total)
                return self._row_to_dict(row)
            raise DatabaseOperationError("create invoice", Exception("No row returned"))
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create invoice: %s", e)
            raise DatabaseOperationError("create invoice", e)

    async def get_by_id(self, invoice_id: UUID) -> Optional[dict]:
        """Get an invoice by ID."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get invoice by id")

        try:
            row = await pool.fetchrow("SELECT * FROM invoices WHERE id = $1", invoice_id)
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get invoice by id", e)

    async def get_by_source_ref(self, source_ref: str) -> Optional[dict]:
        """Get an invoice by source_ref (for deduplication)."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get invoice by source_ref")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM invoices WHERE source_ref = $1", source_ref
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get invoice by source_ref", e)

    async def get_by_contact_and_period(
        self, contact_id: UUID, billing_period: str
    ) -> Optional[dict]:
        """Return an existing non-void recurring invoice for one contact/period,
        or a synthetic hit if that contact/period is quarantined (an
        ambiguous historical collision -- see migration 385's Backfill 2/2)
        rather than backfilled.

        Scoped to the two recurring auto-invoice sources (monthly_auto,
        eom_commercial_billing) so ad-hoc mcp_tool invoices and voided
        invoices never block a new recurring invoice. Used by both recurring
        writers as an app-level pre-check ahead of
        idx_invoices_recurring_contact_period_source (migration 385), which
        is the authoritative DB-enforced guarantee for every UNAMBIGUOUS
        period. A quarantined period has no row claiming
        idx_invoices_recurring_contact_period_source's slot (nothing does,
        by design -- see invoices_billing_period_reservations' migration
        comment), so this pre-check is that period's only guard; callers only
        read the returned dict's "source"/"invoice_number" keys, which the
        reservation branch synthesizes with a clearly-labeled placeholder.
        """
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get invoice by contact and period")

        try:
            row = await pool.fetchrow(
                """
                SELECT source, invoice_number FROM invoices
                WHERE contact_id = $1
                  AND billing_period = $2
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
                contact_id,
                billing_period,
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get invoice by contact and period", e)

    async def get_by_number(self, invoice_number: str) -> Optional[dict]:
        """Get an invoice by invoice number (e.g. INV-2026-0001)."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get invoice by number")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM invoices WHERE lower(invoice_number) = lower($1)",
                invoice_number.strip(),
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get invoice by number", e)

    async def get_by_contact_id(self, contact_id: UUID, limit: int = 20) -> list[dict]:
        """Get invoices for a CRM contact."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get invoices by contact")

        try:
            rows = await pool.fetch(
                """
                SELECT * FROM invoices
                WHERE contact_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                contact_id,
                limit,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get invoices by contact", e)

    async def update_status(
        self,
        invoice_id: UUID,
        status: str,
        sent_at: Optional[datetime] = None,
        sent_via: Optional[str] = None,
        paid_at: Optional[datetime] = None,
        voided_at: Optional[datetime] = None,
        void_reason: Optional[str] = None,
    ) -> None:
        """Update invoice status and related timestamps."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update invoice status")

        try:
            await pool.execute(
                """
                UPDATE invoices
                SET status = $2,
                    sent_at = COALESCE($3, sent_at),
                    sent_via = COALESCE($4, sent_via),
                    paid_at = COALESCE($5, paid_at),
                    voided_at = COALESCE($6, voided_at),
                    void_reason = COALESCE($7, void_reason),
                    updated_at = $8
                WHERE id = $1
                """,
                invoice_id,
                status,
                sent_at,
                sent_via,
                paid_at,
                voided_at,
                void_reason,
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update invoice status", e)

    async def update_invoice(
        self,
        invoice_id: UUID,
        line_items: Optional[list[dict]] = None,
        due_date: Optional[date] = None,
        notes: Optional[str] = None,
        tax_rate: Optional[float] = None,
        discount_amount: Optional[float] = None,
        invoice_for: Optional[str] = None,
        contact_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Update a draft invoice. Only draft invoices can be edited."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update invoice")

        # Verify draft status
        current = await self.get_by_id(invoice_id)
        if not current:
            return None
        if current["status"] != "draft":
            raise DatabaseOperationError(
                "update invoice",
                Exception(f"Cannot edit invoice with status '{current['status']}' (must be 'draft')"),
            )

        # Recalculate amounts if line_items or rates change
        items = _line_items_with_amounts(
            line_items if line_items is not None else current["line_items"],
            overwrite=line_items is not None,
        )
        tax_r = tax_rate if tax_rate is not None else float(current["tax_rate"])
        disc = discount_amount if discount_amount is not None else float(current["discount_amount"])
        metadata = current.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if (
            line_items is not None
            and metadata.get("needs_hours")
            and _line_items_are_billable(items)
        ):
            metadata = {**metadata, "needs_hours": False}

        subtotal = _line_items_subtotal(
            items,
            prefer_recorded_amounts=(
                line_items is None
                and _has_trusted_commercial_billing_line_amounts(current)
            ),
        )
        tax_amt = subtotal * Decimal(str(tax_r))
        total = subtotal + tax_amt - Decimal(str(disc))

        try:
            row = await pool.fetchrow(
                """
                UPDATE invoices
                SET line_items = $2::jsonb,
                    due_date = COALESCE($3, due_date),
                    notes = COALESCE($4, notes),
                    tax_rate = $5,
                    tax_amount = $6,
                    discount_amount = $7,
                    subtotal = $8,
                    total_amount = $9,
                    invoice_for = COALESCE($10, invoice_for),
                    contact_name = COALESCE($11, contact_name),
                    metadata = $12::jsonb,
                    updated_at = $13
                WHERE id = $1
                RETURNING *
                """,
                invoice_id,
                json.dumps(items),
                due_date,
                notes,
                float(tax_r),
                float(tax_amt),
                float(disc),
                float(subtotal),
                float(total),
                invoice_for,
                contact_name,
                json.dumps(metadata),
                datetime.now(timezone.utc),
            )
            return self._row_to_dict(row) if row else None
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            raise DatabaseOperationError("update invoice", e)

    async def get_outstanding(
        self,
        business_context_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get outstanding invoices (sent, partial, overdue)."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get outstanding invoices")

        try:
            if business_context_id:
                rows = await pool.fetch(
                    """
                    SELECT * FROM invoices
                    WHERE status IN ('sent', 'partial', 'overdue')
                      AND business_context_id = $1
                    ORDER BY due_date ASC
                    LIMIT $2
                    """,
                    business_context_id,
                    limit,
                )
            else:
                rows = await pool.fetch(
                    """
                    SELECT * FROM invoices
                    WHERE status IN ('sent', 'partial', 'overdue')
                    ORDER BY due_date ASC
                    LIMIT $1
                    """,
                    limit,
                )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get outstanding invoices", e)

    async def get_overdue(self, as_of_date: Optional[date] = None) -> list[dict]:
        """Get invoices past due date that are still unpaid."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get overdue invoices")

        check_date = as_of_date or date.today()
        try:
            rows = await pool.fetch(
                """
                SELECT * FROM invoices
                WHERE due_date < $1
                  AND status IN ('sent', 'partial', 'overdue')
                  AND amount_due > 0
                ORDER BY due_date ASC
                LIMIT 500
                """,
                check_date,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get overdue invoices", e)

    async def mark_overdue(self, invoice_id: UUID) -> None:
        """Mark an invoice as overdue."""
        await self.update_status(invoice_id, "overdue")

    async def update_reminder(self, invoice_id: UUID) -> None:
        """Increment reminder count and update last_reminder_at."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update invoice reminder")

        try:
            await pool.execute(
                """
                UPDATE invoices
                SET reminder_count = reminder_count + 1,
                    last_reminder_at = $2,
                    updated_at = $2
                WHERE id = $1
                """,
                invoice_id,
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update invoice reminder", e)

    async def search(
        self,
        keyword: Optional[str] = None,
        contact_id: Optional[UUID] = None,
        status: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search invoices with multiple filters."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("search invoices")

        conditions = []
        params: list = []
        idx = 1

        if keyword:
            conditions.append(
                f"(invoice_number ILIKE ${idx} OR customer_name ILIKE ${idx} OR notes ILIKE ${idx})"
            )
            params.append(f"%{keyword}%")
            idx += 1
        if contact_id:
            conditions.append(f"contact_id = ${idx}")
            params.append(contact_id)
            idx += 1
        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if from_date:
            conditions.append(f"issue_date >= ${idx}")
            params.append(from_date)
            idx += 1
        if to_date:
            conditions.append(f"issue_date <= ${idx}")
            params.append(to_date)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        try:
            rows = await pool.fetch(
                f"""
                SELECT * FROM invoices
                {where}
                ORDER BY created_at DESC
                LIMIT ${idx}
                """,
                *params,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("search invoices", e)

    # -- Payments --------------------------------------------------

    async def record_payment(
        self,
        invoice_id: UUID,
        amount: float,
        payment_method: str = "other",
        payment_date: Optional[date] = None,
        reference: Optional[str] = None,
        notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        metadata: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Compatibility wrapper for a one-invoice receipt."""
        outcome = await self.record_payment_with_outcome(
            invoice_id=invoice_id,
            amount=amount,
            payment_method=payment_method,
            payment_date=payment_date,
            reference=reference,
            notes=notes,
            recorded_by=recorded_by,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )
        return outcome.payment

    async def record_payment_with_outcome(
        self,
        invoice_id: UUID,
        amount: float,
        payment_method: str = "other",
        payment_date: Optional[date] = None,
        reference: Optional[str] = None,
        notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        metadata: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
    ) -> InvoicePaymentRecordOutcome:
        """Record one invoice payment and retain first-write/replay state internally."""
        invoice = await self.get_by_id(invoice_id)
        if not invoice:
            raise DatabaseOperationError(
                "record payment", Exception(f"Invoice not found: {invoice_id}")
            )
        try:
            # The long-lived singular MCP tool historically accepted calls without
            # an idempotency key. Preserve that surface and its append semantics:
            # two identical-looking receipts may be two real payments. Callers that
            # need retry deduplication must supply and reuse an operation key.
            compatibility_key = idempotency_key or f"invoice-repository-{uuid4()}"

            receipt_outcome = (
                await self._get_receivables_service().create_payment_with_outcome(
                    contact_id=invoice.get("contact_id"),
                    payer_name=invoice["customer_name"],
                    total_amount=Decimal(str(amount)),
                    payment_method=payment_method,
                    received_date=payment_date or date.today(),
                    allocations=[{"invoice_id": invoice_id, "amount": amount}],
                    reference=reference,
                    notes=notes,
                    recorded_by=recorded_by,
                    metadata=metadata,
                    source="invoice_repository",
                    idempotency_key=compatibility_key,
                    enforce_api_methods=False,
                )
            )
            receipt = receipt_outcome.payment
            allocation = next(
                item
                for item in receipt["allocations"]
                if str(item["invoice_id"]) == str(invoice_id)
            )
            return InvoicePaymentRecordOutcome(
                payment={
                    **allocation,
                    "payment_id": receipt["id"],
                    "payment_date": receipt["received_date"],
                    "payment_method": receipt["payment_method"],
                    "reference": receipt.get("reference"),
                    "notes": receipt.get("notes"),
                    "recorded_by": receipt.get("recorded_by"),
                    "idempotency_key": receipt.get("idempotency_key"),
                    "status": receipt["status"],
                },
                replayed=receipt_outcome.replayed,
            )
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to record payment: %s", e)
            raise DatabaseOperationError("record payment", e)

    async def get_payments(self, invoice_id: UUID) -> list[dict]:
        """Get all payments for an invoice."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get payments")

        try:
            rows = await pool.fetch(
                """
                SELECT ip.*, COALESCE(cp.status, 'legacy') AS payment_status,
                       COALESCE(cp.total_amount, ip.amount) AS receipt_total_amount
                FROM invoice_payments ip
                LEFT JOIN customer_payments cp ON cp.id = ip.payment_id
                WHERE ip.invoice_id = $1
                  AND ip.reversed_at IS NULL
                ORDER BY payment_date DESC
                LIMIT 200
                """,
                invoice_id,
            )
            return [self._payment_row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get payments", e)

    async def get_customer_balance(self, contact_id: UUID) -> dict:
        """Get aggregate balance for a customer."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get customer balance")

        try:
            row = await pool.fetchrow(
                """
                SELECT
                    COALESCE(SUM(total_amount), 0) AS total_invoiced,
                    COALESCE(SUM(amount_paid), 0) AS total_paid,
                    COALESCE(SUM(total_amount - amount_paid), 0) AS outstanding_balance
                FROM invoices
                WHERE contact_id = $1
                  AND status NOT IN ('void', 'draft')
                """,
                contact_id,
            )
            return {
                "contact_id": str(contact_id),
                "total_invoiced": float(row["total_invoiced"]) if row else 0,
                "total_paid": float(row["total_paid"]) if row else 0,
                "outstanding_balance": float(row["outstanding_balance"]) if row else 0,
            }
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get customer balance", e)

    async def get_payment_behavior(self, contact_id: UUID) -> dict:
        """Analyze payment behavior: on-time rate, avg days to pay."""
        pool = self._get_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get payment behavior")

        try:
            row = await pool.fetchrow(
                """
                WITH invoice_stats AS (
                    SELECT
                        i.id,
                        i.due_date,
                        i.status,
                        i.total_amount,
                        i.amount_paid,
                        MIN(p.payment_date) FILTER (
                            WHERE p.payment_id IS NULL
                               OR cp.status IN ('legacy', 'received', 'deposited', 'cleared')
                        ) AS first_payment_date
                    FROM invoices i
                    LEFT JOIN invoice_payments p
                      ON p.invoice_id = i.id AND p.reversed_at IS NULL
                    LEFT JOIN customer_payments cp ON cp.id = p.payment_id
                    WHERE i.contact_id = $1
                      AND i.status NOT IN ('void', 'draft')
                    GROUP BY i.id
                )
                SELECT
                    COUNT(*) AS total_invoices,
                    COUNT(*) FILTER (WHERE status = 'paid' AND first_payment_date <= due_date) AS paid_on_time,
                    COUNT(*) FILTER (WHERE status = 'paid' AND first_payment_date > due_date) AS paid_late,
                    COALESCE(AVG(first_payment_date - due_date) FILTER (WHERE first_payment_date IS NOT NULL), 0) AS avg_days_to_pay,
                    COALESCE(SUM(total_amount - amount_paid) FILTER (WHERE status IN ('sent', 'partial', 'overdue')), 0) AS outstanding_balance
                FROM invoice_stats
                """,
                contact_id,
            )
            return {
                "contact_id": str(contact_id),
                "total_invoices": row["total_invoices"] if row else 0,
                "paid_on_time": row["paid_on_time"] if row else 0,
                "paid_late": row["paid_late"] if row else 0,
                "avg_days_to_pay": float(row["avg_days_to_pay"]) if row else 0,
                "outstanding_balance": float(row["outstanding_balance"]) if row else 0,
            }
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get payment behavior", e)

    # -- Helpers ---------------------------------------------------

    async def _recalculate_amount_paid(self, invoice_id: UUID) -> None:
        """Recalculate via the lifecycle-aware receivables ledger."""
        await self._get_receivables_service().recalculate_invoice(invoice_id)

    def _row_to_dict(self, row) -> dict:
        """Convert an invoice database row to a dict."""
        result = dict(row)
        # JSONB fields
        for key in ("line_items",):
            val = result.get(key)
            if val is None:
                result[key] = []
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = []
        for key in ("metadata",):
            val = result.get(key)
            if val is None:
                result[key] = {}
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = {}
        # Convert Decimal to float for JSON serialization
        for key in ("subtotal", "tax_rate", "tax_amount", "discount_amount",
                     "total_amount", "amount_paid", "amount_due"):
            val = result.get(key)
            if isinstance(val, Decimal):
                result[key] = float(val)
        return result

    def _payment_row_to_dict(self, row) -> dict:
        """Convert a payment database row to a dict."""
        result = dict(row)
        for key in ("metadata",):
            val = result.get(key)
            if val is None:
                result[key] = {}
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = {}
        for key in ("amount",):
            val = result.get(key)
            if isinstance(val, Decimal):
                result[key] = float(val)
        return result


_invoice_repo: Optional[InvoiceRepository] = None


def get_invoice_repo() -> InvoiceRepository:
    """Get the global invoice repository."""
    global _invoice_repo
    if _invoice_repo is None:
        _invoice_repo = InvoiceRepository()
    return _invoice_repo
