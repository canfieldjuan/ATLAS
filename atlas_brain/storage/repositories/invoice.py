"""
Invoice repository for billing and payment tracking.

Provides CRUD operations for invoices and payments stored in PostgreSQL.
"""

import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.invoice")


class InvoiceRepository:
    """Repository for invoice and payment storage and retrieval."""

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
    ) -> dict:
        """Create a new invoice with auto-generated invoice number."""
        pool = get_db_pool()
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
                    (SELECT $19 || '-' || to_char(CURRENT_DATE, 'YYYY') || '-' || lpad(nextval('invoice_number_seq')::text, 4, '0')),
                    $2, $3, $4, $5, $6,
                    $7::jsonb, $8, $9, $10, $11, $12,
                    $13, $14, 'draft', $15, $16, $17,
                    $18, $20, $21::jsonb, $22, $23,
                    $24, $24
                )
                RETURNING *
                """,
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
        pool = get_db_pool()
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
        pool = get_db_pool()
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

    async def get_by_number(self, invoice_number: str) -> Optional[dict]:
        """Get an invoice by invoice number (e.g. INV-2026-0001)."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get invoice by number")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM invoices WHERE invoice_number = $1",
                invoice_number.upper(),
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get invoice by number", e)

    async def get_by_contact_id(self, contact_id: UUID, limit: int = 20) -> list[dict]:
        """Get invoices for a CRM contact."""
        pool = get_db_pool()
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
        pool = get_db_pool()
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
        pool = get_db_pool()
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
        items = line_items if line_items is not None else current["line_items"]
        tax_r = tax_rate if tax_rate is not None else float(current["tax_rate"])
        disc = discount_amount if discount_amount is not None else float(current["discount_amount"])

        subtotal = sum(
            Decimal(str(item.get("quantity", 1))) * Decimal(str(item.get("unit_price", 0)))
            for item in items
        )
        for item in items:
            if "amount" not in item:
                item["amount"] = float(
                    Decimal(str(item.get("quantity", 1))) * Decimal(str(item.get("unit_price", 0)))
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
                    updated_at = $12
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
        pool = get_db_pool()
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
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get overdue invoices")

        check_date = as_of_date or date.today()
        try:
            rows = await pool.fetch(
                """
                SELECT * FROM invoices
                WHERE due_date < $1
                  AND status IN ('sent', 'partial')
                ORDER BY due_date ASC
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
        pool = get_db_pool()
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
        pool = get_db_pool()
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
    ) -> dict:
        """Record a payment and auto-update invoice status."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("record payment")

        payment_id = uuid4()
        pay_date = payment_date or date.today()
        now = datetime.now(timezone.utc)

        try:
            # Insert payment
            pay_row = await pool.fetchrow(
                """
                INSERT INTO invoice_payments (
                    id, invoice_id, amount, payment_date, payment_method,
                    reference, notes, recorded_by, created_at, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                RETURNING *
                """,
                payment_id,
                invoice_id,
                amount,
                pay_date,
                payment_method,
                reference,
                notes,
                recorded_by,
                now,
                json.dumps(metadata or {}),
            )

            # Recalculate amount_paid from all payments
            await self._recalculate_amount_paid(invoice_id)

            # Auto-derive status
            inv = await self.get_by_id(invoice_id)
            if inv:
                if float(inv["amount_due"]) <= 0:
                    await self.update_status(invoice_id, "paid", paid_at=now)
                elif float(inv["amount_paid"]) > 0 and inv["status"] in ("sent", "overdue"):
                    await self.update_status(invoice_id, "partial")

            logger.info(
                "Recorded payment %s on invoice %s: $%.2f via %s",
                payment_id, invoice_id, amount, payment_method,
            )
            return self._payment_row_to_dict(pay_row) if pay_row else {}
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to record payment: %s", e)
            raise DatabaseOperationError("record payment", e)

    async def get_payments(self, invoice_id: UUID) -> list[dict]:
        """Get all payments for an invoice."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get payments")

        try:
            rows = await pool.fetch(
                """
                SELECT * FROM invoice_payments
                WHERE invoice_id = $1
                ORDER BY payment_date DESC
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
        pool = get_db_pool()
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
        pool = get_db_pool()
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
                        MIN(p.payment_date) AS first_payment_date
                    FROM invoices i
                    LEFT JOIN invoice_payments p ON p.invoice_id = i.id
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
        """Recalculate amount_paid from sum of payments."""
        pool = get_db_pool()
        await pool.execute(
            """
            UPDATE invoices
            SET amount_paid = COALESCE(
                (SELECT SUM(amount) FROM invoice_payments WHERE invoice_id = $1), 0
            ),
            updated_at = $2
            WHERE id = $1
            """,
            invoice_id,
            datetime.now(timezone.utc),
        )

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
