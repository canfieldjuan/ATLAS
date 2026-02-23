"""
Customer service agreement repository.

Provides CRUD operations for recurring service agreements that link
CRM contacts to calendar-based auto-invoicing.
"""

import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.customer_service")


class CustomerServiceRepository:
    """Repository for customer service agreements."""

    async def create(
        self,
        contact_id: UUID,
        service_name: str,
        rate: float,
        calendar_keyword: str,
        service_description: Optional[str] = None,
        rate_label: str = "Per Visit",
        tax_rate: float = 0.0,
        calendar_id: Optional[str] = None,
        auto_invoice: bool = True,
        start_date: Optional[date] = None,
        business_context_id: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Create a new customer service agreement."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("create customer service")

        service_id = uuid4()
        now = datetime.now(timezone.utc)
        sdate = start_date or date.today()

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO customer_services (
                    id, contact_id, service_name, service_description,
                    rate, rate_label, tax_rate,
                    calendar_keyword, calendar_id,
                    auto_invoice, start_date,
                    business_context_id, notes, metadata,
                    created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15, $15)
                RETURNING *
                """,
                service_id,
                contact_id,
                service_name,
                service_description,
                float(rate),
                rate_label,
                float(tax_rate),
                calendar_keyword,
                calendar_id,
                auto_invoice,
                sdate,
                business_context_id,
                notes,
                json.dumps(metadata or {}),
                now,
            )
            if row:
                logger.info(
                    "Created service %s for contact %s: %s @ $%.2f",
                    service_id, contact_id, service_name, rate,
                )
                return self._row_to_dict(row)
            raise DatabaseOperationError("create customer service", Exception("No row returned"))
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create customer service: %s", e)
            raise DatabaseOperationError("create customer service", e)

    async def get_by_id(self, service_id: UUID) -> Optional[dict]:
        """Get a service agreement by ID."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get customer service by id")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM customer_services WHERE id = $1", service_id
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get customer service by id", e)

    async def get_by_contact(self, contact_id: UUID) -> list[dict]:
        """Get all service agreements for a contact."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get services by contact")

        try:
            rows = await pool.fetch(
                """
                SELECT * FROM customer_services
                WHERE contact_id = $1
                ORDER BY created_at DESC
                """,
                contact_id,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get services by contact", e)

    async def list_active(self, auto_invoice_only: bool = False) -> list[dict]:
        """List active service agreements."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("list active services")

        try:
            if auto_invoice_only:
                rows = await pool.fetch(
                    """
                    SELECT * FROM customer_services
                    WHERE status = 'active' AND auto_invoice = TRUE
                    ORDER BY created_at
                    """
                )
            else:
                rows = await pool.fetch(
                    """
                    SELECT * FROM customer_services
                    WHERE status = 'active'
                    ORDER BY created_at
                    """
                )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("list active services", e)

    async def update(self, service_id: UUID, **fields) -> Optional[dict]:
        """Update a service agreement. Only active/paused services can be edited."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update customer service")

        current = await self.get_by_id(service_id)
        if not current:
            return None
        if current["status"] not in ("active", "paused"):
            raise DatabaseOperationError(
                "update customer service",
                Exception(f"Cannot edit service with status '{current['status']}'"),
            )

        allowed = {
            "service_name", "service_description", "rate", "rate_label",
            "tax_rate", "calendar_keyword", "calendar_id", "auto_invoice",
            "notes", "business_context_id", "start_date", "end_date",
        }
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return current

        set_clauses = []
        params = []
        idx = 1
        for key, val in updates.items():
            set_clauses.append(f"{key} = ${idx}")
            params.append(float(val) if key in ("rate", "tax_rate") else val)
            idx += 1

        set_clauses.append(f"updated_at = ${idx}")
        params.append(datetime.now(timezone.utc))
        idx += 1

        params.append(service_id)

        try:
            row = await pool.fetchrow(
                f"""
                UPDATE customer_services
                SET {', '.join(set_clauses)}
                WHERE id = ${idx}
                RETURNING *
                """,
                *params,
            )
            return self._row_to_dict(row) if row else None
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            raise DatabaseOperationError("update customer service", e)

    async def update_status(self, service_id: UUID, status: str) -> None:
        """Update service status (active, paused, cancelled)."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update service status")

        try:
            await pool.execute(
                """
                UPDATE customer_services
                SET status = $2, updated_at = $3
                WHERE id = $1
                """,
                service_id,
                status,
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update service status", e)

    async def mark_invoiced(
        self, service_id: UUID, invoiced_date: date, next_date: Optional[date] = None
    ) -> None:
        """Record that a service was invoiced for a period."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark service invoiced")

        try:
            await pool.execute(
                """
                UPDATE customer_services
                SET last_invoiced_at = $2,
                    next_invoice_date = $3,
                    updated_at = $4
                WHERE id = $1
                """,
                service_id,
                invoiced_date,
                next_date,
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark service invoiced", e)

    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        result = dict(row)
        # JSONB
        for key in ("metadata",):
            val = result.get(key)
            if val is None:
                result[key] = {}
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = {}
        # Decimal to float
        for key in ("rate", "tax_rate"):
            val = result.get(key)
            if isinstance(val, Decimal):
                result[key] = float(val)
        return result


_customer_service_repo: Optional[CustomerServiceRepository] = None


def get_customer_service_repo() -> CustomerServiceRepository:
    """Get the global customer service repository."""
    global _customer_service_repo
    if _customer_service_repo is None:
        _customer_service_repo = CustomerServiceRepository()
    return _customer_service_repo
