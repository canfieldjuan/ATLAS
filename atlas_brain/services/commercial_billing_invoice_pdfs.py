"""Durable PDF artifacts for explicitly approved EOM commercial invoices.

This service owns only one narrow transition: a linked commercial-billing
approval's still-draft invoice becomes one immutable, database-retained PDF
artifact.  It does not create an invoice, draft or send Gmail, mutate financial
status, write a filesystem path, invoke CRM, or mark a service invoiced.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional
from uuid import UUID, uuid4

import asyncpg

from .invoice_pdf import render_invoice_pdf
from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError


_ARTIFACT_KIND = "invoice_pdf"
_ARTIFACT_SOURCE = "eom_admin"
_CONTENT_TYPE = "application/pdf"
_INVOICE_SOURCE = "eom_commercial_billing"
_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
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


class CommercialBillingInvoicePDFError(Exception):
    code = "commercial_billing_invoice_pdf_error"


class CommercialBillingInvoicePDFValidationError(CommercialBillingInvoicePDFError):
    code = "invalid_commercial_billing_invoice_pdf"


class CommercialBillingInvoicePDFNotFoundError(CommercialBillingInvoicePDFError):
    code = "commercial_billing_approval_not_found"


class CommercialBillingInvoicePDFConflictError(CommercialBillingInvoicePDFError):
    code = "commercial_billing_invoice_pdf_conflict"


class CommercialBillingInvoicePDFUnavailableError(CommercialBillingInvoicePDFError):
    code = "commercial_billing_invoice_pdf_unavailable"


class CommercialBillingInvoicePDFRenderError(CommercialBillingInvoicePDFError):
    code = "commercial_billing_invoice_pdf_render_failed"


@dataclass(frozen=True)
class _ApprovedInvoice:
    approval_id: UUID
    invoice_id: UUID
    invoice: dict[str, Any]
    render_fingerprint: str


@dataclass(frozen=True)
class ReadyCommercialBillingInvoicePDFArtifact:
    """One verified stored artifact for a server-side delivery transition.

    The raw bytes intentionally remain an internal service result.  Public
    receivables routes continue to expose metadata only; a later server-side
    delivery service may attach the verified bytes without exposing them to a
    browser or trusting a caller-supplied file.
    """

    approval_id: UUID
    artifact_id: UUID
    invoice_id: UUID
    content_type: str
    filename: str
    pdf_bytes: bytes
    pdf_sha256: str
    render_fingerprint: str
    invoice: dict[str, Any]


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _required_text(value: Any, field: str, *, limit: int = 256) -> str:
    text = _text(value)
    if text is None or not text.strip() or len(text.strip()) > limit:
        raise CommercialBillingInvoicePDFConflictError(
            f"Approved invoice {field} is invalid"
        )
    return text.strip()


def _optional_text(value: Any, field: str, *, limit: int = 256) -> str | None:
    if value is None:
        return None
    return _required_text(value, field, limit=limit)


def _uuid(value: Any, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise CommercialBillingInvoicePDFUnavailableError(
            f"Commercial billing {field} is invalid"
        ) from exc


def _json_value(value: Any, field: str, expected_type: type) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CommercialBillingInvoicePDFConflictError(
                f"Approved invoice {field} is invalid"
            ) from exc
    if not isinstance(value, expected_type):
        raise CommercialBillingInvoicePDFConflictError(
            f"Approved invoice {field} is invalid"
        )
    return value


def _exact_money(value: Any, field: str, *, positive: bool = False) -> Decimal:
    try:
        decimal = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise CommercialBillingInvoicePDFConflictError(
            f"Approved invoice {field} is invalid"
        ) from exc
    cents = decimal * Decimal(100)
    if (
        not decimal.is_finite()
        or cents != cents.to_integral_value()
        or (positive and decimal <= 0)
    ):
        raise CommercialBillingInvoicePDFConflictError(
            f"Approved invoice {field} is invalid"
        )
    return decimal


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Unsupported commercial billing PDF value: {type(value).__name__}")


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            default=_json_default,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommercialBillingInvoicePDFConflictError(
            "Approved invoice render evidence is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_key(value: str, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingInvoicePDFValidationError(f"{field} is required")
    key = value.strip()
    if not key or len(key) > limit:
        raise CommercialBillingInvoicePDFValidationError(
            f"{field} must contain 1 to {limit} characters"
        )
    return key


def _render_snapshot(invoice: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the complete invoice field set read by ``render_invoice_pdf``.

    Keeping this in one place means every later reuse compares the same
    customer, line-item, money, date, note, metadata, and status values that
    determine the generated document.
    """

    return {
        "amount_due": invoice["amount_due"],
        "contact_name": invoice["contact_name"],
        "customer_address": invoice["customer_address"],
        "customer_email": invoice["customer_email"],
        "customer_name": invoice["customer_name"],
        "customer_phone": invoice["customer_phone"],
        "discount_amount": invoice["discount_amount"],
        "due_date": invoice["due_date"],
        "invoice_for": invoice["invoice_for"],
        "invoice_number": invoice["invoice_number"],
        "issue_date": invoice["issue_date"],
        "line_items": invoice["line_items"],
        "metadata": invoice["metadata"],
        "notes": invoice["notes"],
        "status": invoice["status"],
        "subtotal": invoice["subtotal"],
        "tax_amount": invoice["tax_amount"],
        "total_amount": invoice["total_amount"],
    }


def _filename(invoice_number: str) -> str:
    filename = f"{invoice_number}.pdf"
    if len(filename) > 128:
        raise CommercialBillingInvoicePDFConflictError(
            "Approved invoice filename is invalid"
        )
    return filename


class CommercialBillingInvoicePDFService:
    """Create or reuse one immutable PDF artifact for an approved draft invoice."""

    def __init__(
        self,
        *,
        pool: Optional[DatabasePool] = None,
        renderer: Callable[[dict[str, Any]], bytes] = render_invoice_pdf,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._configured_pool = pool
        self._renderer = renderer
        self._now = now

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise CommercialBillingInvoicePDFUnavailableError(
                "Commercial billing database unavailable"
            )
        return pool

    async def generate_or_reuse(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        if not isinstance(approval_id, UUID):
            raise CommercialBillingInvoicePDFValidationError("Approval id is invalid")
        key = _validate_key(
            idempotency_key,
            field="Idempotency key",
            limit=_MAX_IDEMPOTENCY_KEY_LENGTH,
        )
        requested_by = _required_text(actor, "authenticated actor", limit=128)
        request_fingerprint = _fingerprint({"approvalId": str(approval_id)})
        try:
            async with self.pool.transaction() as conn:
                await self._lock(conn, f"operation:{key}")
                operation = await self._find_operation(conn, key)
                if operation is not None:
                    self._assert_operation(operation, request_fingerprint)
                    return {
                        "artifact": self.view(operation),
                        "replayed": True,
                        "reused": True,
                    }

                await self._lock(conn, f"approval:{approval_id}")
                approved = await self._approved_invoice(conn, approval_id)
                artifact = await self._find_artifact(conn, approval_id)
                if artifact is not None:
                    if artifact["render_fingerprint"] != approved.render_fingerprint:
                        raise CommercialBillingInvoicePDFConflictError(
                            "Approved invoice changed after PDF generation; resolve it before reuse"
                        )
                    await self._insert_operation(
                        conn,
                        artifact_id=_uuid(artifact["artifact_id"], "PDF artifact id"),
                        idempotency_key=key,
                        request_fingerprint=request_fingerprint,
                        actor=requested_by,
                    )
                    return {
                        "artifact": self.view(artifact),
                        "replayed": False,
                        "reused": True,
                    }

                pdf_bytes = self._render(approved.invoice)
                artifact = await self._insert_artifact(
                    conn,
                    approved=approved,
                    pdf_bytes=pdf_bytes,
                    actor=requested_by,
                )
                await self._insert_operation(
                    conn,
                    artifact_id=_uuid(artifact["artifact_id"], "PDF artifact id"),
                    idempotency_key=key,
                    request_fingerprint=request_fingerprint,
                    actor=requested_by,
                )
                return {
                    "artifact": self.view(artifact),
                    "replayed": False,
                    "reused": False,
                }
        except CommercialBillingInvoicePDFError:
            raise
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise CommercialBillingInvoicePDFConflictError(
                "Commercial billing invoice PDF could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingInvoicePDFUnavailableError(
                "Commercial billing database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingInvoicePDFUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def load_ready_artifact_for_delivery(
        self,
        *,
        conn: Any,
        approval_id: UUID,
    ) -> ReadyCommercialBillingInvoicePDFArtifact:
        """Read one unchanged ready artifact for a server-side delivery step.

        The caller owns the surrounding transaction and approval lock.  Keeping
        the read inside that transaction makes the approval/invoice evidence,
        immutable artifact fingerprint, and eventual external-delivery intent
        one preflight decision rather than a read-then-write race.  This method
        never renders, inserts, updates, or exposes bytes through a route.
        """

        if not isinstance(approval_id, UUID):
            raise CommercialBillingInvoicePDFValidationError("Approval id is invalid")
        approved = await self._approved_invoice(conn, approval_id)
        artifact = await self._find_artifact_with_bytes(conn, approval_id)
        if artifact is None:
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice PDF has not been generated"
            )
        if artifact["render_fingerprint"] != approved.render_fingerprint:
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice changed after PDF generation; resolve it before delivery"
            )
        artifact_kind = _required_text(artifact["artifact_kind"], "PDF artifact kind", limit=32)
        state = _required_text(artifact["state"], "PDF artifact state", limit=32)
        content_type = _required_text(
            artifact["content_type"], "PDF artifact content type", limit=128
        )
        filename = _required_text(artifact["filename"], "PDF artifact filename", limit=128)
        sha256 = _required_text(artifact["pdf_sha256"], "PDF artifact hash", limit=64)
        if (
            artifact_kind != _ARTIFACT_KIND
            or state != "ready"
            or content_type != _CONTENT_TYPE
            or _FINGERPRINT.fullmatch(sha256) is None
        ):
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice PDF artifact is invalid"
            )
        raw_bytes = artifact["pdf_bytes"]
        if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice PDF artifact is invalid"
            )
        pdf_bytes = bytes(raw_bytes)
        size = artifact["byte_size"]
        if (
            not isinstance(size, int)
            or size != len(pdf_bytes)
            or len(pdf_bytes) < 8
            or len(pdf_bytes) > _MAX_ARTIFACT_BYTES
            or not pdf_bytes.startswith(b"%PDF-")
            or b"%%EOF" not in pdf_bytes[-1024:]
            or hashlib.sha256(pdf_bytes).hexdigest() != sha256
        ):
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice PDF artifact is invalid"
            )
        return ReadyCommercialBillingInvoicePDFArtifact(
            approval_id=approved.approval_id,
            artifact_id=_uuid(artifact["artifact_id"], "PDF artifact id"),
            invoice_id=approved.invoice_id,
            content_type=content_type,
            filename=filename,
            pdf_bytes=pdf_bytes,
            pdf_sha256=sha256,
            render_fingerprint=approved.render_fingerprint,
            invoice=approved.invoice,
        )

    def _render(self, invoice: dict[str, Any]) -> bytes:
        try:
            value = self._renderer(invoice)
        except Exception as exc:
            raise CommercialBillingInvoicePDFRenderError(
                "Approved invoice PDF could not be rendered; retry is safe"
            ) from exc
        if not isinstance(value, (bytes, bytearray)):
            raise CommercialBillingInvoicePDFRenderError(
                "Approved invoice PDF could not be rendered; retry is safe"
            )
        pdf_bytes = bytes(value)
        if (
            len(pdf_bytes) < 8
            or len(pdf_bytes) > _MAX_ARTIFACT_BYTES
            or not pdf_bytes.startswith(b"%PDF-")
            or b"%%EOF" not in pdf_bytes[-1024:]
        ):
            raise CommercialBillingInvoicePDFRenderError(
                "Approved invoice PDF could not be rendered; retry is safe"
            )
        return pdf_bytes

    @staticmethod
    async def _lock(conn: Any, scope: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-invoice-pdf:{scope}",
        )

    async def _approved_invoice(self, conn: Any, approval_id: UUID) -> _ApprovedInvoice:
        row = await conn.fetchrow(
            """
            SELECT a.id AS approval_id, a.billing_run_id, a.candidate_key,
                   a.source_fingerprint, a.state AS approval_state, a.invoice_id,
                   i.id AS invoice_id, i.invoice_number, i.customer_name,
                   i.customer_email, i.customer_phone, i.customer_address,
                   i.line_items, i.subtotal, i.tax_amount, i.discount_amount,
                   i.total_amount, i.amount_due, i.issue_date, i.due_date,
                   i.status AS invoice_status, i.source AS invoice_source,
                   i.notes, i.metadata, i.invoice_for, i.contact_name
            FROM commercial_billing_candidate_approvals AS a
            JOIN invoices AS i ON i.id = a.invoice_id
            WHERE a.id = $1
            """,
            approval_id,
        )
        if row is None:
            raise CommercialBillingInvoicePDFNotFoundError(
                "Commercial billing approval not found"
            )
        return self._approved_invoice_from_row(row)

    @classmethod
    def render_fingerprint_from_invoice_row(cls, row: Mapping[str, Any]) -> str:
        """Derive the PDF-service fingerprint for a current invoice read row.

        Read-only recovery projections use this shared parser so a stale
        artifact is judged by exactly the same invoice fields and canonical
        serialization as PDF generation and delivery reuse.  The caller may
        inspect a non-draft lifecycle row, so only the writer keeps the
        still-draft requirement.
        """

        return cls._approved_invoice_from_row(
            row,
            require_draft=False,
        ).render_fingerprint

    @staticmethod
    def _approved_invoice_from_row(
        row: Mapping[str, Any], *, require_draft: bool = True
    ) -> _ApprovedInvoice:
        approval_state = _required_text(row["approval_state"], "approval state", limit=32)
        if approval_state != "invoice_created":
            raise CommercialBillingInvoicePDFConflictError(
                "Commercial billing approval is not ready for PDF generation"
            )
        source_fingerprint = _required_text(
            row["source_fingerprint"], "source fingerprint", limit=64
        )
        if _FINGERPRINT.fullmatch(source_fingerprint) is None:
            raise CommercialBillingInvoicePDFConflictError(
                "Commercial billing approval evidence is invalid"
            )
        if _required_text(row["invoice_source"], "invoice source", limit=32) != _INVOICE_SOURCE:
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice is not an EOM commercial billing invoice"
            )
        status = _required_text(row["invoice_status"], "invoice status", limit=32)
        if require_draft and status != "draft":
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice is no longer a draft"
            )

        metadata = dict(_json_value(row["metadata"], "metadata", Mapping))
        candidate_key = _required_text(row["candidate_key"], "candidate key", limit=512)
        billing_run_id = _uuid(row["billing_run_id"], "billing run id")
        if (
            metadata.get("candidateKey") != candidate_key
            or metadata.get("commercialBillingRunId") != str(billing_run_id)
            or metadata.get("sourceFingerprint") != source_fingerprint
        ):
            raise CommercialBillingInvoicePDFConflictError(
                "Approved invoice evidence no longer matches its approval"
            )

        invoice = {
            "amount_due": _exact_money(row["amount_due"], "amount due"),
            "contact_name": _optional_text(row["contact_name"], "contact name"),
            "customer_address": _optional_text(row["customer_address"], "customer address"),
            "customer_email": _optional_text(row["customer_email"], "customer email"),
            "customer_name": _required_text(row["customer_name"], "customer name"),
            "customer_phone": _optional_text(row["customer_phone"], "customer phone", limit=32),
            "discount_amount": _exact_money(row["discount_amount"], "discount amount"),
            "due_date": row["due_date"],
            "invoice_for": _optional_text(row["invoice_for"], "invoice for"),
            "invoice_number": _required_text(row["invoice_number"], "invoice number", limit=32),
            "issue_date": row["issue_date"],
            "line_items": _json_value(row["line_items"], "line items", list),
            "metadata": metadata,
            "notes": _optional_text(row["notes"], "notes", limit=10_000),
            "status": status,
            "subtotal": _exact_money(row["subtotal"], "subtotal"),
            "tax_amount": _exact_money(row["tax_amount"], "tax amount"),
            "total_amount": _exact_money(row["total_amount"], "total amount", positive=True),
        }
        return _ApprovedInvoice(
            approval_id=_uuid(row["approval_id"], "approval id"),
            invoice_id=_uuid(row["invoice_id"], "invoice id"),
            invoice=invoice,
            render_fingerprint=_fingerprint(_render_snapshot(invoice)),
        )

    @staticmethod
    async def _find_operation(conn: Any, idempotency_key: str) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT o.request_fingerprint AS operation_request_fingerprint,
                   artifact.id AS artifact_id, artifact.approval_id,
                   approval.invoice_id, artifact.artifact_kind, artifact.state,
                   artifact.content_type, artifact.filename, artifact.byte_size,
                   artifact.pdf_sha256, artifact.render_fingerprint,
                   artifact.generated_by, artifact.generated_at
            FROM commercial_billing_invoice_pdf_operations AS o
            JOIN commercial_billing_invoice_pdf_artifacts AS artifact ON artifact.id = o.artifact_id
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = artifact.approval_id
            WHERE o.source = $1 AND o.idempotency_key = $2
            """,
            _ARTIFACT_SOURCE,
            idempotency_key,
        )
        return row

    @staticmethod
    async def _find_artifact(conn: Any, approval_id: UUID) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT artifact.id AS artifact_id, artifact.approval_id,
                   approval.invoice_id, artifact.artifact_kind, artifact.state,
                   artifact.content_type, artifact.filename, artifact.byte_size,
                   artifact.pdf_sha256, artifact.render_fingerprint,
                   artifact.generated_by, artifact.generated_at
            FROM commercial_billing_invoice_pdf_artifacts AS artifact
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = artifact.approval_id
            WHERE artifact.approval_id = $1
            """,
            approval_id,
        )

    @staticmethod
    async def _find_artifact_with_bytes(conn: Any, approval_id: UUID) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT artifact.id AS artifact_id, artifact.approval_id,
                   approval.invoice_id, artifact.artifact_kind, artifact.state,
                   artifact.content_type, artifact.filename, artifact.pdf_bytes,
                   artifact.byte_size, artifact.pdf_sha256,
                   artifact.render_fingerprint, artifact.generated_by,
                   artifact.generated_at
            FROM commercial_billing_invoice_pdf_artifacts AS artifact
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = artifact.approval_id
            WHERE artifact.approval_id = $1
            """,
            approval_id,
        )

    @staticmethod
    def _assert_operation(row: Any, request_fingerprint: str) -> None:
        if row["operation_request_fingerprint"] != request_fingerprint:
            raise CommercialBillingInvoicePDFConflictError(
                "Idempotency key was already used with a different commercial billing approval"
            )

    async def _insert_artifact(
        self,
        conn: Any,
        *,
        approved: _ApprovedInvoice,
        pdf_bytes: bytes,
        actor: str,
    ) -> Any:
        now = self._now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise CommercialBillingInvoicePDFUnavailableError(
                "Commercial billing PDF clock is invalid"
            )
        row = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_invoice_pdf_artifacts (
                id, approval_id, artifact_kind, state, content_type,
                filename, pdf_bytes, byte_size, pdf_sha256, render_fingerprint,
                generated_by, generated_at, created_at
            )
            VALUES (
                $1, $2, $3, 'ready', $4, $5, $6, $7, $8, $9, $10, $11, $11
            )
            RETURNING id AS artifact_id, approval_id, artifact_kind,
                      state, content_type, filename, byte_size, pdf_sha256,
                      render_fingerprint, generated_by, generated_at
            """,
            uuid4(),
            approved.approval_id,
            _ARTIFACT_KIND,
            _CONTENT_TYPE,
            _filename(_required_text(approved.invoice["invoice_number"], "invoice number", limit=32)),
            pdf_bytes,
            len(pdf_bytes),
            hashlib.sha256(pdf_bytes).hexdigest(),
            approved.render_fingerprint,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingInvoicePDFUnavailableError(
                "Commercial billing PDF artifact could not be reconciled"
            )
        return {**dict(row), "invoice_id": approved.invoice_id}

    async def _insert_operation(
        self,
        conn: Any,
        *,
        artifact_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
    ) -> None:
        now = self._now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise CommercialBillingInvoicePDFUnavailableError(
                "Commercial billing PDF clock is invalid"
            )
        await conn.execute(
            """
            INSERT INTO commercial_billing_invoice_pdf_operations (
                id, artifact_id, source, idempotency_key,
                request_fingerprint, requested_by, requested_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
            """,
            uuid4(),
            artifact_id,
            _ARTIFACT_SOURCE,
            idempotency_key,
            request_fingerprint,
            actor,
            now,
        )

    @staticmethod
    def view(row: Any) -> dict[str, Any]:
        return {
            "approvalId": str(row["approval_id"]),
            "contentType": row["content_type"],
            "filename": row["filename"],
            "generatedAt": row["generated_at"].isoformat(),
            "generatedBy": row["generated_by"],
            "id": str(row["artifact_id"]),
            "invoiceId": str(row["invoice_id"]),
            "kind": row["artifact_kind"],
            "renderFingerprint": row["render_fingerprint"],
            "sha256": row["pdf_sha256"],
            "sizeBytes": row["byte_size"],
            "state": row["state"],
        }


def get_commercial_billing_invoice_pdf_service() -> CommercialBillingInvoicePDFService:
    return CommercialBillingInvoicePDFService()


__all__ = [
    "CommercialBillingInvoicePDFConflictError",
    "CommercialBillingInvoicePDFError",
    "CommercialBillingInvoicePDFNotFoundError",
    "CommercialBillingInvoicePDFRenderError",
    "CommercialBillingInvoicePDFService",
    "CommercialBillingInvoicePDFUnavailableError",
    "CommercialBillingInvoicePDFValidationError",
    "ReadyCommercialBillingInvoicePDFArtifact",
    "get_commercial_billing_invoice_pdf_service",
]
