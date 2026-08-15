"""Durable no-send Gmail drafts for approved EOM commercial invoices.

This service is the delivery boundary after an explicit commercial-billing
approval and its immutable PDF artifact already exist.  It creates or recovers
one Gmail *draft* only; it does not send mail, change an invoice status, write
CRM/service evidence, or invoke Square.  Gmail has no caller-provided draft
idempotency key, so an intent plus stable RFC Message-ID is committed before
the external call.  An uncertain create is recovered by lookup, never retried
as another create.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional, Protocol
from uuid import UUID, uuid4

import asyncpg

from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError
from ..templates.email.invoice import BUSINESS_NAME, render_invoice_html, render_invoice_text
from ..tools.gmail import (
    GmailDraftCreateError,
    get_gmail_transport,
)
from .commercial_billing_invoice_pdfs import (
    CommercialBillingInvoicePDFConflictError,
    CommercialBillingInvoicePDFError,
    CommercialBillingInvoicePDFNotFoundError,
    CommercialBillingInvoicePDFService,
    CommercialBillingInvoicePDFUnavailableError,
    CommercialBillingInvoicePDFValidationError,
    ReadyCommercialBillingInvoicePDFArtifact,
)
from .eom_crm_mutations import normalize_contact_email


logger = logging.getLogger("atlas.services.commercial_billing_invoice_gmail_drafts")

_DRAFT_SOURCE = "eom_admin"
_DELIVERY_METHOD = "gmail_pdf"
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_ACTOR_LENGTH = 128
_MAX_GMAIL_ID_LENGTH = 256
_DRAFT_STATES = frozenset(
    {"creating", "retryable", "recovery_required", "draft_created"}
)
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


class CommercialBillingGmailDraftError(Exception):
    code = "commercial_billing_gmail_draft_error"


class CommercialBillingGmailDraftValidationError(CommercialBillingGmailDraftError):
    code = "invalid_commercial_billing_gmail_draft"


class CommercialBillingGmailDraftNotFoundError(CommercialBillingGmailDraftError):
    code = "commercial_billing_gmail_draft_not_found"


class CommercialBillingGmailDraftConflictError(CommercialBillingGmailDraftError):
    code = "commercial_billing_gmail_draft_conflict"


class CommercialBillingGmailDraftUnavailableError(CommercialBillingGmailDraftError):
    code = "commercial_billing_gmail_draft_unavailable"


class CommercialBillingGmailDraftRecoveryRequiredError(CommercialBillingGmailDraftError):
    code = "commercial_billing_gmail_draft_recovery_required"


class _GmailDraftGateway(Protocol):
    async def create_draft(
        self,
        *,
        to: list[str],
        subject: str,
        body: str,
        attachments: list[dict[str, Any]],
        html: str,
        headers: Mapping[str, str],
    ) -> dict[str, Any]: ...

    async def find_draft_by_rfc_message_id(
        self,
        rfc_message_id: str,
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class _DraftContext:
    artifact: ReadyCommercialBillingInvoicePDFArtifact
    body: str
    html: str
    recipient_email: str
    rfc_message_id: str
    subject: str


@dataclass(frozen=True)
class _PreparedDraft:
    context: _DraftContext | None
    record: Mapping[str, Any]
    replayed: bool
    action: str


def _request_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingGmailDraftValidationError(f"{field} is required")
    text = value.strip()
    if not text or len(text) > limit:
        raise CommercialBillingGmailDraftValidationError(
            f"{field} must contain 1 to {limit} characters"
        )
    return text


def _stored_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingGmailDraftConflictError(
            f"Commercial billing Gmail draft {field} is invalid"
        )
    text = value.strip()
    if not text or len(text) > limit:
        raise CommercialBillingGmailDraftConflictError(
            f"Commercial billing Gmail draft {field} is invalid"
        )
    return text


def _uuid(value: Any, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CommercialBillingGmailDraftUnavailableError(
            f"Commercial billing Gmail draft {field} is invalid"
        ) from exc


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommercialBillingGmailDraftValidationError(
            "Commercial billing Gmail request is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _rfc_message_id(approval_id: UUID) -> str:
    """One stable mailbox-search identity per immutable approval record."""

    return (
        f"<atlas-eom-commercial-billing-{approval_id}"
        "@effinghamofficemaids.com>"
    )


def _subject(invoice: Mapping[str, Any]) -> str:
    invoice_number = _stored_text(
        invoice.get("invoice_number"), "invoice number", limit=32
    )
    try:
        total = Decimal(str(invoice.get("total_amount")))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CommercialBillingGmailDraftConflictError(
            "Approved invoice total is invalid"
        ) from exc
    cents = total * Decimal(100)
    if not total.is_finite() or cents != cents.to_integral_value() or total <= 0:
        raise CommercialBillingGmailDraftConflictError(
            "Approved invoice total is invalid"
        )
    subject = f"Invoice {invoice_number} - {BUSINESS_NAME} - ${total:,.2f}"
    if len(subject) > 512 or "\r" in subject or "\n" in subject:
        raise CommercialBillingGmailDraftConflictError(
            "Approved invoice email subject is invalid"
        )
    return subject


def _gateway_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CommercialBillingGmailDraftRecoveryRequiredError(
            "Gmail draft response is invalid; recover it by its Message-ID"
        )
    message = value.get("message")
    if not isinstance(message, Mapping):
        raise CommercialBillingGmailDraftRecoveryRequiredError(
            "Gmail draft response is invalid; recover it by its Message-ID"
        )
    fields = {
        "gmail_draft_id": value.get("id"),
        "gmail_message_id": message.get("id"),
        "gmail_thread_id": message.get("threadId"),
    }
    result: dict[str, str] = {}
    for field, raw in fields.items():
        if not isinstance(raw, str):
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft response is invalid; recover it by its Message-ID"
            )
        normalized = raw.strip()
        if (
            not normalized
            or len(normalized) > _MAX_GMAIL_ID_LENGTH
            or "\r" in normalized
            or "\n" in normalized
            or "\x00" in normalized
        ):
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft response is invalid; recover it by its Message-ID"
            )
        result[field] = normalized
    return result


class CommercialBillingInvoiceGmailDraftService:
    """Create, recover, or reuse one no-send Gmail draft per approval/PDF."""

    def __init__(
        self,
        *,
        pool: Optional[DatabasePool] = None,
        pdf_service: CommercialBillingInvoicePDFService | None = None,
        gateway_loader: Callable[[], _GmailDraftGateway] = get_gmail_transport,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._configured_pool = pool
        self._configured_pdf_service = pdf_service
        self._gateway_loader = gateway_loader
        self._now = now

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing database unavailable"
            )
        return pool

    @property
    def pdf_service(self) -> CommercialBillingInvoicePDFService:
        return self._configured_pdf_service or CommercialBillingInvoicePDFService(
            pool=self.pool
        )

    async def create_or_reuse(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        """Create one Gmail draft, or recover/reuse its durable identity.

        The external call is intentionally outside PostgreSQL transactions.  A
        precommitted ``creating`` record is enough to make every later caller
        reconcile the same RFC Message-ID before another create could occur.
        """

        if not isinstance(approval_id, UUID):
            raise CommercialBillingGmailDraftValidationError("Approval id is invalid")
        key = _request_text(
            idempotency_key,
            "Idempotency key",
            limit=_MAX_IDEMPOTENCY_KEY_LENGTH,
        )
        requested_by = _request_text(actor, "authenticated actor", limit=_MAX_ACTOR_LENGTH)
        request_fingerprint = _fingerprint({"approvalId": str(approval_id)})
        try:
            prepared = await self._prepare(
                approval_id=approval_id,
                idempotency_key=key,
                request_fingerprint=request_fingerprint,
                actor=requested_by,
            )
            if prepared.action == "return":
                return {
                    "draft": self._view(prepared.record),
                    "replayed": prepared.replayed,
                    "reused": True,
                }
            if prepared.action == "lookup":
                return await self._recover(
                    prepared=prepared,
                    actor=requested_by,
                )
            if prepared.action != "create" or prepared.context is None:
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft action is invalid"
                )
            return await self._create(
                prepared=prepared,
                actor=requested_by,
            )
        except CommercialBillingGmailDraftError:
            raise
        except CommercialBillingInvoicePDFValidationError as exc:
            raise CommercialBillingGmailDraftValidationError(str(exc)) from exc
        except CommercialBillingInvoicePDFNotFoundError as exc:
            raise CommercialBillingGmailDraftNotFoundError(str(exc)) from exc
        except CommercialBillingInvoicePDFConflictError as exc:
            raise CommercialBillingGmailDraftConflictError(str(exc)) from exc
        except CommercialBillingInvoicePDFUnavailableError as exc:
            raise CommercialBillingGmailDraftUnavailableError(str(exc)) from exc
        except CommercialBillingInvoicePDFError as exc:
            raise CommercialBillingGmailDraftUnavailableError(str(exc)) from exc
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def _prepare(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
    ) -> _PreparedDraft:
        async with self.pool.transaction() as conn:
            await self._lock(conn, f"operation:{idempotency_key}")
            operation = await self._find_operation(conn, idempotency_key)
            if operation is not None:
                self._assert_operation(operation, approval_id, request_fingerprint)
                state = self._state(operation)
                if state == "draft_created":
                    return _PreparedDraft(
                        context=None,
                        record=operation,
                        replayed=True,
                        action="return",
                    )
                await self._lock(conn, f"approval:{approval_id}")
                if state == "retryable":
                    context = await self._current_context(conn, approval_id)
                    self._assert_context(operation, context)
                    claimed = await self._claim_retryable(
                        conn, _uuid(operation["id"], "record id"), actor
                    )
                    return _PreparedDraft(
                        context=context,
                        record=claimed,
                        replayed=True,
                        action="create",
                    )
                return _PreparedDraft(
                    context=None,
                    record=operation,
                    replayed=True,
                    action="lookup",
                )

            await self._lock(conn, f"approval:{approval_id}")
            context = await self._current_context(conn, approval_id)
            record = await self._find_record_for_approval(conn, approval_id)
            if record is None:
                record = await self._insert_record(conn, context, actor)
                await self._insert_operation(
                    conn,
                    record_id=_uuid(record["id"], "record id"),
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    actor=actor,
                )
                return _PreparedDraft(
                    context=context,
                    record=record,
                    replayed=False,
                    action="create",
                )

            self._assert_context(record, context)
            await self._insert_operation(
                conn,
                record_id=_uuid(record["id"], "record id"),
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                actor=actor,
            )
            state = self._state(record)
            if state == "draft_created":
                return _PreparedDraft(
                    context=None,
                    record=record,
                    replayed=False,
                    action="return",
                )
            if state == "retryable":
                claimed = await self._claim_retryable(
                    conn, _uuid(record["id"], "record id"), actor
                )
                return _PreparedDraft(
                    context=context,
                    record=claimed,
                    replayed=False,
                    action="create",
                )
            return _PreparedDraft(
                context=None,
                record=record,
                replayed=False,
                action="lookup",
            )

    async def _create(
        self,
        *,
        prepared: _PreparedDraft,
        actor: str,
    ) -> dict[str, Any]:
        if prepared.context is None:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft create context is invalid"
            )
        record_id = _uuid(prepared.record["id"], "record id")
        try:
            gateway = self._gateway_loader()
        except Exception as exc:
            await self._best_effort_transition(record_id, "retryable", actor)
            raise CommercialBillingGmailDraftUnavailableError(
                "Gmail draft transport is unavailable; retry is safe"
            ) from exc
        try:
            result = await gateway.create_draft(
                to=[prepared.context.recipient_email],
                subject=prepared.context.subject,
                body=prepared.context.body,
                html=prepared.context.html,
                attachments=[
                    {
                        "filename": prepared.context.artifact.filename,
                        "content": base64.b64encode(
                            prepared.context.artifact.pdf_bytes
                        ).decode("ascii"),
                    }
                ],
                headers={
                    "Message-ID": prepared.context.rfc_message_id,
                    "X-Atlas-Commercial-Billing-Approval": str(
                        prepared.context.artifact.approval_id
                    ),
                    "X-Atlas-Commercial-Billing-Invoice": str(
                        prepared.context.artifact.invoice_id
                    ),
                },
            )
        except GmailDraftCreateError as exc:
            if exc.definitely_not_created:
                await self._best_effort_transition(record_id, "retryable", actor)
                raise CommercialBillingGmailDraftUnavailableError(
                    "Gmail draft creation failed before acceptance; retry is safe"
                ) from exc
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft creation outcome is uncertain; recover it by its Message-ID"
            ) from exc
        except Exception as exc:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft creation outcome is uncertain; recover it by its Message-ID"
            ) from exc

        try:
            identity = _gateway_identity(result)
        except CommercialBillingGmailDraftRecoveryRequiredError as exc:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise exc
        try:
            confirmed = await self._confirm_draft(record_id, identity, actor)
        except CommercialBillingGmailDraftError:
            raise
        except Exception as exc:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft may exist but confirmation failed; recover it by its Message-ID"
            ) from exc
        return {
            "draft": self._view(confirmed),
            "replayed": prepared.replayed,
            "reused": False,
        }

    async def _recover(
        self,
        *,
        prepared: _PreparedDraft,
        actor: str,
    ) -> dict[str, Any]:
        record_id = _uuid(prepared.record["id"], "record id")
        rfc_message_id = _stored_text(
            prepared.record["rfc_message_id"], "RFC Message-ID", limit=320
        )
        try:
            gateway = self._gateway_loader()
            result = await gateway.find_draft_by_rfc_message_id(rfc_message_id)
        except Exception as exc:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft lookup failed; recover it by its Message-ID"
            ) from exc
        if result is None:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft was not found; it may be missing or require reconciliation"
            )
        try:
            identity = _gateway_identity(result)
        except CommercialBillingGmailDraftRecoveryRequiredError as exc:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise exc
        try:
            confirmed = await self._confirm_draft(record_id, identity, actor)
        except CommercialBillingGmailDraftError:
            raise
        except Exception as exc:
            await self._best_effort_transition(record_id, "recovery_required", actor)
            raise CommercialBillingGmailDraftRecoveryRequiredError(
                "Gmail draft was found but could not be confirmed; retry recovery"
            ) from exc
        return {
            "draft": self._view(confirmed),
            "replayed": prepared.replayed,
            "reused": True,
        }

    async def _current_context(
        self,
        conn: Any,
        approval_id: UUID,
    ) -> _DraftContext:
        artifact = await self.pdf_service.load_ready_artifact_for_delivery(
            conn=conn,
            approval_id=approval_id,
        )
        metadata = artifact.invoice.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("deliveryMethod") != _DELIVERY_METHOD
        ):
            raise CommercialBillingGmailDraftConflictError(
                "Approved invoice is not configured for Gmail PDF delivery"
            )
        recipient_email = normalize_contact_email(
            artifact.invoice.get("customer_email")
        )
        if recipient_email is None:
            raise CommercialBillingGmailDraftConflictError(
                "Approved invoice billing recipient is invalid"
            )
        try:
            body = render_invoice_text(artifact.invoice)
            html = render_invoice_html(artifact.invoice)
        except Exception as exc:
            raise CommercialBillingGmailDraftUnavailableError(
                "Approved invoice email could not be rendered"
            ) from exc
        if (
            not isinstance(body, str)
            or not body.strip()
            or not isinstance(html, str)
            or not html.strip()
        ):
            raise CommercialBillingGmailDraftUnavailableError(
                "Approved invoice email could not be rendered"
            )
        return _DraftContext(
            artifact=artifact,
            body=body,
            html=html,
            recipient_email=recipient_email,
            rfc_message_id=_rfc_message_id(artifact.approval_id),
            subject=_subject(artifact.invoice),
        )

    @staticmethod
    async def _lock(conn: Any, scope: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-invoice-gmail-draft:{scope}",
        )

    @staticmethod
    async def _find_operation(conn: Any, idempotency_key: str) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT operation.request_fingerprint AS operation_request_fingerprint,
                   draft.id, draft.approval_id, draft.artifact_id, draft.state,
                   draft.recipient_email, draft.subject, draft.rfc_message_id,
                   draft.gmail_draft_id, draft.gmail_message_id,
                   draft.gmail_thread_id, draft.created_by, draft.created_at,
                   draft.last_attempt_by, draft.last_attempt_at,
                   draft.draft_created_at, draft.recovery_required_at,
                   approval.invoice_id
            FROM commercial_billing_invoice_gmail_draft_operations AS operation
            JOIN commercial_billing_invoice_gmail_drafts AS draft
              ON draft.id = operation.gmail_draft_record_id
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = draft.approval_id
            WHERE operation.source = $1 AND operation.idempotency_key = $2
            """,
            _DRAFT_SOURCE,
            idempotency_key,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_record_for_approval(conn: Any, approval_id: UUID) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT draft.id, draft.approval_id, draft.artifact_id, draft.state,
                   draft.recipient_email, draft.subject, draft.rfc_message_id,
                   draft.gmail_draft_id, draft.gmail_message_id,
                   draft.gmail_thread_id, draft.created_by, draft.created_at,
                   draft.last_attempt_by, draft.last_attempt_at,
                   draft.draft_created_at, draft.recovery_required_at,
                   approval.invoice_id
            FROM commercial_billing_invoice_gmail_drafts AS draft
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = draft.approval_id
            WHERE draft.approval_id = $1
            """,
            approval_id,
        )
        return dict(row) if row is not None else None

    async def _insert_record(
        self,
        conn: Any,
        context: _DraftContext,
        actor: str,
    ) -> Any:
        now = self._timestamp()
        row = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_invoice_gmail_drafts (
                id, approval_id, artifact_id, state, recipient_email, subject,
                rfc_message_id, created_by, created_at, last_attempt_by,
                last_attempt_at
            )
            VALUES ($1, $2, $3, 'creating', $4, $5, $6, $7, $8, $7, $8)
            RETURNING id, approval_id, artifact_id, state, recipient_email,
                      subject, rfc_message_id, gmail_draft_id, gmail_message_id,
                      gmail_thread_id, created_by, created_at, last_attempt_by,
                      last_attempt_at, draft_created_at, recovery_required_at
            """,
            uuid4(),
            context.artifact.approval_id,
            context.artifact.artifact_id,
            context.recipient_email,
            context.subject,
            context.rfc_message_id,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft intent could not be reconciled"
            )
        return {**dict(row), "invoice_id": context.artifact.invoice_id}

    async def _insert_operation(
        self,
        conn: Any,
        *,
        record_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
    ) -> None:
        now = self._timestamp()
        await conn.execute(
            """
            INSERT INTO commercial_billing_invoice_gmail_draft_operations (
                id, gmail_draft_record_id, source, idempotency_key,
                request_fingerprint, requested_by, requested_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
            """,
            uuid4(),
            record_id,
            _DRAFT_SOURCE,
            idempotency_key,
            request_fingerprint,
            actor,
            now,
        )

    async def _claim_retryable(self, conn: Any, record_id: UUID, actor: str) -> Any:
        now = self._timestamp()
        row = await conn.fetchrow(
            """
            UPDATE commercial_billing_invoice_gmail_drafts
               SET state = 'creating', last_attempt_by = $2, last_attempt_at = $3,
                   recovery_required_at = NULL
             WHERE id = $1 AND state = 'retryable'
            RETURNING id, approval_id, artifact_id, state, recipient_email,
                      subject, rfc_message_id, gmail_draft_id, gmail_message_id,
                      gmail_thread_id, created_by, created_at, last_attempt_by,
                      last_attempt_at, draft_created_at, recovery_required_at
            """,
            record_id,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft retry could not be claimed"
            )
        return dict(row)

    async def _confirm_draft(
        self,
        record_id: UUID,
        identity: Mapping[str, str],
        actor: str,
    ) -> Any:
        now = self._timestamp()
        async with self.pool.transaction() as conn:
            record = await self._find_record_for_id(conn, record_id)
            if record is None:
                raise CommercialBillingGmailDraftUnavailableError(
                    "Commercial billing Gmail draft intent is unavailable"
                )
            approval_id = _uuid(record["approval_id"], "approval id")
            await self._lock(conn, f"approval:{approval_id}")
            record = await self._find_record_for_id(conn, record_id)
            if record is None:
                raise CommercialBillingGmailDraftUnavailableError(
                    "Commercial billing Gmail draft intent is unavailable"
                )
            state = self._state(record)
            if state == "draft_created":
                if all(record[field] == identity[field] for field in identity):
                    return record
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft identity conflicts with a prior result"
                )
            row = await conn.fetchrow(
                """
                UPDATE commercial_billing_invoice_gmail_drafts
                   SET state = 'draft_created', gmail_draft_id = $2,
                       gmail_message_id = $3, gmail_thread_id = $4,
                       last_attempt_by = $5, last_attempt_at = $6,
                       draft_created_at = $6, recovery_required_at = NULL
                 WHERE id = $1 AND state IN ('creating', 'retryable', 'recovery_required')
                RETURNING id, approval_id, artifact_id, state, recipient_email,
                          subject, rfc_message_id, gmail_draft_id, gmail_message_id,
                          gmail_thread_id, created_by, created_at, last_attempt_by,
                          last_attempt_at, draft_created_at, recovery_required_at
                """,
                record_id,
                identity["gmail_draft_id"],
                identity["gmail_message_id"],
                identity["gmail_thread_id"],
                actor,
                now,
            )
            if row is None:
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft could not be confirmed"
                )
            return {**dict(row), "invoice_id": record["invoice_id"]}

    @staticmethod
    async def _find_record_for_id(conn: Any, record_id: UUID) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT draft.id, draft.approval_id, draft.artifact_id, draft.state,
                   draft.recipient_email, draft.subject, draft.rfc_message_id,
                   draft.gmail_draft_id, draft.gmail_message_id,
                   draft.gmail_thread_id, draft.created_by, draft.created_at,
                   draft.last_attempt_by, draft.last_attempt_at,
                   draft.draft_created_at, draft.recovery_required_at,
                   approval.invoice_id
            FROM commercial_billing_invoice_gmail_drafts AS draft
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = draft.approval_id
            WHERE draft.id = $1
            """,
            record_id,
        )
        return dict(row) if row is not None else None

    async def _best_effort_transition(
        self,
        record_id: UUID,
        state: str,
        actor: str,
    ) -> None:
        """Record recovery evidence without masking the original transport error."""

        try:
            now = self._timestamp()
            async with self.pool.transaction() as conn:
                record = await self._find_record_for_id(conn, record_id)
                if record is None or self._state(record) == "draft_created":
                    return
                approval_id = _uuid(record["approval_id"], "approval id")
                await self._lock(conn, f"approval:{approval_id}")
                if state == "retryable":
                    await conn.execute(
                        """
                        UPDATE commercial_billing_invoice_gmail_drafts
                           SET state = 'retryable', last_attempt_by = $2,
                               last_attempt_at = $3, recovery_required_at = NULL
                         WHERE id = $1 AND state = 'creating'
                        """,
                        record_id,
                        actor,
                        now,
                    )
                elif state == "recovery_required":
                    await conn.execute(
                        """
                        UPDATE commercial_billing_invoice_gmail_drafts
                           SET state = 'recovery_required', last_attempt_by = $2,
                               last_attempt_at = $3, recovery_required_at = $3
                         WHERE id = $1
                           AND state IN ('creating', 'retryable', 'recovery_required')
                        """,
                        record_id,
                        actor,
                        now,
                    )
                else:
                    raise CommercialBillingGmailDraftConflictError(
                        "Commercial billing Gmail draft recovery state is invalid"
                    )
        except Exception:
            logger.warning(
                "Commercial billing Gmail draft %s could not record %s recovery state",
                record_id,
                state,
                exc_info=True,
            )

    @staticmethod
    def _assert_operation(
        operation: Mapping[str, Any],
        approval_id: UUID,
        request_fingerprint: str,
    ) -> None:
        if operation.get("operation_request_fingerprint") != request_fingerprint:
            raise CommercialBillingGmailDraftConflictError(
                "Idempotency key was already used with a different commercial billing approval"
            )
        if _uuid(operation.get("approval_id"), "approval id") != approval_id:
            raise CommercialBillingGmailDraftConflictError(
                "Idempotency key was already used with a different commercial billing approval"
            )

    @staticmethod
    def _assert_context(record: Mapping[str, Any], context: _DraftContext) -> None:
        expected = {
            "approval_id": context.artifact.approval_id,
            "artifact_id": context.artifact.artifact_id,
            "recipient_email": context.recipient_email,
            "subject": context.subject,
            "rfc_message_id": context.rfc_message_id,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft intent no longer matches its approval PDF"
                )

    @staticmethod
    def _state(record: Mapping[str, Any]) -> str:
        state = _stored_text(record.get("state"), "state", limit=32)
        if state not in _DRAFT_STATES:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft state is invalid"
            )
        return state

    def _timestamp(self) -> datetime:
        value = self._now()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft clock is invalid"
            )
        return value

    @staticmethod
    def _view(record: Mapping[str, Any]) -> dict[str, Any]:
        state = CommercialBillingInvoiceGmailDraftService._state(record)
        created_at = record.get("created_at")
        last_attempt_at = record.get("last_attempt_at")
        if not isinstance(created_at, datetime) or not isinstance(last_attempt_at, datetime):
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft timestamps are invalid"
            )
        result = {
            "approvalId": str(_uuid(record.get("approval_id"), "approval id")),
            "artifactId": str(_uuid(record.get("artifact_id"), "artifact id")),
            "createdAt": created_at.isoformat(),
            "createdBy": _stored_text(record.get("created_by"), "creator", limit=128),
            "gmailDraftId": record.get("gmail_draft_id"),
            "gmailMessageId": record.get("gmail_message_id"),
            "gmailThreadId": record.get("gmail_thread_id"),
            "id": str(_uuid(record.get("id"), "record id")),
            "invoiceId": str(_uuid(record.get("invoice_id"), "invoice id")),
            "lastAttemptAt": last_attempt_at.isoformat(),
            "lastAttemptBy": _stored_text(
                record.get("last_attempt_by"), "last attempt actor", limit=128
            ),
            "recipientEmail": _stored_text(
                record.get("recipient_email"), "recipient", limit=256
            ),
            "rfcMessageId": _stored_text(
                record.get("rfc_message_id"), "RFC Message-ID", limit=320
            ),
            "state": state,
            "subject": _stored_text(record.get("subject"), "subject", limit=512),
        }
        for field, key in (
            ("draft_created_at", "draftCreatedAt"),
            ("recovery_required_at", "recoveryRequiredAt"),
        ):
            value = record.get(field)
            if value is not None:
                if not isinstance(value, datetime):
                    raise CommercialBillingGmailDraftUnavailableError(
                        "Commercial billing Gmail draft timestamps are invalid"
                    )
                result[key] = value.isoformat()
            else:
                result[key] = None
        return result


def get_commercial_billing_invoice_gmail_draft_service() -> CommercialBillingInvoiceGmailDraftService:
    return CommercialBillingInvoiceGmailDraftService()


__all__ = [
    "CommercialBillingGmailDraftConflictError",
    "CommercialBillingGmailDraftError",
    "CommercialBillingGmailDraftNotFoundError",
    "CommercialBillingGmailDraftRecoveryRequiredError",
    "CommercialBillingGmailDraftUnavailableError",
    "CommercialBillingGmailDraftValidationError",
    "CommercialBillingInvoiceGmailDraftService",
    "get_commercial_billing_invoice_gmail_draft_service",
]
