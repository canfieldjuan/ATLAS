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
_REPLACEMENT_DRAFT_SOURCE = "eom_admin_draft_replacement"
_DELIVERY_METHOD = "gmail_pdf"
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_ACTOR_LENGTH = 128
_MAX_GMAIL_ID_LENGTH = 256
_DRAFT_STATES = frozenset(
    {"creating", "retryable", "recovery_required", "draft_created"}
)
_RECONCILIATION_STATES = frozenset(
    {"not_reconciled", "draft_present", "draft_missing", "sent_confirmed"}
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

_RECORD_SELECT = """
    SELECT draft.id, draft.approval_id, draft.artifact_id, draft.state,
           draft.recipient_email, draft.subject, draft.rfc_message_id,
           draft.gmail_draft_id, draft.gmail_message_id,
           draft.gmail_thread_id, draft.created_by, draft.created_at,
           draft.last_attempt_by, draft.last_attempt_at,
           draft.draft_created_at, draft.recovery_required_at,
           draft.draft_generation, draft.last_replaced_by,
           draft.last_replaced_at, draft.reconciliation_state,
           draft.gmail_sent_message_id, draft.gmail_sent_thread_id,
           draft.gmail_sent_at, draft.sent_reconciled_by,
           draft.sent_reconciled_at, draft.last_reconciled_by,
           draft.last_reconciled_at, draft.draft_missing_by,
           draft.draft_missing_at, approval.invoice_id,
           invoice.status AS invoice_status,
           invoice.sent_at AS invoice_sent_at,
           invoice.sent_via AS invoice_sent_via
    FROM commercial_billing_invoice_gmail_drafts AS draft
    JOIN commercial_billing_candidate_approvals AS approval
      ON approval.id = draft.approval_id
    JOIN invoices AS invoice ON invoice.id = approval.invoice_id
"""


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
    replacement: Mapping[str, Any] | None = None


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


def _draft_generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CommercialBillingGmailDraftConflictError(
            "Commercial billing Gmail draft generation is invalid"
        )
    return value


def _rfc_message_id(approval_id: UUID, *, generation: int = 1) -> str:
    """One stable mailbox-search identity per current draft generation."""

    generation = _draft_generation(generation)
    suffix = "" if generation == 1 else f"-g{generation}"

    return (
        f"<atlas-eom-commercial-billing-{approval_id}{suffix}@effinghamofficemaids.com>"
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
                    "draft": self.view(prepared.record),
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

    async def replace_missing(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        """Explicitly replace one reconciliation-proven missing Gmail draft.

        This is deliberately separate from ``create_or_reuse``: only an
        operator action after durable ``draft_missing`` evidence can advance
        the root record to a new current generation.  The predecessor is
        captured in an append-only event before the no-send Gmail create.
        """

        if not isinstance(approval_id, UUID):
            raise CommercialBillingGmailDraftValidationError("Approval id is invalid")
        key = _request_text(
            idempotency_key,
            "Idempotency key",
            limit=_MAX_IDEMPOTENCY_KEY_LENGTH,
        )
        requested_by = _request_text(actor, "actor", limit=_MAX_ACTOR_LENGTH)
        request_fingerprint = _fingerprint(
            {"approvalId": str(approval_id), "operation": "replace_missing"}
        )
        try:
            prepared = await self._prepare_replacement(
                approval_id=approval_id,
                idempotency_key=key,
                request_fingerprint=request_fingerprint,
                actor=requested_by,
            )
            if prepared.action == "return":
                result = {
                    "draft": self.view(prepared.record),
                    "replayed": prepared.replayed,
                    "reused": True,
                }
            elif prepared.action == "lookup":
                result = await self._recover(prepared=prepared, actor=requested_by)
            elif prepared.action == "create" and prepared.context is not None:
                result = await self._create(prepared=prepared, actor=requested_by)
            else:
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft replacement action is invalid"
                )
            replacement = prepared.replacement
            if replacement is None:
                raise CommercialBillingGmailDraftUnavailableError(
                    "Commercial billing Gmail draft replacement evidence is unavailable"
                )
            result["replacement"] = self._replacement_view(replacement)
            return result
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
                "Commercial billing Gmail draft replacement could not be reconciled"
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
                    context = await self._current_context(
                        conn,
                        approval_id,
                        generation=_draft_generation(
                            operation.get("draft_generation", 1)
                        ),
                    )
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
            record = await self._find_record_for_approval(conn, approval_id)
            if record is None:
                context = await self._current_context(conn, approval_id)
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

            context = await self._current_context(
                conn,
                approval_id,
                generation=_draft_generation(record.get("draft_generation", 1)),
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

    async def _prepare_replacement(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
    ) -> _PreparedDraft:
        """Commit one new current-draft intent only after missing-draft proof.

        The replacement operation and its predecessor snapshot commit together
        before Gmail is called.  This keeps the retry key tied to a durable,
        non-sending intent even if the external request times out.
        """

        async with self.pool.transaction() as conn:
            await self._lock(conn, f"replacement-operation:{idempotency_key}")
            operation = await self._find_operation(
                conn,
                idempotency_key,
                source=_REPLACEMENT_DRAFT_SOURCE,
            )
            if operation is not None:
                self._assert_operation(operation, approval_id, request_fingerprint)
                await self._lock(conn, f"approval:{approval_id}")
                operation = await self._find_operation(
                    conn,
                    idempotency_key,
                    source=_REPLACEMENT_DRAFT_SOURCE,
                )
                if operation is None:
                    raise CommercialBillingGmailDraftUnavailableError(
                        "Commercial billing Gmail draft replacement operation is unavailable"
                    )
                replacement = await self._find_replacement_event_for_operation(
                    conn,
                    _uuid(operation["operation_id"], "replacement operation id"),
                )
                if replacement is None:
                    raise CommercialBillingGmailDraftConflictError(
                        "Commercial billing Gmail draft replacement evidence is invalid"
                    )
                replacement_generation = self._replacement_generation_for_current_record(
                    operation,
                    replacement,
                )
                state = self._state(operation)
                if state == "draft_created":
                    return _PreparedDraft(
                        context=None,
                        record=operation,
                        replayed=True,
                        action="return",
                        replacement=replacement,
                    )
                context = await self._current_context(
                    conn,
                    approval_id,
                    generation=replacement_generation,
                )
                self._assert_context(operation, context)
                if state == "retryable":
                    claimed = await self._claim_retryable(
                        conn,
                        _uuid(operation["id"], "record id"),
                        actor,
                    )
                    return _PreparedDraft(
                        context=context,
                        record=claimed,
                        replayed=True,
                        action="create",
                        replacement=replacement,
                    )
                return _PreparedDraft(
                    context=None,
                    record=operation,
                    replayed=True,
                    action="lookup",
                    replacement=replacement,
                )

            await self._lock(conn, f"approval:{approval_id}")
            record = await self._find_record_for_approval(
                conn,
                approval_id,
                for_update=True,
            )
            if record is None:
                raise CommercialBillingGmailDraftNotFoundError(
                    "Commercial billing Gmail draft not found"
                )
            self._assert_replaceable(record)
            if await self._has_pending_sent_reconciliation(
                conn,
                _uuid(record["id"], "record id"),
            ):
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft replacement must wait for pending sent-mail reconciliation"
                )
            prior_generation = _draft_generation(record["draft_generation"])
            context = await self._current_context(
                conn,
                approval_id,
                generation=prior_generation + 1,
            )
            self._assert_replacement_context(record, context)
            operation_id = await self._insert_operation(
                conn,
                record_id=_uuid(record["id"], "record id"),
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                actor=actor,
                source=_REPLACEMENT_DRAFT_SOURCE,
            )
            replaced, replacement = await self._snapshot_and_claim_replacement(
                conn,
                record=record,
                context=context,
                operation_id=operation_id,
                actor=actor,
            )
            return _PreparedDraft(
                context=context,
                record=replaced,
                replayed=False,
                action="create",
                replacement=replacement,
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
            "draft": self.view(confirmed),
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
            "draft": self.view(confirmed),
            "replayed": prepared.replayed,
            "reused": True,
        }

    async def _current_context(
        self,
        conn: Any,
        approval_id: UUID,
        *,
        generation: int = 1,
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
            rfc_message_id=_rfc_message_id(
                artifact.approval_id,
                generation=generation,
            ),
            subject=_subject(artifact.invoice),
        )

    @staticmethod
    async def _lock(conn: Any, scope: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-invoice-gmail-draft:{scope}",
        )

    @staticmethod
    async def _find_operation(
        conn: Any,
        idempotency_key: str,
        *,
        source: str = _DRAFT_SOURCE,
    ) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT operation.id AS operation_id,
                   operation.request_fingerprint AS operation_request_fingerprint,
                   draft.id, draft.approval_id, draft.artifact_id, draft.state,
                   draft.recipient_email, draft.subject, draft.rfc_message_id,
                   draft.gmail_draft_id, draft.gmail_message_id,
                   draft.gmail_thread_id, draft.created_by, draft.created_at,
                   draft.last_attempt_by, draft.last_attempt_at,
                   draft.draft_created_at, draft.recovery_required_at,
                   draft.draft_generation, draft.last_replaced_by,
                   draft.last_replaced_at, draft.reconciliation_state,
                   draft.gmail_sent_message_id, draft.gmail_sent_thread_id,
                   draft.gmail_sent_at, draft.sent_reconciled_by,
                   draft.sent_reconciled_at, draft.last_reconciled_by,
                   draft.last_reconciled_at, draft.draft_missing_by,
                   draft.draft_missing_at, approval.invoice_id,
                   invoice.status AS invoice_status,
                   invoice.sent_at AS invoice_sent_at,
                   invoice.sent_via AS invoice_sent_via
            FROM commercial_billing_invoice_gmail_draft_operations AS operation
            JOIN commercial_billing_invoice_gmail_drafts AS draft
              ON draft.id = operation.gmail_draft_record_id
            JOIN commercial_billing_candidate_approvals AS approval
              ON approval.id = draft.approval_id
            JOIN invoices AS invoice ON invoice.id = approval.invoice_id
            WHERE operation.source = $1 AND operation.idempotency_key = $2
            """,
            source,
            idempotency_key,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_record_for_approval(
        conn: Any,
        approval_id: UUID,
        *,
        for_update: bool = False,
    ) -> Any | None:
        suffix = " FOR UPDATE OF draft" if for_update else ""
        row = await conn.fetchrow(
            _RECORD_SELECT + " WHERE draft.approval_id = $1" + suffix,
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
                      last_attempt_at, draft_created_at, recovery_required_at,
                      draft_generation, last_replaced_by, last_replaced_at,
                      reconciliation_state, gmail_sent_message_id,
                      gmail_sent_thread_id, gmail_sent_at, sent_reconciled_by,
                      sent_reconciled_at, last_reconciled_by, last_reconciled_at,
                      draft_missing_by, draft_missing_at
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
        source: str = _DRAFT_SOURCE,
    ) -> UUID:
        now = self._timestamp()
        operation_id = uuid4()
        row = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_invoice_gmail_draft_operations (
                id, gmail_draft_record_id, source, idempotency_key,
                request_fingerprint, requested_by, requested_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
            RETURNING id
            """,
            operation_id,
            record_id,
            source,
            idempotency_key,
            request_fingerprint,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft operation could not be reconciled"
            )
        returned = _uuid(row["id"], "operation id")
        if returned != operation_id:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft operation identity is invalid"
            )
        return returned

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
                      last_attempt_at, draft_created_at, recovery_required_at,
                      draft_generation, last_replaced_by, last_replaced_at,
                      reconciliation_state, gmail_sent_message_id,
                      gmail_sent_thread_id, gmail_sent_at, sent_reconciled_by,
                      sent_reconciled_at, last_reconciled_by, last_reconciled_at,
                      draft_missing_by, draft_missing_at
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
                          last_attempt_at, draft_created_at, recovery_required_at,
                          draft_generation, last_replaced_by, last_replaced_at,
                          reconciliation_state, gmail_sent_message_id,
                          gmail_sent_thread_id, gmail_sent_at, sent_reconciled_by,
                          sent_reconciled_at, last_reconciled_by,
                          last_reconciled_at, draft_missing_by, draft_missing_at
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
            _RECORD_SELECT + " WHERE draft.id = $1",
            record_id,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_replacement_event_for_operation(
        conn: Any,
        operation_id: UUID,
    ) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT id, gmail_draft_record_id, operation_id, prior_generation,
                   replacement_generation, replaced_by, replaced_at, created_at
            FROM commercial_billing_invoice_gmail_draft_replacement_events
            WHERE operation_id = $1
            """,
            operation_id,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _has_pending_sent_reconciliation(conn: Any, record_id: UUID) -> bool:
        value = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM commercial_billing_gmail_sent_reconciliation_operations
                WHERE gmail_draft_record_id = $1 AND state = 'pending'
            )
            """,
            record_id,
        )
        if not isinstance(value, bool):
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft reconciliation state is unavailable"
            )
        return value

    async def _snapshot_and_claim_replacement(
        self,
        conn: Any,
        *,
        record: Mapping[str, Any],
        context: _DraftContext,
        operation_id: UUID,
        actor: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Preserve the missing generation and atomically reset current intent."""

        record_id = _uuid(record["id"], "record id")
        prior_generation = _draft_generation(record["draft_generation"])
        replacement_generation = prior_generation + 1
        snapshot = await conn.fetchval(
            """
            SELECT to_jsonb(draft)::text
            FROM commercial_billing_invoice_gmail_drafts AS draft
            WHERE draft.id = $1
            """,
            record_id,
        )
        if not isinstance(snapshot, str):
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft replacement snapshot is unavailable"
            )
        try:
            snapshot_value = json.loads(snapshot)
        except json.JSONDecodeError as exc:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement snapshot is invalid"
            ) from exc
        if (
            not isinstance(snapshot_value, Mapping)
            or snapshot_value.get("id") != str(record_id)
            or snapshot_value.get("draft_generation") != prior_generation
        ):
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement snapshot is invalid"
            )
        now = self._timestamp()
        event = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_invoice_gmail_draft_replacement_events (
                id, gmail_draft_record_id, operation_id, prior_generation,
                replacement_generation, prior_snapshot, replaced_by,
                replaced_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $8)
            RETURNING id, gmail_draft_record_id, operation_id, prior_generation,
                      replacement_generation, replaced_by, replaced_at, created_at
            """,
            uuid4(),
            record_id,
            operation_id,
            prior_generation,
            replacement_generation,
            snapshot,
            actor,
            now,
        )
        if event is None:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft replacement evidence is unavailable"
            )
        row = await conn.fetchrow(
            """
            UPDATE commercial_billing_invoice_gmail_drafts
               SET artifact_id = $2, state = 'creating', recipient_email = $3,
                   subject = $4, rfc_message_id = $5, gmail_draft_id = NULL,
                   gmail_message_id = NULL, gmail_thread_id = NULL,
                   last_attempt_by = $6, last_attempt_at = $8,
                   draft_created_at = NULL, recovery_required_at = NULL,
                   draft_generation = $7, last_replaced_by = $6,
                   last_replaced_at = $8, reconciliation_state = 'not_reconciled',
                   gmail_sent_message_id = NULL, gmail_sent_thread_id = NULL,
                   gmail_sent_at = NULL, sent_reconciled_by = NULL,
                   sent_reconciled_at = NULL, last_reconciled_by = NULL,
                   last_reconciled_at = NULL, draft_missing_by = NULL,
                   draft_missing_at = NULL
             WHERE id = $1 AND state = 'draft_created'
               AND reconciliation_state = 'draft_missing'
               AND draft_generation = $9
            RETURNING id, approval_id, artifact_id, state, recipient_email,
                      subject, rfc_message_id, gmail_draft_id, gmail_message_id,
                      gmail_thread_id, created_by, created_at, last_attempt_by,
                      last_attempt_at, draft_created_at, recovery_required_at,
                      draft_generation, last_replaced_by, last_replaced_at,
                      reconciliation_state, gmail_sent_message_id,
                      gmail_sent_thread_id, gmail_sent_at, sent_reconciled_by,
                      sent_reconciled_at, last_reconciled_by, last_reconciled_at,
                      draft_missing_by, draft_missing_at
            """,
            record_id,
            context.artifact.artifact_id,
            context.recipient_email,
            context.subject,
            context.rfc_message_id,
            actor,
            replacement_generation,
            now,
            prior_generation,
        )
        if row is None:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement could not be claimed"
            )
        return (
            {**dict(row), "invoice_id": context.artifact.invoice_id},
            dict(event),
        )

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
    def _assert_replaceable(record: Mapping[str, Any]) -> None:
        if CommercialBillingInvoiceGmailDraftService._state(record) != "draft_created":
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft is not ready for replacement"
            )
        if (
            CommercialBillingInvoiceGmailDraftService._reconciliation_state(record)
            != "draft_missing"
        ):
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement requires confirmed missing-draft evidence"
            )
        invoice_status = _stored_text(
            record.get("invoice_status"),
            "invoice status",
            limit=32,
        )
        if (
            invoice_status != "draft"
            or record.get("invoice_sent_at") is not None
            or record.get("invoice_sent_via") is not None
        ):
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing invoice is no longer eligible for Gmail draft replacement"
            )
        if any(
            record.get(field) is not None
            for field in (
                "gmail_sent_message_id",
                "gmail_sent_thread_id",
                "gmail_sent_at",
                "sent_reconciled_by",
                "sent_reconciled_at",
            )
        ):
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft has conflicting sent-mail evidence"
            )
        _stored_text(record.get("draft_missing_by"), "missing-draft actor", limit=128)
        CommercialBillingInvoiceGmailDraftService._recorded_timestamp(
            record.get("draft_missing_at"),
            "missing-draft timestamp",
        )
        generation = _draft_generation(record.get("draft_generation"))
        approval_id = _uuid(record.get("approval_id"), "approval id")
        if _stored_text(
            record.get("rfc_message_id"),
            "RFC Message-ID",
            limit=320,
        ) != _rfc_message_id(approval_id, generation=generation):
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft identity is invalid for replacement"
            )

    @staticmethod
    def _assert_replacement_context(
        record: Mapping[str, Any],
        context: _DraftContext,
    ) -> None:
        expected = {
            "approval_id": context.artifact.approval_id,
            "artifact_id": context.artifact.artifact_id,
            "recipient_email": context.recipient_email,
            "subject": context.subject,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise CommercialBillingGmailDraftConflictError(
                    "Commercial billing Gmail draft predecessor no longer matches its approval PDF"
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
    def _replacement_generation_for_current_record(
        record: Mapping[str, Any],
        replacement: Mapping[str, Any],
    ) -> int:
        """Reject an old replacement key after a later generation takes over."""

        record_id = _uuid(record.get("id"), "record id")
        if _uuid(
            replacement.get("gmail_draft_record_id"),
            "replacement record id",
        ) != record_id:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement evidence is invalid"
            )
        prior_generation = _draft_generation(replacement.get("prior_generation"))
        replacement_generation = _draft_generation(
            replacement.get("replacement_generation")
        )
        if replacement_generation != prior_generation + 1:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement evidence is invalid"
            )
        if _draft_generation(record.get("draft_generation")) != replacement_generation:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement replay is stale"
            )
        return replacement_generation

    @staticmethod
    def _reconciliation_state(record: Mapping[str, Any]) -> str:
        state = _stored_text(
            record.get("reconciliation_state"),
            "reconciliation state",
            limit=32,
        )
        if state not in _RECONCILIATION_STATES:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft reconciliation state is invalid"
            )
        return state

    @staticmethod
    def _state(record: Mapping[str, Any]) -> str:
        state = _stored_text(record.get("state"), "state", limit=32)
        if state not in _DRAFT_STATES:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft state is invalid"
            )
        return state

    @staticmethod
    def _recorded_timestamp(value: Any, field: str) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise CommercialBillingGmailDraftConflictError(
                f"Commercial billing Gmail draft {field} is invalid"
            )
        return value

    @staticmethod
    def _replacement_view(replacement: Mapping[str, Any]) -> dict[str, Any]:
        prior_generation = _draft_generation(replacement.get("prior_generation"))
        replacement_generation = _draft_generation(
            replacement.get("replacement_generation")
        )
        if replacement_generation != prior_generation + 1:
            raise CommercialBillingGmailDraftConflictError(
                "Commercial billing Gmail draft replacement evidence is invalid"
            )
        replaced_at = CommercialBillingInvoiceGmailDraftService._recorded_timestamp(
            replacement.get("replaced_at"),
            "replacement timestamp",
        )
        return {
            "draftId": str(
                _uuid(replacement.get("gmail_draft_record_id"), "record id")
            ),
            "id": str(_uuid(replacement.get("id"), "replacement event id")),
            "priorGeneration": prior_generation,
            "replacedAt": replaced_at.isoformat(),
            "replacedBy": _stored_text(
                replacement.get("replaced_by"),
                "replacement actor",
                limit=128,
            ),
            "replacementGeneration": replacement_generation,
        }

    def _timestamp(self) -> datetime:
        value = self._now()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise CommercialBillingGmailDraftUnavailableError(
                "Commercial billing Gmail draft clock is invalid"
            )
        return value

    @staticmethod
    def view(record: Mapping[str, Any]) -> dict[str, Any]:
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
