"""Proof-gated Sent-mail reconciliation for EOM commercial Gmail drafts.

The earlier Gmail-draft slice persists only a no-send draft identity.  Gmail
replaces that draft with a different message when an operator sends it, so this
service searches Sent mail by the durable RFC Message-ID and marks the linked
ATLAS invoice sent only after it verifies the mailbox evidence.  A disappeared
draft without that proof is explicitly retained as missing, never inferred sent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Protocol
from uuid import UUID, uuid4

import asyncpg

from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError
from ..tools.gmail import (
    GmailSentMessageLookupError,
    get_gmail_transport,
)


_DELIVERY_METHOD = "gmail_pdf"
_DRAFT_SOURCE = "eom_admin"
_INVOICE_SOURCE = "eom_commercial_billing"
_MAX_ACTOR_LENGTH = 128
_MAX_GMAIL_ID_LENGTH = 256
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_RFC_MESSAGE_ID_LENGTH = 320
_OPERATION_STATES = frozenset({"pending", "completed"})
_OUTCOME_STATES = frozenset({"draft_present", "draft_missing", "sent_confirmed"})
_RECONCILIATION_STATES = frozenset(
    {"not_reconciled", "draft_present", "draft_missing", "sent_confirmed"}
)
_POST_SEND_INVOICE_STATUSES = frozenset(
    {"sent", "partial", "overdue", "paid", "void"}
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
    SELECT draft.id AS draft_id, draft.approval_id, draft.state AS draft_state,
           draft.rfc_message_id, draft.reconciliation_state,
           draft.gmail_draft_id, draft.gmail_message_id, draft.gmail_thread_id,
           draft.gmail_sent_message_id, draft.gmail_sent_thread_id,
           draft.gmail_sent_at, draft.sent_reconciled_by,
           draft.sent_reconciled_at, draft.last_reconciled_by,
           draft.last_reconciled_at, draft.draft_missing_by,
           draft.draft_missing_at,
           approval.state AS approval_state,
           approval.invoice_id AS approval_invoice_id,
           invoice.id AS invoice_id, invoice.status AS invoice_status,
           invoice.sent_at AS invoice_sent_at, invoice.sent_via AS invoice_sent_via,
           invoice.source AS invoice_source, invoice.metadata AS invoice_metadata
    FROM commercial_billing_invoice_gmail_drafts AS draft
    JOIN commercial_billing_candidate_approvals AS approval
      ON approval.id = draft.approval_id
    JOIN invoices AS invoice ON invoice.id = approval.invoice_id
"""

_OPERATION_RECORD_SELECT = """
    SELECT operation.id AS operation_id,
           operation.request_fingerprint AS operation_request_fingerprint,
           operation.state AS operation_state,
           operation.outcome_state AS operation_outcome_state,
           draft.id AS draft_id, draft.approval_id, draft.state AS draft_state,
           draft.rfc_message_id, draft.reconciliation_state,
           draft.gmail_draft_id, draft.gmail_message_id, draft.gmail_thread_id,
           draft.gmail_sent_message_id, draft.gmail_sent_thread_id,
           draft.gmail_sent_at, draft.sent_reconciled_by,
           draft.sent_reconciled_at, draft.last_reconciled_by,
           draft.last_reconciled_at, draft.draft_missing_by,
           draft.draft_missing_at,
           approval.state AS approval_state,
           approval.invoice_id AS approval_invoice_id,
           invoice.id AS invoice_id, invoice.status AS invoice_status,
           invoice.sent_at AS invoice_sent_at, invoice.sent_via AS invoice_sent_via,
           invoice.source AS invoice_source, invoice.metadata AS invoice_metadata
    FROM commercial_billing_gmail_sent_reconciliation_operations AS operation
    JOIN commercial_billing_invoice_gmail_drafts AS draft
      ON draft.id = operation.gmail_draft_record_id
    JOIN commercial_billing_candidate_approvals AS approval
      ON approval.id = draft.approval_id
    JOIN invoices AS invoice ON invoice.id = approval.invoice_id
"""


class CommercialBillingGmailSentReconciliationError(Exception):
    code = "commercial_billing_gmail_sent_reconciliation_error"


class CommercialBillingGmailSentReconciliationValidationError(
    CommercialBillingGmailSentReconciliationError
):
    code = "invalid_commercial_billing_gmail_sent_reconciliation"


class CommercialBillingGmailSentReconciliationNotFoundError(
    CommercialBillingGmailSentReconciliationError
):
    code = "commercial_billing_gmail_draft_not_found"


class CommercialBillingGmailSentReconciliationConflictError(
    CommercialBillingGmailSentReconciliationError
):
    code = "commercial_billing_gmail_sent_reconciliation_conflict"


class CommercialBillingGmailSentReconciliationUnavailableError(
    CommercialBillingGmailSentReconciliationError
):
    code = "commercial_billing_gmail_sent_reconciliation_unavailable"


class _GmailSentReconciliationGateway(Protocol):
    async def find_draft_by_rfc_message_id(
        self, rfc_message_id: str
    ) -> dict[str, Any] | None: ...

    async def find_sent_message_by_rfc_message_id(
        self, rfc_message_id: str
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class _Context:
    approval_id: UUID
    draft_id: UUID
    draft_state: str
    invoice_id: UUID
    invoice_sent_at: datetime | None
    invoice_sent_via: str | None
    invoice_status: str
    reconciliation_state: str
    rfc_message_id: str


@dataclass(frozen=True)
class _PreparedReconciliation:
    context: _Context
    operation: Mapping[str, Any]
    record: Mapping[str, Any]
    replayed: bool
    action: str


@dataclass(frozen=True)
class _SentProof:
    gmail_message_id: str
    gmail_sent_at: datetime
    gmail_thread_id: str


@dataclass(frozen=True)
class _LookupOutcome:
    state: str
    proof: _SentProof | None = None


def _request_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingGmailSentReconciliationValidationError(
            f"{field} is required"
        )
    text = value.strip()
    if (
        not text
        or len(text) > limit
        or "\r" in text
        or "\n" in text
        or "\x00" in text
    ):
        raise CommercialBillingGmailSentReconciliationValidationError(
            f"{field} must contain 1 to {limit} safe characters"
        )
    return text


def _stored_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingGmailSentReconciliationConflictError(
            f"Commercial billing Gmail reconciliation {field} is invalid"
        )
    text = value.strip()
    if (
        not text
        or len(text) > limit
        or "\r" in text
        or "\n" in text
        or "\x00" in text
    ):
        raise CommercialBillingGmailSentReconciliationConflictError(
            f"Commercial billing Gmail reconciliation {field} is invalid"
        )
    return text


def _uuid(value: Any, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CommercialBillingGmailSentReconciliationUnavailableError(
            f"Commercial billing Gmail reconciliation {field} is invalid"
        ) from exc


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommercialBillingGmailSentReconciliationValidationError(
            "Commercial billing Gmail reconciliation request is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CommercialBillingGmailSentReconciliationConflictError(
                f"Commercial billing Gmail reconciliation {field} is invalid"
            ) from exc
    if not isinstance(value, Mapping):
        raise CommercialBillingGmailSentReconciliationConflictError(
            f"Commercial billing Gmail reconciliation {field} is invalid"
        )
    return value


def _state(value: Any) -> str:
    state = _stored_text(value, "state", limit=32)
    if state not in _RECONCILIATION_STATES:
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Commercial billing Gmail reconciliation state is invalid"
        )
    return state


def _operation_state(value: Any) -> str:
    state = _stored_text(value, "operation state", limit=32)
    if state not in _OPERATION_STATES:
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Commercial billing Gmail reconciliation operation state is invalid"
        )
    return state


def _external_id(value: Any, field: str) -> str:
    return _stored_text(value, field, limit=_MAX_GMAIL_ID_LENGTH)


def _rfc_message_id(value: Any) -> str:
    message_id = _stored_text(value, "RFC Message-ID", limit=_MAX_RFC_MESSAGE_ID_LENGTH)
    if (
        len(message_id) < 5
        or not message_id.startswith("<")
        or not message_id.endswith(">")
        or message_id.count("@") != 1
        or any(character.isspace() for character in message_id)
    ):
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Commercial billing Gmail reconciliation RFC Message-ID is invalid"
        )
    return message_id


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CommercialBillingGmailSentReconciliationUnavailableError(
            f"Commercial billing Gmail reconciliation {field} is invalid"
        )
    return value


def _optional_timestamp(value: Any, field: str) -> datetime | None:
    return None if value is None else _timestamp(value, field)


def _sent_timestamp(value: Any) -> datetime:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Gmail sent-message timestamp is invalid"
        )
    try:
        milliseconds = int(value)
        if milliseconds <= 0:
            raise ValueError("must be positive")
        seconds, remainder = divmod(milliseconds, 1000)
        return datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
            milliseconds=remainder
        )
    except (OverflowError, OSError, ValueError) as exc:
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Gmail sent-message timestamp is invalid"
        ) from exc


def _exact_header(headers: Any, name: str, expected: str) -> None:
    if not isinstance(headers, list):
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Gmail sent-message headers are invalid"
        )
    values: list[str] = []
    for header in headers:
        if not isinstance(header, Mapping):
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Gmail sent-message headers are invalid"
            )
        raw_name = header.get("name")
        raw_value = header.get("value")
        if (
            not isinstance(raw_name, str)
            or not isinstance(raw_value, str)
            or "\r" in raw_name
            or "\n" in raw_name
            or "\x00" in raw_name
            or "\r" in raw_value
            or "\n" in raw_value
            or "\x00" in raw_value
        ):
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Gmail sent-message headers are invalid"
            )
        if raw_name.casefold() == name.casefold():
            values.append(raw_value)
    if len(values) != 1 or values != [expected]:
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Gmail sent-message does not match its approved invoice identity"
        )


class CommercialBillingInvoiceGmailSentReconciliationService:
    """Reconcile one manually sent Gmail draft without sending another email."""

    def __init__(
        self,
        *,
        pool: Optional[DatabasePool] = None,
        gateway_loader: Callable[[], _GmailSentReconciliationGateway] = get_gmail_transport,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._configured_pool = pool
        self._gateway_loader = gateway_loader
        self._now = now

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Commercial billing database unavailable"
            )
        return pool

    async def reconcile(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        """Record one proof-gated reconciliation or a non-financial observation."""

        if not isinstance(approval_id, UUID):
            raise CommercialBillingGmailSentReconciliationValidationError(
                "Approval id is invalid"
            )
        key = _request_text(
            idempotency_key, "Idempotency key", limit=_MAX_IDEMPOTENCY_KEY_LENGTH
        )
        requested_by = _request_text(actor, "authenticated actor", limit=_MAX_ACTOR_LENGTH)
        fingerprint = _fingerprint({"approvalId": str(approval_id)})
        try:
            prepared = await self._prepare(
                approval_id=approval_id,
                idempotency_key=key,
                request_fingerprint=fingerprint,
                actor=requested_by,
            )
            if prepared.action == "return":
                return self._result(
                    prepared.record,
                    outcome_state=_stored_text(
                        prepared.operation.get("operation_outcome_state"),
                        "operation outcome",
                        limit=32,
                    ),
                    replayed=prepared.replayed,
                    reused=True,
                )
            if prepared.action != "lookup":
                raise CommercialBillingGmailSentReconciliationConflictError(
                    "Commercial billing Gmail reconciliation action is invalid"
                )
            outcome = await self._lookup(prepared.context)
            return await self._finalize(
                prepared=prepared,
                outcome=outcome,
                actor=requested_by,
            )
        except CommercialBillingGmailSentReconciliationError:
            raise
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing Gmail reconciliation could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Commercial billing database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def _prepare(
        self,
        *,
        approval_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
    ) -> _PreparedReconciliation:
        async with self.pool.transaction() as conn:
            await self._lock_operation(conn, idempotency_key)
            operation = await self._find_operation_by_key(conn, idempotency_key)
            if operation is not None:
                self._assert_operation(operation, approval_id, request_fingerprint)
                context = self._context(operation)
                await self._lock_approval(conn, context.approval_id)
                operation = await self._find_operation_by_key(conn, idempotency_key)
                if operation is None:
                    raise CommercialBillingGmailSentReconciliationUnavailableError(
                        "Commercial billing Gmail reconciliation operation is unavailable"
                    )
                context = self._context(operation)
                if _operation_state(operation["operation_state"]) == "completed":
                    return _PreparedReconciliation(
                        context=context,
                        operation=operation,
                        record=operation,
                        replayed=True,
                        action="return",
                    )
                if context.reconciliation_state == "sent_confirmed":
                    self._assert_confirmed_financial_state(context)
                    operation = await self._complete_operation(
                        conn,
                        operation_id=_uuid(operation["operation_id"], "operation id"),
                        outcome_state="sent_confirmed",
                        now=self._now_timestamp(),
                    )
                    return _PreparedReconciliation(
                        context=context,
                        operation=operation,
                        record=operation,
                        replayed=True,
                        action="return",
                    )
                self._assert_reconcilable(context)
                return _PreparedReconciliation(
                    context=context,
                    operation=operation,
                    record=operation,
                    replayed=True,
                    action="lookup",
                )

            await self._lock_approval(conn, approval_id)
            record = await self._find_record_for_approval(conn, approval_id)
            if record is None:
                raise CommercialBillingGmailSentReconciliationNotFoundError(
                    "Commercial billing Gmail draft not found"
                )
            context = self._context(record)
            operation = await self._insert_operation(
                conn,
                draft_id=context.draft_id,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                actor=actor,
                now=self._now_timestamp(),
            )
            if context.reconciliation_state == "sent_confirmed":
                self._assert_confirmed_financial_state(context)
                operation = await self._complete_operation(
                    conn,
                    operation_id=_uuid(operation["operation_id"], "operation id"),
                    outcome_state="sent_confirmed",
                    now=self._now_timestamp(),
                )
                return _PreparedReconciliation(
                    context=context,
                    operation=operation,
                    record=record,
                    replayed=False,
                    action="return",
                )
            self._assert_reconcilable(context)
            return _PreparedReconciliation(
                context=context,
                operation=operation,
                record=record,
                replayed=False,
                action="lookup",
            )

    async def _lookup(self, context: _Context) -> _LookupOutcome:
        try:
            gateway = self._gateway_loader()
            sent_message = await gateway.find_sent_message_by_rfc_message_id(
                context.rfc_message_id
            )
        except GmailSentMessageLookupError as exc:
            if exc.retryable:
                raise CommercialBillingGmailSentReconciliationUnavailableError(
                    "Gmail Sent-mail lookup is unavailable; retry is safe"
                ) from exc
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Gmail Sent-mail evidence is ambiguous or invalid"
            ) from exc
        except Exception as exc:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Gmail Sent-mail lookup is unavailable; retry is safe"
            ) from exc
        if sent_message is not None:
            return _LookupOutcome(
                state="sent_confirmed",
                proof=self._sent_proof(sent_message, context),
            )

        try:
            draft = await gateway.find_draft_by_rfc_message_id(context.rfc_message_id)
        except Exception as exc:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Gmail draft lookup is unavailable; retry is safe"
            ) from exc
        return _LookupOutcome(
            state="draft_present" if draft is not None else "draft_missing"
        )

    async def _finalize(
        self,
        *,
        prepared: _PreparedReconciliation,
        outcome: _LookupOutcome,
        actor: str,
    ) -> dict[str, Any]:
        now = self._now_timestamp()
        async with self.pool.transaction() as conn:
            await self._lock_approval(conn, prepared.context.approval_id)
            record = await self._find_record_for_id(conn, prepared.context.draft_id)
            if record is None:
                raise CommercialBillingGmailSentReconciliationUnavailableError(
                    "Commercial billing Gmail draft is unavailable"
                )
            context = self._context(record)
            operation = await self._find_operation_by_id(
                conn, _uuid(prepared.operation["operation_id"], "operation id")
            )
            if operation is None:
                raise CommercialBillingGmailSentReconciliationUnavailableError(
                    "Commercial billing Gmail reconciliation operation is unavailable"
                )
            operation_state = _operation_state(operation["operation_state"])
            if operation_state == "completed":
                return self._result(
                    record,
                    outcome_state=_stored_text(
                        operation["operation_outcome_state"],
                        "operation outcome",
                        limit=32,
                    ),
                    replayed=True,
                    reused=True,
                )
            if context.reconciliation_state == "sent_confirmed":
                self._assert_confirmed_financial_state(context)
                operation = await self._complete_operation(
                    conn,
                    operation_id=_uuid(operation["operation_id"], "operation id"),
                    outcome_state="sent_confirmed",
                    now=now,
                )
                return self._result(
                    record,
                    outcome_state=_stored_text(
                        operation["operation_outcome_state"],
                        "operation outcome",
                        limit=32,
                    ),
                    replayed=prepared.replayed,
                    reused=True,
                )

            self._assert_reconcilable(context)
            if outcome.state == "sent_confirmed":
                if outcome.proof is None:
                    raise CommercialBillingGmailSentReconciliationConflictError(
                        "Gmail sent-mail proof is invalid"
                    )
                record = await self._confirm_sent(
                    conn,
                    context=context,
                    proof=outcome.proof,
                    actor=actor,
                    now=now,
                )
            elif outcome.state in {"draft_present", "draft_missing"}:
                record = await self._record_nonfinancial_outcome(
                    conn,
                    context=context,
                    outcome_state=outcome.state,
                    actor=actor,
                    now=now,
                )
            else:
                raise CommercialBillingGmailSentReconciliationConflictError(
                    "Commercial billing Gmail reconciliation outcome is invalid"
                )
            operation = await self._complete_operation(
                conn,
                operation_id=_uuid(operation["operation_id"], "operation id"),
                outcome_state=outcome.state,
                now=now,
            )
            return self._result(
                record,
                outcome_state=_stored_text(
                    operation["operation_outcome_state"],
                    "operation outcome",
                    limit=32,
                ),
                replayed=prepared.replayed,
                reused=False,
            )

    @staticmethod
    async def _lock_operation(conn: Any, idempotency_key: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-gmail-sent-reconciliation:operation:{idempotency_key}",
        )

    @staticmethod
    async def _lock_approval(conn: Any, approval_id: UUID) -> None:
        """Share the original draft service's approval lock across both flows."""

        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-invoice-gmail-draft:approval:{approval_id}",
        )

    @staticmethod
    async def _find_record_for_approval(conn: Any, approval_id: UUID) -> Any | None:
        row = await conn.fetchrow(
            _RECORD_SELECT + " WHERE draft.approval_id = $1",
            approval_id,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_record_for_id(conn: Any, draft_id: UUID) -> Any | None:
        row = await conn.fetchrow(
            _RECORD_SELECT + " WHERE draft.id = $1",
            draft_id,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_operation_by_key(conn: Any, idempotency_key: str) -> Any | None:
        row = await conn.fetchrow(
            _OPERATION_RECORD_SELECT
            + " WHERE operation.source = $1 AND operation.idempotency_key = $2",
            _DRAFT_SOURCE,
            idempotency_key,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _find_operation_by_id(conn: Any, operation_id: UUID) -> Any | None:
        row = await conn.fetchrow(
            """
            SELECT id AS operation_id, state AS operation_state,
                   outcome_state AS operation_outcome_state
            FROM commercial_billing_gmail_sent_reconciliation_operations
            WHERE id = $1
            """,
            operation_id,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def _insert_operation(
        conn: Any,
        *,
        draft_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
        now: datetime,
    ) -> Any:
        row = await conn.fetchrow(
            """
            INSERT INTO commercial_billing_gmail_sent_reconciliation_operations (
                id, gmail_draft_record_id, source, idempotency_key,
                request_fingerprint, requested_by, requested_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
            RETURNING id AS operation_id, state AS operation_state,
                      outcome_state AS operation_outcome_state
            """,
            uuid4(),
            draft_id,
            _DRAFT_SOURCE,
            idempotency_key,
            request_fingerprint,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Commercial billing Gmail reconciliation operation could not be recorded"
            )
        return dict(row)

    @staticmethod
    async def _complete_operation(
        conn: Any,
        *,
        operation_id: UUID,
        outcome_state: str,
        now: datetime,
    ) -> Any:
        if outcome_state not in _OUTCOME_STATES:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing Gmail reconciliation outcome is invalid"
            )
        row = await conn.fetchrow(
            """
            UPDATE commercial_billing_gmail_sent_reconciliation_operations
               SET state = 'completed', outcome_state = $2, completed_at = $3
             WHERE id = $1 AND state = 'pending'
            RETURNING id AS operation_id, state AS operation_state,
                      outcome_state AS operation_outcome_state
            """,
            operation_id,
            outcome_state,
            now,
        )
        if row is not None:
            return dict(row)
        existing = await CommercialBillingInvoiceGmailSentReconciliationService._find_operation_by_id(
            conn, operation_id
        )
        if existing is not None and _operation_state(existing["operation_state"]) == "completed":
            return existing
        raise CommercialBillingGmailSentReconciliationConflictError(
            "Commercial billing Gmail reconciliation operation could not be completed"
        )

    async def _confirm_sent(
        self,
        conn: Any,
        *,
        context: _Context,
        proof: _SentProof,
        actor: str,
        now: datetime,
    ) -> Mapping[str, Any]:
        invoice = await conn.fetchrow(
            "SELECT id, status, source FROM invoices WHERE id = $1 FOR UPDATE",
            context.invoice_id,
        )
        if invoice is None:
            raise CommercialBillingGmailSentReconciliationNotFoundError(
                "Commercial billing invoice not found"
            )
        if _stored_text(invoice["source"], "invoice source", limit=32) != _INVOICE_SOURCE:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing invoice is not eligible for Gmail reconciliation"
            )
        if _stored_text(invoice["status"], "invoice status", limit=32) != "draft":
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing invoice changed outside Gmail sent-mail reconciliation"
            )
        updated_invoice = await conn.fetchrow(
            """
            UPDATE invoices
               SET status = 'sent', sent_at = $2, sent_via = 'gmail', updated_at = $3
             WHERE id = $1 AND status = 'draft'
            RETURNING id
            """,
            context.invoice_id,
            proof.gmail_sent_at,
            now,
        )
        if updated_invoice is None:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing invoice changed outside Gmail sent-mail reconciliation"
            )
        row = await conn.fetchrow(
            """
            UPDATE commercial_billing_invoice_gmail_drafts
               SET reconciliation_state = 'sent_confirmed',
                   gmail_sent_message_id = $2, gmail_sent_thread_id = $3,
                   gmail_sent_at = $4, sent_reconciled_by = $5,
                   sent_reconciled_at = $6, last_reconciled_by = $5,
                   last_reconciled_at = $6
             WHERE id = $1 AND state = 'draft_created'
               AND reconciliation_state <> 'sent_confirmed'
            RETURNING id
            """,
            context.draft_id,
            proof.gmail_message_id,
            proof.gmail_thread_id,
            proof.gmail_sent_at,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing Gmail reconciliation changed concurrently"
            )
        return await self._record_with_context(conn, context.draft_id)

    async def _record_nonfinancial_outcome(
        self,
        conn: Any,
        *,
        context: _Context,
        outcome_state: str,
        actor: str,
        now: datetime,
    ) -> Mapping[str, Any]:
        row = await conn.fetchrow(
            """
            UPDATE commercial_billing_invoice_gmail_drafts
               SET reconciliation_state = $2::varchar, last_reconciled_by = $3,
                   last_reconciled_at = $4,
                   draft_missing_by = CASE
                       WHEN $2::varchar = 'draft_missing' THEN $3
                       ELSE draft_missing_by
                   END,
                   draft_missing_at = CASE
                       WHEN $2::varchar = 'draft_missing' THEN $4
                       ELSE draft_missing_at
                   END
             WHERE id = $1 AND state = 'draft_created'
               AND reconciliation_state <> 'sent_confirmed'
            RETURNING id
            """,
            context.draft_id,
            outcome_state,
            actor,
            now,
        )
        if row is None:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing Gmail reconciliation changed concurrently"
            )
        return await self._record_with_context(conn, context.draft_id)

    @staticmethod
    async def _record_with_context(conn: Any, draft_id: UUID) -> Mapping[str, Any]:
        context_row = await CommercialBillingInvoiceGmailSentReconciliationService._find_record_for_id(
            conn, draft_id
        )
        if context_row is None:
            raise CommercialBillingGmailSentReconciliationUnavailableError(
                "Commercial billing Gmail draft is unavailable"
            )
        return context_row

    @staticmethod
    def _assert_operation(
        operation: Mapping[str, Any], approval_id: UUID, request_fingerprint: str
    ) -> None:
        if operation.get("operation_request_fingerprint") != request_fingerprint:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Idempotency key was already used with a different commercial billing approval"
            )
        if _uuid(operation.get("approval_id"), "approval id") != approval_id:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Idempotency key was already used with a different commercial billing approval"
            )

    @staticmethod
    def _context(record: Mapping[str, Any]) -> _Context:
        approval_id = _uuid(record.get("approval_id"), "approval id")
        invoice_id = _uuid(record.get("invoice_id"), "invoice id")
        if _uuid(record.get("approval_invoice_id"), "approval invoice id") != invoice_id:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing approval no longer matches its invoice"
            )
        if _stored_text(record.get("approval_state"), "approval state", limit=32) != "invoice_created":
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing approval is not ready for Gmail reconciliation"
            )
        if _stored_text(record.get("invoice_source"), "invoice source", limit=32) != _INVOICE_SOURCE:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing invoice is not eligible for Gmail reconciliation"
            )
        metadata = _mapping(record.get("invoice_metadata"), "invoice metadata")
        if metadata.get("deliveryMethod") != _DELIVERY_METHOD:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing invoice is not configured for Gmail PDF delivery"
            )
        return _Context(
            approval_id=approval_id,
            draft_id=_uuid(record.get("draft_id"), "draft id"),
            draft_state=_stored_text(record.get("draft_state"), "draft state", limit=32),
            invoice_id=invoice_id,
            invoice_sent_at=_optional_timestamp(
                record.get("invoice_sent_at"), "invoice sent timestamp"
            ),
            invoice_sent_via=(
                None
                if record.get("invoice_sent_via") is None
                else _stored_text(record.get("invoice_sent_via"), "invoice sent via", limit=32)
            ),
            invoice_status=_stored_text(record.get("invoice_status"), "invoice status", limit=32),
            reconciliation_state=_state(record.get("reconciliation_state")),
            rfc_message_id=_rfc_message_id(record.get("rfc_message_id")),
        )

    @staticmethod
    def _assert_reconcilable(context: _Context) -> None:
        if context.draft_state != "draft_created":
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing Gmail draft is not ready for sent-mail reconciliation"
            )
        if context.invoice_status != "draft":
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing invoice changed outside Gmail sent-mail reconciliation"
            )

    @staticmethod
    def _assert_confirmed_financial_state(context: _Context) -> None:
        if context.invoice_status not in _POST_SEND_INVOICE_STATUSES:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Confirmed Gmail sent-mail evidence no longer matches invoice lifecycle"
            )
        if context.invoice_sent_via != "gmail" or context.invoice_sent_at is None:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Confirmed Gmail sent-mail evidence no longer matches invoice lifecycle"
            )

    @staticmethod
    def _sent_proof(value: Any, context: _Context) -> _SentProof:
        if not isinstance(value, Mapping):
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Gmail sent-message evidence is invalid"
            )
        labels = value.get("labelIds")
        if (
            not isinstance(labels, list)
            or not all(isinstance(label, str) and label for label in labels)
            or "SENT" not in labels
        ):
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Gmail sent-message is not verifiably in Sent mail"
            )
        headers = value.get("headers")
        _exact_header(headers, "Message-ID", context.rfc_message_id)
        _exact_header(
            headers,
            "X-Atlas-Commercial-Billing-Approval",
            str(context.approval_id),
        )
        _exact_header(
            headers,
            "X-Atlas-Commercial-Billing-Invoice",
            str(context.invoice_id),
        )
        return _SentProof(
            gmail_message_id=_external_id(value.get("id"), "sent message id"),
            gmail_thread_id=_external_id(value.get("threadId"), "sent thread id"),
            gmail_sent_at=_sent_timestamp(value.get("internalDate")),
        )

    def _now_timestamp(self) -> datetime:
        return _timestamp(self._now(), "clock")

    @staticmethod
    def _view(record: Mapping[str, Any]) -> dict[str, Any]:
        state = _state(record.get("reconciliation_state"))
        result = {
            "approvalId": str(_uuid(record.get("approval_id"), "approval id")),
            "draftId": str(_uuid(record.get("draft_id"), "draft id")),
            "invoiceId": str(_uuid(record.get("invoice_id"), "invoice id")),
            "state": state,
            "rfcMessageId": _rfc_message_id(record.get("rfc_message_id")),
            "gmailDraftId": record.get("gmail_draft_id"),
            "gmailMessageId": record.get("gmail_message_id"),
            "gmailThreadId": record.get("gmail_thread_id"),
            "gmailSentMessageId": record.get("gmail_sent_message_id"),
            "gmailSentThreadId": record.get("gmail_sent_thread_id"),
            "gmailSentAt": CommercialBillingInvoiceGmailSentReconciliationService._iso_optional(
                record.get("gmail_sent_at"), "Gmail sent timestamp"
            ),
            "sentReconciledAt": CommercialBillingInvoiceGmailSentReconciliationService._iso_optional(
                record.get("sent_reconciled_at"), "sent reconciliation timestamp"
            ),
            "sentReconciledBy": record.get("sent_reconciled_by"),
            "lastReconciledAt": CommercialBillingInvoiceGmailSentReconciliationService._iso_optional(
                record.get("last_reconciled_at"), "last reconciliation timestamp"
            ),
            "lastReconciledBy": record.get("last_reconciled_by"),
            "draftMissingAt": CommercialBillingInvoiceGmailSentReconciliationService._iso_optional(
                record.get("draft_missing_at"), "draft missing timestamp"
            ),
            "draftMissingBy": record.get("draft_missing_by"),
            "recoveryAction": {
                "draft_present": "wait_for_operator_send",
                "draft_missing": "review_missing_draft",
                "not_reconciled": "reconcile_sent_mail",
                "sent_confirmed": "none",
            }[state],
        }
        if state == "sent_confirmed":
            _external_id(result["gmailSentMessageId"], "sent message id")
            _external_id(result["gmailSentThreadId"], "sent thread id")
            _stored_text(result["sentReconciledBy"], "sent reconciliation actor", limit=128)
            if result["gmailSentAt"] is None or result["sentReconciledAt"] is None:
                raise CommercialBillingGmailSentReconciliationUnavailableError(
                    "Commercial billing Gmail sent reconciliation timestamps are invalid"
                )
        return result

    @staticmethod
    def _iso_optional(value: Any, field: str) -> str | None:
        return None if value is None else _timestamp(value, field).isoformat()

    @classmethod
    def _result(
        cls,
        record: Mapping[str, Any],
        *,
        outcome_state: str,
        replayed: bool,
        reused: bool,
    ) -> dict[str, Any]:
        if outcome_state not in _OUTCOME_STATES:
            raise CommercialBillingGmailSentReconciliationConflictError(
                "Commercial billing Gmail reconciliation outcome is invalid"
            )
        return {
            "outcome": outcome_state,
            "reconciliation": cls._view(record),
            "replayed": replayed,
            "reused": reused,
        }


def get_commercial_billing_invoice_gmail_sent_reconciliation_service(
) -> CommercialBillingInvoiceGmailSentReconciliationService:
    return CommercialBillingInvoiceGmailSentReconciliationService()


__all__ = [
    "CommercialBillingGmailSentReconciliationConflictError",
    "CommercialBillingGmailSentReconciliationError",
    "CommercialBillingGmailSentReconciliationNotFoundError",
    "CommercialBillingGmailSentReconciliationUnavailableError",
    "CommercialBillingGmailSentReconciliationValidationError",
    "CommercialBillingInvoiceGmailSentReconciliationService",
    "get_commercial_billing_invoice_gmail_sent_reconciliation_service",
]
