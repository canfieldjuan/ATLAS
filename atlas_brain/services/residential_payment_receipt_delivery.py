"""Explicit, durable Gmail delivery for residential payment receipts.

Payment creation owns the financial transaction and only queues a deterministic
receipt snapshot.  This service is deliberately downstream of that commit: it
may change receipt-delivery evidence, but never changes a payment, allocation,
invoice, deposit, clearing, return, or void.

Gmail does not accept a caller-owned idempotency key for ``messages.send``.  A
durable operation therefore enters ``attempting`` before the external request.
If the process dies at any point after that marker, a later caller is limited to
Sent-mail reconciliation by the immutable RFC Message-ID; it never sends again.
That conservative recovery behavior prevents duplicate customer receipts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import getaddresses
from typing import Any, Callable, Mapping, Protocol
from uuid import UUID, uuid4

import asyncpg

from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError


_SOURCE = "eom_admin"
_MAX_ACTOR_LENGTH = 128
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_GMAIL_ID_LENGTH = 256
_MAX_RFC_MESSAGE_ID_LENGTH = 320
_MAX_SUBJECT_LENGTH = 500
_MAX_BODY_LENGTH = 20_000
_ATTEMPT_LEASE = timedelta(minutes=5)
_DELIVERY_STATES = frozenset({"pending", "sent", "failed", "skipped"})
_OPERATION_STATES = frozenset(
    {"prepared", "attempting", "completed", "recovery_required"}
)
_OUTCOMES = frozenset({"sent", "failed", "already_sent"})
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


class ResidentialPaymentReceiptDeliveryError(Exception):
    """Base error with a stable public code."""

    code = "residential_payment_receipt_delivery_error"


class ResidentialPaymentReceiptDeliveryValidationError(
    ResidentialPaymentReceiptDeliveryError
):
    code = "invalid_residential_payment_receipt_delivery"


class ResidentialPaymentReceiptDeliveryNotFoundError(
    ResidentialPaymentReceiptDeliveryError
):
    code = "residential_payment_receipt_delivery_not_found"


class ResidentialPaymentReceiptDeliveryConflictError(
    ResidentialPaymentReceiptDeliveryError
):
    code = "residential_payment_receipt_delivery_conflict"


class ResidentialPaymentReceiptDeliveryUnavailableError(
    ResidentialPaymentReceiptDeliveryError
):
    code = "residential_payment_receipt_delivery_unavailable"


class _GmailReceiptGateway(Protocol):
    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        *,
        headers: Mapping[str, str],
    ) -> dict[str, Any]: ...

    async def find_sent_message_by_rfc_message_id(
        self,
        rfc_message_id: str,
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class _DeliveryContext:
    delivery_id: UUID
    payment_id: UUID
    recipient_email: str
    receipt_number: str
    rfc_message_id: str
    subject: str
    body: str


@dataclass(frozen=True)
class _PreparedOperation:
    context: _DeliveryContext
    delivery: Mapping[str, Any]
    operation: Mapping[str, Any]
    replayed: bool
    action: str


@dataclass(frozen=True)
class _SentProof:
    gmail_message_id: str
    gmail_thread_id: str
    sent_at: datetime


def _request_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ResidentialPaymentReceiptDeliveryValidationError(f"{field} is required")
    text = value.strip()
    if (
        not text
        or len(text) > limit
        or "\r" in text
        or "\n" in text
        or "\x00" in text
    ):
        raise ResidentialPaymentReceiptDeliveryValidationError(
            f"{field} must contain 1 to {limit} safe characters"
        )
    return text


def _stored_text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            f"Residential payment receipt {field} is invalid"
        )
    text = value.strip()
    if not text or len(text) > limit or "\x00" in text:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            f"Residential payment receipt {field} is invalid"
        )
    return text


def _stored_subject(value: Any) -> str:
    subject = _stored_text(value, "subject", limit=_MAX_SUBJECT_LENGTH)
    if "\r" in subject or "\n" in subject:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt subject is invalid"
        )
    return subject


def _stored_body(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_BODY_LENGTH:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt body is invalid"
        )
    if "\x00" in value:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt body is invalid"
        )
    return value


def _uuid(value: Any, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            f"Residential payment receipt {field} is invalid"
        ) from exc


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            f"Residential payment receipt {field} is invalid"
        )
    return value


def _optional_timestamp(value: Any, field: str) -> datetime | None:
    return None if value is None else _timestamp(value, field)


def _delivery_state(value: Any) -> str:
    state = _stored_text(value, "delivery status", limit=32)
    if state not in _DELIVERY_STATES:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt delivery status is invalid"
        )
    return state


def _operation_state(value: Any) -> str:
    state = _stored_text(value, "operation state", limit=32)
    if state not in _OPERATION_STATES:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt operation state is invalid"
        )
    return state


def _outcome(value: Any) -> str | None:
    if value is None:
        return None
    outcome = _stored_text(value, "operation outcome", limit=32)
    if outcome not in _OUTCOMES:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt operation outcome is invalid"
        )
    return outcome


def _rfc_message_id(value: Any) -> str:
    message_id = _stored_text(value, "RFC Message-ID", limit=_MAX_RFC_MESSAGE_ID_LENGTH)
    if (
        len(message_id) < 5
        or not message_id.startswith("<")
        or not message_id.endswith(">")
        or message_id.count("@") != 1
        or any(character.isspace() for character in message_id)
    ):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt RFC Message-ID is invalid"
        )
    return message_id


def _email(value: Any) -> str:
    if not isinstance(value, str):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt recipient is invalid"
        )
    # Keep the slim EOM profile free of the broader operator-mutation import
    # graph until an explicit delivery request actually needs email validation.
    from .eom_crm_mutations import normalize_contact_email

    try:
        normalized = normalize_contact_email(value)
    except Exception as exc:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt recipient is invalid"
        ) from exc
    if normalized is None or normalized != value:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Residential payment receipt recipient is invalid"
        )
    return normalized


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidentialPaymentReceiptDeliveryValidationError(
            "Residential payment receipt request is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _external_id(value: Any, field: str) -> str:
    return _stored_text(value, field, limit=_MAX_GMAIL_ID_LENGTH)


def _gmail_sent_timestamp(value: Any) -> datetime:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent timestamp is invalid"
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
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent timestamp is invalid"
        ) from exc


def _is_definite_gmail_send_rejection(exc: Exception) -> bool:
    """Identify only the transport's explicit pre-acceptance failure signal.

    This lazy import keeps the slim EOM API's startup import graph free of the
    Gmail client while still refusing to interpret an arbitrary gateway error as
    proof that customer mail was not accepted.
    """

    from ..tools.gmail import GmailSendError

    return isinstance(exc, GmailSendError) and exc.definitely_not_sent


def _exact_header(headers: Any, name: str, expected: str) -> None:
    if not isinstance(headers, list):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent headers are invalid"
        )
    values: list[str] = []
    for header in headers:
        if not isinstance(header, Mapping):
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Gmail receipt sent headers are invalid"
            )
        header_name = header.get("name")
        header_value = header.get("value")
        if not isinstance(header_name, str) or not isinstance(header_value, str):
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Gmail receipt sent headers are invalid"
            )
        if header_name.casefold() == name.casefold():
            values.append(header_value)
    if values != [expected]:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            f"Gmail receipt sent {name} header is invalid"
        )


def _exact_recipient_header(headers: Any, expected: str) -> None:
    if not isinstance(headers, list):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent headers are invalid"
        )
    values = [
        header.get("value")
        for header in headers
        if isinstance(header, Mapping)
        and isinstance(header.get("name"), str)
        and header["name"].casefold() == "to"
    ]
    if len(values) != 1:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent To header is invalid"
        )
    (recipient_header,) = values
    if not isinstance(recipient_header, str):
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent To header is invalid"
        )
    from .eom_crm_mutations import normalize_contact_email

    try:
        parsed = [
            normalize_contact_email(address)
            for _, address in getaddresses([recipient_header])
        ]
    except Exception as exc:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent To header is invalid"
        ) from exc
    if parsed != [expected]:
        raise ResidentialPaymentReceiptDeliveryConflictError(
            "Gmail receipt sent To header is invalid"
        )


class ResidentialPaymentReceiptDeliveryService:
    """Own the non-financial receipt-send state machine."""

    def __init__(
        self,
        *,
        pool: DatabasePool | None = None,
        gmail_gateway: _GmailReceiptGateway | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._configured_pool = pool
        self._gmail_gateway = gmail_gateway
        self._now = now or (lambda: datetime.now(timezone.utc))

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise ResidentialPaymentReceiptDeliveryUnavailableError(
                "Residential payment receipt database unavailable"
            )
        return pool

    @property
    def gmail_gateway(self) -> _GmailReceiptGateway:
        if self._gmail_gateway is not None:
            return self._gmail_gateway
        # Import Gmail only at the explicit operator send boundary.  The slim
        # EOM API imports this service solely to expose its protected route and
        # must not load credentials, provider clients, or the full tool graph
        # during application startup.
        from ..tools.gmail import get_gmail_transport

        return get_gmail_transport()

    async def dispatch(
        self,
        *,
        payment_id: UUID,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        """Send, recover, or return one persisted residential receipt operation."""

        if not isinstance(payment_id, UUID):
            raise ResidentialPaymentReceiptDeliveryValidationError(
                "Payment id is invalid"
            )
        key = _request_text(
            idempotency_key, "Idempotency key", limit=_MAX_IDEMPOTENCY_KEY_LENGTH
        )
        requested_by = _request_text(actor, "authenticated actor", limit=_MAX_ACTOR_LENGTH)
        fingerprint = _fingerprint({"paymentId": str(payment_id)})
        try:
            prepared = await self._prepare(
                payment_id=payment_id,
                idempotency_key=key,
                request_fingerprint=fingerprint,
                actor=requested_by,
            )
            if prepared.action == "return":
                return self._result(
                    prepared.delivery,
                    prepared.operation,
                    replayed=prepared.replayed,
                    reused=True,
                )
            if prepared.action == "recover":
                return await self._recover_existing(prepared, actor=requested_by)
            if prepared.action != "preflight":
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation action is invalid"
                )

            proof = await self._lookup_sent(prepared.context)
            if proof is not None:
                return await self._confirm_sent(
                    prepared=prepared,
                    proof=proof,
                    actor=requested_by,
                    replayed=prepared.replayed,
                    reconciliation=False,
                )

            claimed = await self._claim_attempt(prepared, actor=requested_by)
            if claimed is None:
                refreshed = await self._reload_prepared(prepared)
                return await self._recover_existing(refreshed, actor=requested_by)

            try:
                response = await self.gmail_gateway.send(
                    [claimed.context.recipient_email],
                    claimed.context.subject,
                    claimed.context.body,
                    headers={
                        "Message-ID": claimed.context.rfc_message_id,
                        "X-Atlas-EOM-Payment-Receipt": str(
                            claimed.context.delivery_id
                        ),
                    },
                )
                proof = self._send_response_proof(response, now=self._now())
            except Exception as exc:
                if _is_definite_gmail_send_rejection(exc):
                    return await self._record_definite_failure(
                        prepared=claimed,
                        actor=requested_by,
                        replayed=prepared.replayed,
                    )
                # A gateway exception after the durable attempting marker is
                # necessarily ambiguous.  Do not create another customer email.
                return await self._recover_uncertain(
                    prepared=claimed,
                    actor=requested_by,
                    replayed=prepared.replayed,
                )
            return await self._confirm_sent(
                prepared=claimed,
                proof=proof,
                actor=requested_by,
                replayed=prepared.replayed,
                reconciliation=False,
            )
        except ResidentialPaymentReceiptDeliveryError:
            raise
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Residential payment receipt delivery could not be recorded"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise ResidentialPaymentReceiptDeliveryUnavailableError(
                "Residential payment receipt database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise ResidentialPaymentReceiptDeliveryUnavailableError(
                "Residential payment receipt database unavailable"
            ) from exc

    async def reconcile(
        self,
        *,
        payment_id: UUID,
        actor: str,
    ) -> dict[str, Any]:
        """Check Gmail Sent evidence for an already-ambiguous receipt attempt.

        This deliberately has no dispatch idempotency key and never reaches
        ``GmailTransport.send``.  It is an operator-visible recovery action:
        repeat calls may perform the same read-only Sent-mail lookup, but can
        neither make a second customer email nor change a financial record.
        """

        if not isinstance(payment_id, UUID):
            raise ResidentialPaymentReceiptDeliveryValidationError(
                "Payment id is invalid"
            )
        requested_by = _request_text(actor, "authenticated actor", limit=_MAX_ACTOR_LENGTH)
        try:
            prepared = await self._prepare_reconciliation(payment_id=payment_id)
            return await self._recover_existing(prepared, actor=requested_by)
        except ResidentialPaymentReceiptDeliveryError:
            raise
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Residential payment receipt delivery could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise ResidentialPaymentReceiptDeliveryUnavailableError(
                "Residential payment receipt database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise ResidentialPaymentReceiptDeliveryUnavailableError(
                "Residential payment receipt database unavailable"
            ) from exc

    async def _prepare(
        self,
        *,
        payment_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
    ) -> _PreparedOperation:
        async with self.pool.transaction() as conn:
            # Every state transition takes locks in this order: optional
            # idempotency-key advisory lock, delivery advisory lock, operation
            # row, delivery row.  The initial identity lookups below are
            # deliberately non-locking so a fresh key cannot hold the delivery
            # row while a claimant holds an operation row for that delivery.
            await self._lock_operation(conn, idempotency_key)
            operation = await self._find_operation_by_key(conn, idempotency_key)
            if operation is not None:
                delivery_id = _uuid(operation["receipt_delivery_id"], "operation receipt id")
                await self._lock_delivery(conn, delivery_id)
                operation = await self._find_operation_by_id(
                    conn, _uuid(operation["id"], "operation id")
                )
                delivery = await self._find_delivery_by_id(conn, delivery_id)
                if operation is None or delivery is None:
                    raise ResidentialPaymentReceiptDeliveryConflictError(
                        "Residential payment receipt operation evidence is invalid"
                    )
                context = self._context(delivery)
                self._assert_operation(
                    operation,
                    delivery_payment_id=context.payment_id,
                    requested_payment_id=payment_id,
                    request_fingerprint=request_fingerprint,
                    delivery_id=context.delivery_id,
                )
                state = _operation_state(operation["state"])
                if state == "completed":
                    return _PreparedOperation(
                        context=context,
                        delivery=delivery,
                        operation=operation,
                        replayed=True,
                        action="return",
                    )
                return _PreparedOperation(
                    context=context,
                    delivery=delivery,
                    operation=operation,
                    replayed=True,
                    action="preflight" if state == "prepared" else "recover",
                )

            delivery_id = await self._find_delivery_id_by_payment(conn, payment_id)
            if delivery_id is None:
                raise ResidentialPaymentReceiptDeliveryNotFoundError(
                    "Residential payment receipt delivery not found"
                )
            await self._lock_delivery(conn, delivery_id)
            active = await self._find_active_operation(conn, delivery_id)
            delivery = await self._find_delivery_by_id(conn, delivery_id)
            if delivery is None:
                raise ResidentialPaymentReceiptDeliveryNotFoundError(
                    "Residential payment receipt delivery not found"
                )
            context = self._context(delivery)
            status = _delivery_state(delivery["delivery_status"])
            if status == "skipped":
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt delivery has no recipient"
                )
            if active is not None:
                active_state = _operation_state(active["state"])
                if active_state == "recovery_required":
                    raise ResidentialPaymentReceiptDeliveryConflictError(
                        "Residential payment receipt delivery requires reconciliation"
                    )
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt delivery is already in progress"
                )
            now = self._now()
            if status == "sent":
                operation = await self._insert_operation(
                    conn,
                    delivery_id=context.delivery_id,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    actor=actor,
                    state="completed",
                    outcome="already_sent",
                    result_delivery_status="sent",
                    result_sent_at=_timestamp(delivery["sent_at"], "sent time"),
                    now=now,
                )
                return _PreparedOperation(
                    context=context,
                    delivery=delivery,
                    operation=operation,
                    replayed=False,
                    action="return",
                )
            if status not in {"pending", "failed"}:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt delivery status is not dispatchable"
                )
            if status == "failed":
                delivery = await conn.fetchrow(
                    """
                    UPDATE payment_receipt_deliveries
                    SET delivery_status = 'pending', updated_at = $2
                    WHERE id = $1
                    RETURNING *
                    """,
                    context.delivery_id,
                    now,
                )
                context = self._context(delivery)
            operation = await self._insert_operation(
                conn,
                delivery_id=context.delivery_id,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                actor=actor,
                state="prepared",
                outcome=None,
                now=now,
            )
            return _PreparedOperation(
                context=context,
                delivery=delivery,
                operation=operation,
                replayed=False,
                action="preflight",
            )

    async def _prepare_reconciliation(
        self,
        *,
        payment_id: UUID,
    ) -> _PreparedOperation:
        """Load only an active ambiguous operation; never create one or send."""

        async with self.pool.transaction() as conn:
            delivery_id = await self._find_delivery_id_by_payment(conn, payment_id)
            if delivery_id is None:
                raise ResidentialPaymentReceiptDeliveryNotFoundError(
                    "Residential payment receipt delivery not found"
                )
            await self._lock_delivery(conn, delivery_id)
            active = await self._find_active_operation(conn, delivery_id)
            delivery = await self._find_delivery_by_id(conn, delivery_id)
            if delivery is None:
                raise ResidentialPaymentReceiptDeliveryNotFoundError(
                    "Residential payment receipt delivery not found"
                )
            context = self._context(delivery)
            if active is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt delivery has no ambiguous attempt to reconcile"
                )
            state = _operation_state(active["state"])
            if state == "prepared":
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt delivery is not yet attempting Gmail delivery"
                )
            if state not in {"attempting", "recovery_required"}:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt delivery has no ambiguous attempt to reconcile"
                )
            return _PreparedOperation(
                context=context,
                delivery=delivery,
                operation=active,
                replayed=True,
                action="recover",
            )

    async def _claim_attempt(
        self,
        prepared: _PreparedOperation,
        *,
        actor: str,
    ) -> _PreparedOperation | None:
        """Atomically install the no-second-send marker before Gmail I/O."""

        operation_id = _uuid(prepared.operation["id"], "operation id")
        async with self.pool.transaction() as conn:
            await self._lock_delivery(conn, prepared.context.delivery_id)
            operation = await self._find_operation_by_id(conn, operation_id)
            delivery = await self._find_delivery_by_id(conn, prepared.context.delivery_id)
            if operation is None or delivery is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation evidence is unavailable"
                )
            if _operation_state(operation["state"]) != "prepared":
                return None
            if _delivery_state(delivery["delivery_status"]) != "pending":
                return None
            now = self._now()
            operation = await conn.fetchrow(
                """
                UPDATE payment_receipt_delivery_operations
                SET state = 'attempting', attempt_started_at = $2
                WHERE id = $1 AND state = 'prepared'
                RETURNING *
                """,
                operation_id,
                now,
            )
            if operation is None:
                return None
            delivery = await conn.fetchrow(
                """
                UPDATE payment_receipt_deliveries
                SET last_attempt_by = $2, last_attempt_at = $3, updated_at = $3
                WHERE id = $1
                RETURNING *
                """,
                prepared.context.delivery_id,
                actor,
                now,
            )
            return _PreparedOperation(
                context=self._context(delivery),
                delivery=delivery,
                operation=operation,
                replayed=prepared.replayed,
                action="attempting",
            )

    async def _recover_existing(
        self,
        prepared: _PreparedOperation,
        *,
        actor: str,
    ) -> dict[str, Any]:
        """Read Gmail only for an already-attempting/recovery operation."""

        state = _operation_state(prepared.operation["state"])
        if state == "completed":
            return self._result(
                prepared.delivery, prepared.operation, replayed=prepared.replayed, reused=True
            )
        if state == "prepared":
            # A reentrant caller may resume a safely unattempted operation.
            return await self.dispatch(
                payment_id=prepared.context.payment_id,
                idempotency_key=_stored_text(
                    prepared.operation["idempotency_key"],
                    "operation idempotency key",
                    limit=_MAX_IDEMPOTENCY_KEY_LENGTH,
                ),
                actor=actor,
            )
        if state == "attempting":
            started = _timestamp(
                prepared.operation["attempt_started_at"], "attempt start time"
            )
            if self._now() - started < _ATTEMPT_LEASE:
                return self._result(
                    prepared.delivery,
                    prepared.operation,
                    replayed=prepared.replayed,
                    reused=True,
                )
        try:
            proof = await self._lookup_sent(prepared.context)
        except ResidentialPaymentReceiptDeliveryUnavailableError:
            return await self._mark_recovery_required(
                prepared=prepared,
                actor=actor,
                replayed=prepared.replayed,
                reconciliation=True,
            )
        if proof is not None:
            return await self._confirm_sent(
                prepared=prepared,
                proof=proof,
                actor=actor,
                replayed=prepared.replayed,
                reconciliation=True,
            )
        return await self._mark_recovery_required(
            prepared=prepared,
            actor=actor,
            replayed=prepared.replayed,
            reconciliation=True,
        )

    async def _recover_uncertain(
        self,
        *,
        prepared: _PreparedOperation,
        actor: str,
        replayed: bool,
    ) -> dict[str, Any]:
        try:
            proof = await self._lookup_sent(prepared.context)
        except ResidentialPaymentReceiptDeliveryUnavailableError:
            proof = None
        if proof is not None:
            return await self._confirm_sent(
                prepared=prepared,
                proof=proof,
                actor=actor,
                replayed=replayed,
                reconciliation=True,
            )
        return await self._mark_recovery_required(
            prepared=prepared,
            actor=actor,
            replayed=replayed,
            reconciliation=True,
        )

    async def _lookup_sent(self, context: _DeliveryContext) -> _SentProof | None:
        try:
            result = await self.gmail_gateway.find_sent_message_by_rfc_message_id(
                context.rfc_message_id
            )
        except Exception as exc:
            raise ResidentialPaymentReceiptDeliveryUnavailableError(
                "Gmail receipt sent-mail lookup is unavailable"
            ) from exc
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Gmail receipt sent-mail evidence is invalid"
            )
        labels = result.get("labelIds")
        if (
            not isinstance(labels, list)
            or not all(isinstance(label, str) and label for label in labels)
            or "SENT" not in labels
        ):
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Gmail receipt sent-mail evidence is invalid"
            )
        _exact_header(result.get("headers"), "Message-ID", context.rfc_message_id)
        _exact_header(
            result.get("headers"),
            "X-Atlas-EOM-Payment-Receipt",
            str(context.delivery_id),
        )
        _exact_recipient_header(result.get("headers"), context.recipient_email)
        return _SentProof(
            gmail_message_id=_external_id(result.get("id"), "Gmail message id"),
            gmail_thread_id=_external_id(result.get("threadId"), "Gmail thread id"),
            sent_at=_gmail_sent_timestamp(result.get("internalDate")),
        )

    @staticmethod
    def _send_response_proof(response: Any, *, now: datetime) -> _SentProof:
        if not isinstance(response, Mapping):
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Gmail receipt send response is invalid"
            )
        return _SentProof(
            gmail_message_id=_external_id(response.get("id"), "Gmail message id"),
            gmail_thread_id=_external_id(response.get("threadId"), "Gmail thread id"),
            sent_at=now,
        )

    async def _confirm_sent(
        self,
        *,
        prepared: _PreparedOperation,
        proof: _SentProof,
        actor: str,
        replayed: bool,
        reconciliation: bool,
    ) -> dict[str, Any]:
        operation_id = _uuid(prepared.operation["id"], "operation id")
        async with self.pool.transaction() as conn:
            await self._lock_delivery(conn, prepared.context.delivery_id)
            operation = await self._find_operation_by_id(conn, operation_id)
            delivery = await self._find_delivery_by_id(conn, prepared.context.delivery_id)
            if operation is None or delivery is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation evidence is unavailable"
                )
            delivery_status = _delivery_state(delivery["delivery_status"])
            if delivery_status == "sent":
                if (
                    _external_id(delivery.get("gmail_message_id"), "Gmail message id")
                    != proof.gmail_message_id
                    or _external_id(
                        delivery.get("gmail_thread_id"), "Gmail thread id"
                    )
                    != proof.gmail_thread_id
                ):
                    raise ResidentialPaymentReceiptDeliveryConflictError(
                        "Residential payment receipt sent-mail identity conflicts"
                    )
                if reconciliation:
                    await self._record_reconciliation_event(
                        conn,
                        delivery_id=prepared.context.delivery_id,
                        operation_id=operation_id,
                        actor=actor,
                        outcome="sent",
                        now=self._now(),
                    )
                return self._result(delivery, operation, replayed=replayed, reused=True)
            state = _operation_state(operation["state"])
            if state not in {"prepared", "attempting", "recovery_required"}:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation cannot record sent mail"
                )
            now = self._now()
            outcome = "already_sent" if state == "prepared" else "sent"
            delivery = await conn.fetchrow(
                """
                UPDATE payment_receipt_deliveries
                SET delivery_status = 'sent', gmail_message_id = $2,
                    gmail_thread_id = $3, sent_at = $4,
                    recovery_required_at = NULL, updated_at = $5
                WHERE id = $1
                RETURNING *
                """,
                prepared.context.delivery_id,
                proof.gmail_message_id,
                proof.gmail_thread_id,
                proof.sent_at,
                now,
            )
            operation = await conn.fetchrow(
                """
                UPDATE payment_receipt_delivery_operations
                SET state = 'completed', outcome = $2, completed_at = $3,
                    recovery_required_at = NULL, result_delivery_status = 'sent',
                    result_sent_at = $4
                WHERE id = $1
                RETURNING *
                """,
                operation_id,
                outcome,
                now,
                proof.sent_at,
            )
            if reconciliation:
                await self._record_reconciliation_event(
                    conn,
                    delivery_id=prepared.context.delivery_id,
                    operation_id=operation_id,
                    actor=actor,
                    outcome="sent",
                    now=now,
                )
            return self._result(delivery, operation, replayed=replayed, reused=False)

    async def _record_definite_failure(
        self,
        *,
        prepared: _PreparedOperation,
        actor: str,
        replayed: bool,
    ) -> dict[str, Any]:
        operation_id = _uuid(prepared.operation["id"], "operation id")
        async with self.pool.transaction() as conn:
            await self._lock_delivery(conn, prepared.context.delivery_id)
            operation = await self._find_operation_by_id(conn, operation_id)
            delivery = await self._find_delivery_by_id(conn, prepared.context.delivery_id)
            if operation is None or delivery is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation evidence is unavailable"
                )
            if _operation_state(operation["state"]) != "attempting":
                return self._result(delivery, operation, replayed=replayed, reused=True)
            now = self._now()
            delivery = await conn.fetchrow(
                """
                UPDATE payment_receipt_deliveries
                SET delivery_status = 'failed', last_attempt_by = $2,
                    last_attempt_at = $3, last_failure_code = 'gmail_rejected',
                    last_failure_at = $3, updated_at = $3
                WHERE id = $1
                RETURNING *
                """,
                prepared.context.delivery_id,
                actor,
                now,
            )
            operation = await conn.fetchrow(
                """
                UPDATE payment_receipt_delivery_operations
                SET state = 'completed', outcome = 'failed', completed_at = $2,
                    result_delivery_status = 'failed', result_sent_at = NULL
                WHERE id = $1
                RETURNING *
                """,
                operation_id,
                now,
            )
            return self._result(delivery, operation, replayed=replayed, reused=False)

    async def _mark_recovery_required(
        self,
        *,
        prepared: _PreparedOperation,
        actor: str,
        replayed: bool,
        reconciliation: bool,
    ) -> dict[str, Any]:
        operation_id = _uuid(prepared.operation["id"], "operation id")
        async with self.pool.transaction() as conn:
            await self._lock_delivery(conn, prepared.context.delivery_id)
            operation = await self._find_operation_by_id(conn, operation_id)
            delivery = await self._find_delivery_by_id(conn, prepared.context.delivery_id)
            if operation is None or delivery is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation evidence is unavailable"
                )
            state = _operation_state(operation["state"])
            if state == "completed":
                return self._result(delivery, operation, replayed=replayed, reused=True)
            if state not in {"attempting", "recovery_required"}:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation cannot require recovery"
                )
            now = self._now()
            if state == "attempting":
                operation = await conn.fetchrow(
                    """
                    UPDATE payment_receipt_delivery_operations
                    SET state = 'recovery_required', recovery_required_at = $2
                    WHERE id = $1
                    RETURNING *
                    """,
                    operation_id,
                    now,
                )
            if delivery["recovery_required_at"] is None:
                delivery = await conn.fetchrow(
                    """
                    UPDATE payment_receipt_deliveries
                    SET recovery_required_at = $2, updated_at = $2
                    WHERE id = $1
                    RETURNING *
                    """,
                    prepared.context.delivery_id,
                    now,
                )
            if reconciliation:
                await self._record_reconciliation_event(
                    conn,
                    delivery_id=prepared.context.delivery_id,
                    operation_id=operation_id,
                    actor=actor,
                    outcome="recovery_required",
                    now=now,
                )
            return self._result(delivery, operation, replayed=replayed, reused=True)

    async def _reload_prepared(self, prepared: _PreparedOperation) -> _PreparedOperation:
        operation_id = _uuid(prepared.operation["id"], "operation id")
        async with self.pool.transaction() as conn:
            await self._lock_delivery(conn, prepared.context.delivery_id)
            operation = await self._find_operation_by_id(conn, operation_id)
            delivery = await self._find_delivery_by_id(conn, prepared.context.delivery_id)
            if operation is None or delivery is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation evidence is unavailable"
                )
            return _PreparedOperation(
                context=self._context(delivery),
                delivery=delivery,
                operation=operation,
                replayed=prepared.replayed,
                action="recover",
            )

    @staticmethod
    async def _lock_operation(conn: Any, idempotency_key: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"residential-payment-receipt-delivery:operation:{idempotency_key}",
        )

    @staticmethod
    async def _lock_delivery(conn: Any, delivery_id: UUID) -> None:
        """Acquire the first shared lock for every receipt-state transition."""

        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"residential-payment-receipt-delivery:delivery:{delivery_id}",
        )

    @staticmethod
    async def _find_delivery_id_by_payment(conn: Any, payment_id: UUID) -> UUID | None:
        value = await conn.fetchval(
            """
            SELECT id
            FROM payment_receipt_deliveries
            WHERE payment_id = $1
            """,
            payment_id,
        )
        return None if value is None else _uuid(value, "delivery id")

    @staticmethod
    async def _find_delivery_by_id(conn: Any, delivery_id: UUID) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT *
            FROM payment_receipt_deliveries
            WHERE id = $1
            FOR UPDATE
            """,
            delivery_id,
        )

    @staticmethod
    async def _find_operation_by_key(conn: Any, idempotency_key: str) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT *
            FROM payment_receipt_delivery_operations
            WHERE source = $1 AND idempotency_key = $2
            """,
            _SOURCE,
            idempotency_key,
        )

    @staticmethod
    async def _find_operation_by_id(conn: Any, operation_id: UUID) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT *
            FROM payment_receipt_delivery_operations
            WHERE id = $1
            FOR UPDATE
            """,
            operation_id,
        )

    @staticmethod
    async def _find_active_operation(conn: Any, delivery_id: UUID) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT *
            FROM payment_receipt_delivery_operations
            WHERE receipt_delivery_id = $1
              AND state IN ('prepared', 'attempting', 'recovery_required')
            ORDER BY requested_at DESC, id DESC
            LIMIT 1
            FOR UPDATE
            """,
            delivery_id,
        )

    @staticmethod
    async def _insert_operation(
        conn: Any,
        *,
        delivery_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        actor: str,
        state: str,
        outcome: str | None,
        now: datetime,
        result_delivery_status: str | None = None,
        result_sent_at: datetime | None = None,
    ) -> Any:
        return await conn.fetchrow(
            """
            INSERT INTO payment_receipt_delivery_operations (
                id, receipt_delivery_id, source, idempotency_key,
                request_fingerprint, state, outcome, requested_by, requested_at,
                completed_at, result_delivery_status, result_sent_at, created_at
            )
            VALUES (
                $1, $2, $3::varchar, $4::varchar, $5::varchar, $6::varchar,
                $7::varchar, $8::varchar, $9::timestamptz,
                CASE WHEN $6::varchar = 'completed' THEN $9::timestamptz ELSE NULL END,
                $10::varchar, $11::timestamptz, $9::timestamptz
            )
            RETURNING *
            """,
            uuid4(),
            delivery_id,
            _SOURCE,
            idempotency_key,
            request_fingerprint,
            state,
            outcome,
            actor,
            now,
            result_delivery_status,
            result_sent_at,
        )

    @staticmethod
    async def _record_reconciliation_event(
        conn: Any,
        *,
        delivery_id: UUID,
        operation_id: UUID,
        actor: str,
        outcome: str,
        now: datetime,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO payment_receipt_delivery_reconciliation_events (
                receipt_delivery_id, operation_id, actor, outcome, reconciled_at,
                created_at
            )
            VALUES ($1, $2, $3::varchar, $4::varchar, $5::timestamptz, $5::timestamptz)
            """,
            delivery_id,
            operation_id,
            actor,
            outcome,
            now,
        )

    @staticmethod
    def _context(delivery: Mapping[str, Any]) -> _DeliveryContext:
        status = _delivery_state(delivery.get("delivery_status"))
        if status == "skipped":
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Residential payment receipt delivery has no recipient"
            )
        return _DeliveryContext(
            delivery_id=_uuid(delivery.get("id"), "delivery id"),
            payment_id=_uuid(delivery.get("payment_id"), "payment id"),
            recipient_email=_email(delivery.get("recipient_email")),
            receipt_number=_stored_text(
                delivery.get("receipt_number"), "receipt number", limit=64
            ),
            rfc_message_id=_rfc_message_id(delivery.get("rfc_message_id")),
            subject=_stored_subject(delivery.get("subject")),
            body=_stored_body(delivery.get("body")),
        )

    @staticmethod
    def _assert_operation(
        operation: Mapping[str, Any],
        *,
        delivery_payment_id: UUID,
        requested_payment_id: UUID,
        request_fingerprint: str,
        delivery_id: UUID,
    ) -> None:
        if _uuid(operation.get("receipt_delivery_id"), "operation receipt id") != delivery_id:
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Residential payment receipt operation evidence is invalid"
            )
        if _stored_text(
            operation.get("request_fingerprint"), "operation fingerprint", limit=64
        ) != request_fingerprint:
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Idempotency key was already used for a different receipt delivery"
            )
        if delivery_payment_id != requested_payment_id:
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Idempotency key was already used for a different receipt delivery"
            )
        if not isinstance(requested_payment_id, UUID):
            raise ResidentialPaymentReceiptDeliveryValidationError(
                "Payment id is invalid"
            )

    @staticmethod
    def _result(
        delivery: Mapping[str, Any],
        operation: Mapping[str, Any],
        *,
        replayed: bool,
        reused: bool,
    ) -> dict[str, Any]:
        state = _operation_state(operation.get("state"))
        outcome = _outcome(operation.get("outcome"))
        result_delivery = ResidentialPaymentReceiptDeliveryService._result_delivery(
            delivery, operation, state=state, outcome=outcome
        )
        context = ResidentialPaymentReceiptDeliveryService._context(result_delivery)
        sent_at = _optional_timestamp(result_delivery.get("sent_at"), "sent time")
        recovery_required_at = _optional_timestamp(
            result_delivery.get("recovery_required_at"), "recovery time"
        )
        return {
            "payment_id": str(context.payment_id),
            "receipt_delivery": {
                "receipt_number": context.receipt_number,
                "recipient_email": context.recipient_email,
                "status": _delivery_state(result_delivery.get("delivery_status")),
                "skip_reason": result_delivery.get("skip_reason"),
                "sent_at": sent_at.isoformat() if sent_at is not None else None,
                "recovery_required_at": (
                    recovery_required_at.isoformat()
                    if recovery_required_at is not None
                    else None
                ),
            },
            "operation": {
                "state": state,
                "outcome": outcome,
                "requested_at": _timestamp(
                    operation.get("requested_at"), "operation requested time"
                ).isoformat(),
                "completed_at": (
                    _optional_timestamp(
                        operation.get("completed_at"), "operation completed time"
                    ).isoformat()
                    if operation.get("completed_at") is not None
                    else None
                ),
            },
            "replayed": replayed,
            "reused": reused,
        }

    @staticmethod
    def _result_delivery(
        delivery: Mapping[str, Any],
        operation: Mapping[str, Any],
        *,
        state: str,
        outcome: str | None,
    ) -> Mapping[str, Any]:
        """Render completed-key replays from their immutable recorded outcome.

        The receipt row may legitimately advance after a definite failed send
        when a later idempotency key retries it.  A replay of the earlier key
        must nevertheless reproduce the receipt result that key completed
        with, rather than combining its completed operation outcome with the
        current mutable delivery row.
        """

        if state != "completed":
            return delivery
        result_status = _delivery_state(
            operation.get("result_delivery_status")
        )
        result_sent_at = _optional_timestamp(
            operation.get("result_sent_at"), "operation result sent time"
        )
        if outcome in {"sent", "already_sent"}:
            if result_status != "sent" or result_sent_at is None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation result is invalid"
                )
        elif outcome == "failed":
            if result_status != "failed" or result_sent_at is not None:
                raise ResidentialPaymentReceiptDeliveryConflictError(
                    "Residential payment receipt operation result is invalid"
                )
        else:
            raise ResidentialPaymentReceiptDeliveryConflictError(
                "Residential payment receipt operation result is invalid"
            )
        return {
            **dict(delivery),
            "delivery_status": result_status,
            "sent_at": result_sent_at,
            "recovery_required_at": None,
        }


_service: ResidentialPaymentReceiptDeliveryService | None = None


def get_residential_payment_receipt_delivery_service(
) -> ResidentialPaymentReceiptDeliveryService:
    """Return the process-local receipt dispatch service."""

    global _service
    if _service is None:
        _service = ResidentialPaymentReceiptDeliveryService()
    return _service
