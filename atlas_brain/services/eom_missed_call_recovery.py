"""Durable EOM missed-call recovery state and delivery worker.

The public estimate acknowledgement is intentionally outside this module.  A
recovery sequence begins only after the authenticated office records a no-answer
call against a canonical EOM lead.  The database stores every operator action,
sequence transition, and delivery result before a caller sees success.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import asyncpg
import httpx

from ..eom_api.config import EOMFunnelConfig, funnel_settings
from ..templates.email.estimate_confirmation import (
    BUSINESS_EMAIL,
    BUSINESS_NAME,
)
from ..templates.email.missed_call_recovery import (
    MissedCallRecoveryEmail,
    render_missed_call_recovery_email,
)


logger = logging.getLogger("atlas.eom.missed_call_recovery")

_EOM_CONTEXT = "effingham_maids"
_MAX_ACTOR_NAME_LENGTH = 128
_MAX_OPERATION_KEY_LENGTH = 128
_MAX_STATUS_IDS = 200
_MAX_WORK_BATCH = 25
_ATTEMPT_LEASE = timedelta(minutes=5)
# Resend retains idempotency keys for 24 hours.  Leave a one-hour safety margin:
# a delayed process must become visibly recoverable instead of sending a second
# customer email after provider-side de-duplication has expired.
_PROVIDER_KEY_WINDOW = timedelta(hours=23)
_RETRY_DELAYS = (timedelta(minutes=5), timedelta(minutes=30), timedelta(hours=2))
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_OPERATION_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_MANUAL_CANCEL_REASONS = frozenset(
    {"callback_recorded", "response_recorded", "opt_out", "manual"}
)
_TRACKED_RESPONSE_INTERACTIONS = frozenset(
    {
        "email_inbound",
        "lead_response",
        "callback_completed",
        "conversation_completed",
        "opt_out",
    }
)
# CLOSED / ENUMERATED: these are the only browser-originated mutations this
# recovery slice accepts. The database receipt records this finite vocabulary
# with one globally unique operation key, so a retry can never move to another
# lead or mutation kind after an interrupted browser request.
_OPERATION_KINDS = frozenset({"no_answer", "resume", "cancel"})


class EOMMissedCallRecoveryError(Exception):
    """Base class for a stable API-safe recovery error."""

    status_code = 409
    code = "missed_call_recovery_error"


class EOMMissedCallRecoveryValidationError(EOMMissedCallRecoveryError):
    status_code = 422
    code = "invalid_missed_call_recovery_request"


class EOMMissedCallRecoveryNotFoundError(EOMMissedCallRecoveryError):
    status_code = 404
    code = "missed_call_recovery_not_found"


class EOMMissedCallRecoveryConflictError(EOMMissedCallRecoveryError):
    status_code = 409
    code = "missed_call_recovery_conflict"


class EOMMissedCallRecoveryUnavailableError(EOMMissedCallRecoveryError):
    status_code = 503
    code = "missed_call_recovery_unavailable"


class _DefiniteDeliveryError(Exception):
    """The provider positively rejected a request before accepting mail."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        recovery_required_if_exhausted: bool = False,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        # A retryable transport rejection can safely become ``failed`` after
        # its bounded attempts are exhausted. A provider response that says
        # this idempotency key is currently being processed is different: its
        # eventual acceptance remains unknown, so exhaustion must preserve
        # ambiguity for an operator rather than claim a definite failure.
        self.recovery_required_if_exhausted = recovery_required_if_exhausted


class _AmbiguousDeliveryError(Exception):
    """The provider outcome cannot prove that mail was not accepted."""


class _RecoveryEmailGateway(Protocol):
    async def send(
        self,
        *,
        recipient_email: str,
        subject: str,
        body: str,
        idempotency_key: str,
    ) -> str: ...


class ResendMissedCallRecoveryGateway:
    """Small Resend adapter with no customer data in application logs.

    The generic email tool predates delivery idempotency and does not expose an
    accepted-vs-ambiguous result.  This narrow adapter reuses its authoritative
    deploy-time Resend configuration while preserving the provider key required
    for a durable outbox.
    """

    _URL = "https://api.resend.com/emails"

    def __init__(
        self,
        *,
        timeout_seconds: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        # Production uses httpx's normal transport. The optional injected
        # transport is deliberately narrow test plumbing so this exact adapter
        # (headers and result classification included) can be proven without a
        # real provider call.
        self._transport = transport

    async def send(
        self,
        *,
        recipient_email: str,
        subject: str,
        body: str,
        idempotency_key: str,
    ) -> str:
        from ..config import settings

        email_config = settings.email
        api_key = (email_config.api_key or "").strip()
        if not email_config.enabled or not api_key:
            raise _DefiniteDeliveryError("email_not_configured", retryable=False)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        }
        payload = {
            "from": f"{BUSINESS_NAME} <{BUSINESS_EMAIL}>",
            "to": [recipient_email],
            "subject": subject,
            "text": body,
            "reply_to": BUSINESS_EMAIL,
        }
        try:
            async with httpx.AsyncClient(
                timeout=float(self._timeout_seconds), transport=self._transport
            ) as client:
                response = await client.post(self._URL, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
            raise _AmbiguousDeliveryError("resend_transport_unknown") from exc

        if 200 <= response.status_code < 300:
            try:
                payload = response.json()
            except ValueError as exc:
                raise _AmbiguousDeliveryError("resend_success_body_unknown") from exc
            message_id = payload.get("id") if isinstance(payload, Mapping) else None
            if not isinstance(message_id, str) or not message_id.strip():
                raise _AmbiguousDeliveryError("resend_success_identity_unknown")
            return message_id.strip()

        provider_code = ""
        try:
            error_payload = response.json()
            if isinstance(error_payload, Mapping):
                raw = error_payload.get("name") or error_payload.get("code")
                if isinstance(raw, str):
                    provider_code = raw.strip().casefold()
        except ValueError:
            pass
        if response.status_code == 409:
            if provider_code == "concurrent_idempotent_requests":
                raise _DefiniteDeliveryError(
                    "resend_request_in_progress",
                    retryable=True,
                    recovery_required_if_exhausted=True,
                )
            # An idempotency conflict can mean another in-flight request
            # accepted this exact key. It is not proof that Resend rejected
            # mail, so leave recovery evidence instead of inviting an unsafe
            # manual resend from a false terminal "failed" state.
            raise _AmbiguousDeliveryError("resend_idempotency_conflict_unknown")
        if response.status_code in (408, 429) or response.status_code >= 500:
            # The stable provider key keeps the retry safe while it remains in
            # Resend's retention window.
            raise _DefiniteDeliveryError("resend_transient_rejection", retryable=True)
        if response.status_code in (400, 401, 403, 404, 409, 422):
            raise _DefiniteDeliveryError("resend_rejected", retryable=False)
        raise _AmbiguousDeliveryError("resend_unclassified_response")


@dataclass(frozen=True)
class _Eligibility:
    eligible: bool
    reason: str | None
    recipient_email: str | None
    full_name: str | None


@dataclass(frozen=True)
class _ClaimedStep:
    step_id: UUID
    sequence_id: UUID
    contact_id: UUID
    claim_token: UUID
    recipient_email: str
    subject: str
    body: str
    provider_idempotency_key: str
    provider_key_expires_at: datetime
    attempt_count: int


@dataclass(frozen=True)
class _ClaimResult:
    """One dispatch iteration's durable claim outcome."""

    processed: bool
    claim: _ClaimedStep | None = None


@dataclass(frozen=True)
class _SentHistory:
    contact_id: UUID
    recipient_email: str
    subject: str
    body: str
    provider_message_id: str
    sequence_id: UUID
    step_id: UUID


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_operation_key(value: object) -> str:
    if not isinstance(value, str):
        raise EOMMissedCallRecoveryValidationError("Idempotency key is required")
    value = value.strip()
    if not _OPERATION_KEY_PATTERN.fullmatch(value):
        raise EOMMissedCallRecoveryValidationError("Idempotency key is invalid")
    return value


def _safe_actor(actor_id: object, actor_name: object) -> tuple[int, str]:
    if isinstance(actor_id, bool):
        raise EOMMissedCallRecoveryValidationError("Authenticated actor is invalid")
    try:
        parsed_id = int(actor_id)
    except (TypeError, ValueError) as exc:
        raise EOMMissedCallRecoveryValidationError(
            "Authenticated actor is invalid"
        ) from exc
    if parsed_id <= 0:
        raise EOMMissedCallRecoveryValidationError("Authenticated actor is invalid")
    if not isinstance(actor_name, str):
        raise EOMMissedCallRecoveryValidationError("Authenticated actor is invalid")
    parsed_name = actor_name.strip()
    if (
        not parsed_name
        or len(parsed_name) > _MAX_ACTOR_NAME_LENGTH
        or "\x00" in parsed_name
    ):
        raise EOMMissedCallRecoveryValidationError("Authenticated actor is invalid")
    return parsed_id, parsed_name


def _uuid(value: object, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EOMMissedCallRecoveryValidationError(f"{field} is invalid") from exc


def _datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EOMMissedCallRecoveryConflictError(
            f"Missed-call recovery {field} is invalid"
        )
    return value


def _normalize_recipient(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > 254
        or not _EMAIL_PATTERN.fullmatch(normalized)
    ):
        return None
    return normalized


def next_business_day_due(
    from_time: datetime,
    *,
    timezone_name: str,
) -> datetime:
    """Return 09:00 local on the next Monday-Friday date.

    The configured zone—not the worker machine zone—owns the policy.  EOM has
    no holiday calendar or office-hours configuration today, so weekday is the
    explicit, tested initial business-day definition.
    """

    current = _datetime(from_time, "due time")
    try:
        zone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise EOMMissedCallRecoveryValidationError(
            "Missed-call recovery time zone is invalid"
        ) from exc
    local = current.astimezone(zone)
    candidate = local.date() + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return datetime(
        candidate.year,
        candidate.month,
        candidate.day,
        9,
        0,
        tzinfo=zone,
    ).astimezone(timezone.utc)


def _third_step_due(second_step_due: datetime, *, timezone_name: str) -> datetime:
    """Return the same configured local clock time three calendar days later.

    Calendar-day follow-up is an EOM business rule, not a raw 72-hour duration.
    Converting back through the configured zone means a Friday 09:00 message
    remains a Monday 09:00 message when a daylight-saving boundary sits between
    the two dates.
    """

    current = _datetime(second_step_due, "second step due")
    try:
        zone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise EOMMissedCallRecoveryValidationError(
            "Missed-call recovery time zone is invalid"
        ) from exc
    return (current.astimezone(zone) + timedelta(days=3)).astimezone(timezone.utc)


async def missed_call_recovery_schema_ready(pool: Any) -> bool:
    """Return whether the additive provider schema is safe to serve."""

    try:
        return bool(
            await pool.fetchval(
                """
                SELECT to_regclass('eom_missed_call_attempts') IS NOT NULL
                   AND to_regclass('eom_missed_call_operation_receipts') IS NOT NULL
                   AND to_regclass('eom_missed_call_contact_suppressions') IS NOT NULL
                   AND to_regclass('eom_missed_call_sequences') IS NOT NULL
                   AND to_regclass('eom_missed_call_sequence_steps') IS NOT NULL
                   AND to_regclass('eom_missed_call_sequence_events') IS NOT NULL
                   AND (
                       SELECT COUNT(*) = 6
                          AND BOOL_AND(
                              table_owner.rolname = 'atlas_eom_handoff_owner'
                              AND has_table_privilege(current_user, relation.oid, 'SELECT')
                              AND has_table_privilege(current_user, relation.oid, 'INSERT')
                              AND (
                                  (
                                      required_relation.requires_update
                                      AND has_table_privilege(
                                          current_user, relation.oid, 'UPDATE'
                                      )
                                  )
                                  OR (
                                      NOT required_relation.requires_update
                                      AND NOT has_table_privilege(
                                          current_user, relation.oid, 'UPDATE'
                                      )
                                      AND NOT has_any_column_privilege(
                                          current_user, relation.oid, 'UPDATE'
                                      )
                                  )
                              )
                              AND NOT has_table_privilege(
                                  current_user, relation.oid, 'DELETE'
                              )
                              AND NOT has_table_privilege(
                                  current_user, relation.oid, 'TRUNCATE'
                              )
                              AND NOT has_table_privilege(
                                  current_user, relation.oid, 'REFERENCES'
                              )
                              AND NOT has_any_column_privilege(
                                  current_user, relation.oid, 'REFERENCES'
                              )
                              AND NOT has_table_privilege(
                                  current_user, relation.oid, 'TRIGGER'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'SELECT'
                              )
                              AND NOT has_any_column_privilege(
                                  'atlas_nocodb', relation.oid, 'SELECT'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'INSERT'
                              )
                              AND NOT has_any_column_privilege(
                                  'atlas_nocodb', relation.oid, 'INSERT'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'UPDATE'
                              )
                              AND NOT has_any_column_privilege(
                                  'atlas_nocodb', relation.oid, 'UPDATE'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'DELETE'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'TRUNCATE'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'REFERENCES'
                              )
                              AND NOT has_any_column_privilege(
                                  'atlas_nocodb', relation.oid, 'REFERENCES'
                              )
                              AND NOT has_table_privilege(
                                  'atlas_nocodb', relation.oid, 'TRIGGER'
                              )
                          )
                       FROM (
                           VALUES
                               ('eom_missed_call_operation_receipts', TRUE),
                               ('eom_missed_call_attempts', TRUE),
                               ('eom_missed_call_contact_suppressions', FALSE),
                               ('eom_missed_call_sequences', TRUE),
                               ('eom_missed_call_sequence_steps', TRUE),
                               ('eom_missed_call_sequence_events', FALSE)
                       ) AS required_relation(relation_name, requires_update)
                       JOIN pg_class AS relation
                         ON relation.oid = to_regclass(
                             format(
                                 '%I.%I', current_schema(),
                                 required_relation.relation_name
                             )
                         )
                       JOIN pg_namespace AS namespace
                         ON namespace.oid = relation.relnamespace
                      JOIN pg_roles AS table_owner
                         ON table_owner.oid = relation.relowner
                      WHERE namespace.nspname = current_schema()
                   )
                   AND EXISTS (
                       SELECT 1
                       FROM pg_roles AS guard_role
                       WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
                         AND NOT guard_role.rolcanlogin
                         AND NOT guard_role.rolinherit
                         AND NOT guard_role.rolsuper
                         AND NOT guard_role.rolcreaterole
                         AND NOT guard_role.rolcreatedb
                         AND NOT guard_role.rolreplication
                         AND NOT guard_role.rolbypassrls
                         AND has_schema_privilege(
                             guard_role.oid, current_schema(), 'USAGE'
                         )
                         AND has_schema_privilege(
                             guard_role.oid, current_schema(), 'CREATE'
                         )
                         AND NOT EXISTS (
                             SELECT 1
                             FROM pg_roles AS member_role
                             WHERE member_role.rolcanlogin
                               AND NOT member_role.rolsuper
                               AND pg_has_role(
                                   member_role.oid, guard_role.oid, 'MEMBER'
                               )
                         )
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_index AS idx
                       JOIN pg_class AS rel ON rel.oid = idx.indexrelid
                       WHERE idx.indrelid = 'eom_missed_call_sequences'::regclass
                         AND rel.relname = 'uq_eom_missed_call_sequences_active_contact'
                         AND idx.indisunique
                         AND idx.indpred IS NOT NULL
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_attribute AS attribute
                       WHERE attribute.attrelid = 'eom_missed_call_sequence_steps'::regclass
                         AND attribute.attname = 'claim_token'
                         AND NOT attribute.attisdropped
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_trigger AS trigger
                       WHERE trigger.tgrelid = 'eom_missed_call_operation_receipts'::regclass
                         AND trigger.tgname = 'trg_prevent_eom_missed_call_operation_receipt_mutation'
                         AND trigger.tgfoid = to_regprocedure(
                             'prevent_eom_missed_call_operation_receipt_mutation()'
                         )
                         AND trigger.tgtype = 27
                         AND trigger.tgenabled = 'O'
                         AND NOT trigger.tgisinternal
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_trigger AS trigger
                       WHERE trigger.tgrelid = 'eom_missed_call_attempts'::regclass
                         AND trigger.tgname = 'trg_prevent_eom_missed_call_attempt_mutation'
                         AND trigger.tgfoid = to_regprocedure(
                             'prevent_eom_missed_call_attempt_mutation()'
                         )
                         AND trigger.tgtype = 27
                         AND trigger.tgenabled = 'O'
                         AND NOT trigger.tgisinternal
                   )
                   AND (
                       SELECT COUNT(*) = 2
                          AND BOOL_AND(
                              language_state.lanname = 'plpgsql'
                              AND procedure.prosrc = expected_function.body
                          )
                       FROM (
                           VALUES
                               (
                                   'prevent_eom_missed_call_operation_receipt_mutation()',
                                   E'\\nBEGIN\\n    RAISE EXCEPTION ''eom_missed_call_operation_receipts is append-only'';\\nEND;\\n'
                               ),
                               (
                                   'prevent_eom_missed_call_attempt_mutation()',
                                   E'\\nBEGIN\\n    RAISE EXCEPTION ''eom_missed_call_attempts is append-only'';\\nEND;\\n'
                               )
                       ) AS expected_function(signature, body)
                       JOIN pg_proc AS procedure
                         ON procedure.oid = to_regprocedure(
                             expected_function.signature
                         )
                       JOIN pg_language AS language_state
                         ON language_state.oid = procedure.prolang
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_trigger AS trigger
                       WHERE trigger.tgrelid = 'contacts'::regclass
                         AND trigger.tgname = 'trg_cancel_eom_missed_call_on_contact_change'
                         AND trigger.tgfoid = to_regprocedure(
                             'cancel_eom_missed_call_on_contact_change()'
                         )
                         AND trigger.tgtype = 17
                         AND trigger.tgenabled = 'O'
                         AND NOT trigger.tgisinternal
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_trigger AS trigger
                       WHERE trigger.tgrelid = 'contact_interactions'::regclass
                         AND trigger.tgname = 'trg_lock_eom_missed_call_interaction_contact'
                         AND trigger.tgfoid = to_regprocedure(
                             'lock_eom_missed_call_interaction_contact()'
                         )
                         AND trigger.tgtype = 31
                         AND trigger.tgenabled = 'O'
                         AND NOT trigger.tgisinternal
                   )
                   AND EXISTS (
                       SELECT 1 FROM pg_trigger AS trigger
                       WHERE trigger.tgrelid = 'contact_interactions'::regclass
                         AND trigger.tgname = 'trg_cancel_eom_missed_call_on_interaction'
                         AND trigger.tgfoid = to_regprocedure(
                             'cancel_eom_missed_call_on_interaction()'
                         )
                         AND trigger.tgtype = 29
                         AND trigger.tgenabled = 'O'
                         AND NOT trigger.tgisinternal
                   )
                   AND (
                       SELECT COUNT(*) = 6
                          AND BOOL_AND(
                              procedure.prosecdef
                              AND function_owner.rolname = 'atlas_eom_handoff_owner'
                              AND COALESCE(
                                  procedure.proconfig @> ARRAY[
                                      format(
                                          'search_path=pg_catalog, %I, pg_temp',
                                          current_schema()
                                      )
                                  ],
                                  FALSE
                              )
                              AND NOT has_function_privilege(
                                  'atlas_nocodb', procedure.oid, 'EXECUTE'
                              )
                          )
                       FROM pg_proc AS procedure
                       JOIN pg_namespace AS function_namespace
                         ON function_namespace.oid = procedure.pronamespace
                       JOIN pg_roles AS function_owner
                         ON function_owner.oid = procedure.proowner
                      WHERE function_namespace.nspname = current_schema()
                        AND procedure.proname = ANY (ARRAY[
                            'cancel_eom_missed_call_sequences_for_contact',
                            'lock_eom_missed_call_interaction_contact',
                            'eom_missed_call_effective_recipient',
                            'cancel_eom_missed_call_on_recipient_change',
                            'cancel_eom_missed_call_on_contact_change',
                            'cancel_eom_missed_call_on_interaction'
                        ])
                   )
                   AND to_regprocedure(
                       'eom_missed_call_has_proven_inbound_sms(jsonb)'
                   ) IS NOT NULL
                """
            )
        )
    except Exception:
        return False


class EOMMissedCallRecoveryService:
    """Owns the EOM call evidence, eligibility, outbox, and worker state."""

    def __init__(
        self,
        *,
        pool: Any,
        config: EOMFunnelConfig | None = None,
        gateway: _RecoveryEmailGateway | None = None,
        now: Callable[[], datetime] | None = None,
        email_history: Any | None = None,
    ) -> None:
        self._pool = pool
        self._config = config or funnel_settings
        self._gateway = gateway
        self._now = now or _now_utc
        self._email_history = email_history

    @property
    def pool(self) -> Any:
        if not bool(getattr(self._pool, "is_initialized", True)):
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery database is unavailable"
            )
        return self._pool

    @property
    def gateway(self) -> _RecoveryEmailGateway:
        if self._gateway is not None:
            return self._gateway
        return ResendMissedCallRecoveryGateway(
            timeout_seconds=self._config.missed_call_delivery_timeout_seconds
        )

    async def require_schema_ready(self) -> None:
        if not await missed_call_recovery_schema_ready(self.pool):
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery schema is unavailable"
            )

    def delivery_block_reason(self) -> str | None:
        """Return why this service must not render or send customer mail.

        The sequence state records the reason instead of discovering it only
        after a due worker has claimed a customer email. Tests supply a fake
        gateway deliberately, which is a configured transport seam rather than
        permission to consult a real Resend credential.
        """

        configured_reason = self._config.missed_call_recovery_delivery_block_reason
        if configured_reason is not None:
            return configured_reason
        if self._gateway is not None:
            return None
        from ..config import settings

        email_config = settings.email
        if not email_config.enabled or not (email_config.api_key or "").strip():
            return "email_transport_unavailable"
        return None

    async def _bind_operation_receipt(
        self,
        conn: Any,
        *,
        contact_id: UUID,
        operation_key: str,
        operation_kind: str,
        fingerprint: str,
    ) -> bool:
        """Durably bind one browser operation key before mutating a lead.

        ``True`` means the exact same completed operation owns the key and the
        caller must return its current authoritative status. A different lead,
        mutation kind, or request fingerprint is a conflict, never a second
        action. ``ON CONFLICT DO NOTHING`` keeps two concurrent contacts from
        poisoning their transactions with a unique-violation before the loser
        can inspect the winning receipt.
        """

        if operation_kind not in _OPERATION_KINDS:
            raise RuntimeError("Unsupported missed-call recovery operation kind")

        existing = await conn.fetchrow(
            """
            SELECT contact_id, operation_kind, request_fingerprint
            FROM eom_missed_call_operation_receipts
            WHERE operation_key = $1
            FOR UPDATE
            """,
            operation_key,
        )
        if existing is None:
            inserted = await conn.fetchrow(
                """
                INSERT INTO eom_missed_call_operation_receipts (
                    operation_key, contact_id, operation_kind, request_fingerprint
                ) VALUES ($1, $2, $3, $4)
                ON CONFLICT (operation_key) DO NOTHING
                RETURNING operation_key
                """,
                operation_key,
                contact_id,
                operation_kind,
                fingerprint,
            )
            if inserted is not None:
                return False
            # A concurrent transaction just committed the winning receipt.
            # Re-read it under lock before deciding whether this is a replay or
            # a cross-contact/key-reuse conflict.
            existing = await conn.fetchrow(
                """
                SELECT contact_id, operation_kind, request_fingerprint
                FROM eom_missed_call_operation_receipts
                WHERE operation_key = $1
                FOR UPDATE
                """,
                operation_key,
            )
            if existing is None:
                raise EOMMissedCallRecoveryUnavailableError(
                    "Missed-call recovery operation receipt is unavailable"
                )

        if (
            _uuid(existing["contact_id"], "Receipt contact id") != contact_id
            or existing["operation_kind"] != operation_kind
            or existing["request_fingerprint"] != fingerprint
        ):
            raise EOMMissedCallRecoveryConflictError(
                "Idempotency key belongs to a different missed-call recovery operation"
            )
        return True

    async def block_active_sequences_for_configuration(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        """Persist a deployment pause before a worker can quiesce delivery.

        A configuration change can happen while a sequence has pending steps.
        Those rows must become visibly blocked and require an explicit resume
        after configuration is restored; silently retaining ``active`` would
        otherwise let a restart send overdue messages automatically.
        """

        block_reason = reason or self.delivery_block_reason()
        if block_reason is None:
            return 0
        try:
            async with self.pool.transaction() as conn:
                now = self._now()
                blocked = await conn.fetch(
                    """
                    UPDATE eom_missed_call_sequences
                       SET state = 'blocked_configuration', blocked_reason = $1,
                           updated_at = $2
                     WHERE state = 'active'
                     RETURNING id
                    """,
                    block_reason,
                    now,
                )
                for row in blocked:
                    await self._event(
                        conn,
                        sequence_id=row["id"],
                        event_type="sequence_blocked",
                        reason_code=block_reason,
                        actor_id=None,
                        actor_name="system",
                        source="worker",
                        metadata={"configuration_block": True},
                        occurred_at=now,
                    )
                return len(blocked)
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery configuration block could not be recorded"
            ) from exc

    async def record_no_answer(
        self,
        *,
        contact_id: UUID,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Record one real operator no-answer and create/reuse its sequence."""

        contact_id = _uuid(contact_id, "Contact id")
        operation_key = _safe_operation_key(operation_key)
        actor_id, actor_name = _safe_actor(actor_id, actor_name)
        fingerprint = _fingerprint(
            {"contactId": str(contact_id), "operation": "no_answer"}
        )
        try:
            async with self.pool.transaction() as conn:
                contact = await self._contact_for_update(conn, contact_id)
                replayed = await self._bind_operation_receipt(
                    conn,
                    contact_id=contact_id,
                    operation_key=operation_key,
                    operation_kind="no_answer",
                    fingerprint=fingerprint,
                )
                if replayed:
                    previous = await conn.fetchrow(
                        """
                        SELECT id FROM eom_missed_call_attempts
                        WHERE operation_key = $1
                        FOR UPDATE
                        """,
                        operation_key,
                    )
                    if previous is None:
                        raise EOMMissedCallRecoveryUnavailableError(
                            "Missed-call recovery operation evidence is unavailable"
                        )
                    status = await self._status_for_contact(conn, contact_id)
                    return {
                        "attemptId": str(previous["id"]),
                        "idempotent": True,
                        "sequence": status,
                    }

                eligibility = await self._evaluate_contact_eligibility(
                    conn, contact, sequence_created_at=None, recipient_snapshot=None
                )
                now = self._now()
                attempt_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO eom_missed_call_attempts (
                        id, contact_id, operation_key, request_fingerprint,
                        actor_id, actor_name, source, occurred_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'time_tracker', $7)
                    """,
                    attempt_id,
                    contact_id,
                    operation_key,
                    fingerprint,
                    actor_id,
                    actor_name,
                    now,
                )
                await conn.execute(
                    """
                    INSERT INTO contact_interactions (
                        id, contact_id, interaction_type, summary, intent,
                        occurred_at, metadata, interaction_dedupe_key
                    ) VALUES ($1, $2, 'call',
                              'Operator recorded unanswered call', 'no_answer',
                              $3, $4::jsonb, $5)
                    """,
                    uuid4(),
                    contact_id,
                    now,
                    json.dumps(
                        {
                            "missed_call_attempt_id": str(attempt_id),
                            "operation_key": operation_key,
                            "actor_id": actor_id,
                        }
                    ),
                    _fingerprint(
                        {
                            "contactId": str(contact_id),
                            "operationKey": operation_key,
                            "interaction": "no_answer",
                        }
                    ),
                )
                await conn.execute(
                    """
                    INSERT INTO eom_lead_lifecycle_events (
                        contact_id, event_type, from_stage, to_stage, actor,
                        source, operation_key, metadata, occurred_at
                    ) VALUES ($1, 'call_attempt_no_answer', $2, $2, $3,
                              'time_tracker', $4, $5::jsonb, $6)
                    """,
                    contact_id,
                    contact["lead_stage"],
                    actor_name,
                    operation_key,
                    json.dumps({"attempt_id": str(attempt_id), "actor_id": actor_id}),
                    now,
                )

                active = await conn.fetchrow(
                    """
                    SELECT * FROM eom_missed_call_sequences
                    WHERE contact_id = $1
                      AND state IN ('active', 'blocked_configuration')
                    FOR UPDATE
                    """,
                    contact_id,
                )
                if active is not None:
                    # The database trigger normally terminalizes a sequence as
                    # soon as lead evidence changes. Recheck here too so a
                    # historical/manual data repair that bypassed that trigger
                    # cannot make a stale sequence reusable.
                    if not eligibility.eligible:
                        await self._cancel_for_reason(
                            conn,
                            sequence_id=active["id"],
                            reason=eligibility.reason or "ineligible",
                            source="time_tracker",
                            actor_id=actor_id,
                            actor_name=actor_name,
                        )
                        return {
                            "attemptId": str(attempt_id),
                            "idempotent": False,
                            "sequence": await self._status_for_contact(
                                conn, contact_id
                            ),
                        }
                    await self._event(
                        conn,
                        sequence_id=active["id"],
                        event_type="sequence_reused",
                        reason_code="active_sequence_exists",
                        actor_id=actor_id,
                        actor_name=actor_name,
                        source="time_tracker",
                        metadata={"attempt_id": str(attempt_id)},
                        occurred_at=now,
                    )
                    return {
                        "attemptId": str(attempt_id),
                        "idempotent": False,
                        "sequence": await self._status_for_contact(conn, contact_id),
                    }

                if not eligibility.eligible:
                    return {
                        "attemptId": str(attempt_id),
                        "idempotent": False,
                        "sequence": None,
                        "notStartedReason": eligibility.reason,
                    }

                sequence_id = uuid4()
                recipient = eligibility.recipient_email
                if recipient is None or eligibility.full_name is None:
                    raise EOMMissedCallRecoveryConflictError(
                        "Missed-call recovery eligibility is incomplete"
                    )
                delivery_block_reason = self.delivery_block_reason()
                if delivery_block_reason is not None:
                    await conn.execute(
                        """
                        INSERT INTO eom_missed_call_sequences (
                            id, contact_id, initiating_attempt_id, recipient_email,
                            state, blocked_reason, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, 'blocked_configuration',
                                  $5, $6, $6)
                        """,
                        sequence_id,
                        contact_id,
                        attempt_id,
                        recipient,
                        delivery_block_reason,
                        now,
                    )
                    await self._event(
                        conn,
                        sequence_id=sequence_id,
                        event_type="sequence_blocked",
                        reason_code=delivery_block_reason,
                        actor_id=actor_id,
                        actor_name=actor_name,
                        source="time_tracker",
                        metadata={},
                        occurred_at=now,
                    )
                else:
                    await conn.execute(
                        """
                        INSERT INTO eom_missed_call_sequences (
                            id, contact_id, initiating_attempt_id, recipient_email,
                            state, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, 'active', $5, $5)
                        """,
                        sequence_id,
                        contact_id,
                        attempt_id,
                        recipient,
                        now,
                    )
                    await self._insert_steps(
                        conn,
                        sequence_id=sequence_id,
                        full_name=eligibility.full_name,
                        now=now,
                    )
                    await self._event(
                        conn,
                        sequence_id=sequence_id,
                        event_type="sequence_started",
                        reason_code=None,
                        actor_id=actor_id,
                        actor_name=actor_name,
                        source="time_tracker",
                        metadata={},
                        occurred_at=now,
                    )
                return {
                    "attemptId": str(attempt_id),
                    "idempotent": False,
                    "sequence": await self._status_for_contact(conn, contact_id),
                }
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery could not be recorded"
            ) from exc

    async def resume_blocked_sequence(
        self,
        *,
        contact_id: UUID,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Explicitly recover a sequence blocked by missing delivery config."""

        contact_id = _uuid(contact_id, "Contact id")
        operation_key = _safe_operation_key(operation_key)
        actor_id, actor_name = _safe_actor(actor_id, actor_name)
        fingerprint = _fingerprint(
            {"contactId": str(contact_id), "operation": "resume"}
        )
        try:
            async with self.pool.transaction() as conn:
                contact = await self._contact_for_update(conn, contact_id)
                replayed = await self._bind_operation_receipt(
                    conn,
                    contact_id=contact_id,
                    operation_key=operation_key,
                    operation_kind="resume",
                    fingerprint=fingerprint,
                )
                if replayed:
                    return {
                        "idempotent": True,
                        "sequence": await self._status_for_contact(conn, contact_id),
                    }
                delivery_block_reason = self.delivery_block_reason()
                if delivery_block_reason is not None:
                    raise EOMMissedCallRecoveryConflictError(
                        "Missed-call recovery delivery is not configured"
                    )
                sequence = await conn.fetchrow(
                    """
                    SELECT * FROM eom_missed_call_sequences
                    WHERE contact_id = $1 AND state = 'blocked_configuration'
                    FOR UPDATE
                    """,
                    contact_id,
                )
                if sequence is None:
                    raise EOMMissedCallRecoveryConflictError(
                        "No blocked missed-call recovery sequence exists"
                    )
                eligibility = await self._evaluate_contact_eligibility(
                    conn,
                    contact,
                    sequence_created_at=sequence["created_at"],
                    recipient_snapshot=sequence["recipient_email"],
                )
                if not eligibility.eligible or eligibility.full_name is None:
                    await self._cancel_for_reason(
                        conn,
                        sequence_id=sequence["id"],
                        reason=eligibility.reason or "ineligible",
                        source="time_tracker",
                        actor_id=actor_id,
                        actor_name=actor_name,
                    )
                    return {
                        "idempotent": False,
                        "sequence": await self._status_for_contact(conn, contact_id),
                    }
                now = self._now()
                await conn.execute(
                    """
                    UPDATE eom_missed_call_sequences
                       SET state = 'active', blocked_reason = NULL, updated_at = $2
                     WHERE id = $1 AND state = 'blocked_configuration'
                    """,
                    sequence["id"],
                    now,
                )
                await self._insert_steps(
                    conn,
                    sequence_id=sequence["id"],
                    full_name=eligibility.full_name,
                    now=now,
                )
                await self._event(
                    conn,
                    sequence_id=sequence["id"],
                    event_type="sequence_resumed",
                    reason_code=None,
                    actor_id=actor_id,
                    actor_name=actor_name,
                    source="time_tracker",
                    metadata={},
                    occurred_at=now,
                )
                await conn.execute(
                    """
                    INSERT INTO eom_lead_lifecycle_events (
                        contact_id, event_type, from_stage, to_stage, actor,
                        source, operation_key, metadata, occurred_at
                    ) VALUES ($1, 'missed_call_recovery_resumed', $2, $2, $3,
                              'time_tracker', $4, $5::jsonb, $6)
                    """,
                    contact_id,
                    contact["lead_stage"],
                    actor_name,
                    operation_key,
                    json.dumps(
                        {"sequence_id": str(sequence["id"]), "actor_id": actor_id}
                    ),
                    now,
                )
                return {
                    "idempotent": False,
                    "sequence": await self._status_for_contact(conn, contact_id),
                }
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery could not be resumed"
            ) from exc

    async def cancel_sequence(
        self,
        *,
        contact_id: UUID,
        operation_key: str,
        actor_id: int,
        actor_name: str,
        reason: str,
    ) -> dict[str, Any]:
        """Record a manually verified stop condition without rewriting history."""

        contact_id = _uuid(contact_id, "Contact id")
        operation_key = _safe_operation_key(operation_key)
        actor_id, actor_name = _safe_actor(actor_id, actor_name)
        if reason not in _MANUAL_CANCEL_REASONS:
            raise EOMMissedCallRecoveryValidationError("Cancellation reason is invalid")
        fingerprint = _fingerprint(
            {
                "contactId": str(contact_id),
                "operation": "cancel",
                "reason": reason,
            }
        )
        interaction_type = {
            "callback_recorded": "callback_completed",
            "response_recorded": "lead_response",
            "opt_out": "opt_out",
            "manual": "lead_response",
        }[reason]
        try:
            async with self.pool.transaction() as conn:
                contact = await self._contact_for_update(conn, contact_id)
                replayed = await self._bind_operation_receipt(
                    conn,
                    contact_id=contact_id,
                    operation_key=operation_key,
                    operation_kind="cancel",
                    fingerprint=fingerprint,
                )
                if replayed:
                    return {
                        "idempotent": True,
                        "sequence": await self._status_for_contact(conn, contact_id),
                    }
                now = self._now()
                # Cancel through the service before recording the standard CRM
                # interaction. Both rows commit together, but this preserves the
                # authenticated actor in the sequence-event ledger; the later
                # interaction trigger then sees no current sequence to cancel.
                current_sequence = await conn.fetchrow(
                    """
                    SELECT id FROM eom_missed_call_sequences
                    WHERE contact_id = $1
                      AND state IN ('active', 'blocked_configuration')
                    FOR UPDATE
                    """,
                    contact_id,
                )
                if current_sequence is not None:
                    await self._cancel_for_reason(
                        conn,
                        sequence_id=current_sequence["id"],
                        reason=reason,
                        source="time_tracker",
                        actor_id=actor_id,
                        actor_name=actor_name,
                    )
                await conn.execute(
                    """
                    INSERT INTO contact_interactions (
                        id, contact_id, interaction_type, summary, intent,
                        occurred_at, metadata, interaction_dedupe_key
                    ) VALUES ($1, $2, $3, 'Operator recorded recovery stop condition',
                              'missed_call_recovery', $4, $5::jsonb, $6)
                    """,
                    uuid4(),
                    contact_id,
                    interaction_type,
                    now,
                    json.dumps(
                        {
                            "reason": reason,
                            "actor_id": actor_id,
                            "missed_call_recovery_cancel_reason": reason,
                        }
                    ),
                    _fingerprint(
                        {
                            "contactId": str(contact_id),
                            "operationKey": operation_key,
                            "interaction": interaction_type,
                        }
                    ),
                )
                if reason == "opt_out":
                    await conn.execute(
                        """
                        INSERT INTO eom_missed_call_contact_suppressions (
                            contact_id, reason_code, actor_id, actor_name, source
                        ) VALUES ($1, 'opt_out', $2, $3, 'time_tracker')
                        ON CONFLICT (contact_id) DO NOTHING
                        """,
                        contact_id,
                        actor_id,
                        actor_name,
                    )
                await conn.execute(
                    """
                    INSERT INTO eom_lead_lifecycle_events (
                        contact_id, event_type, from_stage, to_stage, actor,
                        source, operation_key, metadata, occurred_at
                    ) VALUES ($1, 'missed_call_recovery_cancelled', $2, $2, $3,
                              'time_tracker', $4, $5::jsonb, $6)
                    """,
                    contact_id,
                    contact["lead_stage"],
                    actor_name,
                    operation_key,
                    json.dumps({"reason": reason, "actor_id": actor_id}),
                    now,
                )
                return {
                    "idempotent": False,
                    "sequence": await self._status_for_contact(conn, contact_id),
                }
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery could not be cancelled"
            ) from exc

    async def statuses(self, *, contact_ids: Sequence[UUID]) -> list[dict[str, Any]]:
        """Return bounded, non-PII sequence status for known EOM contact ids."""

        unique_ids = list(
            dict.fromkeys(_uuid(value, "Contact id") for value in contact_ids)
        )
        if not unique_ids:
            raise EOMMissedCallRecoveryValidationError(
                "At least one contact id is required"
            )
        if len(unique_ids) > _MAX_STATUS_IDS:
            raise EOMMissedCallRecoveryValidationError("Too many contact ids")
        try:
            rows = await self.pool.fetch(
                """
                SELECT
                    s.contact_id, s.id AS sequence_id, s.state, s.blocked_reason,
                    s.cancellation_reason, s.created_at, s.terminal_at,
                    next_step.step_number AS next_step_number,
                    next_step.due_at AS next_due_at,
                    last_event.event_type AS last_event_type,
                    last_event.reason_code AS last_event_reason
                FROM eom_missed_call_sequences AS s
                JOIN contacts AS c ON c.id = s.contact_id
                LEFT JOIN LATERAL (
                    SELECT step_number, COALESCE(next_attempt_at, due_at) AS due_at
                    FROM eom_missed_call_sequence_steps
                    WHERE sequence_id = s.id
                      AND state IN ('pending', 'attempting')
                    ORDER BY step_number ASC
                    LIMIT 1
                ) AS next_step ON TRUE
                LEFT JOIN LATERAL (
                    SELECT event_type, reason_code
                    FROM eom_missed_call_sequence_events
                    WHERE sequence_id = s.id
                    ORDER BY occurred_at DESC, id DESC
                    LIMIT 1
                ) AS last_event ON TRUE
                WHERE s.contact_id = ANY($1::uuid[])
                  AND c.business_context_id = $2
                ORDER BY s.created_at DESC, s.id DESC
                """,
                unique_ids,
                _EOM_CONTEXT,
            )
            newest: dict[UUID, dict[str, Any]] = {}
            for row in rows:
                contact_id = _uuid(row["contact_id"], "Contact id")
                newest.setdefault(contact_id, self._serialize_status(row))
            return [newest[value] for value in unique_ids if value in newest]
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery status is unavailable"
            ) from exc

    async def dispatch_due_steps(self, *, limit: int = _MAX_WORK_BATCH) -> int:
        """Recover stale claims then deliver a bounded number of due steps."""

        if limit < 1 or limit > _MAX_WORK_BATCH:
            raise EOMMissedCallRecoveryValidationError("Worker limit is invalid")
        delivery_block_reason = self.delivery_block_reason()
        if delivery_block_reason is not None:
            await self.block_active_sequences_for_configuration(
                reason=delivery_block_reason
            )
            return 0
        await self._recover_stale_claims(limit=limit)
        # This return value represents rows that crossed the durable claim
        # boundary and entered the delivery phase (including a definite
        # provider rejection that scheduled a retry). A pending row that is
        # cancelled by the first authoritative eligibility read is useful
        # worker progress, but it is not a delivery attempt.
        delivery_attempts = 0
        for _ in range(limit):
            claim_result = await self._claim_one_due_step()
            if not claim_result.processed:
                break
            if claim_result.claim is None:
                continue
            delivery_attempts += 1
            history = await self._deliver_claim(claim_result.claim)
            if history is not None and history.provider_message_id:
                await self._write_sent_history(history)
        return delivery_attempts

    async def _contact_for_update(
        self,
        conn: Any,
        contact_id: UUID,
        *,
        allow_non_eom: bool = False,
    ) -> Mapping[str, Any]:
        """Lock one canonical contact before reading delivery evidence.

        Worker delivery uses ``allow_non_eom`` so a contact that was reassigned
        after its EOM sequence began can be terminalized rather than being
        mistaken for a missing row. Operator routes retain the stricter EOM
        lookup and therefore never mutate a lead outside this tenant.
        """

        locked = await conn.fetchrow(
            """
            SELECT id
            FROM contacts
            WHERE id = $1
              AND (business_context_id = $2 OR $3::boolean)
            FOR UPDATE
            """,
            contact_id,
            _EOM_CONTEXT,
            allow_non_eom,
        )
        if locked is None:
            raise EOMMissedCallRecoveryNotFoundError("EOM lead was not found")

        # This must be a new statement after the row lock. Under PostgreSQL's
        # read-committed isolation a statement snapshot can predate waiting on
        # a ``FOR UPDATE`` lock held by an interaction correction. Reading the
        # latest form evidence only after that wait gives the worker the
        # correction's committed recipient/variant rather than a stale
        # snapshot.
        contact = await conn.fetchrow(
            """
            SELECT c.*, latest.submitted_email, latest.ack_variant,
                   latest.latest_intake_at
            FROM contacts AS c
            LEFT JOIN LATERAL (
                SELECT
                    NULLIF(ci.metadata->>'submitted_email', '') AS submitted_email,
                    NULLIF(ci.metadata->>'ack_variant', '') AS ack_variant,
                    ci.occurred_at AS latest_intake_at
                FROM contact_interactions AS ci
                WHERE ci.contact_id = c.id
                  AND ci.interaction_type = 'web_form'
                  AND ci.intent = 'estimate_request'
                ORDER BY ci.occurred_at DESC, ci.created_at DESC, ci.id DESC
                LIMIT 1
            ) AS latest ON TRUE
            WHERE c.id = $1
            """,
            contact_id,
        )
        if contact is None:
            raise EOMMissedCallRecoveryNotFoundError("EOM lead was not found")
        return contact

    async def _evaluate_contact_eligibility(
        self,
        conn: Any,
        contact: Mapping[str, Any],
        *,
        sequence_created_at: datetime | None,
        recipient_snapshot: str | None,
    ) -> _Eligibility:
        if contact.get("business_context_id") != _EOM_CONTEXT:
            return _Eligibility(False, "tenant_changed", None, None)
        if contact.get("contact_type") != "lead":
            return _Eligibility(False, "became_customer", None, None)
        if contact.get("status") != "active":
            return _Eligibility(False, "contact_inactive", None, None)
        if contact.get("lead_stage") != "new":
            return _Eligibility(False, "lead_advanced", None, None)
        if contact.get("customer_type") == "commercial":
            return _Eligibility(False, "non_residential", None, None)
        if contact.get("ack_variant") != "residential":
            return _Eligibility(False, "not_residential_estimate", None, None)
        recipient = _normalize_recipient(
            contact.get("submitted_email") or contact.get("email")
        )
        if recipient is None:
            return _Eligibility(False, "missing_or_invalid_recipient", None, None)
        if recipient_snapshot is not None and recipient != recipient_snapshot:
            return _Eligibility(False, "recipient_changed", None, None)
        if await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM eom_missed_call_contact_suppressions
                WHERE contact_id = $1 AND business_context_id = $2
            )
            """,
            contact["id"],
            _EOM_CONTEXT,
        ):
            return _Eligibility(False, "suppressed_recipient", None, None)
        # Before a sequence exists, response evidence is measured from the
        # latest qualifying estimate request. Once it exists, only newer
        # evidence can cancel it because the start path already rejected a
        # response that preceded the call attempt.
        response_since = sequence_created_at or contact.get("latest_intake_at")
        if isinstance(response_since, datetime):
            response = await conn.fetchval(
                """
                SELECT interaction_type
                FROM contact_interactions
                WHERE contact_id = $1
                  AND occurred_at > $2
                  AND (
                    interaction_type = ANY($3::varchar[])
                    OR (
                        interaction_type = 'sms'
                        AND eom_missed_call_has_proven_inbound_sms(metadata)
                    )
                    OR (interaction_type = 'web_form' AND intent = 'estimate_request')
                  )
                ORDER BY occurred_at DESC, created_at DESC, id DESC
                LIMIT 1
                """,
                contact["id"],
                response_since,
                list(_TRACKED_RESPONSE_INTERACTIONS),
            )
            if response is not None:
                return _Eligibility(
                    False, "tracked_response_or_new_request", None, None
                )
        full_name = contact.get("full_name")
        if not isinstance(full_name, str) or not full_name.strip():
            return _Eligibility(False, "invalid_contact_name", None, None)
        return _Eligibility(True, None, recipient, full_name.strip())

    async def _insert_steps(
        self,
        conn: Any,
        *,
        sequence_id: UUID,
        full_name: str,
        now: datetime,
    ) -> None:
        booking_link = self._config.missed_call_booking_link.strip()
        if not booking_link:
            raise EOMMissedCallRecoveryConflictError(
                "Missed-call recovery booking link is not configured"
            )
        step_one_due = _datetime(now, "current time")
        step_two_due = next_business_day_due(
            step_one_due, timezone_name=self._config.missed_call_timezone
        )
        due_times = (
            step_one_due,
            step_two_due,
            _third_step_due(
                step_two_due,
                timezone_name=self._config.missed_call_timezone,
            ),
        )
        for step_number, due_at in enumerate(due_times, start=1):
            message: MissedCallRecoveryEmail = render_missed_call_recovery_email(
                step_number=step_number,
                full_name=full_name,
                booking_link=booking_link,
            )
            step_id = uuid4()
            provider_key = f"eom-missed-call/{step_id}"
            await conn.execute(
                """
                INSERT INTO eom_missed_call_sequence_steps (
                    id, sequence_id, step_number, due_at, subject, body,
                    provider_idempotency_key, state, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending', $8, $8)
                ON CONFLICT (sequence_id, step_number) DO NOTHING
                """,
                step_id,
                sequence_id,
                step_number,
                due_at,
                message.subject,
                message.body,
                provider_key,
                now,
            )

    async def _event(
        self,
        conn: Any,
        *,
        sequence_id: UUID,
        event_type: str,
        reason_code: str | None,
        actor_id: int | None,
        actor_name: str,
        source: str,
        metadata: Mapping[str, Any],
        occurred_at: datetime,
        step_id: UUID | None = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO eom_missed_call_sequence_events (
                sequence_id, step_id, event_type, reason_code, actor_id,
                actor_name, source, metadata, occurred_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
            """,
            sequence_id,
            step_id,
            event_type,
            reason_code,
            actor_id,
            actor_name,
            source,
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")),
            occurred_at,
        )

    async def _status_for_contact(
        self, conn: Any, contact_id: UUID
    ) -> dict[str, Any] | None:
        row = await conn.fetchrow(
            """
            SELECT
                s.contact_id, s.id AS sequence_id, s.state, s.blocked_reason,
                s.cancellation_reason, s.created_at, s.terminal_at,
                next_step.step_number AS next_step_number,
                next_step.due_at AS next_due_at,
                last_event.event_type AS last_event_type,
                last_event.reason_code AS last_event_reason
            FROM eom_missed_call_sequences AS s
            LEFT JOIN LATERAL (
                SELECT step_number, COALESCE(next_attempt_at, due_at) AS due_at
                FROM eom_missed_call_sequence_steps
                WHERE sequence_id = s.id
                  AND state IN ('pending', 'attempting')
                ORDER BY step_number ASC
                LIMIT 1
            ) AS next_step ON TRUE
            LEFT JOIN LATERAL (
                SELECT event_type, reason_code
                FROM eom_missed_call_sequence_events
                WHERE sequence_id = s.id
                ORDER BY occurred_at DESC, id DESC
                LIMIT 1
            ) AS last_event ON TRUE
            WHERE s.contact_id = $1
            ORDER BY s.created_at DESC, s.id DESC
            LIMIT 1
            """,
            contact_id,
        )
        return None if row is None else self._serialize_status(row)

    @staticmethod
    def _serialize_status(row: Mapping[str, Any]) -> dict[str, Any]:
        def timestamp(value: object) -> str | None:
            return value.isoformat() if isinstance(value, datetime) else None

        return {
            "contactId": str(row["contact_id"]),
            "sequenceId": str(row["sequence_id"]),
            "state": str(row["state"]),
            "blockedReason": row.get("blocked_reason"),
            "cancellationReason": row.get("cancellation_reason"),
            "nextStepNumber": (
                int(row["next_step_number"])
                if row.get("next_step_number") is not None
                else None
            ),
            "nextFollowUpAt": timestamp(row.get("next_due_at")),
            "lastEvent": row.get("last_event_type"),
            "lastReason": row.get("last_event_reason"),
            "createdAt": timestamp(row.get("created_at")),
            "terminalAt": timestamp(row.get("terminal_at")),
        }

    async def _cancel_for_reason(
        self,
        conn: Any,
        *,
        sequence_id: UUID,
        reason: str,
        source: str,
        actor_id: int | None,
        actor_name: str,
    ) -> None:
        sequence = await conn.fetchrow(
            "SELECT * FROM eom_missed_call_sequences WHERE id = $1 FOR UPDATE",
            sequence_id,
        )
        if sequence is None or sequence["state"] not in {
            "active",
            "blocked_configuration",
        }:
            return
        now = self._now()
        await conn.execute(
            """
            UPDATE eom_missed_call_sequences
               SET state = 'cancelled', blocked_reason = NULL,
                   cancellation_reason = $2,
                   terminal_at = $3, updated_at = $3
             WHERE id = $1
            """,
            sequence_id,
            reason,
            now,
        )
        skipped = await conn.fetch(
            """
            UPDATE eom_missed_call_sequence_steps
               SET state = 'skipped', terminal_reason = $2,
                   next_attempt_at = NULL, claim_token = NULL, claimed_at = NULL,
                   claim_expires_at = NULL, updated_at = $3
             WHERE sequence_id = $1 AND state IN ('pending', 'attempting')
            RETURNING id
            """,
            sequence_id,
            reason,
            now,
        )
        await self._event(
            conn,
            sequence_id=sequence_id,
            event_type="sequence_cancelled",
            reason_code=reason,
            actor_id=actor_id,
            actor_name=actor_name,
            source=source,
            metadata={},
            occurred_at=now,
        )
        for row in skipped:
            await self._event(
                conn,
                sequence_id=sequence_id,
                step_id=row["id"],
                event_type="step_skipped",
                reason_code=reason,
                actor_id=actor_id,
                actor_name=actor_name,
                source=source,
                metadata={},
                occurred_at=now,
            )

    async def _recover_stale_claims(self, *, limit: int) -> None:
        now = self._now()
        try:
            async with self.pool.transaction() as conn:
                rows = await conn.fetch(
                    """
                    SELECT step.*, sequence.contact_id, sequence.state AS sequence_state
                    FROM eom_missed_call_sequence_steps AS step
                    JOIN eom_missed_call_sequences AS sequence ON sequence.id = step.sequence_id
                    WHERE step.state = 'attempting'
                      AND step.claim_expires_at <= $1
                    ORDER BY step.claim_expires_at ASC, step.id ASC
                    FOR UPDATE OF step, sequence SKIP LOCKED
                    LIMIT $2
                    """,
                    now,
                    limit,
                )
                for row in rows:
                    if row["sequence_state"] == "blocked_configuration":
                        # A configuration pause can race an already persisted
                        # claim. It is not proof that the provider did or did
                        # not accept the email, so preserve the claim/key until
                        # an explicit resume restores ``active`` and the normal
                        # recovery rules can safely decide its outcome.
                        continue
                    if row["sequence_state"] != "active":
                        skipped = await conn.fetchrow(
                            """
                            UPDATE eom_missed_call_sequence_steps
                               SET state = 'skipped', terminal_reason = 'sequence_terminal',
                                   claim_token = NULL, claimed_at = NULL, claim_expires_at = NULL,
                                   next_attempt_at = NULL, updated_at = $2
                             WHERE id = $1 AND state = 'attempting'
                            RETURNING id
                            """,
                            row["id"],
                            now,
                        )
                        if skipped is not None:
                            await self._event(
                                conn,
                                sequence_id=row["sequence_id"],
                                step_id=skipped["id"],
                                event_type="step_skipped",
                                reason_code="sequence_terminal",
                                actor_id=None,
                                actor_name="system",
                                source="worker",
                                metadata={},
                                occurred_at=now,
                            )
                        continue
                    expires_at = row["provider_key_expires_at"]
                    if (
                        isinstance(expires_at, datetime)
                        and now < expires_at
                        and int(row["attempt_count"])
                        < self._config.missed_call_max_delivery_attempts
                    ):
                        await conn.execute(
                            """
                            UPDATE eom_missed_call_sequence_steps
                               SET state = 'pending', next_attempt_at = $2,
                                   claim_token = NULL, claimed_at = NULL, claim_expires_at = NULL,
                                   updated_at = $2
                             WHERE id = $1 AND state = 'attempting'
                            """,
                            row["id"],
                            now,
                        )
                        await self._event(
                            conn,
                            sequence_id=row["sequence_id"],
                            step_id=row["id"],
                            event_type="step_retry_scheduled",
                            reason_code="claim_lease_expired",
                            actor_id=None,
                            actor_name="system",
                            source="worker",
                            metadata={},
                            occurred_at=now,
                        )
                    else:
                        await self._mark_recovery_required(
                            conn,
                            sequence_id=row["sequence_id"],
                            step_id=row["id"],
                            reason="provider_idempotency_window_expired",
                            now=now,
                        )
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery stale claim check is unavailable"
            ) from exc

    async def _claim_one_due_step(self) -> _ClaimResult:
        """Durably claim one due step before any external email request.

        The first short transaction makes the provider key's expiry durable.
        If a process dies after Resend accepts mail but before the second
        transaction can confirm it, a later worker knows it may only reuse that
        same key inside the provider's retention window; after the window it
        leaves operator-visible recovery evidence instead of risking a second
        customer message.
        """

        now = self._now()
        try:
            async with self.pool.transaction() as conn:
                # Claim the step first so peer workers still use SKIP LOCKED,
                # then lock the canonical contact before the sequence. Contact
                # and interaction mutations use that same contact-first order
                # in migration 389. The second, fresh contact query below is
                # therefore the linearization point for every mutable piece of
                # delivery eligibility (including a corrected form recipient).
                candidate = await conn.fetchrow(
                    """
                    SELECT
                        step.id AS step_id,
                        step.sequence_id,
                        sequence.contact_id
                    FROM eom_missed_call_sequence_steps AS step
                    JOIN eom_missed_call_sequences AS sequence ON sequence.id = step.sequence_id
                    WHERE step.state = 'pending'
                      AND sequence.state = 'active'
                      AND COALESCE(step.next_attempt_at, step.due_at) <= $1
                      AND NOT EXISTS (
                          SELECT 1 FROM eom_missed_call_sequence_steps AS earlier
                          WHERE earlier.sequence_id = step.sequence_id
                            AND earlier.step_number < step.step_number
                            AND earlier.state NOT IN ('sent', 'skipped')
                      )
                    ORDER BY COALESCE(step.next_attempt_at, step.due_at), step.step_number, step.id
                    FOR UPDATE OF step SKIP LOCKED
                    LIMIT 1
                    """,
                    now,
                )
                if candidate is None:
                    return _ClaimResult(processed=False)
                sequence_id = _uuid(candidate["sequence_id"], "sequence id")
                contact_id = _uuid(candidate["contact_id"], "contact id")
                contact = await self._contact_for_update(
                    conn,
                    contact_id,
                    allow_non_eom=True,
                )
                sequence = await conn.fetchrow(
                    """
                    SELECT id, contact_id, created_at, recipient_email
                    FROM eom_missed_call_sequences
                    WHERE id = $1
                      AND contact_id = $2
                      AND state = 'active'
                    FOR UPDATE
                    """,
                    sequence_id,
                    contact_id,
                )
                if sequence is None:
                    # Another contact-first mutation terminalized this exact
                    # sequence after the step was selected. It is progress for
                    # this bounded sweep, but it is not a sendable claim.
                    return _ClaimResult(processed=True)
                eligibility = await self._evaluate_contact_eligibility(
                    conn,
                    contact,
                    sequence_created_at=sequence["created_at"],
                    recipient_snapshot=sequence["recipient_email"],
                )
                if not eligibility.eligible:
                    await self._cancel_for_reason(
                        conn,
                        sequence_id=sequence_id,
                        reason=eligibility.reason or "ineligible",
                        source="worker",
                        actor_id=None,
                        actor_name="system",
                    )
                    return _ClaimResult(processed=True)

                claim_token = uuid4()
                claimed = await conn.fetchrow(
                    """
                    UPDATE eom_missed_call_sequence_steps
                       SET state = 'attempting', attempt_count = attempt_count + 1,
                           claim_token = $2, claimed_at = $3, claim_expires_at = $4,
                           provider_key_expires_at = COALESCE(provider_key_expires_at, $5),
                           next_attempt_at = NULL, updated_at = $3
                     WHERE id = $1 AND state = 'pending'
                     RETURNING *
                    """,
                    candidate["step_id"],
                    claim_token,
                    now,
                    now + _ATTEMPT_LEASE,
                    now + _PROVIDER_KEY_WINDOW,
                )
                if claimed is None:
                    return _ClaimResult(processed=False)
                await self._event(
                    conn,
                    sequence_id=sequence_id,
                    step_id=candidate["step_id"],
                    event_type="step_claimed",
                    reason_code=None,
                    actor_id=None,
                    actor_name="system",
                    source="worker",
                    metadata={"attempt": int(claimed["attempt_count"])},
                    occurred_at=now,
                )
                recipient = eligibility.recipient_email
                if recipient is None:
                    raise EOMMissedCallRecoveryConflictError(
                        "Missed-call recovery recipient is invalid"
                    )
                return _ClaimResult(
                    processed=True,
                    claim=_ClaimedStep(
                        step_id=_uuid(claimed["id"], "step id"),
                        sequence_id=_uuid(claimed["sequence_id"], "sequence id"),
                        contact_id=contact_id,
                        claim_token=claim_token,
                        recipient_email=recipient,
                        subject=str(claimed["subject"]),
                        body=str(claimed["body"]),
                        provider_idempotency_key=str(
                            claimed["provider_idempotency_key"]
                        ),
                        provider_key_expires_at=_datetime(
                            claimed["provider_key_expires_at"], "provider key expiry"
                        ),
                        attempt_count=int(claimed["attempt_count"]),
                    ),
                )
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery claim is unavailable"
            ) from exc

    async def _deliver_claim(self, claim: _ClaimedStep) -> _SentHistory | None:
        """Re-read authoritative state, then send while it remains locked."""

        now = self._now()
        try:
            async with self.pool.transaction() as conn:
                # The migration's BEFORE interaction trigger takes this same
                # contact row lock before it writes any intake/response
                # evidence. Taking it before the sequence means either that
                # mutation commits before this fresh recheck, or it waits until
                # this provider call finishes; no correction can slip between
                # the final snapshot and external delivery.
                contact = await self._contact_for_update(
                    conn,
                    claim.contact_id,
                    allow_non_eom=True,
                )
                row = await conn.fetchrow(
                    """
                    SELECT
                        step.sequence_id,
                        step.claim_expires_at,
                        step.provider_key_expires_at,
                        step.provider_idempotency_key,
                        step.subject,
                        step.body,
                        sequence.contact_id,
                        sequence.created_at AS sequence_created_at,
                        sequence.recipient_email
                    FROM eom_missed_call_sequence_steps AS step
                    JOIN eom_missed_call_sequences AS sequence ON sequence.id = step.sequence_id
                    WHERE step.id = $1
                      AND step.state = 'attempting'
                      AND step.claim_token = $2
                      AND sequence.state = 'active'
                    FOR UPDATE OF step, sequence
                    """,
                    claim.step_id,
                    claim.claim_token,
                )
                if row is None:
                    return None
                if row["claim_expires_at"] is None or row["claim_expires_at"] <= now:
                    await conn.execute(
                        """
                        UPDATE eom_missed_call_sequence_steps
                           SET state = 'pending', claim_token = NULL,
                               claimed_at = NULL, claim_expires_at = NULL,
                               next_attempt_at = $2, updated_at = $2
                         WHERE id = $1 AND state = 'attempting' AND claim_token = $3
                        """,
                        claim.step_id,
                        now,
                        claim.claim_token,
                    )
                    await self._event(
                        conn,
                        sequence_id=claim.sequence_id,
                        step_id=claim.step_id,
                        event_type="step_retry_scheduled",
                        reason_code="claim_lease_expired_before_delivery",
                        actor_id=None,
                        actor_name="system",
                        source="worker",
                        metadata={},
                        occurred_at=now,
                    )
                    return None
                if (
                    row["provider_key_expires_at"] is None
                    or now >= row["provider_key_expires_at"]
                ):
                    await self._mark_recovery_required(
                        conn,
                        sequence_id=claim.sequence_id,
                        step_id=claim.step_id,
                        reason="provider_idempotency_window_expired",
                        now=now,
                    )
                    return None
                if (
                    _uuid(row["sequence_id"], "sequence id") != claim.sequence_id
                    or _uuid(row["contact_id"], "contact id") != claim.contact_id
                    or str(row["provider_idempotency_key"])
                    != claim.provider_idempotency_key
                    or str(row["subject"]) != claim.subject
                    or str(row["body"]) != claim.body
                ):
                    await self._mark_recovery_required(
                        conn,
                        sequence_id=claim.sequence_id,
                        step_id=claim.step_id,
                        reason="claimed_payload_changed",
                        now=now,
                    )
                    return None

                eligibility = await self._evaluate_contact_eligibility(
                    conn,
                    contact,
                    sequence_created_at=row["sequence_created_at"],
                    recipient_snapshot=row["recipient_email"],
                )
                if not eligibility.eligible:
                    await self._cancel_for_reason(
                        conn,
                        sequence_id=claim.sequence_id,
                        reason=eligibility.reason or "ineligible",
                        source="worker",
                        actor_id=None,
                        actor_name="system",
                    )
                    return None
                if eligibility.recipient_email != claim.recipient_email:
                    await self._cancel_for_reason(
                        conn,
                        sequence_id=claim.sequence_id,
                        reason="recipient_changed",
                        source="worker",
                        actor_id=None,
                        actor_name="system",
                    )
                    return None

                try:
                    message_id = await self.gateway.send(
                        recipient_email=claim.recipient_email,
                        subject=claim.subject,
                        body=claim.body,
                        idempotency_key=claim.provider_idempotency_key,
                    )
                except _DefiniteDeliveryError as exc:
                    return await self._handle_definite_delivery_error(
                        conn, claim=claim, error=exc, now=now
                    )
                except _AmbiguousDeliveryError as exc:
                    await self._mark_recovery_required(
                        conn,
                        sequence_id=claim.sequence_id,
                        step_id=claim.step_id,
                        reason=str(exc),
                        now=now,
                    )
                    return None

                sent_at = self._now()
                updated = await conn.fetchrow(
                    """
                    UPDATE eom_missed_call_sequence_steps
                       SET state = 'sent', provider_message_id = $2, sent_at = $3,
                           claim_token = NULL, claimed_at = NULL, claim_expires_at = NULL,
                           terminal_reason = NULL, updated_at = $3
                     WHERE id = $1 AND state = 'attempting' AND claim_token = $4
                     RETURNING step_number
                    """,
                    claim.step_id,
                    message_id,
                    sent_at,
                    claim.claim_token,
                )
                if updated is None:
                    raise EOMMissedCallRecoveryConflictError(
                        "Missed-call recovery delivery claim was lost"
                    )
                await self._event(
                    conn,
                    sequence_id=claim.sequence_id,
                    step_id=claim.step_id,
                    event_type="step_sent",
                    reason_code="provider_accepted",
                    actor_id=None,
                    actor_name="system",
                    source="worker",
                    metadata={"attempt": claim.attempt_count},
                    occurred_at=sent_at,
                )
                if int(updated["step_number"]) == 2:
                    # Step 3 is intentionally relative to successful Email 2,
                    # not merely Email 2's original due time. A delayed/retried
                    # second message must never compress the promised three-day
                    # final follow-up interval.
                    await conn.execute(
                        """
                        UPDATE eom_missed_call_sequence_steps
                           SET due_at = $2, next_attempt_at = NULL, updated_at = $3
                         WHERE sequence_id = $1
                           AND step_number = 3
                           AND state = 'pending'
                        """,
                        claim.sequence_id,
                        _third_step_due(
                            sent_at,
                            timezone_name=self._config.missed_call_timezone,
                        ),
                        sent_at,
                    )
                elif int(updated["step_number"]) == 3:
                    await conn.execute(
                        """
                        UPDATE eom_missed_call_sequences
                           SET state = 'completed', terminal_at = $2, updated_at = $2
                         WHERE id = $1 AND state = 'active'
                        """,
                        claim.sequence_id,
                        sent_at,
                    )
                    await self._event(
                        conn,
                        sequence_id=claim.sequence_id,
                        event_type="sequence_completed",
                        reason_code=None,
                        actor_id=None,
                        actor_name="system",
                        source="worker",
                        metadata={},
                        occurred_at=sent_at,
                    )
                return _SentHistory(
                    contact_id=claim.contact_id,
                    recipient_email=claim.recipient_email,
                    subject=claim.subject,
                    body=claim.body,
                    provider_message_id=message_id,
                    sequence_id=claim.sequence_id,
                    step_id=claim.step_id,
                )
        except EOMMissedCallRecoveryError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery delivery is unavailable"
            ) from exc

    async def _handle_definite_delivery_error(
        self,
        conn: Any,
        *,
        claim: _ClaimedStep,
        error: _DefiniteDeliveryError,
        now: datetime,
    ) -> _SentHistory:
        retry_allowed = (
            error.retryable
            and claim.attempt_count < self._config.missed_call_max_delivery_attempts
            and now < claim.provider_key_expires_at
        )
        if retry_allowed:
            delay = _RETRY_DELAYS[min(claim.attempt_count - 1, len(_RETRY_DELAYS) - 1)]
            next_attempt = now + delay
            await conn.execute(
                """
                UPDATE eom_missed_call_sequence_steps
                   SET state = 'pending', next_attempt_at = $2, claim_token = NULL,
                       claimed_at = NULL,
                       claim_expires_at = NULL, last_error_code = $3, updated_at = $1
                 WHERE id = $4 AND state = 'attempting'
                """,
                now,
                next_attempt,
                error.code,
                claim.step_id,
            )
            await self._event(
                conn,
                sequence_id=claim.sequence_id,
                step_id=claim.step_id,
                event_type="step_retry_scheduled",
                reason_code=error.code,
                actor_id=None,
                actor_name="system",
                source="worker",
                metadata={"attempt": claim.attempt_count},
                occurred_at=now,
            )
        elif error.recovery_required_if_exhausted:
            # Resend explicitly says another request for this stable key is
            # still running. It is safe to retry that same key while its
            # evidence window remains open, but once the bounded retry budget
            # is exhausted we cannot claim the other request was rejected.
            await self._mark_recovery_required(
                conn,
                sequence_id=claim.sequence_id,
                step_id=claim.step_id,
                reason=error.code,
                now=now,
            )
        else:
            await self._mark_failed(
                conn,
                sequence_id=claim.sequence_id,
                step_id=claim.step_id,
                reason=("retry_exhausted" if error.retryable else error.code),
                now=now,
            )
        return _SentHistory(
            contact_id=claim.contact_id,
            recipient_email="",
            subject="",
            body="",
            provider_message_id="",
            sequence_id=claim.sequence_id,
            step_id=claim.step_id,
        )

    async def _mark_failed(
        self,
        conn: Any,
        *,
        sequence_id: UUID,
        step_id: UUID,
        reason: str,
        now: datetime,
    ) -> None:
        await conn.execute(
            """
            UPDATE eom_missed_call_sequence_steps
               SET state = 'failed', terminal_reason = $2, last_error_code = $2,
                   claim_token = NULL, claimed_at = NULL, claim_expires_at = NULL,
                   updated_at = $3
             WHERE id = $1
            """,
            step_id,
            reason,
            now,
        )
        await conn.execute(
            """
            UPDATE eom_missed_call_sequences
               SET state = 'failed', cancellation_reason = $2,
                   terminal_at = $3, updated_at = $3
             WHERE id = $1 AND state = 'active'
            """,
            sequence_id,
            reason,
            now,
        )
        skipped = await conn.fetch(
            """
            UPDATE eom_missed_call_sequence_steps
               SET state = 'skipped', terminal_reason = $2,
                   next_attempt_at = NULL, claim_expires_at = NULL, updated_at = $3
             WHERE sequence_id = $1 AND state = 'pending'
             RETURNING id
            """,
            sequence_id,
            reason,
            now,
        )
        await self._event(
            conn,
            sequence_id=sequence_id,
            step_id=step_id,
            event_type="step_failed",
            reason_code=reason,
            actor_id=None,
            actor_name="system",
            source="worker",
            metadata={},
            occurred_at=now,
        )
        for row in skipped:
            await self._event(
                conn,
                sequence_id=sequence_id,
                step_id=row["id"],
                event_type="step_skipped",
                reason_code=reason,
                actor_id=None,
                actor_name="system",
                source="worker",
                metadata={},
                occurred_at=now,
            )

    async def _mark_recovery_required(
        self,
        conn: Any,
        *,
        sequence_id: UUID,
        step_id: UUID,
        reason: str,
        now: datetime,
    ) -> None:
        await conn.execute(
            """
            UPDATE eom_missed_call_sequence_steps
               SET state = 'recovery_required', terminal_reason = $2,
                   last_error_code = $2, claim_token = NULL, claimed_at = NULL,
                   claim_expires_at = NULL,
                   updated_at = $3
             WHERE id = $1
            """,
            step_id,
            reason,
            now,
        )
        await conn.execute(
            """
            UPDATE eom_missed_call_sequences
               SET state = 'recovery_required', cancellation_reason = $2,
                   terminal_at = $3, updated_at = $3
             WHERE id = $1 AND state = 'active'
            """,
            sequence_id,
            reason,
            now,
        )
        skipped = await conn.fetch(
            """
            UPDATE eom_missed_call_sequence_steps
               SET state = 'skipped', terminal_reason = $2,
                   next_attempt_at = NULL, claim_expires_at = NULL, updated_at = $3
             WHERE sequence_id = $1 AND state = 'pending'
             RETURNING id
            """,
            sequence_id,
            reason,
            now,
        )
        await self._event(
            conn,
            sequence_id=sequence_id,
            step_id=step_id,
            event_type="step_recovery_required",
            reason_code=reason,
            actor_id=None,
            actor_name="system",
            source="worker",
            metadata={},
            occurred_at=now,
        )
        for row in skipped:
            await self._event(
                conn,
                sequence_id=sequence_id,
                step_id=row["id"],
                event_type="step_skipped",
                reason_code=reason,
                actor_id=None,
                actor_name="system",
                source="worker",
                metadata={},
                occurred_at=now,
            )

    async def _write_sent_history(self, history: _SentHistory) -> None:
        """Write secondary sent-history evidence after primary state commits."""

        if not history.provider_message_id:
            return
        repository = self._email_history
        if repository is None:
            # The slim EOM profile imports this module with the funnel router;
            # keep the broader repository/model graph lazy until a confirmed
            # provider acceptance actually needs secondary history evidence.
            from ..storage.repositories.email import EmailRepository

            repository = EmailRepository(pool=self.pool)
        try:
            await repository.create(
                to_addresses=[history.recipient_email],
                subject=history.subject,
                body=history.body,
                template_type="eom_missed_call_recovery",
                resend_message_id=history.provider_message_id,
                metadata={
                    "source": "eom_missed_call_recovery",
                    "contact_id": str(history.contact_id),
                    "sequence_id": str(history.sequence_id),
                    "step_id": str(history.step_id),
                },
                business_context_id=_EOM_CONTEXT,
            )
        except Exception as exc:
            # The provider acceptance and durable sequence result are already
            # committed. History is valuable secondary evidence, never an
            # authorization to resend or to roll back a truthful send.
            logger.warning(
                "Missed-call sent-email history write failed sequence=%s step=%s error=%s",
                history.sequence_id,
                history.step_id,
                type(exc).__name__,
            )


async def prepare_eom_missed_call_recovery_worker(
    *,
    pool: Any,
    config: EOMFunnelConfig | None = None,
) -> tuple[asyncio.Event, asyncio.Task[None]] | None:
    """Fence, durably pause, or start the recovery worker for one app lifespan.

    Both supported Atlas entrypoints call this exact boundary. A disabled or
    incomplete deployment first blocks any existing active sequence; restoring
    configuration later never resumes those steps without the operator's
    explicit recovery action. A first rollout with no recovery schema is safe
    while disabled, whereas an enabled rollout fails startup rather than
    serving an unbacked mutation route.
    """

    effective_config = config or funnel_settings
    recovery = EOMMissedCallRecoveryService(pool=pool, config=effective_config)
    schema_ready = await missed_call_recovery_schema_ready(pool)
    if not schema_ready:
        if effective_config.missed_call_recovery_enabled:
            raise EOMMissedCallRecoveryUnavailableError(
                "Missed-call recovery schema is unavailable"
            )
        logger.info(
            "EOM missed-call recovery schema is absent while delivery is disabled"
        )
        return None

    delivery_block_reason = recovery.delivery_block_reason()
    if delivery_block_reason is not None:
        blocked_count = await recovery.block_active_sequences_for_configuration(
            reason=delivery_block_reason
        )
        logger.warning(
            "EOM missed-call recovery delivery is blocked reason=%s sequences=%s",
            delivery_block_reason,
            blocked_count,
        )
        return None

    worker = start_eom_missed_call_recovery_worker(
        pool=pool,
        config=effective_config,
    )
    logger.info("EOM missed-call recovery worker started")
    return worker


async def run_eom_missed_call_recovery_worker(
    *,
    service: EOMMissedCallRecoveryService,
    stop_event: asyncio.Event,
) -> None:
    """Run a bounded, multi-process-safe EOM delivery loop until shutdown."""

    interval = service._config.missed_call_poll_interval_seconds
    while not stop_event.is_set():
        try:
            await service.dispatch_due_steps()
        except Exception as exc:
            logger.error(
                "Missed-call recovery worker cycle failed error=%s",
                type(exc).__name__,
            )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            pass


def start_eom_missed_call_recovery_worker(
    *,
    pool: Any,
    config: EOMFunnelConfig | None = None,
) -> tuple[asyncio.Event, asyncio.Task[None]]:
    """Start one process-local loop; database locks make many loops safe."""

    service = EOMMissedCallRecoveryService(pool=pool, config=config)
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        run_eom_missed_call_recovery_worker(service=service, stop_event=stop_event),
        name="eom-missed-call-recovery-worker",
    )
    return stop_event, task


async def stop_eom_missed_call_recovery_worker(
    worker: tuple[asyncio.Event, asyncio.Task[None]] | None,
) -> None:
    """Stop a local worker without cancelling an in-flight transactional send."""

    if worker is None:
        return
    stop_event, task = worker
    stop_event.set()
    try:
        await asyncio.wait_for(task, timeout=15)
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
