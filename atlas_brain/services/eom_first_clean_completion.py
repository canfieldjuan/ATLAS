"""Durable first-clean completion evidence for EOM onboarding.

The existing EOM ``first_clean_booked`` lifecycle event proves a Calendar
appointment was reconciled.  This module intentionally does not infer service
delivery from that event, a calendar record, job status, or actual-hours
projection.  It records one authenticated, actor-attributed manager report
anchored to the immutable canonical customer handoff and tracker service ID.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

import asyncpg


_EOM_CONTEXT = "effingham_maids"
_MAX_SIGNED_BIGINT = 2**63 - 1
_MAX_ACTOR_NAME_LENGTH = 128
_MAX_LIFECYCLE_ACTOR_LENGTH = 128
_OPERATION_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_SERVICE_KINDS = frozenset({"job", "planned_visit"})


class EOMFirstCleanCompletionError(Exception):
    """Base class for API-safe first-clean completion errors."""

    status_code = 409
    code = "first_clean_completion_error"


class EOMFirstCleanCompletionValidationError(EOMFirstCleanCompletionError):
    status_code = 422
    code = "invalid_first_clean_completion_request"


class EOMFirstCleanCompletionNotFoundError(EOMFirstCleanCompletionError):
    status_code = 404
    code = "first_clean_completion_not_found"


class EOMFirstCleanCompletionConflictError(EOMFirstCleanCompletionError):
    status_code = 409
    code = "first_clean_completion_conflict"


class EOMFirstCleanCompletionUnavailableError(EOMFirstCleanCompletionError):
    status_code = 503
    code = "first_clean_completion_unavailable"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _uuid(value: object, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EOMFirstCleanCompletionValidationError(f"{field} is invalid") from exc


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EOMFirstCleanCompletionValidationError(f"{field} is invalid")
    if value <= 0 or value > _MAX_SIGNED_BIGINT:
        raise EOMFirstCleanCompletionValidationError(f"{field} is invalid")
    return value


def _operation_key(value: object) -> str:
    if not isinstance(value, str):
        raise EOMFirstCleanCompletionValidationError("Idempotency key is required")
    value = value.strip()
    if not _OPERATION_KEY_PATTERN.fullmatch(value):
        raise EOMFirstCleanCompletionValidationError("Idempotency key is invalid")
    return value


def _lifecycle_actor(actor_id: int, actor_name: str) -> str:
    """Serialize the actor exactly as the lifecycle ledger stores it."""

    return f"employee:{actor_id}:{actor_name}"


def _actor(actor_id: object, actor_name: object) -> tuple[int, str]:
    parsed_id = _positive_int(actor_id, "Authenticated actor")
    if not isinstance(actor_name, str):
        raise EOMFirstCleanCompletionValidationError("Authenticated actor is invalid")
    parsed_name = actor_name.strip()
    if (
        not parsed_name
        or len(parsed_name) > _MAX_ACTOR_NAME_LENGTH
        or "\x00" in parsed_name
        or len(_lifecycle_actor(parsed_id, parsed_name)) > _MAX_LIFECYCLE_ACTOR_LENGTH
    ):
        raise EOMFirstCleanCompletionValidationError("Authenticated actor is invalid")
    return parsed_id, parsed_name


def _service_kind(value: object) -> str:
    if not isinstance(value, str):
        raise EOMFirstCleanCompletionValidationError("Tracker service kind is invalid")
    if value not in _SERVICE_KINDS:
        raise EOMFirstCleanCompletionValidationError("Tracker service kind is invalid")
    return value


def _completed_at(value: object, *, now: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EOMFirstCleanCompletionValidationError(
            "Completion time must include a timezone"
        )
    normalized = value.astimezone(timezone.utc)
    if normalized > now:
        raise EOMFirstCleanCompletionValidationError(
            "Completion time cannot be in the future"
        )
    return normalized


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _receipt_result(row: Mapping[str, Any], *, idempotent: bool) -> dict[str, Any]:
    completed_at = row["completed_at"]
    created_at = row["created_at"]
    if not isinstance(completed_at, datetime) or not isinstance(created_at, datetime):
        raise EOMFirstCleanCompletionUnavailableError(
            "First-clean completion receipt is invalid"
        )
    return {
        "receiptId": str(row["id"]),
        "contactId": str(row["contact_id"]),
        "handoffId": str(row["handoff_id"]),
        "trackerCustomerId": int(row["tracker_customer_id"]),
        "trackerSiteId": int(row["tracker_site_id"]),
        "trackerServiceKind": str(row["tracker_service_kind"]),
        "trackerServiceId": int(row["tracker_service_id"]),
        "completedAt": _utc_iso(completed_at),
        "recordedAt": _utc_iso(created_at),
        "idempotent": idempotent,
    }


async def first_clean_completion_schema_ready(pool: Any) -> bool:
    """Return whether every new database backstop is safe to serve."""

    try:
        return bool(
            await pool.fetchval(
                """
                SELECT to_regclass('eom_customer_handoffs') IS NOT NULL
                   AND to_regclass('eom_lead_lifecycle_events') IS NOT NULL
                   AND to_regclass('eom_first_clean_completion_operation_receipts')
                       IS NOT NULL
                   AND to_regclass('eom_first_clean_completion_receipts') IS NOT NULL
                   -- The pool must be a direct, least-privilege Atlas login.
                   -- A DBA session that SET ROLE atlas could RESET ROLE and
                   -- bypass the protected-object ACL boundary after readiness.
                   AND current_user = 'atlas'
                   AND session_user = 'atlas'
                   AND EXISTS (
                       SELECT 1
                         FROM pg_roles AS runtime_role
                        WHERE runtime_role.rolname = 'atlas'
                          AND runtime_role.rolcanlogin
                          AND NOT runtime_role.rolsuper
                          AND NOT runtime_role.rolcreaterole
                          AND NOT runtime_role.rolcreatedb
                          AND NOT runtime_role.rolreplication
                          AND NOT runtime_role.rolbypassrls
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_roles AS guard_role
                        WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
                          AND NOT guard_role.rolcanlogin
                          AND NOT guard_role.rolinherit
                          AND NOT guard_role.rolsuper
                          AND NOT guard_role.rolcreaterole
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_roles AS member_role
                        WHERE member_role.rolname IN ('atlas', 'atlas_nocodb')
                          AND EXISTS (
                              WITH RECURSIVE role_chain(roleid) AS (
                                  SELECT membership.roleid
                                    FROM pg_auth_members AS membership
                                   WHERE membership.member = member_role.oid
                                  UNION
                                  SELECT membership.roleid
                                    FROM pg_auth_members AS membership
                                    JOIN role_chain
                                      ON membership.member = role_chain.roleid
                              )
                              SELECT 1
                                FROM role_chain
                               WHERE roleid = (
                                   SELECT oid
                                     FROM pg_roles
                                    WHERE rolname = 'atlas_eom_handoff_owner'
                               )
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 2
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname = current_schema()
                          AND relation.relkind = 'r'
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND relation.relowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname = current_schema()
                          AND relation.relkind = 'r'
                          AND relation.relname = 'eom_customer_handoffs'
                          AND relation.relowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname = current_schema()
                          AND relation.relkind = 'r'
                          AND relation.relname = 'eom_lead_lifecycle_events'
                          AND relation.relowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 2
                         FROM pg_proc AS protected_function
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = protected_function.pronamespace
                        WHERE namespace.nspname = current_schema()
                          AND protected_function.proname IN (
                              'require_eom_customer_handoff_finalization',
                              'prevent_eom_customer_handoff_mutation'
                          )
                          AND protected_function.pronargs = 0
                          AND protected_function.proowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 1
                         FROM pg_proc AS protected_function
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = protected_function.pronamespace
                        WHERE namespace.nspname = current_schema()
                          AND protected_function.proname =
                              'prevent_eom_lead_lifecycle_event_mutation'
                          AND protected_function.pronargs = 0
                          AND protected_function.proowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 3
                         FROM pg_proc AS protected_function
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = protected_function.pronamespace
                        WHERE namespace.nspname = current_schema()
                          AND protected_function.proname IN (
                              'prevent_eom_first_clean_completion_mutation',
                              'require_eom_first_clean_completion_operation_scope',
                              'require_eom_first_clean_completion_receipt'
                          )
                          AND protected_function.pronargs = 0
                          AND protected_function.proowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 2
                         FROM pg_proc AS protected_function
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = protected_function.pronamespace
                        WHERE namespace.nspname = current_schema()
                          AND protected_function.proname IN (
                              'require_eom_first_clean_completion_operation_scope',
                              'require_eom_first_clean_completion_receipt'
                          )
                          AND protected_function.pronargs = 0
                          AND protected_function.proconfig @> ARRAY[
                              format(
                                  'search_path=%I, pg_catalog, pg_temp',
                                  current_schema()
                              )
                          ]
                   )
                   AND (
                       SELECT COUNT(*) = 6
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND acl.grantee = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                          AND acl.privilege_type IN ('SELECT', 'INSERT', 'UPDATE')
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND acl.grantee = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                          AND acl.privilege_type NOT IN ('SELECT', 'INSERT', 'UPDATE')
                   )
                   AND (
                       SELECT COUNT(*) = 2
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = 'eom_lead_lifecycle_events'
                          AND acl.grantee = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                          AND acl.privilege_type IN ('SELECT', 'INSERT')
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = 'eom_lead_lifecycle_events'
                          AND acl.grantee = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                          AND acl.privilege_type NOT IN ('SELECT', 'INSERT')
                   )
                   AND (
                       SELECT COUNT(*) = 14
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND acl.grantee = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                          AND acl.privilege_type IN (
                              'SELECT', 'INSERT', 'UPDATE', 'DELETE',
                              'TRUNCATE', 'REFERENCES', 'TRIGGER'
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND acl.grantee = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
                          )
                          AND acl.privilege_type NOT IN (
                              'SELECT', 'INSERT', 'UPDATE', 'DELETE',
                              'TRUNCATE', 'REFERENCES', 'TRIGGER'
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND acl.grantee NOT IN (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname IN (
                                   'atlas', 'atlas_eom_handoff_owner'
                               )
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname IN (
                              'eom_first_clean_completion_operation_receipts',
                              'eom_first_clean_completion_receipts'
                          )
                          AND acl.is_grantable
                   )
                   AND has_schema_privilege(
                       'atlas_eom_handoff_owner', current_schema(), 'USAGE'
                   )
                   AND has_table_privilege(
                       'atlas_eom_handoff_owner',
                       'eom_first_clean_completion_operation_receipts',
                       'SELECT'
                   )
                   AND has_table_privilege(
                       'atlas_eom_handoff_owner',
                       'eom_first_clean_completion_operation_receipts',
                       'UPDATE'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_first_clean_completion_operation_receipts',
                       'SELECT'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_first_clean_completion_operation_receipts',
                       'INSERT'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_first_clean_completion_operation_receipts',
                       'UPDATE'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_first_clean_completion_receipts',
                       'SELECT'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_first_clean_completion_receipts',
                       'INSERT'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_first_clean_completion_receipts',
                       'UPDATE'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_lead_lifecycle_events',
                       'SELECT'
                   )
                   AND has_table_privilege(
                       current_user,
                       'eom_lead_lifecycle_events',
                       'INSERT'
                   )
                   AND (
                       SELECT COUNT(*) = 4
                         FROM pg_constraint
                        WHERE conrelid =
                              'eom_first_clean_completion_receipts'::regclass
                          AND conname IN (
                              'uq_eom_first_clean_completion_receipt_contact',
                              'uq_eom_first_clean_completion_receipt_handoff',
                              'uq_eom_first_clean_completion_receipt_operation',
                              'uq_eom_first_clean_completion_tracker_service'
                          )
                   )
                   AND NOT EXISTS (
                       WITH expected_trigger(
                           relation_name, trigger_name, function_name
                       ) AS (
                           VALUES
                               ('eom_customer_handoffs',
                                'trg_require_eom_customer_handoff_finalization',
                                'require_eom_customer_handoff_finalization'),
                               ('eom_customer_handoffs',
                                'trg_prevent_eom_customer_handoff_mutation',
                                'prevent_eom_customer_handoff_mutation'),
                               ('eom_customer_handoffs',
                                'trg_prevent_eom_customer_handoff_truncate',
                                'prevent_eom_customer_handoff_mutation'),
                               ('eom_lead_lifecycle_events',
                                'trg_prevent_eom_lead_lifecycle_event_mutation',
                                'prevent_eom_lead_lifecycle_event_mutation'),
                               ('eom_lead_lifecycle_events',
                                'trg_prevent_eom_lead_lifecycle_event_truncate',
                                'prevent_eom_lead_lifecycle_event_mutation'),
                               ('eom_first_clean_completion_operation_receipts',
                                'trg_require_eom_first_clean_completion_operation_scope',
                                'require_eom_first_clean_completion_operation_scope'),
                               ('eom_first_clean_completion_operation_receipts',
                                'trg_prevent_eom_first_clean_completion_operation_mutation',
                                'prevent_eom_first_clean_completion_mutation'),
                               ('eom_first_clean_completion_operation_receipts',
                                'trg_prevent_eom_first_clean_completion_operation_truncate',
                                'prevent_eom_first_clean_completion_mutation'),
                               ('eom_first_clean_completion_receipts',
                                'trg_require_eom_first_clean_completion_receipt',
                                'require_eom_first_clean_completion_receipt'),
                               ('eom_first_clean_completion_receipts',
                                'trg_prevent_eom_first_clean_completion_receipt_mutation',
                                'prevent_eom_first_clean_completion_mutation'),
                               ('eom_first_clean_completion_receipts',
                                'trg_prevent_eom_first_clean_completion_receipt_truncate',
                                'prevent_eom_first_clean_completion_mutation')
                       )
                       SELECT 1
                         FROM expected_trigger
                        WHERE NOT EXISTS (
                            SELECT 1
                              FROM pg_trigger AS trigger
                              JOIN pg_proc AS trigger_function
                                ON trigger_function.oid = trigger.tgfoid
                              JOIN pg_namespace AS function_namespace
                                ON function_namespace.oid = trigger_function.pronamespace
                             WHERE trigger.tgrelid = to_regclass(
                                       format(
                                           '%I.%I',
                                           current_schema(),
                                           expected_trigger.relation_name
                                       )
                                   )
                               AND trigger.tgname = expected_trigger.trigger_name
                               AND NOT trigger.tgisinternal
                               AND trigger.tgenabled IN ('O', 'A')
                               AND function_namespace.nspname = current_schema()
                               AND trigger_function.proname = expected_trigger.function_name
                               AND trigger_function.pronargs = 0
                               AND trigger_function.proowner = (
                                   SELECT oid
                                     FROM pg_roles
                                    WHERE rolname = 'atlas_eom_handoff_owner'
                               )
                        )
                   )
                """
            )
        )
    except Exception:
        return False


class EOMFirstCleanCompletionService:
    """Owns immutable first-clean completion admission and idempotency."""

    def __init__(
        self,
        *,
        pool: Any,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._now = now or _now_utc

    @property
    def pool(self) -> Any:
        if not bool(getattr(self._pool, "is_initialized", True)):
            raise EOMFirstCleanCompletionUnavailableError(
                "First-clean completion database is unavailable"
            )
        return self._pool

    async def require_schema_ready(self) -> None:
        if not await first_clean_completion_schema_ready(self.pool):
            raise EOMFirstCleanCompletionUnavailableError(
                "First-clean completion schema is unavailable"
            )

    async def _bind_operation_receipt(
        self,
        conn: Any,
        *,
        contact_id: UUID,
        operation_key: str,
        fingerprint: str,
    ) -> bool:
        """Bind an operation key before receipt/lifecycle writes.

        Returning ``True`` means a previous transaction owns the same complete
        request and the caller must return that immutable receipt.  A key never
        changes contact or request facts after a crashed/retried call.
        """

        existing = await conn.fetchrow(
            """
            SELECT contact_id, request_fingerprint
            FROM eom_first_clean_completion_operation_receipts
            WHERE operation_key = $1
            FOR UPDATE
            """,
            operation_key,
        )
        if existing is None:
            inserted = await conn.fetchrow(
                """
                INSERT INTO eom_first_clean_completion_operation_receipts (
                    operation_key, contact_id, request_fingerprint
                ) VALUES ($1, $2, $3)
                ON CONFLICT (operation_key) DO NOTHING
                RETURNING operation_key
                """,
                operation_key,
                contact_id,
                fingerprint,
            )
            if inserted is not None:
                return False
            existing = await conn.fetchrow(
                """
                SELECT contact_id, request_fingerprint
                FROM eom_first_clean_completion_operation_receipts
                WHERE operation_key = $1
                FOR UPDATE
                """,
                operation_key,
            )
            if existing is None:
                raise EOMFirstCleanCompletionUnavailableError(
                    "First-clean completion operation receipt is unavailable"
                )

        if _uuid(existing["contact_id"], "Receipt contact id") != contact_id or (
            str(existing["request_fingerprint"]) != fingerprint
        ):
            raise EOMFirstCleanCompletionConflictError(
                "Idempotency key belongs to a different first-clean completion"
            )
        return True

    async def _receipt_for_operation(
        self, conn: Any, *, operation_key: str
    ) -> Mapping[str, Any]:
        row = await conn.fetchrow(
            """
            SELECT id, contact_id, handoff_id, tracker_customer_id,
                   tracker_site_id, tracker_service_kind, tracker_service_id,
                   completed_at, created_at
            FROM eom_first_clean_completion_receipts
            WHERE operation_key = $1
            FOR UPDATE
            """,
            operation_key,
        )
        if row is None:
            raise EOMFirstCleanCompletionUnavailableError(
                "First-clean completion operation evidence is unavailable"
            )
        return row

    async def _contact_for_update(
        self, conn: Any, contact_id: UUID
    ) -> Mapping[str, Any]:
        row = await conn.fetchrow(
            """
            SELECT id, business_context_id, contact_type, customer_type, status
            FROM contacts
            WHERE id = $1::uuid
            FOR UPDATE
            """,
            contact_id,
        )
        if row is None or row["business_context_id"] != _EOM_CONTEXT:
            raise EOMFirstCleanCompletionNotFoundError("EOM customer was not found")
        return row

    @staticmethod
    def _require_eligible_contact(contact: Mapping[str, Any]) -> None:
        if contact["contact_type"] != "customer":
            raise EOMFirstCleanCompletionConflictError("EOM contact is not a customer")
        if contact["status"] != "active":
            raise EOMFirstCleanCompletionConflictError("EOM customer must be active")
        if contact["customer_type"] != "residential":
            raise EOMFirstCleanCompletionConflictError(
                "EOM customer is not residential"
            )

    async def _matching_handoff_for_update(
        self,
        conn: Any,
        *,
        contact_id: UUID,
        tracker_customer_id: int,
        tracker_site_id: int,
    ) -> Mapping[str, Any]:
        handoff = await conn.fetchrow(
            """
            SELECT id, contact_id, tracker_customer_id, tracker_site_id
            FROM eom_customer_handoffs
            WHERE contact_id = $1::uuid
            FOR UPDATE
            """,
            contact_id,
        )
        if handoff is None:
            raise EOMFirstCleanCompletionConflictError(
                "EOM customer has no canonical tracker handoff"
            )
        if (
            int(handoff["tracker_customer_id"]) != tracker_customer_id
            or int(handoff["tracker_site_id"]) != tracker_site_id
        ):
            raise EOMFirstCleanCompletionConflictError(
                "Tracker customer or site does not match the canonical handoff"
            )
        return handoff

    async def _assert_no_existing_completion(
        self,
        conn: Any,
        *,
        contact_id: UUID,
        tracker_service_kind: str,
        tracker_service_id: int,
    ) -> None:
        by_contact = await conn.fetchrow(
            """
            SELECT id
            FROM eom_first_clean_completion_receipts
            WHERE contact_id = $1::uuid
            FOR UPDATE
            """,
            contact_id,
        )
        if by_contact is not None:
            raise EOMFirstCleanCompletionConflictError(
                "EOM customer already has first-clean completion evidence"
            )
        by_service = await conn.fetchrow(
            """
            SELECT id
            FROM eom_first_clean_completion_receipts
            WHERE tracker_service_kind = $1
              AND tracker_service_id = $2
            FOR UPDATE
            """,
            tracker_service_kind,
            tracker_service_id,
        )
        if by_service is not None:
            raise EOMFirstCleanCompletionConflictError(
                "Tracker service already belongs to another first-clean completion"
            )

    async def record_completion(
        self,
        *,
        contact_id: UUID | str,
        tracker_customer_id: int,
        tracker_site_id: int,
        tracker_service_kind: str,
        tracker_service_id: int,
        completed_at: datetime,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Persist exactly one actual first-clean completion report.

        The caller must keep the same Idempotency-Key for an unchanged retry.
        A distinct operation key cannot silently reconcile a second source
        identity or changed completion fact; it fails closed for operator
        review, preserving the original immutable evidence.
        """

        contact_id = _uuid(contact_id, "Contact id")
        tracker_customer_id = _positive_int(tracker_customer_id, "Tracker customer id")
        tracker_site_id = _positive_int(tracker_site_id, "Tracker site id")
        tracker_service_kind = _service_kind(tracker_service_kind)
        tracker_service_id = _positive_int(tracker_service_id, "Tracker service id")
        operation_key = _operation_key(operation_key)
        actor_id, actor_name = _actor(actor_id, actor_name)
        lifecycle_actor = _lifecycle_actor(actor_id, actor_name)
        now = self._now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise EOMFirstCleanCompletionUnavailableError(
                "First-clean completion clock is unavailable"
            )
        now = now.astimezone(timezone.utc)
        completed_at = _completed_at(completed_at, now=now)
        fingerprint = _fingerprint(
            {
                "contactId": str(contact_id),
                "trackerCustomerId": tracker_customer_id,
                "trackerSiteId": tracker_site_id,
                "trackerServiceKind": tracker_service_kind,
                "trackerServiceId": tracker_service_id,
                "completedAt": _utc_iso(completed_at),
                "actorId": actor_id,
                "actorName": actor_name,
                "operation": "first_clean_completion",
            }
        )
        receipt_id = uuid4()
        try:
            async with self.pool.transaction() as conn:
                lock_keys = sorted(
                    {
                        f"eom-first-clean-completion:contact:{contact_id}",
                        f"eom-first-clean-completion:operation:{operation_key}",
                        "eom-first-clean-completion:service:"
                        f"{tracker_service_kind}:{tracker_service_id}",
                    }
                )
                for lock_key in lock_keys:
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        lock_key,
                    )

                contact = await self._contact_for_update(conn, contact_id)
                replayed = await self._bind_operation_receipt(
                    conn,
                    contact_id=contact_id,
                    operation_key=operation_key,
                    fingerprint=fingerprint,
                )
                if replayed:
                    return _receipt_result(
                        await self._receipt_for_operation(
                            conn, operation_key=operation_key
                        ),
                        idempotent=True,
                    )

                self._require_eligible_contact(contact)
                handoff = await self._matching_handoff_for_update(
                    conn,
                    contact_id=contact_id,
                    tracker_customer_id=tracker_customer_id,
                    tracker_site_id=tracker_site_id,
                )
                handoff_id = _uuid(handoff["id"], "Canonical handoff id")
                await self._assert_no_existing_completion(
                    conn,
                    contact_id=contact_id,
                    tracker_service_kind=tracker_service_kind,
                    tracker_service_id=tracker_service_id,
                )

                await conn.execute(
                    """
                    INSERT INTO eom_lead_lifecycle_events (
                        contact_id, event_type, actor, source, operation_key,
                        metadata, occurred_at
                    ) VALUES (
                        $1, 'first_clean_completed', $2, 'time_tracker', $3,
                        jsonb_build_object(
                            'completion_receipt_id', $4::text,
                            'handoff_id', $5::text,
                            'tracker_customer_id', $6::bigint,
                            'tracker_site_id', $7::bigint,
                            'tracker_service_kind', $8::text,
                            'tracker_service_id', $9::bigint
                        ),
                        $10
                    )
                    """,
                    contact_id,
                    lifecycle_actor,
                    operation_key,
                    str(receipt_id),
                    str(handoff_id),
                    tracker_customer_id,
                    tracker_site_id,
                    tracker_service_kind,
                    tracker_service_id,
                    completed_at,
                )
                receipt = await conn.fetchrow(
                    """
                    INSERT INTO eom_first_clean_completion_receipts (
                        id, contact_id, handoff_id, tracker_customer_id,
                        tracker_site_id, tracker_service_kind, tracker_service_id,
                        completed_at, operation_key, request_fingerprint,
                        actor_id, actor_name, source
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                        'time_tracker'
                    )
                    RETURNING id, contact_id, handoff_id, tracker_customer_id,
                              tracker_site_id, tracker_service_kind,
                              tracker_service_id, completed_at, created_at
                    """,
                    receipt_id,
                    contact_id,
                    handoff_id,
                    tracker_customer_id,
                    tracker_site_id,
                    tracker_service_kind,
                    tracker_service_id,
                    completed_at,
                    operation_key,
                    fingerprint,
                    actor_id,
                    actor_name,
                )
                if receipt is None:
                    raise EOMFirstCleanCompletionUnavailableError(
                        "First-clean completion receipt could not be recorded"
                    )
                return _receipt_result(receipt, idempotent=False)
        except EOMFirstCleanCompletionError:
            raise
        except (asyncpg.PostgresError, OSError) as exc:
            raise EOMFirstCleanCompletionUnavailableError(
                "First-clean completion could not be recorded"
            ) from exc
