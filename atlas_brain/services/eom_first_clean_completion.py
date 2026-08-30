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

from .eom_card_service_commitment import (
    eom_card_service_commitment_schema_ready,
)


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


def _receipt_result(
    row: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    idempotent: bool,
) -> dict[str, Any]:
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
        "postCleanOnboardingCandidateId": str(candidate["id"]),
        "postCleanOnboardingCandidateStatus": str(candidate["status"]),
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
                   AND to_regclass('eom_post_clean_onboarding_candidates') IS NOT NULL
                   AND (
                       SELECT COUNT(*) = 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname = current_schema()
                          AND relation.relkind = 'r'
                          AND relation.relname =
                              'eom_post_clean_onboarding_candidates'
                          AND relation.relowner = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                   )
                   -- Candidate identity is the at-most-once boundary. Attest
                   -- the actual constrained columns rather than trusting the
                   -- migration or constraint names as evidence.
                   AND (
                       SELECT COUNT(*) = 4
                         FROM pg_constraint AS candidate_constraint
                        WHERE candidate_constraint.conrelid = to_regclass(
                                  format(
                                      '%I.eom_post_clean_onboarding_candidates',
                                      current_schema()
                                  )
                              )
                          AND candidate_constraint.contype IN ('p', 'u')
                          AND candidate_constraint.convalidated
                          AND candidate_constraint.conkey IN (
                              ARRAY[(SELECT attnum FROM pg_attribute
                                      WHERE attrelid = candidate_constraint.conrelid
                                        AND attname = 'id')]::smallint[],
                              ARRAY[(SELECT attnum FROM pg_attribute
                                      WHERE attrelid = candidate_constraint.conrelid
                                        AND attname = 'completion_receipt_id')]::smallint[],
                              ARRAY[(SELECT attnum FROM pg_attribute
                                      WHERE attrelid = candidate_constraint.conrelid
                                        AND attname = 'contact_id')]::smallint[],
                              ARRAY[(SELECT attnum FROM pg_attribute
                                      WHERE attrelid = candidate_constraint.conrelid
                                        AND attname = 'handoff_id')]::smallint[]
                          )
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_constraint AS candidate_contact_fk
                        WHERE candidate_contact_fk.conrelid = to_regclass(
                                  format(
                                      '%I.eom_post_clean_onboarding_candidates',
                                      current_schema()
                                  )
                              )
                          AND candidate_contact_fk.contype = 'f'
                          AND candidate_contact_fk.confrelid = to_regclass(
                                  format('%I.contacts', current_schema())
                              )
                          AND candidate_contact_fk.convalidated
                          AND NOT candidate_contact_fk.condeferrable
                          AND candidate_contact_fk.confdeltype = 'r'
                          AND candidate_contact_fk.conkey = ARRAY[(
                              SELECT attnum FROM pg_attribute
                               WHERE attrelid = candidate_contact_fk.conrelid
                                 AND attname = 'contact_id'
                          )]::smallint[]
                          AND candidate_contact_fk.confkey = ARRAY[(
                              SELECT attnum FROM pg_attribute
                               WHERE attrelid = candidate_contact_fk.confrelid
                                 AND attname = 'id'
                          )]::smallint[]
                   )
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
                          AND NOT EXISTS (
                              SELECT 1
                                FROM pg_database AS database
                               WHERE database.datname = current_database()
                                 AND database.datdba = runtime_role.oid
                          )
                          AND NOT EXISTS (
                              SELECT 1
                                FROM pg_roles AS elevated_role
                               WHERE elevated_role.oid <> runtime_role.oid
                                 AND (
                                     elevated_role.rolsuper
                                     OR elevated_role.rolcreaterole
                                     OR elevated_role.rolcreatedb
                                     OR elevated_role.rolreplication
                                     OR elevated_role.rolbypassrls
                                 )
                                 AND pg_has_role(
                                     runtime_role.oid, elevated_role.oid, 'MEMBER'
                                 )
                          )
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
                   -- `NOLOGIN` prevents a new connection but does not end a
                   -- former direct guard session. That session remains the
                   -- owner of this protected boundary, so serving must fail
                   -- closed until it is gone from the target database.
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_stat_activity AS activity
                         JOIN pg_roles AS guard_role
                           ON guard_role.rolname = 'atlas_eom_handoff_owner'
                        WHERE activity.usesysid = guard_role.oid
                          AND activity.datid = (
                              SELECT database.oid
                                FROM pg_database AS database
                               WHERE database.datname = current_database()
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_roles AS member_role
                        WHERE member_role.rolcanlogin
                          AND NOT member_role.rolsuper
                          AND pg_has_role(
                              member_role.oid,
                              (
                                  SELECT oid
                                    FROM pg_roles
                                   WHERE rolname = 'atlas_eom_handoff_owner'
                              ),
                              'MEMBER'
                          )
                   )
                   -- The runtime must not own the namespace containing the
                   -- guarded relations: schema ownership can DROP SCHEMA ...
                   -- CASCADE despite every table/function ACL below.
                   AND EXISTS (
                       SELECT 1
                         FROM pg_namespace AS namespace
                        WHERE namespace.nspname = current_schema()
                          AND namespace.nspowner = (
                              SELECT oid
                                FROM pg_roles
                               WHERE rolname = 'atlas_eom_handoff_owner'
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
                   AND to_regclass(pg_get_serial_sequence(
                       format('%I.%I', current_schema(), 'eom_lead_lifecycle_events'),
                       'lifecycle_sequence'
                   )) = to_regclass(format(
                       '%I.eom_lead_lifecycle_events_sequence_seq',
                       current_schema()
                   ))
                   AND EXISTS (
                       SELECT 1
                         FROM pg_attrdef AS attribute_default
                         JOIN pg_attribute AS attribute
                           ON attribute.attrelid = attribute_default.adrelid
                          AND attribute.attnum = attribute_default.adnum
                         JOIN pg_depend AS dependency
                           ON dependency.classid = 'pg_attrdef'::regclass
                          AND dependency.objid = attribute_default.oid
                          AND dependency.refclassid = 'pg_class'::regclass
                         JOIN pg_class AS sequence ON sequence.oid = dependency.refobjid
                         JOIN pg_namespace AS sequence_namespace
                           ON sequence_namespace.oid = sequence.relnamespace
                        WHERE attribute_default.adrelid = to_regclass(
                                  format(
                                      '%I.%I',
                                      current_schema(),
                                      'eom_lead_lifecycle_events'
                                  )
                              )
                          AND attribute.attname = 'lifecycle_sequence'
                          AND sequence_namespace.nspname = current_schema()
                          AND sequence.relkind = 'S'
                          AND sequence.relname =
                              'eom_lead_lifecycle_events_sequence_seq'
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
                          AND relation.relkind = 'S'
                          AND relation.relname =
                              'eom_lead_lifecycle_events_sequence_seq'
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
                       SELECT COUNT(*) = 1
                         FROM pg_proc AS protected_function
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = protected_function.pronamespace
                        WHERE namespace.nspname = current_schema()
                          AND protected_function.proname =
                              'require_eom_customer_handoff_finalization'
                          AND protected_function.pronargs = 0
                          AND protected_function.proconfig @> ARRAY[
                              format(
                                  'search_path=pg_catalog, %I, pg_temp',
                                  current_schema()
                              )
                          ]
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
                                  'search_path=pg_catalog, %I, pg_temp',
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
                       SELECT COUNT(*) = 3
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
                          AND relation.relname = 'eom_lead_lifecycle_events'
                          AND acl.grantee = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                          AND acl.privilege_type NOT IN ('SELECT', 'INSERT', 'UPDATE')
                   )
                   AND (
                       SELECT COUNT(*) = 1
                         FROM pg_class AS relation
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN LATERAL aclexplode(
                             COALESCE(relation.relacl, ARRAY[]::aclitem[])
                         ) AS acl
                        WHERE namespace.nspname = current_schema()
                          AND relation.relkind = 'S'
                          AND relation.relname =
                              'eom_lead_lifecycle_events_sequence_seq'
                          AND acl.grantee = (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          )
                          AND acl.privilege_type = 'USAGE'
                          AND NOT acl.is_grantable
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
                          AND relation.relkind = 'S'
                          AND relation.relname =
                              'eom_lead_lifecycle_events_sequence_seq'
                          AND (
                              acl.grantee NOT IN (
                                  SELECT oid
                                    FROM pg_roles
                                   WHERE rolname IN (
                                      'atlas', 'atlas_eom_handoff_owner'
                                   )
                              )
                              OR acl.is_grantable
                              OR (
                                  acl.grantee = (
                                      SELECT oid FROM pg_roles
                                       WHERE rolname = 'atlas'
                                  )
                                  AND acl.privilege_type <> 'USAGE'
                              )
                          )
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
                   AND has_table_privilege(
                       current_user,
                       'eom_lead_lifecycle_events',
                       'UPDATE'
                   )
                   AND has_sequence_privilege(
                       current_user,
                       format(
                           '%I.%I',
                           current_schema(),
                           'eom_lead_lifecycle_events_sequence_seq'
                       ),
                       'USAGE'
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
                   AND (
                       SELECT COUNT(*) = 6
                         FROM (
                             VALUES
                                 ('eom_customer_handoffs', 'contact_id',
                                  'contacts', 'id'),
                                 ('eom_lead_lifecycle_events', 'contact_id',
                                  'contacts', 'id'),
                                 ('eom_first_clean_completion_operation_receipts',
                                  'contact_id', 'contacts', 'id'),
                                 ('eom_first_clean_completion_receipts',
                                  'contact_id', 'contacts', 'id'),
                                 ('eom_first_clean_completion_receipts',
                                  'handoff_id', 'eom_customer_handoffs', 'id'),
                                 ('eom_first_clean_completion_receipts',
                                  'operation_key',
                                  'eom_first_clean_completion_operation_receipts',
                                  'operation_key')
                         ) AS expected(
                             source_relation,
                             source_column,
                             target_relation,
                             target_column
                         )
                        WHERE EXISTS (
                            SELECT 1
                              FROM pg_constraint AS foreign_key
                             WHERE foreign_key.contype = 'f'
                               AND foreign_key.conrelid = to_regclass(
                                   format(
                                       '%I.%I',
                                       current_schema(),
                                       expected.source_relation
                                   )
                               )
                               AND foreign_key.confrelid = to_regclass(
                                   format(
                                       '%I.%I',
                                       current_schema(),
                                       expected.target_relation
                                   )
                               )
                               AND foreign_key.conkey = ARRAY[
                                   (
                                       SELECT attribute.attnum
                                         FROM pg_attribute AS attribute
                                        WHERE attribute.attrelid =
                                              foreign_key.conrelid
                                          AND attribute.attname =
                                              expected.source_column
                                          AND NOT attribute.attisdropped
                                   )
                               ]::smallint[]
                               AND foreign_key.confkey = ARRAY[
                                   (
                                       SELECT attribute.attnum
                                         FROM pg_attribute AS attribute
                                        WHERE attribute.attrelid =
                                              foreign_key.confrelid
                                          AND attribute.attname =
                                              expected.target_column
                                          AND NOT attribute.attisdropped
                                   )
                               ]::smallint[]
                               AND foreign_key.confdeltype = 'r'
                               AND foreign_key.convalidated
                               AND NOT foreign_key.condeferrable
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

    async def _ensure_candidate_for_receipt(
        self,
        conn: Any,
        *,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Create or recover the one non-sendable candidate for a receipt."""

        await conn.execute(
            """
            INSERT INTO eom_post_clean_onboarding_candidates (
                completion_receipt_id, contact_id, handoff_id
            ) VALUES ($1, $2, $3)
            ON CONFLICT (completion_receipt_id) DO NOTHING
            """,
            receipt["id"],
            receipt["contact_id"],
            receipt["handoff_id"],
        )
        candidate = await conn.fetchrow(
            """
            SELECT id, completion_receipt_id, contact_id, handoff_id, status
            FROM eom_post_clean_onboarding_candidates
            WHERE completion_receipt_id = $1
            FOR UPDATE
            """,
            receipt["id"],
        )
        if candidate is None or (
            _uuid(candidate["contact_id"], "Candidate contact id")
            != _uuid(receipt["contact_id"], "Receipt contact id")
            or _uuid(candidate["handoff_id"], "Candidate handoff id")
            != _uuid(receipt["handoff_id"], "Receipt handoff id")
            or str(candidate["status"]) != "pending"
        ):
            raise EOMFirstCleanCompletionUnavailableError(
                "Post-clean onboarding candidate is inconsistent"
            )
        return candidate

    async def list_candidates(
        self,
        *,
        limit: int,
        cursor_created_at: datetime | None = None,
        cursor_candidate_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Read pending candidates with blockers derived from current CRM state."""

        await self.require_schema_ready()
        try:
            commitment_schema_present = bool(
                await self.pool.fetchval(
                    """
                    SELECT to_regclass(format(
                        '%I.eom_post_clean_service_commitments',
                        current_schema()
                    )) IS NOT NULL
                    """
                )
            )
            if commitment_schema_present and not (
                await eom_card_service_commitment_schema_ready(self.pool)
            ):
                raise EOMFirstCleanCompletionUnavailableError(
                    "Service-commitment projection schema is unavailable"
                )
            commitment_columns = (
                "commitment.service_commitment, "
                "commitment.decided_by_name AS service_commitment_decided_by, "
                "commitment.decided_at AS service_commitment_decided_at"
                if commitment_schema_present
                else (
                    "NULL::varchar(16) AS service_commitment, "
                    "NULL::varchar(128) AS service_commitment_decided_by, "
                    "NULL::timestamptz AS service_commitment_decided_at"
                )
            )
            commitment_join = (
                "LEFT JOIN eom_post_clean_service_commitments AS commitment "
                "ON commitment.candidate_id = candidate.id "
                "AND commitment.contact_id = candidate.contact_id"
                if commitment_schema_present
                else ""
            )
            rows = await self.pool.fetch(
                f"""
                SELECT candidate.id AS candidate_id,
                       candidate.completion_receipt_id,
                       candidate.contact_id,
                       candidate.handoff_id,
                       candidate.status,
                       candidate.created_at,
                       contact.full_name,
                       contact.email AS recipient_email,
                       receipt.tracker_service_kind,
                       receipt.tracker_service_id,
                       receipt.completed_at,
                       {commitment_columns},
                       CASE
                           WHEN contact.status <> 'active'
                             OR contact.contact_type <> 'customer'
                               THEN 'inactive_customer'
                           WHEN contact.customer_type <> 'residential'
                               THEN 'not_residential'
                           WHEN contact.email IS NULL OR btrim(contact.email) = ''
                               THEN 'no_email'
                           ELSE NULL
                       END AS blocker
                FROM eom_post_clean_onboarding_candidates AS candidate
                JOIN contacts AS contact ON contact.id = candidate.contact_id
                JOIN eom_first_clean_completion_receipts AS receipt
                  ON receipt.id = candidate.completion_receipt_id
                 AND receipt.contact_id = candidate.contact_id
                 AND receipt.handoff_id = candidate.handoff_id
                {commitment_join}
                WHERE candidate.business_context_id = $1
                  AND contact.business_context_id = $1
                  AND candidate.status = 'pending'
                  AND (
                      $2::timestamptz IS NULL
                      OR (candidate.created_at, candidate.id) < ($2, $3::uuid)
                  )
                ORDER BY candidate.created_at DESC, candidate.id DESC
                LIMIT $4
                """,
                _EOM_CONTEXT,
                cursor_created_at,
                cursor_candidate_id,
                limit,
            )
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMFirstCleanCompletionUnavailableError(
                "Post-clean onboarding candidates are unavailable"
            ) from exc
        return [dict(row) for row in rows]

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
                # The canonical CRM table remains runtime-owned for ordinary
                # CRM writes. Hold a compatible read lock before catalog
                # attestation so a concurrent DROP TABLE contacts CASCADE
                # cannot remove protected foreign keys between readiness and
                # receipt admission.
                await conn.execute(
                    "LOCK TABLE contacts, eom_post_clean_onboarding_candidates "
                    "IN ACCESS SHARE MODE"
                )
                if not await first_clean_completion_schema_ready(conn):
                    raise EOMFirstCleanCompletionUnavailableError(
                        "First-clean completion schema is unavailable"
                    )
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
                    receipt = await self._receipt_for_operation(
                        conn, operation_key=operation_key
                    )
                    candidate = await self._ensure_candidate_for_receipt(
                        conn, receipt=receipt
                    )
                    return _receipt_result(
                        receipt,
                        candidate=candidate,
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
                candidate = await self._ensure_candidate_for_receipt(
                    conn, receipt=receipt
                )
                return _receipt_result(
                    receipt,
                    candidate=candidate,
                    idempotent=False,
                )
        except EOMFirstCleanCompletionError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMFirstCleanCompletionUnavailableError(
                "First-clean completion could not be recorded"
            ) from exc
