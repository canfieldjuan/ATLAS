"""Immutable service-commitment decisions for post-clean card policy."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

import asyncpg


_EOM_CONTEXT = "effingham_maids"
_MAX_SIGNED_BIGINT = 2**63 - 1
_OPERATION_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_SERVICE_COMMITMENTS = frozenset({"recurring", "one_time"})


class EOMCardServiceCommitmentError(Exception):
    """Base class for stable service-commitment API failures."""

    status_code = 409
    code = "card_service_commitment_error"


class EOMCardServiceCommitmentValidationError(EOMCardServiceCommitmentError):
    status_code = 422
    code = "invalid_card_service_commitment_request"


class EOMCardServiceCommitmentNotFoundError(EOMCardServiceCommitmentError):
    status_code = 404
    code = "card_service_commitment_not_found"


class EOMCardServiceCommitmentConflictError(EOMCardServiceCommitmentError):
    status_code = 409
    code = "card_service_commitment_conflict"


class EOMCardServiceCommitmentUnavailableError(EOMCardServiceCommitmentError):
    status_code = 503
    code = "card_service_commitment_unavailable"


def _uuid(value: object, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EOMCardServiceCommitmentValidationError(f"{field} is invalid") from exc


def _operation_key(value: object) -> str:
    if not isinstance(value, str):
        raise EOMCardServiceCommitmentValidationError("Idempotency key is required")
    parsed = value.strip()
    if not _OPERATION_KEY_PATTERN.fullmatch(parsed):
        raise EOMCardServiceCommitmentValidationError("Idempotency key is invalid")
    return parsed


def _service_commitment(value: object) -> str:
    if not isinstance(value, str) or value not in _SERVICE_COMMITMENTS:
        raise EOMCardServiceCommitmentValidationError("Service commitment is invalid")
    return value


def _actor(actor_id: object, actor_name: object) -> tuple[int, str]:
    if (
        isinstance(actor_id, bool)
        or not isinstance(actor_id, int)
        or actor_id <= 0
        or actor_id > _MAX_SIGNED_BIGINT
        or not isinstance(actor_name, str)
    ):
        raise EOMCardServiceCommitmentValidationError("Authenticated actor is invalid")
    parsed_name = actor_name.strip()
    if not parsed_name or len(parsed_name) > 128 or "\x00" in parsed_name:
        raise EOMCardServiceCommitmentValidationError("Authenticated actor is invalid")
    return actor_id, parsed_name


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result(row: Mapping[str, Any], *, idempotent: bool) -> dict[str, Any]:
    decided_at = row["decided_at"]
    if not isinstance(decided_at, datetime) or decided_at.tzinfo is None:
        raise EOMCardServiceCommitmentUnavailableError(
            "Stored service commitment is invalid"
        )
    return {
        "candidateId": str(row["candidate_id"]),
        "contactId": str(row["contact_id"]),
        "serviceCommitment": str(row["service_commitment"]),
        "decidedByName": str(row["decided_by_name"]),
        "decidedAt": decided_at.astimezone(timezone.utc),
        "idempotent": idempotent,
    }


async def eom_card_service_commitment_schema_ready(pool: Any) -> bool:
    """Attest the immutable decision relation and both database guards."""

    try:
        return bool(
            await pool.fetchval(
                """
                /* eom_card_service_commitment_schema_ready */
                WITH expected_triggers(
                    relation_name, function_name, trigger_name, trigger_type,
                    function_body_md5
                ) AS (
                    VALUES
                        ('eom_post_clean_service_commitments',
                         'protect_eom_card_service_commitment',
                         'trg_protect_eom_card_service_commitment', 31,
                         '19c2265038039de0ba5041094e5b9dae'),
                        ('eom_post_clean_service_commitments',
                         'protect_eom_card_service_commitment',
                         'trg_protect_eom_card_service_commitment_truncate', 34,
                         '19c2265038039de0ba5041094e5b9dae'),
                        ('eom_card_vault_enrollments',
                         'require_eom_recurring_card_commitment',
                         'trg_require_eom_recurring_card_commitment', 7,
                         '4cf617f6c456cc863902b1a5e2bbe27a')
                ),
                expected_constraints(name, type, definition) AS (
                    VALUES
                        ('pk_eom_post_clean_service_commitments', 'p',
                         'PRIMARY KEY (id)'),
                        ('uq_eom_post_clean_service_commitment_candidate', 'u',
                         'UNIQUE (candidate_id)'),
                        ('uq_eom_post_clean_service_commitment_contact', 'u',
                         'UNIQUE (contact_id)'),
                        ('uq_eom_post_clean_service_commitment_operation', 'u',
                         'UNIQUE (operation_key)'),
                        ('fk_eom_post_clean_service_commitment_candidate', 'f',
                         'FOREIGN KEY (candidate_id) REFERENCES '
                         'eom_post_clean_onboarding_candidates(id) '
                         'ON DELETE RESTRICT'),
                        ('fk_eom_post_clean_service_commitment_contact', 'f',
                         'FOREIGN KEY (contact_id) REFERENCES contacts(id) '
                         'ON DELETE RESTRICT'),
                        ('ck_eom_post_clean_service_commitment_operation', 'c',
                         'CHECK (operation_key::text ~ '
                         '''^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$''::text)'),
                        ('ck_eom_post_clean_service_commitment_fingerprint', 'c',
                         'CHECK (request_fingerprint::text ~ '
                         '''^[0-9a-f]{64}$''::text)'),
                        ('ck_eom_post_clean_service_commitment_value', 'c',
                         'CHECK (service_commitment::text = ANY '
                         '(ARRAY[''recurring''::character varying, '
                         '''one_time''::character varying]::text[]))'),
                        ('ck_eom_post_clean_service_commitment_context', 'c',
                         'CHECK (business_context_id::text = '
                         '''effingham_maids''::text)'),
                        ('ck_eom_post_clean_service_commitment_actor', 'c',
                         'CHECK (decided_by_employee_id > 0 AND '
                         'char_length(btrim(decided_by_name::text)) >= 1 AND '
                         'char_length(btrim(decided_by_name::text)) <= 128)')
                ),
                expected_columns(name, data_type, not_null, default_expression) AS (
                    VALUES
                        ('id', 'uuid', TRUE, NULL),
                        ('candidate_id', 'uuid', TRUE, NULL),
                        ('contact_id', 'uuid', TRUE, NULL),
                        ('operation_key', 'character varying(128)', TRUE, NULL),
                        ('request_fingerprint', 'character varying(64)', TRUE, NULL),
                        ('service_commitment', 'character varying(16)', TRUE, NULL),
                        ('business_context_id', 'character varying(64)', TRUE,
                         '''effingham_maids''::character varying'),
                        ('decided_by_employee_id', 'bigint', TRUE, NULL),
                        ('decided_by_name', 'character varying(128)', TRUE, NULL),
                        ('decided_at', 'timestamp with time zone', TRUE,
                         'CURRENT_TIMESTAMP')
                ),
                expected_insert_columns(name) AS (
                    VALUES ('id'), ('candidate_id'), ('contact_id'),
                           ('operation_key'), ('request_fingerprint'),
                           ('service_commitment'), ('decided_by_employee_id'),
                           ('decided_by_name')
                ),
                boundary AS (
                    SELECT current_schema() AS schema_name,
                           to_regclass(format(
                               '%I.eom_post_clean_service_commitments',
                               current_schema()
                           )) AS relation_oid,
                           (SELECT oid FROM pg_roles
                             WHERE rolname = 'atlas_eom_handoff_owner') AS guard_oid
                )
                SELECT current_user = 'atlas'
                   AND session_user = 'atlas'
                   AND boundary.relation_oid IS NOT NULL
                   AND boundary.guard_oid IS NOT NULL
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
                          AND NOT pg_has_role(
                              runtime_role.oid, boundary.guard_oid, 'MEMBER'
                          )
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_roles AS guard_role
                        WHERE guard_role.oid = boundary.guard_oid
                          AND NOT guard_role.rolcanlogin
                          AND NOT guard_role.rolinherit
                          AND NOT guard_role.rolsuper
                          AND NOT guard_role.rolcreaterole
                          AND NOT guard_role.rolcreatedb
                          AND NOT guard_role.rolreplication
                          AND NOT guard_role.rolbypassrls
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_namespace AS namespace
                        WHERE namespace.nspname = boundary.schema_name
                          AND namespace.nspowner = boundary.guard_oid
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                        WHERE relation.oid = boundary.relation_oid
                          AND relation.relkind = 'r'
                          AND relation.relowner = boundary.guard_oid
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         CROSS JOIN LATERAL aclexplode(COALESCE(
                             relation.relacl,
                             acldefault('r', relation.relowner)
                         )) AS privilege
                        WHERE relation.oid = boundary.relation_oid
                          AND privilege.grantee = 0
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS column_definition
                         CROSS JOIN LATERAL aclexplode(
                             column_definition.attacl
                         ) AS privilege
                        WHERE column_definition.attrelid = boundary.relation_oid
                          AND column_definition.attnum > 0
                          AND NOT column_definition.attisdropped
                          AND privilege.grantee = 0
                   )
                   AND has_table_privilege(
                       current_user, boundary.relation_oid, 'SELECT'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.relation_oid,
                       'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM expected_columns AS expected
                        WHERE NOT EXISTS (
                            SELECT 1
                              FROM pg_attribute AS column_definition
                              LEFT JOIN pg_attrdef AS stored_default
                                ON stored_default.adrelid =
                                   column_definition.attrelid
                               AND stored_default.adnum = column_definition.attnum
                             WHERE column_definition.attrelid = boundary.relation_oid
                               AND column_definition.attname = expected.name
                               AND column_definition.attnum > 0
                               AND NOT column_definition.attisdropped
                               AND format_type(
                                   column_definition.atttypid,
                                   column_definition.atttypmod
                               ) = expected.data_type
                               AND column_definition.attnotnull = expected.not_null
                               AND column_definition.attidentity = ''
                               AND column_definition.attgenerated = ''
                               AND pg_get_expr(
                                   stored_default.adbin,
                                   stored_default.adrelid
                               ) IS NOT DISTINCT FROM expected.default_expression
                        )
                   )
                   AND (
                       SELECT count(*)
                         FROM pg_attribute AS column_definition
                        WHERE column_definition.attrelid = boundary.relation_oid
                          AND column_definition.attnum > 0
                          AND NOT column_definition.attisdropped
                   ) = (SELECT count(*) FROM expected_columns)
                   AND NOT EXISTS (
                       SELECT 1
                         FROM expected_insert_columns AS expected
                        WHERE NOT has_column_privilege(
                            current_user,
                            boundary.relation_oid,
                            expected.name,
                            'INSERT'
                        )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS column_definition
                        WHERE column_definition.attrelid = boundary.relation_oid
                          AND column_definition.attnum > 0
                          AND NOT column_definition.attisdropped
                          AND (
                              has_column_privilege(
                                  current_user,
                                  boundary.relation_oid,
                                  column_definition.attname,
                                  'UPDATE'
                              )
                              OR (
                                  has_column_privilege(
                                      current_user,
                                      boundary.relation_oid,
                                      column_definition.attname,
                                      'INSERT'
                                  )
                                  AND NOT EXISTS (
                                      SELECT 1
                                        FROM expected_insert_columns AS expected
                                       WHERE expected.name =
                                             column_definition.attname
                                  )
                              )
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM expected_triggers AS expected
                        WHERE NOT EXISTS (
                            SELECT 1
                              FROM pg_trigger AS trigger
                              JOIN pg_class AS relation
                                ON relation.oid = trigger.tgrelid
                              JOIN pg_namespace AS namespace
                                ON namespace.oid = relation.relnamespace
                              JOIN pg_proc AS function
                                ON function.oid = trigger.tgfoid
                              JOIN pg_language AS language
                                ON language.oid = function.prolang
                              JOIN pg_roles AS owner
                                ON owner.oid = function.proowner
                             WHERE namespace.nspname = boundary.schema_name
                               AND relation.relname = expected.relation_name
                               AND trigger.tgname = expected.trigger_name
                               AND NOT trigger.tgisinternal
                               AND trigger.tgenabled = 'O'
                               AND trigger.tgtype = expected.trigger_type
                               AND trigger.tgattr = ''::int2vector
                               AND trigger.tgnargs = 0
                               AND trigger.tgqual IS NULL
                               AND function.proname = expected.function_name
                               AND function.pronargs = 0
                               AND function.prorettype = 'trigger'::regtype
                               AND language.lanname = 'plpgsql'
                               AND NOT function.prosecdef
                               AND NOT function.proleakproof
                               AND NOT function.proisstrict
                               AND function.provolatile = 'v'
                               AND function.proparallel = 'u'
                               AND function.prokind = 'f'
                               AND function.probin IS NULL
                               AND md5(function.prosrc) =
                                   expected.function_body_md5
                               AND function.proconfig = ARRAY[
                                   format(
                                       'search_path=pg_catalog, %I, pg_temp',
                                       boundary.schema_name
                                   )
                               ]
                               AND owner.oid = boundary.guard_oid
                        )
                   )
                   AND (
                       SELECT count(*)
                         FROM pg_trigger AS trigger
                        WHERE trigger.tgrelid = boundary.relation_oid
                          AND NOT trigger.tgisinternal
                   ) = 2
                   AND NOT EXISTS (
                       SELECT 1
                         FROM expected_constraints AS expected
                        WHERE NOT EXISTS (
                            SELECT 1
                              FROM pg_constraint AS actual
                             WHERE actual.conrelid = boundary.relation_oid
                               AND actual.conname = expected.name
                               AND actual.contype = expected.type
                               AND actual.convalidated
                               AND pg_get_constraintdef(actual.oid, true) =
                                   expected.definition
                        )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_constraint AS actual
                        WHERE actual.conrelid = boundary.relation_oid
                          AND actual.contype IN ('p', 'u', 'f', 'c')
                          AND NOT EXISTS (
                              SELECT 1
                                FROM expected_constraints AS expected
                               WHERE expected.name = actual.conname
                                 AND expected.type = actual.contype
                                 AND expected.definition =
                                     pg_get_constraintdef(actual.oid, true)
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM expected_triggers AS expected
                         JOIN pg_proc AS function
                           ON function.proname = expected.function_name
                          AND function.pronargs = 0
                         JOIN pg_namespace AS namespace
                           ON namespace.oid = function.pronamespace
                         CROSS JOIN LATERAL aclexplode(COALESCE(
                             function.proacl,
                             acldefault('f', function.proowner)
                         )) AS privilege
                        WHERE namespace.nspname = boundary.schema_name
                          AND privilege.grantee IN (0, (
                              SELECT oid FROM pg_roles WHERE rolname = 'atlas'
                          ))
                   )
                FROM boundary
                """
            )
        )
    except (asyncpg.PostgresError, OSError, TimeoutError):
        return False


class EOMCardServiceCommitmentService:
    """Own one immutable operator decision per post-clean candidate."""

    def __init__(self, *, pool: Any) -> None:
        self._pool = pool

    @property
    def pool(self) -> Any:
        if not bool(getattr(self._pool, "is_initialized", True)):
            raise EOMCardServiceCommitmentUnavailableError(
                "Service-commitment database is unavailable"
            )
        return self._pool

    async def require_schema_ready(self) -> None:
        if not await eom_card_service_commitment_schema_ready(self.pool):
            raise EOMCardServiceCommitmentUnavailableError(
                "Service-commitment schema is unavailable"
            )

    async def decide(
        self,
        *,
        candidate_id: UUID | str,
        service_commitment: str,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Record once, replay unchanged, and reject every conflicting claim."""

        parsed_candidate_id = _uuid(candidate_id, "Candidate id")
        parsed_commitment = _service_commitment(service_commitment)
        parsed_key = _operation_key(operation_key)
        parsed_actor_id, parsed_actor_name = _actor(actor_id, actor_name)
        await self.require_schema_ready()
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    parsed_key,
                )
                replay_rows = await connection.fetch(
                    """
                    SELECT *
                    FROM eom_post_clean_service_commitments
                    WHERE candidate_id = $1 OR operation_key = $2
                    """,
                    parsed_candidate_id,
                    parsed_key,
                )
                if replay_rows:
                    exact = next(
                        (
                            row
                            for row in replay_rows
                            if UUID(str(row["candidate_id"])) == parsed_candidate_id
                            and str(row["operation_key"]) == parsed_key
                            and str(row["request_fingerprint"])
                            == _fingerprint(
                                {
                                    "candidateId": str(parsed_candidate_id),
                                    "contactId": str(
                                        _uuid(
                                            row["contact_id"],
                                            "Stored decision contact id",
                                        )
                                    ),
                                    "serviceCommitment": parsed_commitment,
                                    "operationKey": parsed_key,
                                    "actorId": parsed_actor_id,
                                    "actorName": parsed_actor_name,
                                }
                            )
                        ),
                        None,
                    )
                    if exact is None or len(replay_rows) != 1:
                        raise EOMCardServiceCommitmentConflictError(
                            "Service-commitment decision conflicts with existing evidence"
                        )
                    return _result(exact, idempotent=True)
                candidate = await connection.fetchrow(
                    """
                    SELECT candidate.id AS candidate_id,
                           candidate.contact_id,
                           candidate.business_context_id,
                           candidate.status AS candidate_status,
                           contact.business_context_id AS contact_context_id,
                           contact.contact_type,
                           contact.customer_type,
                           contact.status AS contact_status
                    FROM eom_post_clean_onboarding_candidates AS candidate
                    JOIN contacts AS contact ON contact.id = candidate.contact_id
                    WHERE candidate.id = $1
                    FOR UPDATE OF candidate, contact
                    """,
                    parsed_candidate_id,
                )
                if candidate is None:
                    raise EOMCardServiceCommitmentNotFoundError(
                        "Post-clean onboarding candidate was not found"
                    )
                if (
                    str(candidate["business_context_id"]) != _EOM_CONTEXT
                    or str(candidate["contact_context_id"]) != _EOM_CONTEXT
                    or str(candidate["candidate_status"]) != "pending"
                    or str(candidate["contact_type"]) != "customer"
                    or str(candidate["customer_type"]) != "residential"
                    or str(candidate["contact_status"]) != "active"
                ):
                    raise EOMCardServiceCommitmentConflictError(
                        "Post-clean onboarding candidate is not eligible"
                    )
                contact_id = _uuid(candidate["contact_id"], "Candidate contact id")
                fingerprint = _fingerprint(
                    {
                        "candidateId": str(parsed_candidate_id),
                        "contactId": str(contact_id),
                        "serviceCommitment": parsed_commitment,
                        "operationKey": parsed_key,
                        "actorId": parsed_actor_id,
                        "actorName": parsed_actor_name,
                    }
                )
                conflicting_rows = await connection.fetch(
                    """
                    SELECT *
                    FROM eom_post_clean_service_commitments
                    WHERE candidate_id = $1
                       OR contact_id = $2
                       OR operation_key = $3
                    """,
                    parsed_candidate_id,
                    contact_id,
                    parsed_key,
                )
                if conflicting_rows:
                    raise EOMCardServiceCommitmentConflictError(
                        "Service-commitment decision conflicts with existing evidence"
                    )
                row = await connection.fetchrow(
                    """
                    INSERT INTO eom_post_clean_service_commitments (
                        id, candidate_id, contact_id, operation_key,
                        request_fingerprint, service_commitment,
                        decided_by_employee_id, decided_by_name
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING *
                    """,
                    uuid4(),
                    parsed_candidate_id,
                    contact_id,
                    parsed_key,
                    fingerprint,
                    parsed_commitment,
                    parsed_actor_id,
                    parsed_actor_name,
                )
                if row is None:
                    raise EOMCardServiceCommitmentUnavailableError(
                        "Service commitment could not be recorded"
                    )
                return _result(row, idempotent=False)
        except EOMCardServiceCommitmentError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMCardServiceCommitmentUnavailableError(
                "Service commitment could not be recorded"
            ) from exc
