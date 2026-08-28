"""Versioned, immutable Terms authority for Effingham Office Maids."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

import asyncpg


EOM_TERMS_AUDIENCES = ("residential", "commercial")
EOM_TERMS_LOCALES = ("en", "es")
EOM_TERMS_DOCUMENT_FIELDS = (
    "terms",
    "servicesWeCannotProvide",
    "additionalWorkAcknowledgement",
)
_VERSION_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_SECTION_LENGTH = 100_000
_MAX_ACTOR_NAME_LENGTH = 128
_MAX_SIGNED_BIGINT = 2**63 - 1
EOM_TERMS_PUBLICATION_LOCK_KEY = "eom-terms-current-version"


class EOMTermsAuthorityError(Exception):
    """Base class for stable private-API failures."""

    status_code = 409
    code = "eom_terms_authority_error"


class EOMTermsValidationError(EOMTermsAuthorityError):
    status_code = 422
    code = "invalid_eom_terms_request"


class EOMTermsConflictError(EOMTermsAuthorityError):
    status_code = 409
    code = "eom_terms_conflict"


class EOMTermsNotFoundError(EOMTermsAuthorityError):
    status_code = 404
    code = "eom_terms_not_found"


class EOMTermsUnavailableError(EOMTermsAuthorityError):
    status_code = 503
    code = "eom_terms_unavailable"


def _uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EOMTermsValidationError("Terms version id is invalid") from exc


def _version_label(value: object) -> str:
    if not isinstance(value, str):
        raise EOMTermsValidationError("Terms version label is invalid")
    normalized = value.strip()
    if not _VERSION_LABEL_PATTERN.fullmatch(normalized):
        raise EOMTermsValidationError("Terms version label is invalid")
    return normalized


def _actor(actor_id: object, actor_name: object) -> tuple[int, str]:
    if (
        isinstance(actor_id, bool)
        or not isinstance(actor_id, int)
        or actor_id <= 0
        or actor_id > _MAX_SIGNED_BIGINT
    ):
        raise EOMTermsValidationError("Authenticated actor is invalid")
    if not isinstance(actor_name, str):
        raise EOMTermsValidationError("Authenticated actor is invalid")
    normalized_name = actor_name.strip()
    if (
        not normalized_name
        or len(normalized_name) > _MAX_ACTOR_NAME_LENGTH
        or "\x00" in normalized_name
        or any(0xD800 <= ord(char) <= 0xDFFF for char in normalized_name)
    ):
        raise EOMTermsValidationError("Authenticated actor is invalid")
    return actor_id, normalized_name


def _closed_mapping(
    value: object, expected: tuple[str, ...], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise EOMTermsValidationError(
            f"{label} must contain exactly: {', '.join(expected)}"
        )
    return value


def _section(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EOMTermsValidationError(f"{label} must be text")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_SECTION_LENGTH
        or "\x00" in normalized
        or any(0xD800 <= ord(char) <= 0xDFFF for char in normalized)
    ):
        raise EOMTermsValidationError(f"{label} is invalid")
    return normalized


def normalize_eom_terms_documents(
    documents: object,
) -> dict[str, dict[str, dict[str, str]]]:
    """Admit one exact audience x locale x section bundle."""

    audiences = _closed_mapping(documents, EOM_TERMS_AUDIENCES, "documents")
    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for audience in EOM_TERMS_AUDIENCES:
        locales = _closed_mapping(audiences[audience], EOM_TERMS_LOCALES, audience)
        normalized[audience] = {}
        for locale in EOM_TERMS_LOCALES:
            fields = _closed_mapping(
                locales[locale],
                EOM_TERMS_DOCUMENT_FIELDS,
                f"{audience}.{locale}",
            )
            normalized[audience][locale] = {
                field: _section(fields[field], f"{audience}.{locale}.{field}")
                for field in EOM_TERMS_DOCUMENT_FIELDS
            }
    return normalized


def canonical_eom_terms_documents(
    documents: object,
) -> tuple[dict[str, dict[str, dict[str, str]]], str, str]:
    normalized = normalize_eom_terms_documents(documents)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        normalized,
        serialized,
        hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )


def _documents_from_row(value: object) -> dict[str, dict[str, dict[str, str]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise EOMTermsUnavailableError(
                "Stored Terms documents are invalid"
            ) from exc
    try:
        return normalize_eom_terms_documents(value)
    except EOMTermsValidationError as exc:
        raise EOMTermsUnavailableError("Stored Terms documents are invalid") from exc


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EOMTermsUnavailableError("Stored Terms timestamp is invalid")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _version_result(row: Mapping[str, Any], *, idempotent: bool) -> dict[str, Any]:
    documents = _documents_from_row(row["documents"])
    _, _, calculated_hash = canonical_eom_terms_documents(documents)
    stored_hash = row["content_hash"]
    if not isinstance(stored_hash, str) or stored_hash != calculated_hash:
        raise EOMTermsUnavailableError("Stored Terms content hash is invalid")
    return {
        "versionId": str(row["id"]),
        "versionLabel": str(row["version_label"]),
        "status": str(row["status"]),
        "materialChange": bool(row["material_change"]),
        "documents": documents,
        "contentHash": stored_hash,
        "createdById": int(row["created_by_id"]),
        "createdByName": str(row["created_by_name"]),
        "createdAt": _iso(row["created_at"]),
        "publishedById": (
            int(row["published_by_id"])
            if row.get("published_by_id") is not None
            else None
        ),
        "publishedByName": row.get("published_by_name"),
        "publishedAt": _iso(row.get("published_at")),
        "idempotent": idempotent,
    }


async def eom_terms_authority_schema_ready(pool: Any) -> bool:
    """Return whether the connected runtime has the exact guarded boundary."""

    try:
        return bool(
            await pool.fetchval(
                """
                WITH boundary AS (
                    SELECT pg_catalog.current_schema() AS schema_name,
                           to_regclass(format(
                               '%I.eom_terms_versions',
                               pg_catalog.current_schema()
                           )) AS versions_oid,
                           to_regclass(format(
                               '%I.eom_terms_current_version',
                               pg_catalog.current_schema()
                           )) AS current_oid,
                           to_regprocedure(format(
                               '%I.protect_eom_terms_version()',
                               pg_catalog.current_schema()
                           )) AS protect_version_oid,
                           to_regprocedure(format(
                               '%I.require_published_eom_terms_current_version()',
                               pg_catalog.current_schema()
                           )) AS require_current_oid,
                           to_regprocedure(format(
                               '%I.prevent_eom_terms_current_removal()',
                               pg_catalog.current_schema()
                           )) AS prevent_current_removal_oid,
                           (
                               SELECT oid
                                 FROM pg_roles
                                WHERE rolname = 'atlas'
                           ) AS runtime_oid,
                           (
                               SELECT oid
                                 FROM pg_roles
                                WHERE rolname = 'atlas_eom_handoff_owner'
                           ) AS guard_oid
                )
                SELECT boundary.versions_oid IS NOT NULL
                   AND boundary.current_oid IS NOT NULL
                   AND current_user = 'atlas'
                   AND session_user = 'atlas'
                   AND EXISTS (
                       SELECT 1
                         FROM pg_roles AS runtime_role
                        WHERE runtime_role.oid = boundary.runtime_oid
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
                                     runtime_role.oid,
                                     elevated_role.oid,
                                     'MEMBER'
                                 )
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
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_stat_activity AS activity
                        WHERE activity.usesysid = boundary.guard_oid
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
                              boundary.guard_oid,
                              'MEMBER'
                          )
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_namespace AS namespace
                        WHERE namespace.nspname = boundary.schema_name
                          AND namespace.nspowner = boundary.guard_oid
                   )
                   AND (
                       SELECT COUNT(*) = 2
                         FROM pg_class AS relation
                        WHERE relation.oid IN (
                                  boundary.versions_oid,
                                  boundary.current_oid
                              )
                          AND relation.relkind = 'r'
                          AND relation.relowner = boundary.guard_oid
                   )
                   AND (
                       SELECT COUNT(*) = 3
                         FROM pg_proc AS protected_function
                        WHERE protected_function.oid IN (
                                  boundary.protect_version_oid,
                                  boundary.require_current_oid,
                                  boundary.prevent_current_removal_oid
                              )
                          AND protected_function.pronargs = 0
                          AND protected_function.proowner = boundary.guard_oid
                          AND NOT protected_function.prosecdef
                          AND protected_function.prorettype = 'trigger'::regtype
                          AND protected_function.proconfig = ARRAY[
                              format(
                                  'search_path=pg_catalog, %I, pg_temp',
                                  boundary.schema_name
                              )
                          ]
                   )
                   AND (
                       SELECT COUNT(*) = 5
                          AND BOOL_AND(
                              guard_trigger.tgenabled = 'O'
                              AND (
                                  (
                                      guard_trigger.tgrelid =
                                          boundary.versions_oid
                                      AND guard_trigger.tgfoid =
                                          boundary.protect_version_oid
                                      AND guard_trigger.tgname IN (
                                          'trg_protect_eom_terms_version',
                                          'trg_protect_eom_terms_version_truncate'
                                      )
                                  )
                                  OR (
                                      guard_trigger.tgrelid =
                                          boundary.current_oid
                                      AND guard_trigger.tgfoid =
                                          boundary.require_current_oid
                                      AND guard_trigger.tgname =
                                          'trg_require_published_eom_terms_current_version'
                                  )
                                  OR (
                                      guard_trigger.tgrelid =
                                          boundary.current_oid
                                      AND guard_trigger.tgfoid =
                                          boundary.prevent_current_removal_oid
                                      AND guard_trigger.tgname IN (
                                          'trg_prevent_eom_terms_current_delete',
                                          'trg_prevent_eom_terms_current_truncate'
                                      )
                                  )
                              )
                          )
                         FROM pg_trigger AS guard_trigger
                        WHERE NOT guard_trigger.tgisinternal
                          AND guard_trigger.tgrelid IN (
                              boundary.versions_oid,
                              boundary.current_oid
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM (
                             VALUES
                                 ('eom_terms_versions',
                                  'pk_eom_terms_versions', 'p'),
                                 ('eom_terms_versions',
                                  'uq_eom_terms_version_label', 'u'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_context', 'c'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_version_label', 'c'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_status', 'c'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_documents_object', 'c'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_content_hash', 'c'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_creator', 'c'),
                                 ('eom_terms_versions',
                                  'ck_eom_terms_publication', 'c'),
                                 ('eom_terms_current_version',
                                  'pk_eom_terms_current_version', 'p'),
                                 ('eom_terms_current_version',
                                  'ck_eom_terms_current_singleton', 'c'),
                                 ('eom_terms_current_version',
                                  'uq_eom_terms_current_version', 'u'),
                                 ('eom_terms_current_version',
                                  'fk_eom_terms_current_version', 'f'),
                                 ('eom_terms_current_version',
                                  'ck_eom_terms_current_selector_id', 'c'),
                                 ('eom_terms_current_version',
                                  'ck_eom_terms_current_selector_name', 'c')
                         ) AS expected_constraint(
                             relation_name,
                             constraint_name,
                             constraint_type
                         )
                        WHERE NOT EXISTS (
                            SELECT 1
                              FROM pg_constraint AS actual_constraint
                             WHERE actual_constraint.conrelid = to_regclass(
                                       format(
                                           '%I.%I',
                                           boundary.schema_name,
                                           expected_constraint.relation_name
                                       )
                                   )
                               AND actual_constraint.conname =
                                   expected_constraint.constraint_name
                               AND actual_constraint.contype::text =
                                   expected_constraint.constraint_type
                               AND actual_constraint.convalidated
                        )
                   )
                   AND (
                       SELECT COUNT(DISTINCT indexed_attribute.attname) = 2
                         FROM pg_index AS unique_index
                         JOIN pg_attribute AS indexed_attribute
                           ON indexed_attribute.attrelid = unique_index.indrelid
                          AND indexed_attribute.attnum = unique_index.indkey[0]
                          AND NOT indexed_attribute.attisdropped
                        WHERE unique_index.indrelid = boundary.versions_oid
                          AND unique_index.indisunique
                          AND unique_index.indisvalid
                          AND unique_index.indpred IS NULL
                          AND unique_index.indexprs IS NULL
                          AND unique_index.indnkeyatts = 1
                          AND unique_index.indnatts = 1
                          AND indexed_attribute.attname IN (
                              'id', 'version_label'
                          )
                   )
                   AND (
                       SELECT COUNT(DISTINCT indexed_attribute.attname) = 2
                         FROM pg_index AS unique_index
                         JOIN pg_attribute AS indexed_attribute
                           ON indexed_attribute.attrelid = unique_index.indrelid
                          AND indexed_attribute.attnum = unique_index.indkey[0]
                          AND NOT indexed_attribute.attisdropped
                        WHERE unique_index.indrelid = boundary.current_oid
                          AND unique_index.indisunique
                          AND unique_index.indisvalid
                          AND unique_index.indpred IS NULL
                          AND unique_index.indexprs IS NULL
                          AND unique_index.indnkeyatts = 1
                          AND unique_index.indnatts = 1
                          AND indexed_attribute.attname IN (
                              'singleton', 'version_id'
                          )
                   )
                   AND EXISTS (
                       SELECT 1
                         FROM pg_constraint AS foreign_key
                        WHERE foreign_key.contype = 'f'
                          AND foreign_key.conrelid = boundary.current_oid
                          AND foreign_key.confrelid = boundary.versions_oid
                          AND foreign_key.conkey = ARRAY[
                              (
                                  SELECT attnum
                                    FROM pg_attribute
                                   WHERE attrelid = boundary.current_oid
                                     AND attname = 'version_id'
                                     AND NOT attisdropped
                              )
                          ]::smallint[]
                          AND foreign_key.confkey = ARRAY[
                              (
                                  SELECT attnum
                                    FROM pg_attribute
                                   WHERE attrelid = boundary.versions_oid
                                     AND attname = 'id'
                                     AND NOT attisdropped
                              )
                          ]::smallint[]
                          AND foreign_key.confdeltype = 'r'
                          AND foreign_key.convalidated
                          AND NOT foreign_key.condeferrable
                   )
                   AND has_schema_privilege(
                       current_user, boundary.schema_name, 'USAGE'
                   )
                   AND has_table_privilege(
                       current_user, boundary.versions_oid, 'SELECT'
                   )
                   AND has_table_privilege(
                       current_user, boundary.current_oid, 'SELECT'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.versions_oid, 'INSERT'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.versions_oid, 'UPDATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.current_oid, 'INSERT'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.current_oid, 'UPDATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.versions_oid, 'DELETE'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.versions_oid, 'TRUNCATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.versions_oid, 'REFERENCES'
                   )
                   AND NOT has_any_column_privilege(
                       current_user, boundary.versions_oid, 'REFERENCES'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.versions_oid, 'TRIGGER'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.current_oid, 'DELETE'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.current_oid, 'TRUNCATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.current_oid, 'REFERENCES'
                   )
                   AND NOT has_any_column_privilege(
                       current_user, boundary.current_oid, 'REFERENCES'
                   )
                   AND NOT has_table_privilege(
                       current_user, boundary.current_oid, 'TRIGGER'
                   )
                   AND (
                       SELECT COUNT(*) = 7
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.versions_oid
                          AND attribute.attname IN (
                              'id',
                              'version_label',
                              'material_change',
                              'documents',
                              'content_hash',
                              'created_by_id',
                              'created_by_name'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.versions_oid,
                              attribute.attnum,
                              'INSERT'
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.versions_oid
                          AND attribute.attnum > 0
                          AND NOT attribute.attisdropped
                          AND attribute.attname NOT IN (
                              'id',
                              'version_label',
                              'material_change',
                              'documents',
                              'content_hash',
                              'created_by_id',
                              'created_by_name'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.versions_oid,
                              attribute.attnum,
                              'INSERT'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 4
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.versions_oid
                          AND attribute.attname IN (
                              'status',
                              'published_by_id',
                              'published_by_name',
                              'published_at'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.versions_oid,
                              attribute.attnum,
                              'UPDATE'
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.versions_oid
                          AND attribute.attnum > 0
                          AND NOT attribute.attisdropped
                          AND attribute.attname NOT IN (
                              'status',
                              'published_by_id',
                              'published_by_name',
                              'published_at'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.versions_oid,
                              attribute.attnum,
                              'UPDATE'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 5
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.current_oid
                          AND attribute.attname IN (
                              'singleton',
                              'version_id',
                              'selected_by_id',
                              'selected_by_name', 'selected_at'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.current_oid,
                              attribute.attnum,
                              'INSERT'
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.current_oid
                          AND attribute.attnum > 0
                          AND NOT attribute.attisdropped
                          AND attribute.attname NOT IN (
                              'singleton',
                              'version_id',
                              'selected_by_id',
                              'selected_by_name', 'selected_at'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.current_oid,
                              attribute.attnum,
                              'INSERT'
                          )
                   )
                   AND (
                       SELECT COUNT(*) = 4
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.current_oid
                          AND attribute.attname IN (
                              'version_id',
                              'selected_by_id',
                              'selected_by_name',
                              'selected_at'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.current_oid,
                              attribute.attnum,
                              'UPDATE'
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS attribute
                        WHERE attribute.attrelid = boundary.current_oid
                          AND attribute.attnum > 0
                          AND NOT attribute.attisdropped
                          AND attribute.attname NOT IN (
                              'version_id',
                              'selected_by_id',
                              'selected_by_name',
                              'selected_at'
                          )
                          AND has_column_privilege(
                              current_user,
                              boundary.current_oid,
                              attribute.attnum,
                              'UPDATE'
                          )
                   )
                   AND NOT has_function_privilege(
                       current_user,
                       boundary.protect_version_oid,
                       'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user,
                       boundary.require_current_oid,
                       'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user,
                       boundary.prevent_current_removal_oid,
                       'EXECUTE'
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_class AS relation
                         CROSS JOIN LATERAL aclexplode(
                             relation.relacl
                         ) AS acl
                        WHERE relation.oid IN (
                                  boundary.versions_oid,
                                  boundary.current_oid
                              )
                          AND (
                              acl.grantee NOT IN (
                                  boundary.runtime_oid,
                                  boundary.guard_oid
                              )
                              OR acl.is_grantable
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_attribute AS attribute
                         CROSS JOIN LATERAL aclexplode(
                             attribute.attacl
                         ) AS acl
                        WHERE attribute.attrelid IN (
                                  boundary.versions_oid,
                                  boundary.current_oid
                              )
                          AND (
                              acl.grantee <> boundary.runtime_oid
                              OR acl.is_grantable
                          )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_proc AS protected_function
                         CROSS JOIN LATERAL aclexplode(
                             protected_function.proacl
                         ) AS acl
                        WHERE protected_function.oid IN (
                                  boundary.protect_version_oid,
                                  boundary.require_current_oid,
                                  boundary.prevent_current_removal_oid
                              )
                          AND (
                              acl.grantee <> boundary.guard_oid
                              OR acl.is_grantable
                          )
                   )
                  FROM boundary
                """
            )
        )
    except Exception:
        return False


class EOMTermsAuthority:
    """Own Terms draft identity, publication serialization, and current reads."""

    def __init__(self, *, pool: Any) -> None:
        self._pool = pool

    @property
    def pool(self) -> Any:
        if not bool(getattr(self._pool, "is_initialized", True)):
            raise EOMTermsUnavailableError("Terms database is unavailable")
        return self._pool

    async def require_schema_ready(self) -> None:
        if not await eom_terms_authority_schema_ready(self.pool):
            raise EOMTermsUnavailableError("Terms authority schema is unavailable")

    async def create_draft(
        self,
        *,
        version_label: object,
        material_change: object,
        documents: object,
        actor_id: object,
        actor_name: object,
    ) -> dict[str, Any]:
        label = _version_label(version_label)
        if not isinstance(material_change, bool):
            raise EOMTermsValidationError("materialChange must be boolean")
        _, serialized, content_hash = canonical_eom_terms_documents(documents)
        parsed_actor_id, parsed_actor_name = _actor(actor_id, actor_name)
        await self.require_schema_ready()
        try:
            async with self.pool.transaction() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO eom_terms_versions (
                        id, version_label, material_change, documents,
                        content_hash, created_by_id, created_by_name
                    ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                    ON CONFLICT (version_label) DO NOTHING
                    RETURNING *
                    """,
                    uuid4(),
                    label,
                    material_change,
                    serialized,
                    content_hash,
                    parsed_actor_id,
                    parsed_actor_name,
                )
                if row is not None:
                    return _version_result(row, idempotent=False)
                row = await conn.fetchrow(
                    "SELECT * FROM eom_terms_versions WHERE version_label = $1",
                    label,
                )
                if row is None:
                    raise EOMTermsUnavailableError("Terms draft is unavailable")
                if (
                    str(row["content_hash"]) != content_hash
                    or bool(row["material_change"]) is not material_change
                    or _documents_from_row(row["documents"])
                    != normalize_eom_terms_documents(documents)
                ):
                    raise EOMTermsConflictError(
                        "Terms version label belongs to different content"
                    )
                return _version_result(row, idempotent=True)
        except EOMTermsAuthorityError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsUnavailableError("Terms draft could not be stored") from exc

    async def publish(
        self,
        *,
        version_id: object,
        actor_id: object,
        actor_name: object,
    ) -> dict[str, Any]:
        parsed_version_id = _uuid(version_id)
        parsed_actor_id, parsed_actor_name = _actor(actor_id, actor_name)
        await self.require_schema_ready()
        try:
            async with self.pool.transaction() as conn:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                publication_timestamp = await conn.fetchval("SELECT clock_timestamp()")
                row = await conn.fetchrow(
                    "SELECT * FROM eom_terms_versions WHERE id = $1 FOR UPDATE",
                    parsed_version_id,
                )
                if row is None:
                    raise EOMTermsNotFoundError("Terms version was not found")
                current_id = await conn.fetchval(
                    "SELECT version_id FROM eom_terms_current_version WHERE singleton"
                )
                if str(row["status"]) == "published":
                    if current_id == parsed_version_id:
                        return _version_result(row, idempotent=True)
                    raise EOMTermsConflictError(
                        "Published Terms version is not the current version"
                    )
                row = await conn.fetchrow(
                    """
                    UPDATE eom_terms_versions
                    SET status = 'published',
                        published_by_id = $2,
                        published_by_name = $3,
                        published_at = $4
                    WHERE id = $1 AND status = 'draft'
                    RETURNING *
                    """,
                    parsed_version_id,
                    parsed_actor_id,
                    parsed_actor_name,
                    publication_timestamp,
                )
                if row is None:
                    raise EOMTermsConflictError("Terms version could not be published")
                await conn.execute(
                    """
                    INSERT INTO eom_terms_current_version (
                        singleton, version_id, selected_by_id, selected_by_name,
                        selected_at
                    ) VALUES (TRUE, $1, $2, $3, $4)
                    ON CONFLICT (singleton) DO UPDATE
                    SET version_id = EXCLUDED.version_id,
                        selected_by_id = EXCLUDED.selected_by_id,
                        selected_by_name = EXCLUDED.selected_by_name,
                        selected_at = EXCLUDED.selected_at
                    """,
                    parsed_version_id,
                    parsed_actor_id,
                    parsed_actor_name,
                    publication_timestamp,
                )
                return _version_result(row, idempotent=False)
        except EOMTermsAuthorityError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsUnavailableError(
                "Terms version could not be published"
            ) from exc

    async def get_current(self) -> dict[str, Any]:
        await self.require_schema_ready()
        try:
            row = await self.pool.fetchrow(
                """
                SELECT version.*
                FROM eom_terms_current_version AS selected
                JOIN eom_terms_versions AS version ON version.id = selected.version_id
                WHERE selected.singleton AND version.status = 'published'
                """
            )
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsUnavailableError(
                "Current Terms version is unavailable"
            ) from exc
        if row is None:
            raise EOMTermsNotFoundError("Current Terms version was not found")
        return _version_result(row, idempotent=True)
