"""Customer-bound EOM Terms invitations, acceptance, and delivery evidence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping
from uuid import UUID, uuid4

import asyncpg

from .eom_onboarding_drafts import (
    EOMOnboardingDraftError,
    _require_transport_configured,
    send_onboarding_email,
)
from .eom_public_onboarding_tokens import (
    eom_public_onboarding_hmac_key_fingerprint,
    validate_eom_public_onboarding_hmac_secret,
)
from .eom_terms_authority import (
    EOM_TERMS_PUBLICATION_LOCK_KEY,
    EOMTermsValidationError,
    canonical_eom_terms_documents,
    normalize_eom_terms_documents,
)


logger = logging.getLogger("atlas.services.eom_terms_acceptance")

EOM_TERMS_TOKEN_VERSION = "eomt1"
EOM_TERMS_DELIVERY_KINDS = ("invitation", "executed_copy")
EOM_TERMS_DELIVERY_STATUSES = ("pending", "sending", "sent")
_REQUEST_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HMAC_DIGEST_BYTES = hashlib.sha256().digest_size
_HMAC_SIGNATURE_LENGTH = len(
    base64.urlsafe_b64encode(b"\0" * _HMAC_DIGEST_BYTES).decode("ascii").rstrip("=")
)
_MAX_TERMS_TOKEN_LENGTH = (
    len(EOM_TERMS_TOKEN_VERSION) + 1 + 36 + 1 + _HMAC_SIGNATURE_LENGTH
)
_TERMS_TOKEN_PATTERN = re.compile(
    rf"^{EOM_TERMS_TOKEN_VERSION}\."
    r"(?P<invitation_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})\."
    rf"(?P<signature>[A-Za-z0-9_-]{{{_HMAC_SIGNATURE_LENGTH}}})$"
)
_MAX_SIGNED_BIGINT = 2**63 - 1
_MAX_ACTOR_NAME_LENGTH = 128
_MAX_SIGNER_NAME_LENGTH = 256
_MAX_EMAIL_LENGTH = 256


class EOMTermsAcceptanceError(Exception):
    """Base class for stable Terms invitation/acceptance API failures."""

    status_code = 409
    code = "eom_terms_acceptance_error"


class EOMTermsAcceptanceValidationError(EOMTermsAcceptanceError):
    status_code = 422
    code = "invalid_eom_terms_acceptance_request"


class EOMTermsAcceptanceConflictError(EOMTermsAcceptanceError):
    status_code = 409
    code = "eom_terms_acceptance_conflict"


class EOMTermsAcceptanceNotFoundError(EOMTermsAcceptanceError):
    status_code = 404
    code = "eom_terms_acceptance_not_found"


class EOMTermsAcceptanceUnavailableError(EOMTermsAcceptanceError):
    status_code = 503
    code = "eom_terms_acceptance_unavailable"


@dataclass(frozen=True)
class AuthenticatedEOMTermsToken:
    """A verified Terms bearer and the configured key that admitted it."""

    invitation_id: UUID
    signing_key_fingerprint: str


def _terms_signature(*, invitation_id: UUID, secret: str) -> str:
    message = f"{EOM_TERMS_TOKEN_VERSION}.{invitation_id}".encode("ascii")
    digest = hmac.new(
        validate_eom_public_onboarding_hmac_secret(secret).encode("utf-8"),
        message,
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def format_eom_terms_token(*, invitation_id: UUID, secret: str) -> str:
    """Build the transient Terms bearer without storing its raw value."""

    return (
        f"{EOM_TERMS_TOKEN_VERSION}.{invitation_id}."
        f"{_terms_signature(invitation_id=invitation_id, secret=secret)}"
    )


def authenticate_eom_terms_token(
    *, token: object, secret: str, previous_secret: str | None = None
) -> AuthenticatedEOMTermsToken:
    """Fail closed on every value outside the one Terms bearer grammar."""

    if not isinstance(token, str) or len(token) > _MAX_TERMS_TOKEN_LENGTH:
        raise EOMTermsAcceptanceNotFoundError("Terms invitation is unavailable")
    match = _TERMS_TOKEN_PATTERN.fullmatch(token)
    if match is None:
        raise EOMTermsAcceptanceNotFoundError("Terms invitation is unavailable")
    try:
        invitation_id = UUID(match.group("invitation_id"))
    except ValueError as exc:  # pragma: no cover - the regex is intentionally strict.
        raise EOMTermsAcceptanceNotFoundError(
            "Terms invitation is unavailable"
        ) from exc
    supplied_signature = match.group("signature")
    candidate_secrets = (secret,)
    if previous_secret is not None:
        candidate_secrets = (secret, previous_secret)
    for candidate_secret in candidate_secrets:
        expected = _terms_signature(
            invitation_id=invitation_id,
            secret=candidate_secret,
        )
        if hmac.compare_digest(expected, supplied_signature):
            return AuthenticatedEOMTermsToken(
                invitation_id=invitation_id,
                signing_key_fingerprint=eom_public_onboarding_hmac_key_fingerprint(
                    secret=candidate_secret
                ),
            )
    raise EOMTermsAcceptanceNotFoundError("Terms invitation is unavailable")


def build_eom_terms_link(*, base_url: str, token: str) -> str:
    """Keep the Terms bearer in a fragment so HTTP referrers omit it."""

    normalized_base_url = base_url.strip()
    if not normalized_base_url or "#" in normalized_base_url:
        raise EOMTermsAcceptanceValidationError(
            "Public Terms acceptance URL is invalid"
        )
    return f"{normalized_base_url}#termsToken={token}"


def eom_terms_delivery_idempotency_key(*, kind: str, delivery_id: UUID) -> str:
    if kind not in EOM_TERMS_DELIVERY_KINDS:
        raise EOMTermsAcceptanceValidationError("Terms delivery kind is invalid")
    return f"eom-terms-{kind}:{delivery_id}"


def _uuid(value: object, label: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EOMTermsAcceptanceValidationError(f"{label} is invalid") from exc


def _request_key(value: object) -> str:
    if not isinstance(value, str):
        raise EOMTermsAcceptanceValidationError("requestKey is invalid")
    normalized = value.strip()
    if not _REQUEST_KEY_PATTERN.fullmatch(normalized):
        raise EOMTermsAcceptanceValidationError("requestKey is invalid")
    return normalized


def _locale(value: object) -> str:
    if value not in ("en", "es"):
        raise EOMTermsAcceptanceValidationError("locale must be en or es")
    return str(value)


def _safe_text(value: object, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise EOMTermsAcceptanceValidationError(f"{label} is invalid")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum
        or "\x00" in normalized
        or any(0xD800 <= ord(character) <= 0xDFFF for character in normalized)
        or any(not character.isprintable() for character in normalized)
    ):
        raise EOMTermsAcceptanceValidationError(f"{label} is invalid")
    return normalized


def _actor(actor_id: object, actor_name: object) -> tuple[int, str]:
    if (
        isinstance(actor_id, bool)
        or not isinstance(actor_id, int)
        or actor_id <= 0
        or actor_id > _MAX_SIGNED_BIGINT
    ):
        raise EOMTermsAcceptanceValidationError("Authenticated actor is invalid")
    return actor_id, _safe_text(
        actor_name,
        label="Authenticated actor",
        maximum=_MAX_ACTOR_NAME_LENGTH,
    )


def _signer_name(value: object) -> str:
    return _safe_text(value, label="signerName", maximum=_MAX_SIGNER_NAME_LENGTH)


def _client_ip(value: object) -> str:
    if not isinstance(value, str):
        raise EOMTermsAcceptanceValidationError("Client IP is invalid")
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError as exc:
        raise EOMTermsAcceptanceValidationError("Client IP is invalid") from exc


def _email(value: object) -> str:
    if not isinstance(value, str):
        raise EOMTermsAcceptanceUnavailableError("Customer email is unavailable")
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > _MAX_EMAIL_LENGTH
        or not _EMAIL_PATTERN.fullmatch(normalized)
    ):
        raise EOMTermsAcceptanceUnavailableError("Customer email is unavailable")
    return normalized


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EOMTermsAcceptanceUnavailableError("Stored Terms timestamp is invalid")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _documents_from_version(
    row: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    raw_documents = row["documents"]
    if isinstance(raw_documents, str):
        try:
            raw_documents = json.loads(raw_documents)
        except json.JSONDecodeError as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Stored Terms documents are invalid"
            ) from exc
    try:
        documents = normalize_eom_terms_documents(raw_documents)
        _, _, calculated_hash = canonical_eom_terms_documents(documents)
    except EOMTermsValidationError as exc:
        raise EOMTermsAcceptanceUnavailableError(
            "Stored Terms documents are invalid"
        ) from exc
    if calculated_hash != row["content_hash"]:
        raise EOMTermsAcceptanceUnavailableError("Stored Terms content hash is invalid")
    return documents


def _render_sections(*, documents: Mapping[str, str], locale: str) -> str:
    labels = {
        "en": (
            "TERMS AND CONDITIONS",
            "SERVICES WE CANNOT PROVIDE",
            "SEPARATE ADDITIONAL-WORK ACKNOWLEDGEMENT",
        ),
        "es": (
            "TERMINOS Y CONDICIONES",
            "SERVICIOS QUE NO PODEMOS PROPORCIONAR",
            "RECONOCIMIENTO SEPARADO DE TRABAJO ADICIONAL",
        ),
    }[locale]
    return (
        f"{labels[0]}\n{documents['terms']}\n\n"
        f"{labels[1]}\n{documents['servicesWeCannotProvide']}\n\n"
        f"{labels[2]}\n{documents['additionalWorkAcknowledgement']}"
    )


def render_eom_terms_invitation(
    *,
    full_name: str,
    version_label: str,
    content_hash: str,
    documents: Mapping[str, str],
    locale: str,
) -> tuple[str, str]:
    """Render the exact published locale without inventing Terms prose."""

    sections = _render_sections(documents=documents, locale=locale)
    if locale == "es":
        subject = "Revise y acepte los terminos de Effingham Office Maids"
        body = (
            f"Hola {full_name},\n\n"
            "Revise los documentos publicados a continuacion. El enlace seguro "
            "para aceptar estos terminos aparece al final de este correo.\n\n"
            f"Version: {version_label}\nHash: {content_hash}\n\n{sections}"
        )
    else:
        subject = "Review and accept Effingham Office Maids terms"
        body = (
            f"Hello {full_name},\n\n"
            "Please review the published documents below. The secure link to "
            "accept these terms appears at the end of this email.\n\n"
            f"Version: {version_label}\nHash: {content_hash}\n\n{sections}"
        )
    return subject, body


def append_eom_terms_acceptance_link(*, body: str, link: str, locale: str) -> str:
    prompt = (
        "Aceptar los terminos de forma segura:"
        if locale == "es"
        else "Accept the terms securely:"
    )
    return f"{body}\n\n{prompt}\n{link}\n"


def render_eom_terms_executed_copy(
    *,
    full_name: str,
    signer_name: str,
    accepted_at: datetime,
    version_label: str,
    content_hash: str,
    documents: Mapping[str, str],
    locale: str,
) -> tuple[str, str]:
    """Render the immutable acceptance receipt plus the accepted documents."""

    accepted_iso = _iso(accepted_at)
    sections = _render_sections(documents=documents, locale=locale)
    if locale == "es":
        subject = "Copia aceptada de los terminos de Effingham Office Maids"
        body = (
            f"Copia ejecutada para {full_name}\n"
            f"Aceptado por: {signer_name}\n"
            f"Aceptado: {accepted_iso}\n"
            "Terminos generales aceptados: Si\n"
            "Reconocimiento de trabajo adicional aceptado: Si\n"
            f"Version: {version_label}\nHash: {content_hash}\n\n{sections}"
        )
    else:
        subject = "Accepted copy of Effingham Office Maids terms"
        body = (
            f"Executed copy for {full_name}\n"
            f"Accepted by: {signer_name}\n"
            f"Accepted at: {accepted_iso}\n"
            "General terms accepted: Yes\n"
            "Additional-work acknowledgement accepted: Yes\n"
            f"Version: {version_label}\nHash: {content_hash}\n\n{sections}"
        )
    return subject, body


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


async def eom_terms_acceptance_schema_ready(pool: Any) -> bool:
    """Attest the guarded acceptance relations before any customer state use."""

    try:
        return bool(
            await pool.fetchval(
                """
                /* eom_terms_acceptance_schema_ready */
                WITH expected_relations(name) AS (
                    VALUES ('eom_terms_invitations'),
                           ('eom_terms_acceptances'),
                           ('eom_terms_deliveries')
                ),
                expected_functions(name) AS (
                    VALUES ('assign_eom_terms_publication_order'),
                           ('validate_eom_terms_invitation'),
                           ('protect_eom_terms_invitation'),
                           ('validate_eom_terms_acceptance'),
                           ('protect_eom_terms_acceptance'),
                           ('protect_eom_terms_delivery')
                ),
                expected_triggers(relation_name, function_name, trigger_name) AS (
                    VALUES
                        ('eom_terms_versions',
                         'assign_eom_terms_publication_order',
                         'trg_assign_eom_terms_publication_order'),
                        ('eom_terms_invitations',
                         'validate_eom_terms_invitation',
                         'trg_validate_eom_terms_invitation'),
                        ('eom_terms_invitations',
                         'protect_eom_terms_invitation',
                         'trg_protect_eom_terms_invitation'),
                        ('eom_terms_invitations',
                         'protect_eom_terms_invitation',
                         'trg_protect_eom_terms_invitation_truncate'),
                        ('eom_terms_acceptances',
                         'validate_eom_terms_acceptance',
                         'trg_validate_eom_terms_acceptance'),
                        ('eom_terms_acceptances',
                         'protect_eom_terms_acceptance',
                         'trg_protect_eom_terms_acceptance'),
                        ('eom_terms_acceptances',
                         'protect_eom_terms_acceptance',
                         'trg_protect_eom_terms_acceptance_truncate'),
                        ('eom_terms_deliveries',
                         'protect_eom_terms_delivery',
                         'trg_protect_eom_terms_delivery'),
                        ('eom_terms_deliveries',
                         'protect_eom_terms_delivery',
                         'trg_protect_eom_terms_delivery_truncate')
                ),
                expected_constraints(relation_name, constraint_name) AS (
                    VALUES
                        ('eom_terms_versions',
                         'uq_eom_terms_publication_order'),
                        ('eom_terms_versions',
                         'ck_eom_terms_publication_order'),
                        ('eom_terms_invitations',
                         'pk_eom_terms_invitations'),
                        ('eom_terms_invitations',
                         'uq_eom_terms_invitations_request_key'),
                        ('eom_terms_invitations',
                         'fk_eom_terms_invitations_contact'),
                        ('eom_terms_invitations',
                         'fk_eom_terms_invitations_version'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_request_key'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_context'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_audience'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_locale'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_customer_name'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_recipient'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_public_url'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_key_fingerprint'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_window'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_issuer'),
                        ('eom_terms_invitations',
                         'ck_eom_terms_invitations_revocation'),
                        ('eom_terms_acceptances',
                         'pk_eom_terms_acceptances'),
                        ('eom_terms_acceptances',
                         'uq_eom_terms_acceptances_invitation'),
                        ('eom_terms_acceptances',
                         'fk_eom_terms_acceptances_invitation'),
                        ('eom_terms_acceptances',
                         'fk_eom_terms_acceptances_contact'),
                        ('eom_terms_acceptances',
                         'fk_eom_terms_acceptances_version'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_context'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_audience'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_locale'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_recipient'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_signer'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_acknowledgements'),
                        ('eom_terms_acceptances',
                         'ck_eom_terms_acceptances_content_hash'),
                        ('eom_terms_deliveries',
                         'pk_eom_terms_deliveries'),
                        ('eom_terms_deliveries',
                         'uq_eom_terms_deliveries_kind_invitation'),
                        ('eom_terms_deliveries',
                         'uq_eom_terms_deliveries_acceptance'),
                        ('eom_terms_deliveries',
                         'fk_eom_terms_deliveries_invitation'),
                        ('eom_terms_deliveries',
                         'fk_eom_terms_deliveries_acceptance'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_kind'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_shape'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_recipient'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_subject'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_body'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_body_hash'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_status'),
                        ('eom_terms_deliveries',
                         'ck_eom_terms_deliveries_state')
                ),
                boundary AS (
                    SELECT pg_catalog.current_schema() AS schema_name,
                           (SELECT oid FROM pg_catalog.pg_roles
                             WHERE rolname = 'atlas_eom_handoff_owner') AS guard_oid
                )
                SELECT boundary.guard_oid IS NOT NULL
                   AND EXISTS (
                       SELECT 1
                       FROM pg_catalog.pg_roles AS guard_role
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
                       FROM pg_catalog.pg_stat_activity AS activity
                       WHERE activity.usesysid = boundary.guard_oid
                         AND activity.datid = (
                             SELECT database.oid
                             FROM pg_catalog.pg_database AS database
                             WHERE database.datname = current_database()
                         )
                   )
                   AND NOT EXISTS (
                       SELECT 1
                       FROM pg_catalog.pg_roles AS member_role
                       WHERE member_role.rolcanlogin
                         AND NOT member_role.rolsuper
                         AND pg_catalog.pg_has_role(
                             member_role.oid, boundary.guard_oid, 'MEMBER'
                         )
                   )
                   AND EXISTS (
                       SELECT 1
                       FROM pg_catalog.pg_namespace AS namespace
                       WHERE namespace.nspname = boundary.schema_name
                         AND namespace.nspowner = boundary.guard_oid
                   )
                   AND (
                       SELECT count(*) = 3
                         FROM expected_relations AS expected
                         JOIN pg_catalog.pg_class AS relation
                           ON relation.relname = expected.name
                         JOIN pg_catalog.pg_namespace AS namespace
                           ON namespace.oid = relation.relnamespace
                         CROSS JOIN boundary AS relation_boundary
                        WHERE namespace.nspname = relation_boundary.schema_name
                          AND relation.relkind = 'r'
                          AND relation.relowner = relation_boundary.guard_oid
                   )
                   AND (
                       SELECT count(*) = 6
                         FROM expected_functions AS expected
                         JOIN pg_catalog.pg_proc AS guarded_function
                           ON guarded_function.proname = expected.name
                          AND guarded_function.pronargs = 0
                         JOIN pg_catalog.pg_namespace AS namespace
                           ON namespace.oid = guarded_function.pronamespace
                         CROSS JOIN boundary AS function_boundary
                        WHERE namespace.nspname = function_boundary.schema_name
                          AND guarded_function.proowner = function_boundary.guard_oid
                          AND guarded_function.prosecdef = FALSE
                          AND guarded_function.prorettype = 'trigger'::regtype
                          AND guarded_function.proconfig = ARRAY[
                              format(
                                  'search_path=pg_catalog, %I, pg_temp',
                                  function_boundary.schema_name
                              )
                          ]
                   )
                   AND NOT EXISTS (
                       SELECT 1
                       FROM expected_triggers AS expected
                       CROSS JOIN boundary AS trigger_boundary
                       WHERE NOT EXISTS (
                           SELECT 1
                           FROM pg_catalog.pg_trigger AS guard_trigger
                           JOIN pg_catalog.pg_class AS relation
                             ON relation.oid = guard_trigger.tgrelid
                           JOIN pg_catalog.pg_namespace AS namespace
                             ON namespace.oid = relation.relnamespace
                           JOIN pg_catalog.pg_proc AS guarded_function
                             ON guarded_function.oid = guard_trigger.tgfoid
                           WHERE namespace.nspname = trigger_boundary.schema_name
                             AND relation.relname = expected.relation_name
                             AND guarded_function.proname = expected.function_name
                             AND guard_trigger.tgname = expected.trigger_name
                             AND guard_trigger.tgenabled = 'O'
                             AND NOT guard_trigger.tgisinternal
                       )
                   )
                   AND (
                       SELECT count(*) = 8
                       FROM pg_catalog.pg_trigger AS guard_trigger
                       JOIN pg_catalog.pg_class AS relation
                         ON relation.oid = guard_trigger.tgrelid
                       JOIN pg_catalog.pg_namespace AS namespace
                         ON namespace.oid = relation.relnamespace
                       CROSS JOIN boundary AS trigger_boundary
                       WHERE namespace.nspname = trigger_boundary.schema_name
                         AND relation.relname IN (
                             'eom_terms_invitations',
                             'eom_terms_acceptances',
                             'eom_terms_deliveries'
                         )
                         AND NOT guard_trigger.tgisinternal
                   )
                   AND NOT EXISTS (
                       SELECT 1
                       FROM expected_constraints AS expected
                       CROSS JOIN boundary AS constraint_boundary
                       WHERE NOT EXISTS (
                           SELECT 1
                           FROM pg_catalog.pg_constraint AS actual
                           JOIN pg_catalog.pg_class AS relation
                             ON relation.oid = actual.conrelid
                           JOIN pg_catalog.pg_namespace AS namespace
                             ON namespace.oid = relation.relnamespace
                           WHERE namespace.nspname = constraint_boundary.schema_name
                             AND relation.relname = expected.relation_name
                             AND actual.conname = expected.constraint_name
                             AND actual.convalidated
                       )
                   )
                   AND has_table_privilege(
                       current_user, 'eom_terms_invitations', 'SELECT'
                   )
                   AND has_table_privilege(
                       current_user, 'eom_terms_acceptances', 'SELECT'
                   )
                   AND has_table_privilege(
                       current_user, 'eom_terms_deliveries', 'SELECT'
                   )
                   AND NOT has_table_privilege(
                       current_user, 'eom_terms_invitations', 'DELETE, TRUNCATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, 'eom_terms_acceptances',
                       'UPDATE, DELETE, TRUNCATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, 'eom_terms_deliveries', 'DELETE, TRUNCATE'
                   )
                   AND NOT has_table_privilege(
                       current_user, 'eom_terms_invitations',
                       'INSERT, UPDATE, REFERENCES, TRIGGER'
                   )
                   AND NOT has_table_privilege(
                       current_user, 'eom_terms_acceptances',
                       'INSERT, UPDATE, REFERENCES, TRIGGER'
                   )
                   AND NOT has_table_privilege(
                       current_user, 'eom_terms_deliveries',
                       'INSERT, UPDATE, REFERENCES, TRIGGER'
                   )
                   AND has_column_privilege(
                       current_user, 'eom_terms_acceptances', 'signer_name', 'INSERT'
                   )
                   AND NOT has_column_privilege(
                       current_user, 'eom_terms_acceptances', 'signer_name', 'UPDATE'
                   )
                   AND has_column_privilege(
                       current_user, 'eom_terms_invitations', 'revoked_at', 'UPDATE'
                   )
                   AND has_column_privilege(
                       current_user, 'eom_terms_deliveries', 'status', 'UPDATE'
                   )
                   AND NOT has_column_privilege(
                       current_user, 'eom_terms_invitations', 'contact_id', 'UPDATE'
                   )
                   AND NOT has_column_privilege(
                       current_user, 'eom_terms_acceptances', 'accepted_at', 'INSERT'
                   )
                   AND NOT has_column_privilege(
                       current_user, 'eom_terms_deliveries', 'body', 'UPDATE'
                   )
                   AND EXISTS (
                       SELECT 1
                       FROM pg_catalog.pg_attribute AS attribute
                       WHERE attribute.attrelid =
                                 'eom_terms_versions'::regclass
                         AND attribute.attname = 'publication_order'
                         AND attribute.atttypid = 'bigint'::regtype
                         AND attribute.attnum > 0
                         AND NOT attribute.attisdropped
                         AND attribute.attgenerated = ''
                         AND attribute.attidentity = ''
                   )
                   AND NOT has_column_privilege(
                       current_user, 'eom_terms_versions',
                       'publication_order', 'INSERT, UPDATE'
                   )
                   AND NOT has_function_privilege(
                       current_user,
                       'assign_eom_terms_publication_order()', 'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user, 'validate_eom_terms_invitation()', 'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user, 'protect_eom_terms_invitation()', 'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user, 'validate_eom_terms_acceptance()', 'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user, 'protect_eom_terms_acceptance()', 'EXECUTE'
                   )
                   AND NOT has_function_privilege(
                       current_user, 'protect_eom_terms_delivery()', 'EXECUTE'
                   )
                  FROM boundary
                """
            )
        )
    except Exception:
        return False


def _secret_for_fingerprint(
    *,
    fingerprint: object,
    secret: str,
    previous_secret: str | None,
) -> str:
    if not isinstance(fingerprint, str):
        raise EOMTermsAcceptanceUnavailableError("Stored Terms signing key is invalid")
    for candidate in (secret, previous_secret):
        if candidate is None:
            continue
        candidate_fingerprint = eom_public_onboarding_hmac_key_fingerprint(
            secret=candidate
        )
        if hmac.compare_digest(candidate_fingerprint, fingerprint):
            return candidate
    raise EOMTermsAcceptanceUnavailableError(
        "Terms invitation signing key is unavailable"
    )


def _invitation_result(
    row: Mapping[str, Any],
    *,
    idempotent: bool,
    delivery_error: bool = False,
) -> dict[str, Any]:
    if row.get("acceptance_id") is not None:
        state = "accepted"
    elif row.get("revoked_at") is not None:
        state = "revoked"
    elif bool(row.get("is_expired")):
        state = "expired"
    else:
        state = "issued"
    delivery_status = str(row["delivery_status"])
    return {
        "invitationId": str(row["invitation_id"]),
        "contactId": str(row["contact_id"]),
        "versionId": str(row["version_id"]),
        "versionLabel": str(row["version_label"]),
        "contentHash": str(row["content_hash"]),
        "audience": str(row["audience"]),
        "locale": str(row["locale"]),
        "recipientEmail": str(row["recipient_email"]),
        "status": state,
        "issuedAt": _iso(row["issued_at"]),
        "expiresAt": _iso(row["expires_at"]),
        "revokedAt": _iso(row.get("revoked_at")),
        "acceptanceId": (
            str(row["acceptance_id"]) if row.get("acceptance_id") is not None else None
        ),
        "deliveryId": str(row["delivery_id"]),
        "deliveryStatus": delivery_status,
        "deliveryNeedsReconciliation": delivery_status == "sending",
        "deliveryError": delivery_error,
        "idempotent": idempotent,
    }


def _acceptance_result(
    row: Mapping[str, Any],
    *,
    idempotent: bool,
    delivery_error: bool = False,
) -> dict[str, Any]:
    delivery_status = str(row["delivery_status"])
    return {
        "acceptanceId": str(row["acceptance_id"]),
        "invitationId": str(row["invitation_id"]),
        "contactId": str(row["contact_id"]),
        "versionId": str(row["version_id"]),
        "versionLabel": str(row["version_label"]),
        "contentHash": str(row["content_hash"]),
        "audience": str(row["audience"]),
        "locale": str(row["locale"]),
        "signerName": str(row["signer_name"]),
        "termsAccepted": bool(row["terms_accepted"]),
        "additionalWorkAccepted": bool(row["additional_work_accepted"]),
        "acceptedAt": _iso(row["accepted_at"]),
        "deliveryId": str(row["delivery_id"]),
        "executedCopyDeliveryStatus": delivery_status,
        "deliveryNeedsReconciliation": delivery_status == "sending",
        "deliveryError": delivery_error,
        "idempotent": idempotent,
    }


Sender = Callable[..., Awaitable[dict[str, Any]]]


class EOMTermsAcceptanceService:
    """Own invitation, acceptance, readiness, and executed-copy delivery."""

    def __init__(self, *, pool: Any) -> None:
        self._pool = pool

    @property
    def pool(self) -> Any:
        if not bool(getattr(self._pool, "is_initialized", True)):
            raise EOMTermsAcceptanceUnavailableError(
                "Terms acceptance database is unavailable"
            )
        return self._pool

    async def require_schema_ready(self) -> None:
        if not await eom_terms_acceptance_schema_ready(self.pool):
            raise EOMTermsAcceptanceUnavailableError(
                "Terms acceptance schema is unavailable"
            )

    @staticmethod
    async def _invitation_by_id(connection: Any, invitation_id: UUID) -> Any:
        return await connection.fetchrow(
            """
            SELECT invitation.id AS invitation_id,
                   invitation.request_key,
                   invitation.contact_id,
                   invitation.version_id,
                   invitation.audience,
                   invitation.locale,
                   invitation.customer_name,
                   invitation.recipient_email,
                   invitation.public_base_url,
                   invitation.signing_key_fingerprint,
                   invitation.issued_at,
                   invitation.expires_at,
                   invitation.revoked_at,
                   version.version_label,
                   version.content_hash,
                   version.documents,
                   acceptance.id AS acceptance_id,
                   delivery.id AS delivery_id,
                   delivery.status AS delivery_status,
                   delivery.subject AS delivery_subject,
                   delivery.body AS delivery_body,
                   delivery.body_hash AS delivery_body_hash,
                   clock_timestamp() > invitation.expires_at AS is_expired
            FROM eom_terms_invitations AS invitation
            JOIN eom_terms_versions AS version ON version.id = invitation.version_id
            JOIN eom_terms_deliveries AS delivery
              ON delivery.invitation_id = invitation.id
             AND delivery.kind = 'invitation'
            LEFT JOIN eom_terms_acceptances AS acceptance
              ON acceptance.invitation_id = invitation.id
            WHERE invitation.id = $1
            """,
            invitation_id,
        )

    @staticmethod
    async def _invitation_by_request(connection: Any, request_key: str) -> Any:
        row = await connection.fetchrow(
            "SELECT id FROM eom_terms_invitations WHERE request_key = $1",
            request_key,
        )
        if row is None:
            return None
        return await EOMTermsAcceptanceService._invitation_by_id(
            connection, UUID(str(row["id"]))
        )

    @staticmethod
    async def _acceptance_by_invitation(connection: Any, invitation_id: UUID) -> Any:
        return await connection.fetchrow(
            """
            SELECT acceptance.id AS acceptance_id,
                   acceptance.invitation_id,
                   acceptance.contact_id,
                   acceptance.version_id,
                   acceptance.audience,
                   acceptance.locale,
                   acceptance.signer_name,
                   acceptance.terms_accepted,
                   acceptance.additional_work_accepted,
                   acceptance.client_ip,
                   acceptance.accepted_at,
                   version.version_label,
                   version.content_hash,
                   delivery.id AS delivery_id,
                   delivery.status AS delivery_status,
                   delivery.subject AS delivery_subject,
                   delivery.body AS delivery_body,
                   delivery.body_hash AS delivery_body_hash
            FROM eom_terms_acceptances AS acceptance
            JOIN eom_terms_versions AS version ON version.id = acceptance.version_id
            JOIN eom_terms_deliveries AS delivery
              ON delivery.acceptance_id = acceptance.id
             AND delivery.kind = 'executed_copy'
            WHERE acceptance.invitation_id = $1
            """,
            invitation_id,
        )

    @staticmethod
    async def _delivery_by_id(connection: Any, delivery_id: UUID) -> Any:
        return await connection.fetchrow(
            """
            SELECT delivery.*,
                   invitation.public_base_url,
                   invitation.signing_key_fingerprint,
                   invitation.locale,
                   invitation.revoked_at AS invitation_revoked_at,
                   clock_timestamp() > invitation.expires_at
                       AS invitation_expired,
                   EXISTS (
                       SELECT 1
                       FROM eom_terms_versions AS invited
                       JOIN eom_terms_versions AS later
                         ON later.status = 'published'
                        AND later.material_change
                        AND later.publication_order > invited.publication_order
                       WHERE invited.id = invitation.version_id
                         AND invited.status = 'published'
                   ) AS invitation_materially_stale,
                   (
                       contact.business_context_id = 'effingham_maids'
                       AND contact.contact_type = 'customer'
                       AND contact.status = 'active'
                       AND contact.customer_type = invitation.audience
                       AND btrim(contact.full_name) = invitation.customer_name
                       AND lower(btrim(contact.email)) =
                           invitation.recipient_email
                   ) AS invitation_contact_matches,
                   EXISTS (
                       SELECT 1
                       FROM eom_terms_acceptances AS accepted
                       WHERE accepted.invitation_id = invitation.id
                   ) AS invitation_accepted
            FROM eom_terms_deliveries AS delivery
            JOIN eom_terms_invitations AS invitation
              ON invitation.id = delivery.invitation_id
            JOIN contacts AS contact ON contact.id = invitation.contact_id
            WHERE delivery.id = $1
            FOR UPDATE OF invitation, delivery
            FOR SHARE OF contact
            """,
            delivery_id,
        )

    @staticmethod
    def _customer(row: Any) -> tuple[UUID, str, str, str]:
        if row is None:
            raise EOMTermsAcceptanceNotFoundError("EOM customer was not found")
        if (
            row["business_context_id"] != "effingham_maids"
            or row["contact_type"] != "customer"
            or row["status"] != "active"
            or row["customer_type"] not in ("residential", "commercial")
        ):
            raise EOMTermsAcceptanceConflictError(
                "Contact is not an active classified EOM customer"
            )
        try:
            customer_name = _safe_text(
                row["full_name"],
                label="Customer name",
                maximum=_MAX_SIGNER_NAME_LENGTH,
            )
            recipient_email = _email(row["email"])
        except (
            EOMTermsAcceptanceValidationError,
            EOMTermsAcceptanceUnavailableError,
        ) as exc:
            raise EOMTermsAcceptanceConflictError(
                "EOM customer needs a valid name and email"
            ) from exc
        return (
            UUID(str(row["id"])),
            str(row["customer_type"]),
            customer_name,
            recipient_email,
        )

    async def _deliver(
        self,
        *,
        delivery_id: UUID,
        secret: str | None,
        previous_secret: str | None,
        sender: Sender,
    ) -> tuple[dict[str, Any], bool]:
        """Durably claim once, then serialize one bounded transport outcome."""

        def prepare(candidate: dict[str, Any]) -> tuple[str, str] | None:
            body = str(candidate["body"])
            if _body_hash(body) != candidate["body_hash"]:
                raise EOMTermsAcceptanceUnavailableError(
                    "Stored Terms delivery payload is invalid"
                )
            kind = str(candidate["kind"])
            if kind not in ("invitation", "executed_copy"):
                raise EOMTermsAcceptanceUnavailableError(
                    "Stored Terms delivery kind is invalid"
                )
            if not bool(candidate["invitation_contact_matches"]):
                raise EOMTermsAcceptanceConflictError(
                    "EOM customer changed before Terms delivery"
                )
            if kind == "invitation":
                if (
                    candidate["invitation_revoked_at"] is not None
                    or bool(candidate["invitation_expired"])
                    or bool(candidate["invitation_accepted"])
                    or bool(candidate["invitation_materially_stale"])
                ):
                    return None
                if secret is None:
                    raise EOMTermsAcceptanceUnavailableError(
                        "Terms invitation signing key is unavailable"
                    )
                invitation_id = UUID(str(candidate["invitation_id"]))
                token_secret = _secret_for_fingerprint(
                    fingerprint=candidate["signing_key_fingerprint"],
                    secret=secret,
                    previous_secret=previous_secret,
                )
                token = format_eom_terms_token(
                    invitation_id=invitation_id,
                    secret=token_secret,
                )
                link = build_eom_terms_link(
                    base_url=str(candidate["public_base_url"]),
                    token=token,
                )
                body = append_eom_terms_acceptance_link(
                    body=body,
                    link=link,
                    locale=str(candidate["locale"]),
                )
            return kind, body

        transport_succeeded = False
        transport_kind = "unknown"
        try:
            # Commit the non-retryable state before crossing the provider boundary.
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                row = await self._delivery_by_id(connection, delivery_id)
                if row is None:
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms delivery was not found"
                    )
                candidate = dict(row)
                state = str(candidate["status"])
                if state in ("sending", "sent"):
                    return candidate, False
                if state != "pending":
                    raise EOMTermsAcceptanceUnavailableError(
                        "Stored Terms delivery state is invalid"
                    )
                if prepare(candidate) is None:
                    return candidate, False
                claimed = await connection.fetchrow(
                    """
                    UPDATE eom_terms_deliveries AS delivery
                    SET status = 'sending', claimed_at = clock_timestamp()
                    WHERE delivery.id = $1
                      AND delivery.status = 'pending'
                      AND (
                          delivery.kind <> 'invitation'
                          OR EXISTS (
                              SELECT 1
                              FROM eom_terms_invitations AS invitation
                              WHERE invitation.id = delivery.invitation_id
                                AND invitation.revoked_at IS NULL
                                AND clock_timestamp() <= invitation.expires_at
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM eom_terms_acceptances AS acceptance
                                    WHERE acceptance.invitation_id = invitation.id
                                )
                          )
                      )
                    RETURNING delivery.*
                    """,
                    delivery_id,
                )
                if claimed is None:
                    current = await self._delivery_by_id(connection, delivery_id)
                    if current is None:
                        raise EOMTermsAcceptanceNotFoundError(
                            "Terms delivery was not found"
                        )
                    return dict(current), False
            # Revalidate under canonical locks and hold them through transport.
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                row = await self._delivery_by_id(connection, delivery_id)
                if row is None:
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms delivery was not found"
                    )
                candidate = dict(row)
                state = str(candidate["status"])
                if state == "sent":
                    return candidate, False
                if state != "sending":
                    raise EOMTermsAcceptanceUnavailableError(
                        "Stored Terms delivery state is invalid"
                    )
                prepared = prepare(candidate)
                if prepared is None:
                    logger.warning(
                        "Terms delivery requires reconciliation after its "
                        "invitation became unavailable for %s",
                        delivery_id,
                    )
                    return candidate, True
                kind, body = prepared
                transport_kind = kind
                try:
                    send_result = await sender(
                        to=str(candidate["recipient_email"]),
                        subject=str(candidate["subject"]),
                        body=body,
                        idempotency_key=eom_terms_delivery_idempotency_key(
                            kind=kind,
                            delivery_id=delivery_id,
                        ),
                    )
                    message_id = send_result.get("message_id")
                    idempotent_replay = send_result.get("idempotent_replay")
                    if not isinstance(idempotent_replay, bool):
                        raise ValueError("Terms delivery transport result is invalid")
                    transport_succeeded = True
                except Exception:
                    logger.warning(
                        "Terms %s delivery requires reconciliation for %s",
                        kind,
                        delivery_id,
                        exc_info=True,
                    )
                    return candidate, True
                confirmed = await connection.fetchrow(
                    """
                    UPDATE eom_terms_deliveries
                    SET status = 'sent',
                        sent_at = clock_timestamp(),
                        resend_message_id = $2,
                        transport_idempotent_replay = $3
                    WHERE id = $1 AND status = 'sending'
                    RETURNING *
                    """,
                    delivery_id,
                    message_id,
                    idempotent_replay,
                )
                if confirmed is None:
                    logger.warning(
                        "Terms %s transport succeeded but confirmation was not "
                        "stored; its durable claim requires reconciliation for %s",
                        kind,
                        delivery_id,
                    )
                    raise EOMTermsAcceptanceUnavailableError(
                        "Terms delivery confirmation was not stored"
                    )
                return dict(confirmed), False
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            if transport_succeeded:
                logger.warning(
                    "Terms %s transport succeeded but database confirmation "
                    "failed; its durable claim requires reconciliation for %s",
                    transport_kind,
                    delivery_id,
                    exc_info=True,
                )
            raise EOMTermsAcceptanceUnavailableError(
                "Terms delivery state is unavailable"
            ) from exc

    async def issue_and_send(
        self,
        *,
        request_key: object,
        contact_id: object,
        locale: object,
        actor_id: object,
        actor_name: object,
        public_base_url: str,
        hmac_secret: str,
        previous_hmac_secret: str | None = None,
        sender: Sender | None = None,
    ) -> dict[str, Any]:
        parsed_request_key = _request_key(request_key)
        parsed_contact_id = _uuid(contact_id, "contactId")
        parsed_locale = _locale(locale)
        parsed_actor_id, parsed_actor_name = _actor(actor_id, actor_name)
        try:
            parsed_secret = validate_eom_public_onboarding_hmac_secret(hmac_secret)
            parsed_previous_secret = (
                validate_eom_public_onboarding_hmac_secret(previous_hmac_secret)
                if previous_hmac_secret is not None
                else None
            )
        except ValueError as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms invitation signing key is unavailable"
            ) from exc
        build_eom_terms_link(base_url=public_base_url, token="configuration-probe")
        if sender is None:
            try:
                _require_transport_configured()
            except EOMOnboardingDraftError as exc:
                raise EOMTermsAcceptanceUnavailableError(str(exc)) from exc
            sender = send_onboarding_email
        await self.require_schema_ready()
        signing_key_fingerprint = eom_public_onboarding_hmac_key_fingerprint(
            secret=parsed_secret
        )
        idempotent = False
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                existing = await self._invitation_by_request(
                    connection, parsed_request_key
                )
                if existing is not None:
                    if (
                        UUID(str(existing["contact_id"])) != parsed_contact_id
                        or existing["locale"] != parsed_locale
                    ):
                        raise EOMTermsAcceptanceConflictError(
                            "requestKey belongs to another Terms invitation"
                        )
                    if (
                        existing["delivery_status"] == "pending"
                        and existing["acceptance_id"] is None
                        and existing["revoked_at"] is None
                        and not bool(existing["is_expired"])
                    ):
                        contact = await connection.fetchrow(
                            """
                            SELECT id, business_context_id, contact_type, status,
                                   customer_type, full_name, email
                            FROM contacts
                            WHERE id = $1
                            FOR SHARE
                            """,
                            parsed_contact_id,
                        )
                        (
                            canonical_contact_id,
                            audience,
                            _customer_name,
                            recipient_email,
                        ) = self._customer(contact)
                        if (
                            canonical_contact_id != UUID(str(existing["contact_id"]))
                            or audience != existing["audience"]
                            or recipient_email != existing["recipient_email"]
                        ):
                            raise EOMTermsAcceptanceConflictError(
                                "EOM customer changed before Terms delivery"
                            )
                    invitation = existing
                    idempotent = True
                else:
                    contact = await connection.fetchrow(
                        """
                        SELECT id, business_context_id, contact_type, status,
                               customer_type, full_name, email
                        FROM contacts
                        WHERE id = $1
                        FOR SHARE
                        """,
                        parsed_contact_id,
                    )
                    (
                        canonical_contact_id,
                        audience,
                        customer_name,
                        recipient_email,
                    ) = self._customer(contact)
                    version = await connection.fetchrow(
                        """
                        SELECT version.*
                        FROM eom_terms_current_version AS current
                        JOIN eom_terms_versions AS version
                          ON version.id = current.version_id
                        WHERE current.singleton AND version.status = 'published'
                        FOR SHARE OF version
                        """
                    )
                    if version is None:
                        raise EOMTermsAcceptanceNotFoundError(
                            "Current Terms version was not found"
                        )
                    documents = _documents_from_version(version)
                    selected_documents = documents[audience][parsed_locale]
                    subject, body = render_eom_terms_invitation(
                        full_name=customer_name,
                        version_label=str(version["version_label"]),
                        content_hash=str(version["content_hash"]),
                        documents=selected_documents,
                        locale=parsed_locale,
                    )
                    invitation_id = uuid4()
                    delivery_id = uuid4()
                    await connection.execute(
                        """
                        INSERT INTO eom_terms_invitations (
                            id, request_key, contact_id, version_id, audience,
                            locale, customer_name, recipient_email,
                            public_base_url, signing_key_fingerprint,
                            issued_by_id, issued_by_name
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12
                        )
                        """,
                        invitation_id,
                        parsed_request_key,
                        canonical_contact_id,
                        version["id"],
                        audience,
                        parsed_locale,
                        customer_name,
                        recipient_email,
                        public_base_url.strip(),
                        signing_key_fingerprint,
                        parsed_actor_id,
                        parsed_actor_name,
                    )
                    await connection.execute(
                        """
                        INSERT INTO eom_terms_deliveries (
                            id, kind, invitation_id, recipient_email, subject,
                            body, body_hash
                        ) VALUES ($1, 'invitation', $2, $3, $4, $5, $6)
                        """,
                        delivery_id,
                        invitation_id,
                        recipient_email,
                        subject,
                        body,
                        _body_hash(body),
                    )
                    invitation = await self._invitation_by_id(connection, invitation_id)
                    if invitation is None:
                        raise EOMTermsAcceptanceUnavailableError(
                            "Terms invitation could not be stored"
                        )
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms invitation could not be stored"
            ) from exc
        delivery, delivery_error = await self._deliver(
            delivery_id=UUID(str(invitation["delivery_id"])),
            secret=parsed_secret,
            previous_secret=parsed_previous_secret,
            sender=sender,
        )
        result_row = dict(invitation)
        result_row["delivery_status"] = delivery["status"]
        return _invitation_result(
            result_row,
            idempotent=idempotent,
            delivery_error=delivery_error,
        )

    @staticmethod
    async def _has_later_material_version(connection: Any, version_id: UUID) -> bool:
        return bool(
            await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM eom_terms_versions AS invited
                    JOIN eom_terms_versions AS later
                      ON later.status = 'published'
                     AND later.material_change
                     AND later.publication_order > invited.publication_order
                    WHERE invited.id = $1
                      AND invited.status = 'published'
                )
                """,
                version_id,
            )
        )

    @staticmethod
    def _require_authenticated_token(
        token: object,
    ) -> AuthenticatedEOMTermsToken:
        if not isinstance(token, AuthenticatedEOMTermsToken):
            raise EOMTermsAcceptanceNotFoundError("Terms invitation is unavailable")
        return token

    async def get_session(self, *, token: AuthenticatedEOMTermsToken) -> dict[str, Any]:
        authenticated = self._require_authenticated_token(token)
        await self.require_schema_ready()
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                invitation = await self._invitation_by_id(
                    connection, authenticated.invitation_id
                )
                if invitation is None or not hmac.compare_digest(
                    str(invitation["signing_key_fingerprint"]),
                    authenticated.signing_key_fingerprint,
                ):
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms invitation is unavailable"
                    )
                documents = _documents_from_version(invitation)
                selected = documents[str(invitation["audience"])][
                    str(invitation["locale"])
                ]
                if invitation["acceptance_id"] is not None:
                    acceptance = await self._acceptance_by_invitation(
                        connection, authenticated.invitation_id
                    )
                    if acceptance is None:
                        raise EOMTermsAcceptanceUnavailableError(
                            "Stored Terms acceptance is unavailable"
                        )
                    return {
                        "status": "accepted",
                        "invitationId": str(invitation["invitation_id"]),
                        "versionId": str(invitation["version_id"]),
                        "versionLabel": str(invitation["version_label"]),
                        "contentHash": str(invitation["content_hash"]),
                        "audience": str(invitation["audience"]),
                        "locale": str(invitation["locale"]),
                        "acceptedAt": _iso(acceptance["accepted_at"]),
                    }
                if (
                    invitation["revoked_at"] is not None
                    or bool(invitation["is_expired"])
                    or await self._has_later_material_version(
                        connection, UUID(str(invitation["version_id"]))
                    )
                ):
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms invitation is unavailable"
                    )
                return {
                    "status": "ready",
                    "invitationId": str(invitation["invitation_id"]),
                    "versionId": str(invitation["version_id"]),
                    "versionLabel": str(invitation["version_label"]),
                    "contentHash": str(invitation["content_hash"]),
                    "audience": str(invitation["audience"]),
                    "locale": str(invitation["locale"]),
                    "customerName": str(invitation["customer_name"]),
                    "documents": selected,
                    "expiresAt": _iso(invitation["expires_at"]),
                }
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms invitation session is unavailable"
            ) from exc

    async def accept_and_send(
        self,
        *,
        token: AuthenticatedEOMTermsToken,
        signer_name: object,
        terms_accepted: object,
        additional_work_accepted: object,
        client_ip: object,
        sender: Sender | None = None,
    ) -> dict[str, Any]:
        authenticated = self._require_authenticated_token(token)
        parsed_signer_name = _signer_name(signer_name)
        if terms_accepted is not True or additional_work_accepted is not True:
            raise EOMTermsAcceptanceValidationError(
                "Both Terms acknowledgements must be accepted"
            )
        parsed_client_ip = _client_ip(client_ip)
        await self.require_schema_ready()
        idempotent = False
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                invitation = await connection.fetchrow(
                    """
                    SELECT invitation.*, version.version_label,
                           version.content_hash, version.documents,
                           invitation_delivery.status
                               AS invitation_delivery_status
                    FROM eom_terms_invitations AS invitation
                    JOIN eom_terms_versions AS version
                      ON version.id = invitation.version_id
                    JOIN eom_terms_deliveries AS invitation_delivery
                      ON invitation_delivery.invitation_id = invitation.id
                     AND invitation_delivery.kind = 'invitation'
                    WHERE invitation.id = $1
                    FOR UPDATE OF invitation
                    """,
                    authenticated.invitation_id,
                )
                if invitation is None or not hmac.compare_digest(
                    str(invitation["signing_key_fingerprint"]),
                    authenticated.signing_key_fingerprint,
                ):
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms invitation is unavailable"
                    )
                existing = await self._acceptance_by_invitation(
                    connection, authenticated.invitation_id
                )
                if existing is not None:
                    if str(existing["signer_name"]) != parsed_signer_name:
                        raise EOMTermsAcceptanceConflictError(
                            "Terms invitation was accepted by another signer"
                        )
                    acceptance = existing
                    idempotent = True
                else:
                    if invitation["invitation_delivery_status"] == "sending":
                        raise EOMTermsAcceptanceConflictError(
                            "Terms invitation delivery requires reconciliation"
                        )
                    accepted_at = await connection.fetchval("SELECT clock_timestamp()")
                    if (
                        not isinstance(accepted_at, datetime)
                        or accepted_at.tzinfo is None
                    ):
                        raise EOMTermsAcceptanceUnavailableError(
                            "Terms database clock is unavailable"
                        )
                    if (
                        invitation["revoked_at"] is not None
                        or accepted_at > invitation["expires_at"]
                        or await self._has_later_material_version(
                            connection, UUID(str(invitation["version_id"]))
                        )
                    ):
                        raise EOMTermsAcceptanceNotFoundError(
                            "Terms invitation is unavailable"
                        )
                    acceptance_id = uuid4()
                    delivery_id = uuid4()
                    accepted_row = await connection.fetchrow(
                        """
                        INSERT INTO eom_terms_acceptances (
                            id, invitation_id, signer_name, terms_accepted,
                            additional_work_accepted, client_ip
                        ) VALUES (
                            $1, $2, $3, TRUE, TRUE, $4::inet
                        )
                        RETURNING accepted_at
                        """,
                        acceptance_id,
                        authenticated.invitation_id,
                        parsed_signer_name,
                        parsed_client_ip,
                    )
                    if accepted_row is None:
                        raise EOMTermsAcceptanceUnavailableError(
                            "Terms acceptance could not be stored"
                        )
                    accepted_at = accepted_row["accepted_at"]
                    if (
                        not isinstance(accepted_at, datetime)
                        or accepted_at.tzinfo is None
                    ):
                        raise EOMTermsAcceptanceUnavailableError(
                            "Terms database clock is unavailable"
                        )
                    documents = _documents_from_version(invitation)
                    selected = documents[str(invitation["audience"])][
                        str(invitation["locale"])
                    ]
                    subject, body = render_eom_terms_executed_copy(
                        full_name=str(invitation["customer_name"]),
                        signer_name=parsed_signer_name,
                        accepted_at=accepted_at,
                        version_label=str(invitation["version_label"]),
                        content_hash=str(invitation["content_hash"]),
                        documents=selected,
                        locale=str(invitation["locale"]),
                    )
                    await connection.execute(
                        """
                        INSERT INTO eom_terms_deliveries (
                            id, kind, invitation_id, acceptance_id,
                            recipient_email, subject, body, body_hash
                        ) VALUES (
                            $1, 'executed_copy', $2, $3, $4, $5, $6, $7
                        )
                        """,
                        delivery_id,
                        authenticated.invitation_id,
                        acceptance_id,
                        invitation["recipient_email"],
                        subject,
                        body,
                        _body_hash(body),
                    )
                    acceptance = await self._acceptance_by_invitation(
                        connection, authenticated.invitation_id
                    )
                    if acceptance is None:
                        raise EOMTermsAcceptanceUnavailableError(
                            "Terms acceptance could not be stored"
                        )
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms acceptance could not be stored"
            ) from exc
        delivery_error = False
        if sender is None:
            try:
                _require_transport_configured()
            except EOMOnboardingDraftError:
                delivery_error = True
            else:
                sender = send_onboarding_email
        if sender is not None:
            delivery, delivery_error = await self._deliver(
                delivery_id=UUID(str(acceptance["delivery_id"])),
                secret=None,
                previous_secret=None,
                sender=sender,
            )
            result_row = dict(acceptance)
            result_row["delivery_status"] = delivery["status"]
        else:
            result_row = dict(acceptance)
        return _acceptance_result(
            result_row,
            idempotent=idempotent,
            delivery_error=delivery_error,
        )

    async def revoke(
        self,
        *,
        invitation_id: object,
        actor_id: object,
        actor_name: object,
    ) -> dict[str, Any]:
        parsed_invitation_id = _uuid(invitation_id, "invitationId")
        parsed_actor_id, parsed_actor_name = _actor(actor_id, actor_name)
        await self.require_schema_ready()
        idempotent = False
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                row = await connection.fetchrow(
                    """
                    SELECT invitation.*,
                           delivery.status AS invitation_delivery_status
                    FROM eom_terms_invitations AS invitation
                    JOIN eom_terms_deliveries AS delivery
                      ON delivery.invitation_id = invitation.id
                     AND delivery.kind = 'invitation'
                    WHERE invitation.id = $1
                    FOR UPDATE OF invitation
                    """,
                    parsed_invitation_id,
                )
                if row is None:
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms invitation was not found"
                    )
                acceptance_id = await connection.fetchval(
                    "SELECT id FROM eom_terms_acceptances WHERE invitation_id = $1",
                    parsed_invitation_id,
                )
                if acceptance_id is not None:
                    raise EOMTermsAcceptanceConflictError(
                        "Accepted Terms invitation cannot be revoked"
                    )
                if row["invitation_delivery_status"] == "sending":
                    raise EOMTermsAcceptanceConflictError(
                        "Terms invitation delivery requires reconciliation"
                    )
                if row["revoked_at"] is not None:
                    idempotent = True
                else:
                    await connection.execute(
                        """
                        UPDATE eom_terms_invitations
                        SET revoked_at = clock_timestamp(),
                            revoked_by_id = $2,
                            revoked_by_name = $3
                        WHERE id = $1 AND revoked_at IS NULL
                        """,
                        parsed_invitation_id,
                        parsed_actor_id,
                        parsed_actor_name,
                    )
                invitation = await self._invitation_by_id(
                    connection, parsed_invitation_id
                )
                if invitation is None:
                    raise EOMTermsAcceptanceUnavailableError(
                        "Terms invitation is unavailable"
                    )
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms invitation could not be revoked"
            ) from exc
        return _invitation_result(invitation, idempotent=idempotent)

    async def confirm_delivery_sent(
        self,
        *,
        delivery_id: object,
        actor_id: object,
        actor_name: object,
    ) -> dict[str, Any]:
        parsed_delivery_id = _uuid(delivery_id, "deliveryId")
        parsed_actor_id, parsed_actor_name = _actor(actor_id, actor_name)
        await self.require_schema_ready()
        try:
            async with self.pool.transaction() as connection:
                delivery = await connection.fetchrow(
                    "SELECT * FROM eom_terms_deliveries WHERE id = $1 FOR UPDATE",
                    parsed_delivery_id,
                )
                if delivery is None:
                    raise EOMTermsAcceptanceNotFoundError(
                        "Terms delivery was not found"
                    )
                if delivery["status"] == "sent":
                    idempotent = True
                    confirmed = delivery
                elif delivery["status"] == "sending":
                    idempotent = False
                    confirmed = await connection.fetchrow(
                        """
                        UPDATE eom_terms_deliveries
                        SET status = 'sent',
                            sent_at = clock_timestamp(),
                            confirmed_by_id = $2,
                            confirmed_by_name = $3
                        WHERE id = $1 AND status = 'sending'
                        RETURNING *
                        """,
                        parsed_delivery_id,
                        parsed_actor_id,
                        parsed_actor_name,
                    )
                else:
                    raise EOMTermsAcceptanceConflictError(
                        "Pending Terms delivery has no transport result to confirm"
                    )
                if confirmed is None:
                    raise EOMTermsAcceptanceUnavailableError(
                        "Terms delivery could not be confirmed"
                    )
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms delivery could not be confirmed"
            ) from exc
        return {
            "deliveryId": str(confirmed["id"]),
            "kind": str(confirmed["kind"]),
            "status": str(confirmed["status"]),
            "sentAt": _iso(confirmed["sent_at"]),
            "idempotent": idempotent,
        }

    async def get_readiness(self, *, contact_id: object) -> dict[str, Any]:
        parsed_contact_id = _uuid(contact_id, "contactId")
        await self.require_schema_ready()
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                contact = await connection.fetchrow(
                    """
                    SELECT id, business_context_id, contact_type, status,
                           customer_type, full_name, email
                    FROM contacts
                    WHERE id = $1
                    FOR SHARE
                    """,
                    parsed_contact_id,
                )
                _, audience, _, _ = self._customer(contact)
                current = await connection.fetchrow(
                    """
                    SELECT version.id, version.version_label,
                           version.content_hash, version.published_at
                    FROM eom_terms_current_version AS selected
                    JOIN eom_terms_versions AS version
                      ON version.id = selected.version_id
                    WHERE selected.singleton AND version.status = 'published'
                    """
                )
                if current is None:
                    raise EOMTermsAcceptanceNotFoundError(
                        "Current Terms version was not found"
                    )
                acceptance = await connection.fetchrow(
                    """
                    SELECT acceptance.id, acceptance.version_id,
                           acceptance.audience, acceptance.accepted_at,
                           version.version_label, version.content_hash,
                           delivery.status AS delivery_status,
                           EXISTS (
                               SELECT 1
                               FROM eom_terms_versions AS later
                               WHERE later.status = 'published'
                                 AND later.material_change
                                 AND later.publication_order
                                     > version.publication_order
                           ) AS later_material
                    FROM eom_terms_acceptances AS acceptance
                    JOIN eom_terms_versions AS version
                      ON version.id = acceptance.version_id
                    JOIN eom_terms_deliveries AS delivery
                      ON delivery.acceptance_id = acceptance.id
                     AND delivery.kind = 'executed_copy'
                    WHERE acceptance.contact_id = $1
                    ORDER BY version.publication_order DESC,
                             acceptance.accepted_at DESC,
                             acceptance.id DESC
                    LIMIT 1
                    """,
                    parsed_contact_id,
                )
        except EOMTermsAcceptanceError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            raise EOMTermsAcceptanceUnavailableError(
                "Terms readiness is unavailable"
            ) from exc
        if acceptance is None:
            reason = "not_accepted"
            ready = False
        elif acceptance["audience"] != audience:
            reason = "audience_changed"
            ready = False
        elif bool(acceptance["later_material"]):
            reason = "reacceptance_required"
            ready = False
        else:
            reason = "accepted"
            ready = True
        return {
            "contactId": str(parsed_contact_id),
            "audience": audience,
            "ready": ready,
            "reason": reason,
            "currentVersionId": str(current["id"]),
            "currentVersionLabel": str(current["version_label"]),
            "currentContentHash": str(current["content_hash"]),
            "acceptedVersionId": (
                str(acceptance["version_id"]) if acceptance is not None else None
            ),
            "acceptedVersionLabel": (
                str(acceptance["version_label"]) if acceptance is not None else None
            ),
            "acceptedAt": (
                _iso(acceptance["accepted_at"]) if acceptance is not None else None
            ),
            "executedCopyDeliveryStatus": (
                str(acceptance["delivery_status"]) if acceptance is not None else None
            ),
        }
