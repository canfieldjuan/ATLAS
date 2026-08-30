"""Provider-confirmed card-on-file authority for EOM onboarding."""

from __future__ import annotations

import hmac
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol
from urllib.parse import urlencode, urlsplit
from uuid import UUID, uuid4

import asyncpg
import stripe

from ..eom_api.config import (
    EOM_CARD_VAULT_RETURN_URL_MAX_BYTES,
    build_eom_card_vault_return_urls,
)
from .eom_terms_acceptance import AuthenticatedEOMTermsToken
from .eom_terms_authority import EOM_TERMS_PUBLICATION_LOCK_KEY

EOM_CARD_VAULT_SOURCE = "eom_card_vault"
EOM_CARD_VAULT_STRIPE_API_VERSION = "2026-05-27.dahlia"
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9_]{4,255}$")
_PROVIDER_RETRY_WINDOW = timedelta(hours=1)
_LOGGER = logging.getLogger(__name__)


def _log_boundary_failure(operation: str, **identifiers: object) -> None:
    """Preserve operational evidence without logging payment or token material."""

    context = {
        key: str(value) for key, value in identifiers.items() if value is not None
    }
    _LOGGER.exception(
        "EOM card-vault boundary failed: %s",
        operation,
        extra={
            "eom_card_vault_operation": operation,
            "eom_card_vault_context": context,
        },
    )


class EOMCardVaultError(Exception):
    """Base class for stable EOM card-vault API failures."""

    status_code = 409
    code = "eom_card_vault_error"


class EOMCardVaultValidationError(EOMCardVaultError):
    status_code = 422
    code = "invalid_eom_card_vault_request"


class EOMCardVaultNotFoundError(EOMCardVaultError):
    status_code = 404
    code = "eom_card_vault_not_found"


class EOMCardVaultConflictError(EOMCardVaultError):
    status_code = 409
    code = "eom_card_vault_conflict"


class EOMCardVaultUnavailableError(EOMCardVaultError):
    status_code = 503
    code = "eom_card_vault_unavailable"


class EOMCardVaultProviderError(Exception):
    """Bounded provider failure safe to translate at the service boundary."""


class EOMCardVaultSignatureError(EOMCardVaultProviderError):
    """Stripe rejected the webhook signature."""


class EOMCardVaultProvider(Protocol):
    async def create_customer(self, **kwargs: Any) -> Any: ...

    async def create_checkout_session(self, **kwargs: Any) -> Any: ...

    async def retrieve_checkout_session(self, session_id: str) -> Any: ...

    async def retrieve_setup_intent(self, setup_intent_id: str) -> Any: ...

    def construct_event(self, payload: bytes, signature: str) -> Any: ...


def _stripe_error_types() -> tuple[type[BaseException], ...]:
    candidates = (
        getattr(stripe, "StripeError", None),
        getattr(getattr(stripe, "error", None), "StripeError", None),
    )
    return tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, type) and issubclass(candidate, BaseException)
    )


def _stripe_signature_error_types() -> tuple[type[BaseException], ...]:
    candidates = (
        getattr(stripe, "SignatureVerificationError", None),
        getattr(getattr(stripe, "error", None), "SignatureVerificationError", None),
    )
    return tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, type) and issubclass(candidate, BaseException)
    )


@dataclass(frozen=True)
class StripeEOMCardVaultProvider:
    """Stripe adapter with per-request credentials and bounded network calls."""

    secret_key: str
    webhook_secret: str
    timeout_seconds: int
    _http_client: Any = field(init=False, repr=False, compare=False)
    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        http_client = stripe.HTTPXClient(timeout=self.timeout_seconds)
        object.__setattr__(self, "_http_client", http_client)
        object.__setattr__(
            self,
            "_client",
            stripe.StripeClient(
                self.secret_key,
                stripe_version=EOM_CARD_VAULT_STRIPE_API_VERSION,
                http_client=http_client,
            ),
        )

    async def _call(self, operation: Any, /, *args: Any, **kwargs: Any) -> Any:
        try:
            return await operation(*args, **kwargs)
        except (*_stripe_error_types(), OSError, TimeoutError) as exc:
            raise EOMCardVaultProviderError("Stripe operation failed") from exc

    async def close(self) -> None:
        await self._http_client.close_async()

    async def create_customer(self, **kwargs: Any) -> Any:
        params = dict(kwargs)
        idempotency_key = params.pop("idempotency_key")
        return await self._call(
            self._client.v1.customers.create_async,
            params,
            {"idempotency_key": idempotency_key},
        )

    async def create_checkout_session(self, **kwargs: Any) -> Any:
        params = dict(kwargs)
        idempotency_key = params.pop("idempotency_key")
        return await self._call(
            self._client.v1.checkout.sessions.create_async,
            params,
            {"idempotency_key": idempotency_key},
        )

    async def retrieve_checkout_session(self, session_id: str) -> Any:
        return await self._call(
            self._client.v1.checkout.sessions.retrieve_async, session_id
        )

    async def retrieve_setup_intent(self, setup_intent_id: str) -> Any:
        return await self._call(
            self._client.v1.setup_intents.retrieve_async, setup_intent_id
        )

    def construct_event(self, payload: bytes, signature: str) -> Any:
        try:
            return stripe.Webhook.construct_event(
                payload,
                signature,
                self.webhook_secret,
            )
        except _stripe_signature_error_types() as exc:
            raise EOMCardVaultSignatureError("Invalid Stripe signature") from exc
        except (ValueError, TypeError) as exc:
            raise EOMCardVaultSignatureError("Invalid Stripe event") from exc


def _value(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _nested_id(value: Any) -> Any:
    if isinstance(value, str):
        return value
    return _value(value, "id")


def _provider_id(value: Any, *, prefix: str, label: str) -> str:
    parsed = _nested_id(value)
    if (
        not isinstance(parsed, str)
        or not parsed.startswith(prefix)
        or len(parsed) <= len(prefix)
        or not _PROVIDER_ID.fullmatch(parsed)
    ):
        raise EOMCardVaultConflictError(f"Stripe {label} is invalid")
    return parsed


def _uuid(value: object, label: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EOMCardVaultValidationError(f"{label} is invalid") from exc


def _reserved_return_url(value: object, outcome: str) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > EOM_CARD_VAULT_RETURN_URL_MAX_BYTES
    ):
        raise EOMCardVaultConflictError("Stored Checkout return URL is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise EOMCardVaultConflictError(
            "Stored Checkout return URL is invalid"
        ) from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port == 0
        or parsed.query != urlencode({"cardVault": outcome})
        or parsed.fragment
    ):
        raise EOMCardVaultConflictError("Stored Checkout return URL is invalid")
    return value


def _expires_at(value: Any) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EOMCardVaultConflictError("Stripe Checkout expiration is invalid")
    try:
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise EOMCardVaultConflictError(
            "Stripe Checkout expiration is invalid"
        ) from exc
    return parsed


async def eom_card_vault_schema_ready(pool: Any) -> bool:
    """Attest the controlled card-vault relations before any provider effect."""

    try:
        return bool(
            await pool.fetchval(
                """
                /* eom_card_vault_schema_ready */
                WITH expected_relations(name) AS (
                    VALUES ('eom_card_vault_enrollments'),
                           ('eom_card_vault_sessions'),
                           ('eom_card_vault_events')
                ),
                expected_triggers(
                    relation_name, function_name, trigger_name, trigger_type,
                    function_body_md5
                ) AS (
                    VALUES
                        ('eom_card_vault_enrollments',
                         'protect_eom_card_vault_enrollment',
                         'trg_protect_eom_card_vault_enrollment', 31,
                         '45dc4afa04586575909eadfb5de4d629'),
                        ('eom_card_vault_enrollments',
                         'protect_eom_card_vault_enrollment',
                         'trg_protect_eom_card_vault_enrollment_truncate', 34,
                         '45dc4afa04586575909eadfb5de4d629'),
                        ('eom_card_vault_sessions',
                         'protect_eom_card_vault_session',
                         'trg_protect_eom_card_vault_session', 31,
                         'f4d5e3cbee0f637f9d9bc377acd74140'),
                        ('eom_card_vault_sessions',
                         'protect_eom_card_vault_session',
                         'trg_protect_eom_card_vault_session_truncate', 34,
                         'f4d5e3cbee0f637f9d9bc377acd74140'),
                        ('eom_card_vault_events',
                         'protect_eom_card_vault_event',
                         'trg_protect_eom_card_vault_event', 31,
                         'c1ed425051daf9cbbb57e221a2c46ae7'),
                        ('eom_card_vault_events',
                         'protect_eom_card_vault_event',
                         'trg_protect_eom_card_vault_event_truncate', 34,
                         'c1ed425051daf9cbbb57e221a2c46ae7')
                ),
                expected_constraints(
                    relation_name, constraint_name, constraint_type,
                    constraint_definition
                ) AS (
                    VALUES
                        ('eom_card_vault_enrollments',
                         'pk_eom_card_vault_enrollments', 'p',
                         'PRIMARY KEY (id)'),
                        ('eom_card_vault_enrollments',
                         'uq_eom_card_vault_enrollment_candidate', 'u',
                         'UNIQUE (candidate_id)'),
                        ('eom_card_vault_enrollments',
                         'uq_eom_card_vault_enrollment_contact', 'u',
                         'UNIQUE (contact_id)'),
                        ('eom_card_vault_enrollments',
                         'uq_eom_card_vault_stripe_customer', 'u',
                         'UNIQUE (stripe_customer_id)'),
                        ('eom_card_vault_enrollments',
                         'uq_eom_card_vault_setup_intent', 'u',
                         'UNIQUE (stripe_setup_intent_id)'),
                        ('eom_card_vault_enrollments',
                         'uq_eom_card_vault_payment_method', 'u',
                         'UNIQUE (stripe_payment_method_id)'),
                        ('eom_card_vault_enrollments',
                         'fk_eom_card_vault_enrollment_candidate', 'f',
                         'FOREIGN KEY (candidate_id) REFERENCES '
                         'eom_post_clean_onboarding_candidates(id) '
                         'ON DELETE RESTRICT'),
                        ('eom_card_vault_enrollments',
                         'fk_eom_card_vault_enrollment_contact', 'f',
                         'FOREIGN KEY (contact_id) REFERENCES contacts(id) '
                         'ON DELETE RESTRICT'),
                        ('eom_card_vault_enrollments',
                         'fk_eom_card_vault_enrollment_initial_acceptance', 'f',
                         'FOREIGN KEY (initial_acceptance_id) REFERENCES '
                         'eom_terms_acceptances(id) ON DELETE RESTRICT'),
                        ('eom_card_vault_enrollments',
                         'ck_eom_card_vault_enrollment_context', 'c',
                         'CHECK (business_context_id::text = '
                         '''effingham_maids''::text)'),
                        ('eom_card_vault_enrollments',
                         'ck_eom_card_vault_enrollment_customer', 'c',
                         'CHECK (stripe_customer_id IS NULL OR '
                         'stripe_customer_id::text ~ '
                         '''^cus_[A-Za-z0-9_]+$''::text)'),
                        ('eom_card_vault_enrollments',
                         'ck_eom_card_vault_enrollment_state', 'c',
                         'CHECK (status::text = ''pending''::text AND '
                         'stripe_setup_intent_id IS NULL AND '
                         'stripe_payment_method_id IS NULL AND ready_at IS NULL '
                         'OR status::text = ''ready''::text AND '
                         'stripe_customer_id IS NOT NULL AND '
                         'stripe_customer_id::text ~ '
                         '''^cus_[A-Za-z0-9_]+$''::text AND '
                         'stripe_setup_intent_id IS NOT NULL AND '
                         'stripe_setup_intent_id::text ~ '
                         '''^seti_[A-Za-z0-9_]+$''::text AND '
                         'stripe_payment_method_id IS NOT NULL AND '
                         'stripe_payment_method_id::text ~ '
                         '''^pm_[A-Za-z0-9_]+$''::text AND ready_at IS NOT NULL)'),
                        ('eom_card_vault_sessions',
                         'pk_eom_card_vault_sessions', 'p',
                         'PRIMARY KEY (id)'),
                        ('eom_card_vault_sessions',
                         'uq_eom_card_vault_checkout_session', 'u',
                         'UNIQUE (stripe_checkout_session_id)'),
                        ('eom_card_vault_sessions',
                         'fk_eom_card_vault_session_enrollment', 'f',
                         'FOREIGN KEY (enrollment_id) REFERENCES '
                         'eom_card_vault_enrollments(id) ON DELETE RESTRICT'),
                        ('eom_card_vault_sessions',
                         'fk_eom_card_vault_session_acceptance', 'f',
                         'FOREIGN KEY (acceptance_id) REFERENCES '
                         'eom_terms_acceptances(id) ON DELETE RESTRICT'),
                        ('eom_card_vault_sessions',
                         'ck_eom_card_vault_session_return_urls', 'c',
                         'CHECK (octet_length(checkout_success_url) >= 1 AND '
                         'octet_length(checkout_success_url) <= 2048 AND '
                         'octet_length(checkout_cancel_url) >= 1 AND '
                         'octet_length(checkout_cancel_url) <= 2048 AND '
                         'checkout_success_url ~~ '
                         '''https://%?cardVault=success''::text AND '
                         'checkout_cancel_url ~~ '
                         '''https://%?cardVault=cancelled''::text AND '
                         'checkout_success_url <> checkout_cancel_url)'),
                        ('eom_card_vault_sessions',
                         'ck_eom_card_vault_session_retry_window', 'c',
                         'CHECK (provider_retry_until > created_at AND '
                         'provider_retry_until <= '
                         '(created_at + ''02:00:00''::interval))'),
                        ('eom_card_vault_sessions',
                         'ck_eom_card_vault_session_state', 'c',
                         'CHECK (state::text = ''creating''::text AND '
                         'stripe_checkout_session_id IS NULL AND '
                         'checkout_expires_at IS NULL OR '
                         'state::text = ''open''::text AND '
                         'stripe_checkout_session_id IS NOT NULL AND '
                         'stripe_checkout_session_id::text ~ '
                         '''^cs_[A-Za-z0-9_]+$''::text AND '
                         'checkout_expires_at IS NOT NULL AND '
                         'checkout_expires_at > created_at)'),
                        ('eom_card_vault_events',
                         'pk_eom_card_vault_events', 'p',
                         'PRIMARY KEY (stripe_event_id)'),
                        ('eom_card_vault_events',
                         'uq_eom_card_vault_event_session', 'u',
                         'UNIQUE (session_id)'),
                        ('eom_card_vault_events',
                         'fk_eom_card_vault_event_enrollment', 'f',
                         'FOREIGN KEY (enrollment_id) REFERENCES '
                         'eom_card_vault_enrollments(id) ON DELETE RESTRICT'),
                        ('eom_card_vault_events',
                         'fk_eom_card_vault_event_session', 'f',
                         'FOREIGN KEY (session_id) REFERENCES '
                         'eom_card_vault_sessions(id) ON DELETE RESTRICT'),
                        ('eom_card_vault_events',
                         'ck_eom_card_vault_event_id', 'c',
                         'CHECK (stripe_event_id::text ~ '
                         '''^evt_[A-Za-z0-9_]+$''::text)'),
                        ('eom_card_vault_events',
                         'ck_eom_card_vault_event_setup_intent', 'c',
                         'CHECK (stripe_setup_intent_id::text ~ '
                         '''^seti_[A-Za-z0-9_]+$''::text)'),
                        ('eom_card_vault_events',
                         'ck_eom_card_vault_event_payment_method', 'c',
                         'CHECK (stripe_payment_method_id::text ~ '
                         '''^pm_[A-Za-z0-9_]+$''::text)')
                ),
                expected_columns(
                    relation_name, column_name, data_type, not_null,
                    default_expression
                ) AS (
                    VALUES
                        ('eom_card_vault_enrollments', 'id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_enrollments', 'candidate_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_enrollments', 'contact_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_enrollments', 'initial_acceptance_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_enrollments', 'business_context_id',
                         'character varying(64)', TRUE,
                         '''effingham_maids''::character varying'),
                        ('eom_card_vault_enrollments', 'stripe_customer_id',
                         'character varying(255)', FALSE, NULL),
                        ('eom_card_vault_enrollments', 'status',
                         'character varying(16)', TRUE,
                         '''pending''::character varying'),
                        ('eom_card_vault_enrollments', 'stripe_setup_intent_id',
                         'character varying(255)', FALSE, NULL),
                        ('eom_card_vault_enrollments', 'stripe_payment_method_id',
                         'character varying(255)', FALSE, NULL),
                        ('eom_card_vault_enrollments', 'ready_at',
                         'timestamp with time zone', FALSE, NULL),
                        ('eom_card_vault_enrollments', 'created_at',
                         'timestamp with time zone', TRUE, 'CURRENT_TIMESTAMP'),
                        ('eom_card_vault_sessions', 'id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_sessions', 'enrollment_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_sessions', 'acceptance_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_sessions', 'checkout_success_url',
                         'text', TRUE, NULL),
                        ('eom_card_vault_sessions', 'checkout_cancel_url',
                         'text', TRUE, NULL),
                        ('eom_card_vault_sessions', 'provider_retry_until',
                         'timestamp with time zone', TRUE, NULL),
                        ('eom_card_vault_sessions', 'state',
                         'character varying(16)', TRUE,
                         '''creating''::character varying'),
                        ('eom_card_vault_sessions',
                         'stripe_checkout_session_id',
                         'character varying(255)', FALSE, NULL),
                        ('eom_card_vault_sessions', 'checkout_expires_at',
                         'timestamp with time zone', FALSE, NULL),
                        ('eom_card_vault_sessions', 'created_at',
                         'timestamp with time zone', TRUE, 'CURRENT_TIMESTAMP'),
                        ('eom_card_vault_events', 'stripe_event_id',
                         'character varying(255)', TRUE, NULL),
                        ('eom_card_vault_events', 'enrollment_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_events', 'session_id',
                         'uuid', TRUE, NULL),
                        ('eom_card_vault_events', 'stripe_setup_intent_id',
                         'character varying(255)', TRUE, NULL),
                        ('eom_card_vault_events', 'stripe_payment_method_id',
                         'character varying(255)', TRUE, NULL),
                        ('eom_card_vault_events', 'received_at',
                         'timestamp with time zone', TRUE, 'CURRENT_TIMESTAMP')
                ),
                expected_runtime_columns(relation_name, column_name, privilege) AS (
                    VALUES
                        ('eom_card_vault_enrollments', 'id', 'INSERT'),
                        ('eom_card_vault_enrollments', 'candidate_id', 'INSERT'),
                        ('eom_card_vault_enrollments', 'contact_id', 'INSERT'),
                        ('eom_card_vault_enrollments', 'initial_acceptance_id', 'INSERT'),
                        ('eom_card_vault_enrollments', 'stripe_customer_id', 'UPDATE'),
                        ('eom_card_vault_enrollments', 'status', 'UPDATE'),
                        ('eom_card_vault_enrollments', 'stripe_setup_intent_id', 'UPDATE'),
                        ('eom_card_vault_enrollments', 'stripe_payment_method_id', 'UPDATE'),
                        ('eom_card_vault_enrollments', 'ready_at', 'UPDATE'),
                        ('eom_card_vault_sessions', 'id', 'INSERT'),
                        ('eom_card_vault_sessions', 'enrollment_id', 'INSERT'),
                        ('eom_card_vault_sessions', 'acceptance_id', 'INSERT'),
                        ('eom_card_vault_sessions', 'checkout_success_url', 'INSERT'),
                        ('eom_card_vault_sessions', 'checkout_cancel_url', 'INSERT'),
                        ('eom_card_vault_sessions', 'provider_retry_until', 'INSERT'),
                        ('eom_card_vault_sessions', 'state', 'UPDATE'),
                        ('eom_card_vault_sessions', 'stripe_checkout_session_id', 'UPDATE'),
                        ('eom_card_vault_sessions', 'checkout_expires_at', 'UPDATE'),
                        ('eom_card_vault_events', 'stripe_event_id', 'INSERT'),
                        ('eom_card_vault_events', 'enrollment_id', 'INSERT'),
                        ('eom_card_vault_events', 'session_id', 'INSERT'),
                        ('eom_card_vault_events', 'stripe_setup_intent_id', 'INSERT'),
                        ('eom_card_vault_events', 'stripe_payment_method_id', 'INSERT')
                )
                SELECT current_user = 'atlas'
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
                          FROM pg_roles AS guard_role
                          WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
                            AND pg_has_role(
                                runtime_role.oid,
                                guard_role.oid,
                                'MEMBER'
                            )
                      )
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_relations AS expected
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_class AS relation
                        JOIN pg_namespace AS namespace
                          ON namespace.oid = relation.relnamespace
                        JOIN pg_roles AS owner ON owner.oid = relation.relowner
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = expected.name
                          AND relation.relkind = 'r'
                          AND owner.rolname = 'atlas_eom_handoff_owner'
                    )
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_relations AS expected
                    WHERE NOT COALESCE(
                        has_table_privilege(
                            current_user,
                            to_regclass(format(
                                '%I.%I', current_schema(), expected.name
                            )),
                            'SELECT'
                        ),
                        FALSE
                    )
                    OR COALESCE(
                        has_table_privilege(
                            current_user,
                            to_regclass(format(
                                '%I.%I', current_schema(), expected.name
                            )),
                            'DELETE'
                        ),
                        FALSE
                    )
                    OR COALESCE(
                        has_table_privilege(
                            current_user,
                            to_regclass(format(
                                '%I.%I', current_schema(), expected.name
                            )),
                            'TRUNCATE'
                        ),
                        FALSE
                    )
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_columns AS expected
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_attribute AS column_definition
                        JOIN pg_class AS relation
                          ON relation.oid = column_definition.attrelid
                        JOIN pg_namespace AS namespace
                          ON namespace.oid = relation.relnamespace
                        LEFT JOIN pg_attrdef AS stored_default
                          ON stored_default.adrelid = relation.oid
                         AND stored_default.adnum = column_definition.attnum
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = expected.relation_name
                          AND column_definition.attname = expected.column_name
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
                    JOIN pg_class AS relation
                      ON relation.oid = column_definition.attrelid
                    JOIN pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = current_schema()
                      AND relation.relname IN (
                          SELECT name FROM expected_relations
                      )
                      AND column_definition.attnum > 0
                      AND NOT column_definition.attisdropped
                ) = (SELECT count(*) FROM expected_columns)
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_runtime_columns AS expected
                    WHERE NOT COALESCE(
                        has_column_privilege(
                            current_user,
                            to_regclass(format(
                                '%I.%I', current_schema(), expected.relation_name
                            )),
                            expected.column_name,
                            expected.privilege
                        ),
                        FALSE
                    )
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_relations AS expected_relation
                    JOIN pg_class AS relation
                      ON relation.oid = to_regclass(format(
                          '%I.%I', current_schema(), expected_relation.name
                      ))
                    JOIN pg_attribute AS column_definition
                      ON column_definition.attrelid = relation.oid
                     AND column_definition.attnum > 0
                     AND NOT column_definition.attisdropped
                    CROSS JOIN (
                        VALUES ('INSERT'), ('UPDATE')
                    ) AS candidate_privilege(privilege)
                    WHERE has_column_privilege(
                        current_user,
                        relation.oid,
                        column_definition.attname,
                        candidate_privilege.privilege
                    )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM expected_runtime_columns AS expected
                          WHERE expected.relation_name = expected_relation.name
                            AND expected.column_name = column_definition.attname
                            AND expected.privilege = candidate_privilege.privilege
                      )
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_triggers AS expected
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_trigger AS trigger
                        JOIN pg_class AS relation ON relation.oid = trigger.tgrelid
                        JOIN pg_namespace AS namespace
                          ON namespace.oid = relation.relnamespace
                        JOIN pg_proc AS function ON function.oid = trigger.tgfoid
                        JOIN pg_language AS language
                          ON language.oid = function.prolang
                        JOIN pg_roles AS owner ON owner.oid = function.proowner
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = expected.relation_name
                          AND function.proname = expected.function_name
                          AND trigger.tgname = expected.trigger_name
                          AND NOT trigger.tgisinternal
                          AND trigger.tgenabled = 'O'
                          AND trigger.tgtype = expected.trigger_type
                          AND trigger.tgattr = ''::int2vector
                          AND trigger.tgnargs = 0
                          AND trigger.tgqual IS NULL
                          AND language.lanname = 'plpgsql'
                          AND NOT function.prosecdef
                          AND NOT function.proleakproof
                          AND NOT function.proisstrict
                          AND function.provolatile = 'v'
                          AND function.proparallel = 'u'
                          AND function.prokind = 'f'
                          AND function.pronargs = 0
                          AND function.prorettype = 'trigger'::regtype
                          AND function.probin IS NULL
                          AND md5(function.prosrc) = expected.function_body_md5
                          AND function.proconfig = ARRAY[
                              format(
                                  'search_path=pg_catalog, %I, pg_temp',
                                  current_schema()
                              )
                          ]
                          AND owner.rolname = 'atlas_eom_handoff_owner'
                    )
                )
                AND (
                    SELECT count(*)
                    FROM pg_trigger AS trigger
                    JOIN pg_class AS relation ON relation.oid = trigger.tgrelid
                    JOIN pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = current_schema()
                      AND relation.relname IN (
                          SELECT name FROM expected_relations
                      )
                      AND NOT trigger.tgisinternal
                ) = (SELECT count(*) FROM expected_triggers)
                AND NOT EXISTS (
                    SELECT 1
                    FROM expected_constraints AS expected
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint AS actual
                        JOIN pg_class AS relation
                          ON relation.oid = actual.conrelid
                        JOIN pg_namespace AS namespace
                          ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = expected.relation_name
                          AND actual.conname = expected.constraint_name
                          AND actual.contype = expected.constraint_type
                          AND actual.convalidated
                          AND pg_get_constraintdef(actual.oid, true)
                              = expected.constraint_definition
                    )
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint AS actual
                    JOIN pg_class AS relation ON relation.oid = actual.conrelid
                    JOIN pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = current_schema()
                      AND relation.relname IN (
                          SELECT name FROM expected_relations
                      )
                      AND actual.contype IN ('p', 'u', 'f', 'c')
                      AND NOT EXISTS (
                          SELECT 1
                          FROM expected_constraints AS expected
                          WHERE expected.relation_name = relation.relname
                            AND expected.constraint_name = actual.conname
                            AND expected.constraint_type = actual.contype
                            AND expected.constraint_definition
                                = pg_get_constraintdef(actual.oid, true)
                      )
                )
                """
            )
        )
    except (asyncpg.PostgresError, OSError, TimeoutError):
        _log_boundary_failure("schema_attestation")
        return False


class EOMCardVaultService:
    """Own enrollment, hosted setup-session, and provider-confirmed readiness."""

    def __init__(
        self,
        *,
        pool: Any,
        provider: EOMCardVaultProvider | None = None,
    ) -> None:
        self._pool = pool
        self._provider = provider

    @property
    def pool(self) -> Any:
        if not bool(getattr(self._pool, "is_initialized", True)):
            raise EOMCardVaultUnavailableError("EOM card-vault database is unavailable")
        return self._pool

    @property
    def provider(self) -> EOMCardVaultProvider:
        if self._provider is None:
            raise EOMCardVaultUnavailableError("Stripe card setup is unavailable")
        return self._provider

    async def require_schema_ready(self) -> None:
        if not await eom_card_vault_schema_ready(self.pool):
            raise EOMCardVaultUnavailableError("EOM card-vault schema is unavailable")

    @staticmethod
    def _require_token(token: object) -> AuthenticatedEOMTermsToken:
        if not isinstance(token, AuthenticatedEOMTermsToken):
            raise EOMCardVaultNotFoundError("Card setup is unavailable")
        return token

    @staticmethod
    def _validate_eligibility(
        row: Mapping[str, Any] | None,
        token: AuthenticatedEOMTermsToken,
    ) -> None:
        if row is None or not hmac.compare_digest(
            str(row["signing_key_fingerprint"]),
            token.signing_key_fingerprint,
        ):
            raise EOMCardVaultNotFoundError("Card setup is unavailable")
        if (
            row["revoked_at"] is not None
            or bool(row["is_expired"])
            or bool(row["later_material"])
            or str(row["business_context_id"]) != "effingham_maids"
            or str(row["contact_type"]) != "customer"
            or str(row["contact_status"]) != "active"
            or str(row["customer_type"]) != "residential"
            or str(row["invitation_audience"]) != "residential"
            or str(row["acceptance_audience"]) != "residential"
            or str(row["full_name"]).strip() != str(row["invitation_customer_name"])
            or str(row["email"] or "").strip().lower()
            != str(row["invitation_recipient_email"])
            or str(row["acceptance_recipient_email"])
            != str(row["invitation_recipient_email"])
            or row["candidate_id"] is None
            or str(row["candidate_status"]) != "pending"
            or not str(row["email"] or "").strip()
        ):
            raise EOMCardVaultNotFoundError("Card setup is unavailable")

    @staticmethod
    def _checkout_result(
        *,
        enrollment: Mapping[str, Any],
        checkout_url: str | None,
        checkout_expires_at: datetime | None,
        idempotent: bool,
    ) -> dict[str, Any]:
        ready = str(enrollment["status"]) == "ready"
        return {
            "enrollmentId": str(enrollment["id"]),
            "contactId": str(enrollment["contact_id"]),
            "candidateId": str(enrollment["candidate_id"]),
            "status": "ready" if ready else "pending",
            "checkoutUrl": None if ready else checkout_url,
            "checkoutExpiresAt": None if ready else checkout_expires_at,
            "providerConfirmedAt": enrollment["ready_at"] if ready else None,
            "idempotent": idempotent,
        }

    async def _reserve_session(
        self,
        authenticated: AuthenticatedEOMTermsToken,
        *,
        checkout_success_url: str,
        checkout_cancel_url: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, bool]:
        """Commit stable operation IDs before making a provider request."""

        try:
            async with self.pool.transaction() as connection:
                # Current Terms are protected only while the attempt is
                # admitted. Provider I/O happens after this transaction, so
                # unrelated customers and Terms publication are not held on a
                # network round trip.
                await connection.execute(
                    "SELECT pg_advisory_xact_lock_shared(hashtextextended($1, 0))",
                    EOM_TERMS_PUBLICATION_LOCK_KEY,
                )
                eligibility = await connection.fetchrow(
                    """
                    /* eom_card_vault_eligibility */
                    SELECT invitation.signing_key_fingerprint,
                           invitation.revoked_at,
                           clock_timestamp() > invitation.expires_at AS is_expired,
                           invitation.audience AS invitation_audience,
                           invitation.customer_name AS invitation_customer_name,
                           invitation.recipient_email AS invitation_recipient_email,
                           acceptance.id AS acceptance_id,
                           acceptance.audience AS acceptance_audience,
                           acceptance.recipient_email AS acceptance_recipient_email,
                           contact.id AS contact_id,
                           contact.business_context_id,
                           contact.contact_type,
                           contact.status AS contact_status,
                           contact.customer_type,
                           contact.full_name,
                           contact.email,
                           candidate.id AS candidate_id,
                           candidate.status AS candidate_status,
                           clock_timestamp() AS database_now,
                           EXISTS (
                               SELECT 1
                               FROM eom_terms_versions AS later
                               JOIN eom_terms_versions AS accepted_version
                                 ON accepted_version.id = acceptance.version_id
                               WHERE later.status = 'published'
                                 AND later.material_change
                                 AND later.publication_order
                                     > accepted_version.publication_order
                           ) AS later_material
                    FROM eom_terms_invitations AS invitation
                    JOIN eom_terms_acceptances AS acceptance
                      ON acceptance.invitation_id = invitation.id
                     AND acceptance.contact_id = invitation.contact_id
                    JOIN contacts AS contact ON contact.id = invitation.contact_id
                    LEFT JOIN eom_post_clean_onboarding_candidates AS candidate
                      ON candidate.contact_id = contact.id
                    WHERE invitation.id = $1
                    FOR UPDATE OF invitation, contact
                    """,
                    authenticated.invitation_id,
                )
                self._validate_eligibility(eligibility, authenticated)
                assert eligibility is not None
                enrollment = await connection.fetchrow(
                    """
                    /* eom_card_vault_reserve_enrollment */
                    INSERT INTO eom_card_vault_enrollments (
                        id, candidate_id, contact_id, initial_acceptance_id
                    ) VALUES ($1, $2, $3, $4)
                    ON CONFLICT (candidate_id) DO NOTHING
                    RETURNING *
                    """,
                    uuid4(),
                    eligibility["candidate_id"],
                    eligibility["contact_id"],
                    eligibility["acceptance_id"],
                )
                if enrollment is None:
                    enrollment = await connection.fetchrow(
                        """
                        /* eom_card_vault_lock_enrollment */
                        SELECT * FROM eom_card_vault_enrollments
                        WHERE candidate_id = $1
                        FOR UPDATE
                        """,
                        eligibility["candidate_id"],
                    )
                if enrollment is None:
                    raise EOMCardVaultUnavailableError(
                        "Card-vault enrollment could not be reserved"
                    )
                if UUID(str(enrollment["contact_id"])) != UUID(
                    str(eligibility["contact_id"])
                ):
                    raise EOMCardVaultConflictError(
                        "Card-vault enrollment subject changed"
                    )
                if str(enrollment["status"]) == "ready":
                    return dict(eligibility), dict(enrollment), None, True

                latest = await connection.fetchrow(
                    """
                    /* eom_card_vault_latest_session */
                    SELECT * FROM eom_card_vault_sessions
                    WHERE enrollment_id = $1 AND acceptance_id = $2
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    enrollment["id"],
                    eligibility["acceptance_id"],
                )
                reused = latest is not None
                if latest is not None and str(latest["state"]) not in {
                    "creating",
                    "open",
                }:
                    raise EOMCardVaultUnavailableError(
                        "Stored card-vault session state is invalid"
                    )
                if latest is None or (
                    str(latest["state"]) == "open"
                    and latest["checkout_expires_at"] <= eligibility["database_now"]
                ):
                    latest = await connection.fetchrow(
                        """
                        /* eom_card_vault_reserve_session */
                        INSERT INTO eom_card_vault_sessions (
                            id, enrollment_id, acceptance_id,
                            checkout_success_url, checkout_cancel_url,
                            provider_retry_until
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING *
                        """,
                        uuid4(),
                        enrollment["id"],
                        eligibility["acceptance_id"],
                        checkout_success_url,
                        checkout_cancel_url,
                        eligibility["database_now"] + _PROVIDER_RETRY_WINDOW,
                    )
                    reused = False
                if latest is None:
                    raise EOMCardVaultUnavailableError(
                        "Card-vault session could not be reserved"
                    )
                return (
                    dict(eligibility),
                    dict(enrollment),
                    dict(latest),
                    reused,
                )
        except EOMCardVaultError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure(
                "reserve_session",
                invitation_id=authenticated.invitation_id,
            )
            raise EOMCardVaultUnavailableError(
                "EOM card setup could not be reserved"
            ) from exc

    async def _store_customer(
        self,
        *,
        enrollment_id: UUID,
        customer_id: str,
    ) -> dict[str, Any]:
        try:
            async with self.pool.transaction() as connection:
                enrollment = await connection.fetchrow(
                    """
                    /* eom_card_vault_store_customer */
                    UPDATE eom_card_vault_enrollments
                    SET stripe_customer_id = $2
                    WHERE id = $1 AND stripe_customer_id IS NULL
                    RETURNING *
                    """,
                    enrollment_id,
                    customer_id,
                )
                if enrollment is None:
                    enrollment = await connection.fetchrow(
                        """
                        /* eom_card_vault_existing_customer */
                        SELECT * FROM eom_card_vault_enrollments
                        WHERE id = $1
                        FOR UPDATE
                        """,
                        enrollment_id,
                    )
                if (
                    enrollment is None
                    or str(enrollment["stripe_customer_id"]) != customer_id
                ):
                    raise EOMCardVaultConflictError(
                        "Stored Stripe customer does not match"
                    )
                return dict(enrollment)
        except EOMCardVaultError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure("store_customer", enrollment_id=enrollment_id)
            raise EOMCardVaultUnavailableError(
                "Stripe customer could not be stored"
            ) from exc

    async def _store_session(
        self,
        *,
        connection: Any,
        session_row_id: UUID,
        enrollment_id: UUID,
        checkout_session_id: str,
        checkout_expires_at: datetime,
    ) -> dict[str, Any]:
        try:
            session = await connection.fetchrow(
                """
                /* eom_card_vault_store_session */
                UPDATE eom_card_vault_sessions AS session
                SET state = 'open',
                    stripe_checkout_session_id = $2,
                    checkout_expires_at = $3
                WHERE session.id = $1
                  AND session.state = 'creating'
                  AND EXISTS (
                      SELECT 1
                      FROM eom_card_vault_enrollments AS enrollment
                      WHERE enrollment.id = $4
                        AND enrollment.status = 'pending'
                  )
                RETURNING session.*
                """,
                session_row_id,
                checkout_session_id,
                checkout_expires_at,
                enrollment_id,
            )
            if session is None:
                session = await connection.fetchrow(
                    """
                    /* eom_card_vault_existing_session */
                    SELECT * FROM eom_card_vault_sessions
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    session_row_id,
                )
            if (
                session is None
                or str(session["state"]) != "open"
                or str(session["stripe_checkout_session_id"]) != checkout_session_id
                or session["checkout_expires_at"] != checkout_expires_at
            ):
                raise EOMCardVaultConflictError(
                    "Stored Stripe Checkout session does not match"
                )
            return dict(session)
        except EOMCardVaultError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure("store_session", session_id=session_row_id)
            raise EOMCardVaultUnavailableError(
                "Stripe Checkout session could not be stored"
            ) from exc

    async def _materialize_session(
        self,
        *,
        provider: EOMCardVaultProvider,
        enrollment_id: UUID,
        session_row_id: UUID,
        reused: bool,
    ) -> dict[str, Any]:
        """Serialize provider materialization with the monotonic ready transition."""

        async with self.pool.transaction() as connection:
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"eom-card-vault:{enrollment_id}",
            )
            subject = await connection.fetchrow(
                """
                /* eom_card_vault_fenced_session */
                SELECT enrollment.*,
                       session.id AS session_row_id,
                       session.state AS session_state,
                       session.stripe_checkout_session_id,
                       session.checkout_expires_at,
                       session.checkout_success_url,
                       session.checkout_cancel_url,
                       session.provider_retry_until,
                       clock_timestamp() AS database_now
                FROM eom_card_vault_enrollments AS enrollment
                JOIN eom_card_vault_sessions AS session
                  ON session.enrollment_id = enrollment.id
                WHERE enrollment.id = $1
                  AND session.id = $2
                FOR UPDATE OF enrollment, session
                """,
                enrollment_id,
                session_row_id,
            )
            if subject is None:
                raise EOMCardVaultNotFoundError("Card-vault session was not found")
            enrollment = dict(subject)
            if str(enrollment["status"]) == "ready":
                return self._checkout_result(
                    enrollment=enrollment,
                    checkout_url=None,
                    checkout_expires_at=None,
                    idempotent=True,
                )
            if str(enrollment["status"]) != "pending":
                raise EOMCardVaultConflictError(
                    "Stored card-vault enrollment state is invalid"
                )
            customer_id = _provider_id(
                enrollment["stripe_customer_id"],
                prefix="cus_",
                label="customer",
            )
            checkout_success_url = _reserved_return_url(
                subject["checkout_success_url"], "success"
            )
            checkout_cancel_url = _reserved_return_url(
                subject["checkout_cancel_url"], "cancelled"
            )
            session_state = str(subject["session_state"])
            if session_state == "open":
                checkout_session_id = _provider_id(
                    subject["stripe_checkout_session_id"],
                    prefix="cs_",
                    label="Checkout session",
                )
                existing = await provider.retrieve_checkout_session(checkout_session_id)
                checkout_url = _value(existing, "url")
                checkout_metadata = _value(existing, "metadata")
                if (
                    _provider_id(
                        _value(existing, "id"),
                        prefix="cs_",
                        label="Checkout session",
                    )
                    != checkout_session_id
                    or _provider_id(
                        _value(existing, "customer"),
                        prefix="cus_",
                        label="customer",
                    )
                    != customer_id
                    or _value(existing, "mode") != "setup"
                    or _value(existing, "status") != "open"
                    or _value(existing, "client_reference_id") != str(enrollment_id)
                    or not isinstance(checkout_metadata, Mapping)
                    or checkout_metadata.get("source") != EOM_CARD_VAULT_SOURCE
                    or checkout_metadata.get("enrollment_id") != str(enrollment_id)
                    or checkout_metadata.get("session_id") != str(session_row_id)
                    or not isinstance(checkout_url, str)
                    or not checkout_url.startswith("https://")
                ):
                    raise EOMCardVaultConflictError(
                        "Stored Stripe Checkout session is no longer reusable"
                    )
                return self._checkout_result(
                    enrollment=enrollment,
                    checkout_url=checkout_url,
                    checkout_expires_at=subject["checkout_expires_at"],
                    idempotent=True,
                )
            if session_state != "creating":
                raise EOMCardVaultConflictError(
                    "Stored card-vault session state is invalid"
                )
            provider_retry_until = subject["provider_retry_until"]
            if not isinstance(provider_retry_until, datetime):
                raise EOMCardVaultConflictError(
                    "Stored Checkout retry deadline is invalid"
                )
            if provider_retry_until <= subject["database_now"]:
                raise EOMCardVaultUnavailableError(
                    "Card setup requires provider reconciliation before retry"
                )
            created = await provider.create_checkout_session(
                mode="setup",
                locale="en",
                payment_method_types=["card"],
                customer=customer_id,
                client_reference_id=str(enrollment_id),
                success_url=checkout_success_url,
                cancel_url=checkout_cancel_url,
                metadata={
                    "source": EOM_CARD_VAULT_SOURCE,
                    "enrollment_id": str(enrollment_id),
                    "session_id": str(session_row_id),
                },
                setup_intent_data={
                    "metadata": {
                        "source": EOM_CARD_VAULT_SOURCE,
                        "enrollment_id": str(enrollment_id),
                    }
                },
                idempotency_key=f"eom-card-vault-session:{session_row_id}",
            )
            checkout_session_id = _provider_id(
                _value(created, "id"), prefix="cs_", label="Checkout session"
            )
            checkout_url = _value(created, "url")
            checkout_expires_at = _expires_at(_value(created, "expires_at"))
            checkout_metadata = _value(created, "metadata")
            if (
                _provider_id(
                    _value(created, "customer"),
                    prefix="cus_",
                    label="customer",
                )
                != customer_id
                or _value(created, "mode") != "setup"
                or _value(created, "status") != "open"
                or _value(created, "client_reference_id") != str(enrollment_id)
                or not isinstance(checkout_metadata, Mapping)
                or checkout_metadata.get("source") != EOM_CARD_VAULT_SOURCE
                or checkout_metadata.get("enrollment_id") != str(enrollment_id)
                or checkout_metadata.get("session_id") != str(session_row_id)
                or not isinstance(checkout_url, str)
                or not checkout_url.startswith("https://")
                or checkout_expires_at <= subject["database_now"]
            ):
                raise EOMCardVaultConflictError("Stripe Checkout session is invalid")
            await self._store_session(
                connection=connection,
                session_row_id=session_row_id,
                enrollment_id=enrollment_id,
                checkout_session_id=checkout_session_id,
                checkout_expires_at=checkout_expires_at,
            )
            return self._checkout_result(
                enrollment=enrollment,
                checkout_url=checkout_url,
                checkout_expires_at=checkout_expires_at,
                idempotent=reused,
            )

    async def start_session(
        self,
        *,
        token: AuthenticatedEOMTermsToken,
        public_base_url: str,
    ) -> dict[str, Any]:
        """Create or reuse one hosted setup session after locked eligibility."""

        provider = self.provider
        authenticated = self._require_token(token)
        await self.require_schema_ready()
        try:
            checkout_success_url, checkout_cancel_url = (
                build_eom_card_vault_return_urls(public_base_url)
            )
        except ValueError as exc:
            raise EOMCardVaultValidationError(
                "Card setup return URL is invalid"
            ) from exc
        eligibility, enrollment, session_row, reused = await self._reserve_session(
            authenticated,
            checkout_success_url=checkout_success_url,
            checkout_cancel_url=checkout_cancel_url,
        )
        if str(enrollment["status"]) == "ready":
            return self._checkout_result(
                enrollment=enrollment,
                checkout_url=None,
                checkout_expires_at=None,
                idempotent=True,
            )
        assert session_row is not None
        try:
            customer_id = enrollment["stripe_customer_id"]
            if customer_id is None:
                customer_metadata = {
                    "source": EOM_CARD_VAULT_SOURCE,
                    "enrollment_id": str(enrollment["id"]),
                    "contact_id": str(enrollment["contact_id"]),
                    "candidate_id": str(enrollment["candidate_id"]),
                }
                customer = await provider.create_customer(
                    name=str(eligibility["invitation_customer_name"]),
                    email=str(eligibility["invitation_recipient_email"]),
                    metadata=customer_metadata,
                    idempotency_key=f"eom-card-vault-customer:{enrollment['id']}",
                )
                customer_id = _provider_id(
                    _value(customer, "id"), prefix="cus_", label="customer"
                )
                returned_customer_metadata = _value(customer, "metadata")
                if not isinstance(returned_customer_metadata, Mapping) or any(
                    returned_customer_metadata.get(key) != value
                    for key, value in customer_metadata.items()
                ):
                    raise EOMCardVaultConflictError(
                        "Stripe customer subject does not match"
                    )
                enrollment = await self._store_customer(
                    enrollment_id=UUID(str(enrollment["id"])),
                    customer_id=customer_id,
                )
            else:
                customer_id = _provider_id(customer_id, prefix="cus_", label="customer")
            return await self._materialize_session(
                provider=provider,
                enrollment_id=UUID(str(enrollment["id"])),
                session_row_id=UUID(str(session_row["id"])),
                reused=reused,
            )
        except EOMCardVaultError:
            raise
        except EOMCardVaultProviderError as exc:
            _log_boundary_failure(
                "create_or_reuse_session",
                enrollment_id=enrollment["id"],
                session_id=session_row["id"],
            )
            raise EOMCardVaultUnavailableError(
                "Stripe card setup is temporarily unavailable"
            ) from exc
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure(
                "materialize_session",
                enrollment_id=enrollment["id"],
                session_id=session_row["id"],
            )
            raise EOMCardVaultUnavailableError(
                "EOM card setup could not be finalized"
            ) from exc

    async def _existing_confirmation_result(
        self,
        *,
        event_id: str,
        enrollment_id: UUID,
        session_row_id: UUID,
        checkout_session_id: str,
        customer_id: str,
        setup_intent_id: str,
    ) -> dict[str, Any] | None:
        """Acknowledge only an exact durable replay without another provider read."""

        try:
            async with self.pool.transaction() as connection:
                existing = await connection.fetchrow(
                    """
                    /* eom_card_vault_existing_confirmation */
                    SELECT event.stripe_event_id,
                           event.enrollment_id,
                           event.session_id,
                           event.stripe_setup_intent_id,
                           event.stripe_payment_method_id,
                           enrollment.status AS enrollment_status,
                           enrollment.stripe_customer_id,
                           enrollment.stripe_setup_intent_id
                               AS enrollment_setup_intent_id,
                           enrollment.stripe_payment_method_id
                               AS enrollment_payment_method_id,
                           session.state AS session_state,
                           session.stripe_checkout_session_id
                    FROM eom_card_vault_events AS event
                    JOIN eom_card_vault_enrollments AS enrollment
                      ON enrollment.id = event.enrollment_id
                    JOIN eom_card_vault_sessions AS session
                      ON session.id = event.session_id
                    WHERE event.stripe_event_id = $1
                    LIMIT 1
                    """,
                    event_id,
                )
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure(
                "read_existing_confirmation",
                event_id=event_id,
                enrollment_id=enrollment_id,
                session_id=session_row_id,
            )
            raise EOMCardVaultUnavailableError(
                "EOM card-vault confirmation is unavailable"
            ) from exc
        if existing is None:
            return None
        if (
            str(existing["stripe_event_id"]) != event_id
            or UUID(str(existing["enrollment_id"])) != enrollment_id
            or UUID(str(existing["session_id"])) != session_row_id
            or str(existing["stripe_setup_intent_id"]) != setup_intent_id
            or str(existing["stripe_customer_id"]) != customer_id
            or str(existing["stripe_checkout_session_id"]) != checkout_session_id
            or str(existing["enrollment_status"]) != "ready"
            or str(existing["session_state"]) != "open"
            or str(existing["enrollment_setup_intent_id"]) != setup_intent_id
            or str(existing["enrollment_payment_method_id"])
            != str(existing["stripe_payment_method_id"])
        ):
            raise EOMCardVaultConflictError("Stripe event replay does not match")
        return {
            "eventId": event_id,
            "enrollmentId": str(enrollment_id),
            "status": "ready",
            "idempotent": True,
        }

    async def confirm_checkout_session(self, *, event: Any) -> dict[str, Any]:
        """Advance one enrollment only from a provider-confirmed SetupIntent."""

        provider = self.provider
        event_id = _provider_id(_value(event, "id"), prefix="evt_", label="event")
        event_type = _value(event, "type")
        if event_type != "checkout.session.completed":
            return {"eventId": event_id, "status": "ignored", "idempotent": True}
        data = _value(event, "data")
        session = _value(data, "object")
        metadata = _value(session, "metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("source") != EOM_CARD_VAULT_SOURCE
        ):
            return {"eventId": event_id, "status": "ignored", "idempotent": True}
        await self.require_schema_ready()
        enrollment_id = _uuid(metadata.get("enrollment_id"), "enrollmentId")
        session_row_id = _uuid(metadata.get("session_id"), "sessionId")
        checkout_session_id = _provider_id(
            _value(session, "id"), prefix="cs_", label="Checkout session"
        )
        customer_id = _provider_id(
            _value(session, "customer"), prefix="cus_", label="customer"
        )
        setup_intent_id = _provider_id(
            _value(session, "setup_intent"),
            prefix="seti_",
            label="SetupIntent",
        )
        if (
            _value(session, "mode") != "setup"
            or _value(session, "status") != "complete"
        ):
            raise EOMCardVaultConflictError(
                "Completed Checkout session has invalid state"
            )
        if _value(session, "client_reference_id") != str(enrollment_id):
            raise EOMCardVaultConflictError(
                "Completed Checkout session subject does not match"
            )
        existing_confirmation = await self._existing_confirmation_result(
            event_id=event_id,
            enrollment_id=enrollment_id,
            session_row_id=session_row_id,
            checkout_session_id=checkout_session_id,
            customer_id=customer_id,
            setup_intent_id=setup_intent_id,
        )
        if existing_confirmation is not None:
            return existing_confirmation
        try:
            setup_intent = await provider.retrieve_setup_intent(setup_intent_id)
        except EOMCardVaultProviderError as exc:
            _log_boundary_failure(
                "retrieve_setup_intent",
                event_id=event_id,
                enrollment_id=enrollment_id,
            )
            raise EOMCardVaultUnavailableError(
                "Stripe SetupIntent confirmation is unavailable"
            ) from exc
        setup_metadata = _value(setup_intent, "metadata")
        payment_method_id = _provider_id(
            _value(setup_intent, "payment_method"),
            prefix="pm_",
            label="payment method",
        )
        if (
            _provider_id(
                _value(setup_intent, "id"),
                prefix="seti_",
                label="SetupIntent",
            )
            != setup_intent_id
            or _provider_id(
                _value(setup_intent, "customer"),
                prefix="cus_",
                label="customer",
            )
            != customer_id
            or _value(setup_intent, "status") != "succeeded"
            or not isinstance(setup_metadata, Mapping)
            or setup_metadata.get("source") != EOM_CARD_VAULT_SOURCE
            or setup_metadata.get("enrollment_id") != str(enrollment_id)
        ):
            raise EOMCardVaultConflictError("Stripe SetupIntent does not match")
        try:
            async with self.pool.transaction() as connection:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"eom-card-vault:{enrollment_id}",
                )
                enrollment = await connection.fetchrow(
                    """
                    /* eom_card_vault_webhook_subject */
                    SELECT enrollment.*, session.id AS session_row_id
                    FROM eom_card_vault_enrollments AS enrollment
                    JOIN eom_card_vault_sessions AS session
                      ON session.enrollment_id = enrollment.id
                    WHERE enrollment.id = $1
                      AND session.id = $2
                      AND session.stripe_checkout_session_id = $3
                      AND session.state = 'open'
                    FOR UPDATE OF enrollment, session
                    """,
                    enrollment_id,
                    session_row_id,
                    checkout_session_id,
                )
                if enrollment is None:
                    raise EOMCardVaultNotFoundError(
                        "Card-vault enrollment was not found"
                    )
                if str(enrollment["stripe_customer_id"]) != customer_id:
                    raise EOMCardVaultConflictError("Stripe customer does not match")
                existing_event = await connection.fetchrow(
                    """
                    /* eom_card_vault_existing_event */
                    SELECT * FROM eom_card_vault_events
                    WHERE stripe_event_id = $1 OR session_id = $2
                    LIMIT 1
                    """,
                    event_id,
                    session_row_id,
                )
                if existing_event is not None:
                    if (
                        UUID(str(existing_event["enrollment_id"])) != enrollment_id
                        or UUID(str(existing_event["session_id"])) != session_row_id
                        or str(existing_event["stripe_setup_intent_id"])
                        != setup_intent_id
                        or str(existing_event["stripe_payment_method_id"])
                        != payment_method_id
                    ):
                        raise EOMCardVaultConflictError(
                            "Stripe event replay does not match"
                        )
                    return {
                        "eventId": event_id,
                        "enrollmentId": str(enrollment_id),
                        "status": "ready",
                        "idempotent": True,
                    }
                if str(enrollment["status"]) == "ready":
                    if (
                        str(enrollment["stripe_setup_intent_id"]) != setup_intent_id
                        or str(enrollment["stripe_payment_method_id"])
                        != payment_method_id
                    ):
                        raise EOMCardVaultConflictError(
                            "Ready enrollment cannot be replaced"
                        )
                    return {
                        "eventId": event_id,
                        "enrollmentId": str(enrollment_id),
                        "status": "ready",
                        "idempotent": True,
                    }
                await connection.execute(
                    """
                    /* eom_card_vault_record_event */
                    INSERT INTO eom_card_vault_events (
                        stripe_event_id, enrollment_id, session_id,
                        stripe_setup_intent_id, stripe_payment_method_id
                    ) VALUES ($1, $2, $3, $4, $5)
                    """,
                    event_id,
                    enrollment_id,
                    session_row_id,
                    setup_intent_id,
                    payment_method_id,
                )
                ready = await connection.fetchrow(
                    """
                    /* eom_card_vault_mark_ready */
                    UPDATE eom_card_vault_enrollments
                    SET status = 'ready',
                        stripe_setup_intent_id = $2,
                        stripe_payment_method_id = $3,
                        ready_at = clock_timestamp()
                    WHERE id = $1 AND status = 'pending'
                    RETURNING *
                    """,
                    enrollment_id,
                    setup_intent_id,
                    payment_method_id,
                )
                if ready is None:
                    raise EOMCardVaultUnavailableError(
                        "Card-vault readiness could not be stored"
                    )
                return {
                    "eventId": event_id,
                    "enrollmentId": str(enrollment_id),
                    "status": "ready",
                    "idempotent": False,
                }
        except EOMCardVaultError:
            raise
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure(
                "confirm_checkout_session",
                event_id=event_id,
                enrollment_id=enrollment_id,
                session_id=session_row_id,
            )
            raise EOMCardVaultUnavailableError(
                "EOM card-vault confirmation is unavailable"
            ) from exc

    async def get_readiness(self, *, contact_id: object) -> dict[str, Any]:
        """Return card-only readiness without changing onboarding state."""

        parsed_contact_id = _uuid(contact_id, "contactId")
        await self.require_schema_ready()
        try:
            row = await self.pool.fetchrow(
                """
                /* eom_card_vault_readiness */
                SELECT contact.id AS contact_id,
                       contact.customer_type,
                       contact.business_context_id,
                       contact.contact_type,
                       contact.status AS contact_status,
                       candidate.id AS candidate_id,
                       acceptance.id AS acceptance_id,
                       acceptance.audience AS acceptance_audience,
                       enrollment.id AS enrollment_id,
                       enrollment.status AS enrollment_status,
                       enrollment.ready_at,
                       EXISTS (
                           SELECT 1
                           FROM eom_terms_versions AS later
                           WHERE acceptance.id IS NOT NULL
                             AND later.status = 'published'
                             AND later.material_change
                             AND later.publication_order > accepted_version.publication_order
                       ) AS later_material
                FROM contacts AS contact
                LEFT JOIN eom_post_clean_onboarding_candidates AS candidate
                  ON candidate.contact_id = contact.id
                 AND candidate.status = 'pending'
                LEFT JOIN LATERAL (
                    SELECT stored.*
                    FROM eom_terms_acceptances AS stored
                    JOIN eom_terms_versions AS version ON version.id = stored.version_id
                    WHERE stored.contact_id = contact.id
                    ORDER BY (stored.audience = contact.customer_type) DESC,
                             version.publication_order DESC,
                             stored.id DESC
                    LIMIT 1
                ) AS acceptance ON TRUE
                LEFT JOIN eom_terms_versions AS accepted_version
                  ON accepted_version.id = acceptance.version_id
                LEFT JOIN eom_card_vault_enrollments AS enrollment
                  ON enrollment.candidate_id = candidate.id
                WHERE contact.id = $1
                """,
                parsed_contact_id,
            )
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log_boundary_failure("get_readiness", contact_id=parsed_contact_id)
            raise EOMCardVaultUnavailableError(
                "EOM card readiness is unavailable"
            ) from exc
        if (
            row is None
            or str(row["business_context_id"]) != "effingham_maids"
            or str(row["contact_type"]) != "customer"
            or str(row["contact_status"]) != "active"
        ):
            raise EOMCardVaultNotFoundError("Card readiness is unavailable")
        audience = str(row["customer_type"])
        if audience == "commercial":
            return {
                "contactId": str(parsed_contact_id),
                "audience": audience,
                "cardRequired": False,
                "cardReady": True,
                "reason": "not_required",
                "candidateId": None,
                "enrollmentId": None,
                "providerConfirmedAt": None,
            }
        if audience != "residential":
            raise EOMCardVaultNotFoundError("Card readiness is unavailable")
        if str(row["enrollment_status"]) == "ready":
            reason = "ready"
        elif (
            row["acceptance_id"] is None
            or str(row["acceptance_audience"]) != "residential"
            or bool(row["later_material"])
        ):
            reason = "terms_not_ready"
        elif row["candidate_id"] is None:
            reason = "first_clean_not_confirmed"
        elif row["enrollment_id"] is None:
            reason = "not_started"
        elif str(row["enrollment_status"]) == "pending":
            reason = "pending"
        else:
            raise EOMCardVaultUnavailableError("Stored card readiness is invalid")
        return {
            "contactId": str(parsed_contact_id),
            "audience": audience,
            "cardRequired": True,
            "cardReady": reason == "ready",
            "reason": reason,
            "candidateId": (
                str(row["candidate_id"]) if row["candidate_id"] is not None else None
            ),
            "enrollmentId": (
                str(row["enrollment_id"]) if row["enrollment_id"] is not None else None
            ),
            "providerConfirmedAt": row["ready_at"] if reason == "ready" else None,
        }
