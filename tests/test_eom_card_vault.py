"""Focused proof for provider-confirmed EOM card-on-file readiness."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import httpx
import pytest
import stripe
from fastapi import HTTPException
from pydantic import ValidationError

from atlas_brain import main_eom
from atlas_brain.eom_api import card_vault as api_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import (
    EOM_CARD_VAULT_RETURN_URL_MAX_BYTES,
    EOMFunnelConfig,
)
from atlas_brain.services.eom_card_vault import (
    EOM_CARD_VAULT_SOURCE,
    EOM_CARD_VAULT_STRIPE_API_VERSION,
    EOMCardVaultConflictError,
    EOMCardVaultNotFoundError,
    EOMCardVaultProviderError,
    EOMCardVaultService,
    EOMCardVaultSignatureError,
    EOMCardVaultUnavailableError,
    EOMCardVaultValidationError,
    StripeEOMCardVaultProvider,
    _provider_id,
    eom_card_vault_schema_ready,
)
from atlas_brain.services.eom_terms_acceptance import (
    AuthenticatedEOMTermsToken,
    authenticate_eom_terms_token,
    format_eom_terms_token,
)

_NOW = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
_INVITATION_ID = UUID("11111111-1111-4111-8111-111111111111")
_ACCEPTANCE_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONTACT_ID = UUID("33333333-3333-4333-8333-333333333333")
_CANDIDATE_ID = UUID("44444444-4444-4444-8444-444444444444")
_ENROLLMENT_ID = UUID("55555555-5555-4555-8555-555555555555")
_SESSION_ROW_ID = UUID("66666666-6666-4666-8666-666666666666")
_SECRET = "current-eom-public-key-material-1234567890"
_FINGERPRINT = "a" * 64
_SERVICE = auth_mod.generate_eom_funnel_service_token()
_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/398_eom_card_vault.sql"
)


def _config(**overrides: Any) -> EOMFunnelConfig:
    values: dict[str, Any] = {
        "api_enabled": True,
        "service_token_sha256": _SERVICE.sha256,
        "public_onboarding_enabled": True,
        "public_onboarding_url": "https://example.test/onboarding",
        "public_onboarding_hmac_secret": _SECRET,
        "card_vault_enabled": True,
        "card_vault_stripe_secret_key": "sk_" + "test_" + "placeholderxxxxxxxx",
        "card_vault_stripe_webhook_secret": "whsec_" + "placeholderxxxxxxxx",
        "_env_file": None,
    }
    values.update(overrides)
    return EOMFunnelConfig(**values)


class _State:
    def __init__(self) -> None:
        self.card_vault_schema_ready = True
        self.commitment_schema_ready = True
        self.lock = asyncio.Lock()
        self.eligibility: dict[str, Any] | None = {
            "signing_key_fingerprint": _FINGERPRINT,
            "revoked_at": None,
            "is_expired": False,
            "invitation_audience": "residential",
            "invitation_customer_name": "EOM Customer",
            "invitation_recipient_email": "customer@example.test",
            "acceptance_id": _ACCEPTANCE_ID,
            "acceptance_audience": "residential",
            "acceptance_recipient_email": "customer@example.test",
            "contact_id": _CONTACT_ID,
            "business_context_id": "effingham_maids",
            "contact_type": "customer",
            "contact_status": "active",
            "customer_type": "residential",
            "full_name": "EOM Customer",
            "email": "customer@example.test",
            "candidate_id": _CANDIDATE_ID,
            "candidate_status": "pending",
            "service_commitment": "recurring",
            "database_now": _NOW,
            "later_material": False,
        }
        self.enrollment: dict[str, Any] | None = None
        self.sessions: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.readiness: dict[str, Any] | None = None
        self.transaction_count = 0
        self.fail_customer_store_once = False
        self.fail_session_store_once = False
        self.fail_event_store_once = False


class _Connection:
    def __init__(self, state: _State) -> None:
        self.state = state

    async def execute(self, query: str, *args: Any) -> str:
        if "pg_advisory_xact_lock" in query:
            return "SELECT 1"
        if "eom_card_vault_record_event" in query:
            if self.state.fail_event_store_once:
                self.state.fail_event_store_once = False
                raise OSError("injected event-store failure")
            self.state.events.append(
                {
                    "stripe_event_id": args[0],
                    "enrollment_id": args[1],
                    "session_id": args[2],
                    "stripe_setup_intent_id": args[3],
                    "stripe_payment_method_id": args[4],
                }
            )
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute: {query}")

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        state = self.state
        if "eom_card_vault_eligibility" in query:
            return dict(state.eligibility) if state.eligibility is not None else None
        if "eom_card_vault_reserve_enrollment" in query:
            if state.enrollment is not None:
                return None
            state.enrollment = {
                "id": _ENROLLMENT_ID,
                "candidate_id": args[1],
                "contact_id": args[2],
                "initial_acceptance_id": args[3],
                "stripe_customer_id": None,
                "stripe_setup_intent_id": None,
                "stripe_payment_method_id": None,
                "status": "pending",
                "ready_at": None,
            }
            return dict(state.enrollment)
        if "eom_card_vault_lock_enrollment" in query:
            return dict(state.enrollment) if state.enrollment is not None else None
        if "eom_card_vault_store_customer" in query:
            assert state.enrollment is not None
            if state.fail_customer_store_once:
                state.fail_customer_store_once = False
                raise OSError("injected customer-store failure")
            if state.enrollment["stripe_customer_id"] is not None:
                return None
            state.enrollment["stripe_customer_id"] = args[1]
            return dict(state.enrollment)
        if "eom_card_vault_existing_customer" in query:
            return dict(state.enrollment) if state.enrollment is not None else None
        if "eom_card_vault_latest_session" in query:
            row = next(
                (
                    item
                    for item in reversed(state.sessions)
                    if item["acceptance_id"] == args[1]
                ),
                None,
            )
            return dict(row) if row is not None else None
        if "eom_card_vault_reserve_session" in query:
            row = {
                "id": _SESSION_ROW_ID if not state.sessions else args[0],
                "enrollment_id": args[1],
                "acceptance_id": args[2],
                "checkout_success_url": args[3],
                "checkout_cancel_url": args[4],
                "provider_retry_until": args[5],
                "state": "creating",
                "stripe_checkout_session_id": None,
                "checkout_expires_at": None,
                "created_at": _NOW,
            }
            state.sessions.append(row)
            return dict(row)
        if "eom_card_vault_fenced_contact" in query:
            if state.enrollment is None:
                return None
            eligibility = state.eligibility or {}
            session = next(
                (
                    item
                    for item in state.sessions
                    if item["id"] == args[1]
                    and item["enrollment_id"] == state.enrollment["id"]
                ),
                None,
            )
            if session is None:
                return None
            return {
                "contact_business_context_id": eligibility.get("business_context_id"),
                "materialization_contact_type": eligibility.get("contact_type"),
                "materialization_contact_status": eligibility.get("contact_status"),
                "materialization_customer_type": eligibility.get("customer_type"),
                "materialization_full_name": eligibility.get("full_name"),
                "materialization_email": eligibility.get("email"),
                "invitation_customer_name": eligibility.get("invitation_customer_name"),
                "invitation_recipient_email": eligibility.get(
                    "invitation_recipient_email"
                ),
            }
        if "eom_card_vault_fenced_session" in query:
            if state.enrollment is None:
                return None
            row = next(
                (
                    item
                    for item in state.sessions
                    if item["id"] == args[1]
                    and item["enrollment_id"] == state.enrollment["id"]
                ),
                None,
            )
            if row is None:
                return None
            return {
                **state.enrollment,
                "session_row_id": row["id"],
                "session_state": row["state"],
                "stripe_checkout_session_id": row["stripe_checkout_session_id"],
                "checkout_expires_at": row["checkout_expires_at"],
                "checkout_success_url": row["checkout_success_url"],
                "checkout_cancel_url": row["checkout_cancel_url"],
                "provider_retry_until": row["provider_retry_until"],
                "database_now": (state.eligibility or {})["database_now"],
            }
        if "eom_card_vault_store_session" in query:
            row = next(item for item in state.sessions if item["id"] == args[0])
            if state.fail_session_store_once:
                state.fail_session_store_once = False
                raise OSError("injected session-store failure")
            if row["state"] != "creating":
                return None
            row.update(
                state="open",
                stripe_checkout_session_id=args[1],
                checkout_expires_at=args[2],
            )
            return dict(row)
        if "eom_card_vault_existing_session" in query:
            row = next(
                (item for item in state.sessions if item["id"] == args[0]),
                None,
            )
            return dict(row) if row is not None else None
        if "eom_card_vault_existing_confirmation" in query:
            event = next(
                (item for item in state.events if item["stripe_event_id"] == args[0]),
                None,
            )
            if event is None or state.enrollment is None:
                return None
            session = next(
                item for item in state.sessions if item["id"] == event["session_id"]
            )
            return {
                **event,
                "enrollment_status": state.enrollment["status"],
                "stripe_customer_id": state.enrollment["stripe_customer_id"],
                "enrollment_setup_intent_id": state.enrollment[
                    "stripe_setup_intent_id"
                ],
                "enrollment_payment_method_id": state.enrollment[
                    "stripe_payment_method_id"
                ],
                "session_state": session["state"],
                "stripe_checkout_session_id": session["stripe_checkout_session_id"],
            }
        if "eom_card_vault_webhook_subject" in query:
            if state.enrollment is None:
                return None
            session = next(
                (
                    item
                    for item in state.sessions
                    if item["id"] == args[1]
                    and item["stripe_checkout_session_id"] == args[2]
                ),
                None,
            )
            if session is None or state.enrollment["id"] != args[0]:
                return None
            return {**state.enrollment, "session_row_id": session["id"]}
        if "eom_card_vault_existing_event" in query:
            row = next(
                (
                    item
                    for item in state.events
                    if item["stripe_event_id"] == args[0]
                    or item["session_id"] == args[1]
                ),
                None,
            )
            return dict(row) if row is not None else None
        if "eom_card_vault_mark_ready" in query:
            assert state.enrollment is not None
            if state.enrollment["status"] != "pending":
                return None
            state.enrollment.update(
                status="ready",
                stripe_setup_intent_id=args[1],
                stripe_payment_method_id=args[2],
                ready_at=_NOW,
            )
            return dict(state.enrollment)
        raise AssertionError(f"unexpected fetchrow: {query}")


class _Pool:
    is_initialized = True

    def __init__(self, state: _State) -> None:
        self.state = state
        self.connection = _Connection(state)

    @asynccontextmanager
    async def transaction(self):
        self.state.transaction_count += 1
        async with self.state.lock:
            yield self.connection

    async def fetchval(self, query: str, *_args: Any) -> bool:
        if "eom_card_service_commitment_schema_ready" in query:
            return self.state.commitment_schema_ready
        assert "eom_card_vault_schema_ready" in query
        return self.state.card_vault_schema_ready

    async def fetchrow(self, query: str, *_args: Any) -> dict[str, Any] | None:
        if "eom_card_vault_public_readiness" in query:
            assert _args == (_INVITATION_ID,)
            eligibility = self.state.eligibility
            if eligibility is None:
                return None
            enrollment = self.state.enrollment or {}
            return {
                "signing_key_fingerprint": eligibility.get("signing_key_fingerprint"),
                "revoked_at": eligibility.get("revoked_at"),
                "is_expired": eligibility.get("is_expired", False),
                "invitation_audience": eligibility.get("invitation_audience"),
                "invitation_customer_name": eligibility.get("invitation_customer_name"),
                "invitation_recipient_email": eligibility.get(
                    "invitation_recipient_email"
                ),
                "contact_id": eligibility.get("contact_id"),
                "customer_type": eligibility.get("customer_type"),
                "business_context_id": eligibility.get("business_context_id"),
                "contact_type": eligibility.get("contact_type"),
                "contact_status": eligibility.get("contact_status"),
                "full_name": eligibility.get("full_name"),
                "email": eligibility.get("email"),
                "candidate_id": eligibility.get("candidate_id"),
                "service_commitment": eligibility.get("service_commitment"),
                "acceptance_id": eligibility.get("acceptance_id"),
                "acceptance_audience": eligibility.get("acceptance_audience"),
                "enrollment_id": enrollment.get("id"),
                "enrollment_status": enrollment.get("status"),
                "ready_at": enrollment.get("ready_at"),
                "later_material": eligibility.get("later_material", False),
            }
        assert "eom_card_vault_readiness" in query
        if self.state.readiness is not None:
            return dict(self.state.readiness)
        enrollment = self.state.enrollment or {}
        eligibility = self.state.eligibility or {}
        return {
            "contact_id": _CONTACT_ID,
            "customer_type": eligibility.get("customer_type", "residential"),
            "business_context_id": "effingham_maids",
            "contact_type": "customer",
            "contact_status": "active",
            "candidate_id": eligibility.get("candidate_id"),
            "service_commitment": eligibility.get("service_commitment"),
            "acceptance_id": eligibility.get("acceptance_id"),
            "acceptance_audience": eligibility.get("acceptance_audience"),
            "enrollment_id": enrollment.get("id"),
            "enrollment_status": enrollment.get("status"),
            "ready_at": enrollment.get("ready_at"),
            "later_material": eligibility.get("later_material", False),
        }


class _Provider:
    def __init__(self) -> None:
        self.customer_calls: list[dict[str, Any]] = []
        self.session_calls: list[dict[str, Any]] = []
        self.retrieve_session_calls: list[str] = []
        self.retrieve_setup_intent_calls: list[str] = []

    async def create_customer(self, **kwargs: Any) -> dict[str, Any]:
        self.customer_calls.append(kwargs)
        await asyncio.sleep(0)
        return {"id": "cus_cardvault123", "metadata": kwargs["metadata"]}

    async def create_checkout_session(self, **kwargs: Any) -> dict[str, Any]:
        self.session_calls.append(kwargs)
        await asyncio.sleep(0)
        return {
            "id": "cs_test_cardvault123",
            "customer": "cus_cardvault123",
            "mode": "setup",
            "status": "open",
            "url": "https://checkout.stripe.test/session",
            "expires_at": int((_NOW + timedelta(days=1)).timestamp()),
            "client_reference_id": kwargs["client_reference_id"],
            "metadata": kwargs["metadata"],
        }

    async def retrieve_checkout_session(self, session_id: str) -> dict[str, Any]:
        self.retrieve_session_calls.append(session_id)
        created = self.session_calls[-1]
        return {
            "id": session_id,
            "customer": "cus_cardvault123",
            "mode": "setup",
            "status": "open",
            "url": "https://checkout.stripe.test/session",
            "client_reference_id": created["client_reference_id"],
            "metadata": created["metadata"],
        }

    async def retrieve_setup_intent(self, setup_intent_id: str) -> dict[str, Any]:
        self.retrieve_setup_intent_calls.append(setup_intent_id)
        return {
            "id": setup_intent_id,
            "customer": "cus_cardvault123",
            "status": "succeeded",
            "payment_method": "pm_cardvault123",
            "metadata": {
                "source": EOM_CARD_VAULT_SOURCE,
                "enrollment_id": str(_ENROLLMENT_ID),
            },
        }

    def construct_event(self, payload: bytes, signature: str) -> dict[str, Any]:
        if signature != "valid-signature":
            raise EOMCardVaultSignatureError("invalid")
        return {"id": "evt_ignored123", "type": "customer.created"}


class _SessionFailsOnceProvider(_Provider):
    async def create_checkout_session(self, **kwargs: Any) -> dict[str, Any]:
        if not self.session_calls:
            self.session_calls.append(kwargs)
            raise EOMCardVaultProviderError("injected provider failure")
        return await super().create_checkout_session(**kwargs)


def _token() -> AuthenticatedEOMTermsToken:
    return AuthenticatedEOMTermsToken(
        invitation_id=_INVITATION_ID,
        signing_key_fingerprint=_FINGERPRINT,
    )


def _completed_event() -> dict[str, Any]:
    return {
        "id": "evt_cardvault123",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_cardvault123",
                "customer": "cus_cardvault123",
                "setup_intent": "seti_cardvault123",
                "mode": "setup",
                "status": "complete",
                "client_reference_id": str(_ENROLLMENT_ID),
                "metadata": {
                    "source": EOM_CARD_VAULT_SOURCE,
                    "enrollment_id": str(_ENROLLMENT_ID),
                    "session_id": str(_SESSION_ROW_ID),
                },
            }
        },
    }


@pytest.mark.asyncio
async def test_schema_attestation_pins_trigger_and_constraint_definitions() -> None:
    class SchemaPool:
        async def fetchval(self, query: str) -> bool:
            assert "trigger.tgtype = expected.trigger_type" in query
            assert "trigger.tgattr = ''::int2vector" in query
            assert "trigger.tgqual IS NULL" in query
            assert "md5(function.prosrc) = expected.function_body_md5" in query
            assert "language.lanname = 'plpgsql'" in query
            assert "pg_get_constraintdef(actual.oid, true)" in query
            assert "format_type(" in query
            assert "column_definition.attnotnull = expected.not_null" in query
            assert "pg_get_expr(" in query
            assert "eom_card_vault_events', 'received_at'" in query
            assert "uq_eom_card_vault_enrollment_candidate" in query
            assert "UNIQUE (candidate_id)" in query
            assert "ck_eom_card_vault_session_retry_window" in query
            assert "ck_eom_card_vault_enrollment_state" in query
            return True

    assert await eom_card_vault_schema_ready(SchemaPool()) is True


def test_card_vault_defaults_disabled_and_requires_a_complete_authority() -> None:
    default = EOMFunnelConfig(_env_file=None)
    assert default.card_vault_enabled is False
    assert default.card_vault_stripe_secret_key.get_secret_value() == ""
    assert default.card_vault_stripe_webhook_secret.get_secret_value() == ""

    with pytest.raises(ValidationError, match="must be set together"):
        _config(
            card_vault_enabled=False,
            card_vault_stripe_webhook_secret="",
        )
    with pytest.raises(ValidationError, match="PUBLIC_ONBOARDING_ENABLED"):
        _config(public_onboarding_enabled=False)
    configured = _config()
    assert configured.card_vault_request_timeout_seconds == 10
    assert auth_mod.require_eom_card_vault_config(
        config=configured
    ).public_base_url == ("https://example.test/onboarding")
    paused = _config(card_vault_enabled=False)
    provider_config = auth_mod.require_eom_card_vault_provider_config(config=paused)
    assert provider_config.request_timeout_seconds == 10
    assert provider_config.secret_key.startswith("sk_test_")
    with pytest.raises(HTTPException) as issuance_paused:
        auth_mod.require_eom_card_vault_config(config=paused)
    assert issuance_paused.value.status_code == 503
    disabled = _config(
        card_vault_enabled=False,
        card_vault_stripe_secret_key="",
        card_vault_stripe_webhook_secret="",
    )
    with pytest.raises(HTTPException) as unavailable:
        auth_mod.require_eom_card_vault_config(config=disabled)
    assert unavailable.value.status_code == 503


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "card_vault_enabled": False,
                "card_vault_stripe_secret_key": "",
            },
            "must be set together",
        ),
        (
            {
                "card_vault_stripe_secret_key": "",
                "card_vault_stripe_webhook_secret": "",
            },
            "requires dedicated Stripe",
        ),
        (
            {"card_vault_stripe_secret_key": "pk_test_not_server_authority"},
            "secret key is invalid",
        ),
        (
            {"card_vault_stripe_webhook_secret": "secret_not_a_webhook_key"},
            "webhook secret is invalid",
        ),
        ({"card_vault_request_timeout_seconds": 0}, "greater than or equal to 1"),
        ({"card_vault_request_timeout_seconds": 31}, "less than or equal to 30"),
    ],
)
def test_card_vault_configuration_rejects_partial_and_boundary_values(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**overrides)


@pytest.mark.parametrize("timeout", [1, 30])
def test_card_vault_configuration_accepts_timeout_boundaries(timeout: int) -> None:
    assert (
        _config(
            card_vault_request_timeout_seconds=timeout
        ).card_vault_request_timeout_seconds
        == timeout
    )


def test_card_vault_configuration_pins_derived_return_url_byte_boundary() -> None:
    prefix = "https://example.test/"
    longest_suffix = "?cardVault=cancelled"
    path_bytes = (
        EOM_CARD_VAULT_RETURN_URL_MAX_BYTES
        - len(prefix.encode("utf-8"))
        - len(longest_suffix.encode("utf-8"))
    )
    exact_base_url = prefix + ("a" * path_bytes)

    admitted = _config(public_onboarding_url=exact_base_url)
    assert (
        auth_mod.require_eom_card_vault_config(config=admitted).public_base_url
        == exact_base_url
    )
    with pytest.raises(
        ValidationError,
        match="card-vault return URLs must not exceed 2048 bytes",
    ):
        _config(public_onboarding_url=exact_base_url + "a")
    state = _State()
    provider = _Provider()
    with pytest.raises(EOMCardVaultValidationError):
        asyncio.run(
            EOMCardVaultService(pool=_Pool(state), provider=provider).start_session(
                token=_token(),
                public_base_url=exact_base_url + "a",
            )
        )
    assert state.enrollment is None
    assert provider.customer_calls == []
    assert provider.session_calls == []


def test_card_vault_config_grammar_is_closed_across_token_and_container_families() -> (
    None
):
    """Spec-derived oracle: only an ASCII identifier in its exact family is safe."""

    families = (
        ("card_vault_stripe_secret_key", "sk_test_"),
        ("card_vault_stripe_secret_key", "sk_live_"),
        ("card_vault_stripe_secret_key", "rk_test_"),
        ("card_vault_stripe_secret_key", "rk_live_"),
        ("card_vault_stripe_webhook_secret", "whsec_"),
    )
    tokens = (
        ("valid", lambda prefix: f"{prefix}AbC_123456789"),
        ("wrong-family", lambda _prefix: "pk_test_AbC_123456789"),
        ("too-short", lambda prefix: f"{prefix}x"),
        (
            "non-ascii",
            lambda prefix: f"{prefix}AbC_12345678e\N{LATIN SMALL LETTER E WITH ACUTE}",
        ),
        ("punctuation", lambda prefix: f"{prefix}AbC-123456789"),
    )
    containers = (
        ("scalar", lambda token: token),
        ("list", lambda token: [token]),
        ("mapping", lambda token: {"value": token}),
    )

    for (field, prefix), (token_class, make_token), (
        container_class,
        wrap,
    ) in product(families, tokens, containers):
        candidate = wrap(make_token(prefix))
        expected = token_class == "valid" and container_class == "scalar"
        if expected:
            configured = _config(**{field: candidate})
            actual = getattr(configured, field).get_secret_value()
            assert actual == candidate
        else:
            with pytest.raises(ValidationError):
                _config(**{field: candidate})


def test_provider_id_grammar_is_closed_across_token_and_container_families() -> None:
    """Spec-derived oracle: family match plus safe representation is mandatory."""

    families = (
        ("cus_", "customer"),
        ("cs_", "Checkout session"),
        ("seti_", "SetupIntent"),
        ("pm_", "payment method"),
        ("evt_", "event"),
    )
    tokens = (
        ("valid", lambda prefix: f"{prefix}AbC_123"),
        ("prefix-only", lambda prefix: prefix),
        ("wrong-family", lambda _prefix: "bad_AbC_123"),
        ("punctuation", lambda prefix: f"{prefix}AbC-123"),
        ("too-long", lambda prefix: prefix + ("A" * 256)),
    )
    containers = (
        ("scalar", lambda token: token),
        ("mapping", lambda token: {"id": token}),
        ("object", lambda token: SimpleNamespace(id=token)),
        ("nested", lambda token: {"id": {"id": token}}),
        ("sequence", lambda token: [token]),
    )
    admitted_containers = frozenset({"scalar", "mapping", "object"})

    for (prefix, label), (token_class, make_token), (
        container_class,
        wrap,
    ) in product(families, tokens, containers):
        candidate = wrap(make_token(prefix))
        expected = token_class == "valid" and container_class in admitted_containers
        if expected:
            assert _provider_id(candidate, prefix=prefix, label=label) == make_token(
                prefix
            )
        else:
            with pytest.raises(EOMCardVaultConflictError):
                _provider_id(candidate, prefix=prefix, label=label)


@pytest.mark.asyncio
async def test_session_creation_is_setup_only_and_concurrency_idempotent() -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    first, second = await asyncio.gather(
        service.start_session(
            token=_token(), public_base_url="https://eom.test/onboard"
        ),
        service.start_session(
            token=_token(), public_base_url="https://eom.test/onboard"
        ),
    )
    readiness = await service.get_readiness(contact_id=_CONTACT_ID)

    assert {first["idempotent"], second["idempotent"]} == {False, True}
    assert readiness["cardRequired"] is True
    assert readiness["cardReady"] is False
    assert readiness["reason"] == "pending"
    assert len(provider.customer_calls) == 1
    assert {call["idempotency_key"] for call in provider.customer_calls} == {
        f"eom-card-vault-customer:{_ENROLLMENT_ID}"
    }
    assert len(provider.session_calls) == 1
    assert {call["idempotency_key"] for call in provider.session_calls} == {
        f"eom-card-vault-session:{_SESSION_ROW_ID}"
    }
    assert provider.retrieve_session_calls == ["cs_test_cardvault123"]
    call = provider.session_calls[0]
    assert call["mode"] == "setup"
    assert call["locale"] == "en"
    assert call["payment_method_types"] == ["card"]
    assert "payment_intent_data" not in call
    assert call["success_url"] == "https://eom.test/onboard?cardVault=success"
    assert call["cancel_url"] == "https://eom.test/onboard?cardVault=cancelled"
    assert call["metadata"]["enrollment_id"] == str(_ENROLLMENT_ID)
    assert call["setup_intent_data"]["metadata"]["source"] == EOM_CARD_VAULT_SOURCE


@pytest.mark.asyncio
async def test_ready_transition_wins_before_new_session_materialization() -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")
    state.sessions[0]["checkout_expires_at"] = _NOW - timedelta(seconds=1)
    original_reserve = service._reserve_session
    new_session_reserved = asyncio.Event()
    release_start = asyncio.Event()

    async def reserve_then_pause(*args: Any, **kwargs: Any):
        result = await original_reserve(*args, **kwargs)
        if len(state.sessions) == 2:
            new_session_reserved.set()
            await release_start.wait()
        return result

    service._reserve_session = reserve_then_pause  # type: ignore[method-assign]
    start_task = asyncio.create_task(
        service.start_session(token=_token(), public_base_url="https://eom.test")
    )
    await new_session_reserved.wait()
    try:
        confirmed = await service.confirm_checkout_session(event=_completed_event())
    finally:
        release_start.set()
    result = await start_task

    assert confirmed["status"] == "ready"
    assert result["status"] == "ready"
    assert result["checkoutUrl"] is None
    assert len(provider.session_calls) == 1
    assert state.sessions[1]["state"] == "creating"


@pytest.mark.asyncio
async def test_provider_failure_keeps_stable_retry_identities(caplog) -> None:
    state = _State()
    provider = _SessionFailsOnceProvider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with caplog.at_level(logging.ERROR, logger="atlas_brain.services.eom_card_vault"):
        with pytest.raises(EOMCardVaultUnavailableError):
            await service.start_session(
                token=_token(), public_base_url="https://eom.test"
            )
    result = await service.start_session(
        token=_token(), public_base_url="https://eom.test"
    )

    assert result["status"] == "pending"
    assert len(provider.customer_calls) == 1
    assert len(provider.session_calls) == 2
    assert (
        provider.session_calls[0]["idempotency_key"]
        == provider.session_calls[1]["idempotency_key"]
    )
    assert len(state.sessions) == 1
    assert state.sessions[0]["state"] == "open"
    record = next(
        item
        for item in caplog.records
        if getattr(item, "eom_card_vault_operation", None) == "create_or_reuse_session"
    )
    assert record.eom_card_vault_context == {
        "enrollment_id": str(_ENROLLMENT_ID),
        "session_id": str(_SESSION_ROW_ID),
    }
    assert _SECRET not in caplog.text


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("source", "another_source"),
        ("enrollment_id", "77777777-7777-4777-8777-777777777777"),
        ("contact_id", "77777777-7777-4777-8777-777777777777"),
        ("candidate_id", "77777777-7777-4777-8777-777777777777"),
        (None, None),
    ],
)
@pytest.mark.asyncio
async def test_created_customer_requires_exact_provider_subject_metadata(
    field: str | None,
    wrong_value: str | None,
) -> None:
    state = _State()
    provider = _Provider()
    original_create_customer = provider.create_customer

    async def create_mismatched_customer(**kwargs: Any) -> dict[str, Any]:
        customer = await original_create_customer(**kwargs)
        if field is None:
            customer["metadata"] = []
        else:
            customer["metadata"] = {
                **customer["metadata"],
                field: wrong_value,
            }
        return customer

    provider.create_customer = create_mismatched_customer  # type: ignore[method-assign]

    with pytest.raises(
        EOMCardVaultConflictError,
        match="Stripe customer subject does not match",
    ):
        await EOMCardVaultService(pool=_Pool(state), provider=provider).start_session(
            token=_token(), public_base_url="https://eom.test"
        )

    assert state.enrollment is not None
    assert state.enrollment["stripe_customer_id"] is None
    assert provider.session_calls == []


@pytest.mark.asyncio
async def test_reused_session_requires_exact_provider_subject() -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")
    reused = await service.start_session(
        token=_token(), public_base_url="https://eom.test"
    )
    assert reused["idempotent"] is True
    assert reused["checkoutUrl"] == "https://checkout.stripe.test/session"
    original_retrieve = provider.retrieve_checkout_session

    async def mismatched(session_id: str) -> dict[str, Any]:
        result = await original_retrieve(session_id)
        result["metadata"] = {
            **result["metadata"],
            "enrollment_id": str(UUID("77777777-7777-4777-8777-777777777777")),
        }
        return result

    provider.retrieve_checkout_session = mismatched  # type: ignore[method-assign]
    with pytest.raises(EOMCardVaultConflictError):
        await service.start_session(token=_token(), public_base_url="https://eom.test")


@pytest.mark.parametrize("failure_point", ["customer", "session"])
@pytest.mark.asyncio
async def test_database_failure_keeps_stable_provider_identities(
    failure_point: str,
    caplog,
) -> None:
    state = _State()
    setattr(state, f"fail_{failure_point}_store_once", True)
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with caplog.at_level(logging.ERROR, logger="atlas_brain.services.eom_card_vault"):
        with pytest.raises(EOMCardVaultUnavailableError):
            await service.start_session(
                token=_token(), public_base_url="https://eom.test"
            )
    if failure_point == "customer":
        assert state.eligibility is not None
        state.eligibility["full_name"] = "  EOM Customer  "
        state.eligibility["email"] = " CUSTOMER@EXAMPLE.TEST "
    result = await service.start_session(
        token=_token(), public_base_url="https://eom.test"
    )

    assert result["status"] == "pending"
    assert {call["idempotency_key"] for call in provider.customer_calls} == {
        f"eom-card-vault-customer:{_ENROLLMENT_ID}"
    }
    assert {call["idempotency_key"] for call in provider.session_calls} == {
        f"eom-card-vault-session:{_SESSION_ROW_ID}"
    }
    expected_customer_calls = 2 if failure_point == "customer" else 1
    expected_session_calls = 1 if failure_point == "customer" else 2
    assert len(provider.customer_calls) == expected_customer_calls
    assert len(provider.session_calls) == expected_session_calls
    assert {call["name"] for call in provider.customer_calls} == {"EOM Customer"}
    assert {call["email"] for call in provider.customer_calls} == {
        "customer@example.test"
    }
    expected_operation = (
        "store_customer" if failure_point == "customer" else "store_session"
    )
    record = next(
        item
        for item in caplog.records
        if getattr(item, "eom_card_vault_operation", None) == expected_operation
    )
    expected_context_key = (
        "enrollment_id" if failure_point == "customer" else "session_id"
    )
    expected_context_value = (
        str(_ENROLLMENT_ID) if failure_point == "customer" else str(_SESSION_ROW_ID)
    )
    assert record.eom_card_vault_context == {
        expected_context_key: expected_context_value
    }
    assert _SECRET not in caplog.text


@pytest.mark.asyncio
async def test_session_store_retry_reuses_reserved_urls_and_key_after_config_drift() -> (
    None
):
    state = _State()
    state.fail_session_store_once = True
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with pytest.raises(EOMCardVaultUnavailableError):
        await service.start_session(
            token=_token(), public_base_url="https://original.eom.test/onboarding"
        )
    result = await service.start_session(
        token=_token(), public_base_url="https://moved.eom.test/new-onboarding"
    )

    assert result["status"] == "pending"
    assert len(provider.session_calls) == 2
    assert {call["idempotency_key"] for call in provider.session_calls} == {
        f"eom-card-vault-session:{_SESSION_ROW_ID}"
    }
    assert {call["success_url"] for call in provider.session_calls} == {
        "https://original.eom.test/onboarding?cardVault=success"
    }
    assert {call["cancel_url"] for call in provider.session_calls} == {
        "https://original.eom.test/onboarding?cardVault=cancelled"
    }


@pytest.mark.asyncio
async def test_creating_session_retry_stops_at_reserved_deadline() -> None:
    state = _State()
    state.fail_session_store_once = True
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with pytest.raises(EOMCardVaultUnavailableError):
        await service.start_session(token=_token(), public_base_url="https://eom.test")
    assert state.eligibility is not None
    state.eligibility["database_now"] = _NOW + timedelta(hours=1)

    with pytest.raises(
        EOMCardVaultUnavailableError,
        match="requires provider reconciliation",
    ):
        await service.start_session(token=_token(), public_base_url="https://eom.test")

    assert len(provider.session_calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("business_context_id", "other"),
        ("contact_type", "lead"),
        ("contact_status", "archived"),
        ("customer_type", "commercial"),
        ("full_name", "Another Customer"),
        ("email", "other@example.test"),
        ("email", None),
    ],
)
@pytest.mark.asyncio
async def test_final_materialization_revalidates_mutable_contact_before_provider(
    field: str,
    value: Any,
) -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    _, enrollment, session, reused = await service._reserve_session(
        _token(),
        checkout_success_url="https://eom.test?cardVault=success",
        checkout_cancel_url="https://eom.test?cardVault=cancelled",
    )
    assert session is not None
    assert state.eligibility is not None
    state.eligibility[field] = value

    with pytest.raises(EOMCardVaultNotFoundError):
        await service._materialize_session(
            provider=provider,
            enrollment_id=UUID(str(enrollment["id"])),
            session_row_id=UUID(str(session["id"])),
            reused=reused,
        )

    assert provider.customer_calls == []
    assert provider.session_calls == []


@pytest.mark.asyncio
async def test_expired_customer_retry_stops_before_any_second_provider_write() -> None:
    state = _State()
    state.fail_customer_store_once = True
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with pytest.raises(EOMCardVaultUnavailableError):
        await service.start_session(token=_token(), public_base_url="https://eom.test")
    assert state.eligibility is not None
    state.eligibility["database_now"] = _NOW + timedelta(hours=1)

    with pytest.raises(
        EOMCardVaultUnavailableError,
        match="requires provider reconciliation",
    ):
        await service.start_session(token=_token(), public_base_url="https://eom.test")

    assert len(provider.customer_calls) == 1
    assert provider.session_calls == []


@pytest.mark.asyncio
async def test_corrupt_reserved_return_url_stops_before_provider_retry() -> None:
    state = _State()
    state.fail_session_store_once = True
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with pytest.raises(EOMCardVaultUnavailableError):
        await service.start_session(token=_token(), public_base_url="https://eom.test")
    state.sessions[0]["checkout_success_url"] = (
        "https://eom.test?cardVault=success&redirect=https://attacker.test"
    )

    with pytest.raises(EOMCardVaultConflictError, match="return URL is invalid"):
        await service.start_session(token=_token(), public_base_url="https://eom.test")

    assert len(provider.session_calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revoked_at", _NOW),
        ("is_expired", True),
        ("later_material", True),
        ("business_context_id", "other"),
        ("contact_type", "lead"),
        ("contact_status", "archived"),
        ("customer_type", "commercial"),
        ("invitation_audience", "commercial"),
        ("acceptance_audience", "commercial"),
        ("full_name", "Another Customer"),
        ("candidate_id", None),
        ("candidate_status", "blocked"),
        ("service_commitment", None),
        ("service_commitment", "one_time"),
        ("email", None),
        ("invitation_recipient_email", "other@example.test"),
        ("acceptance_recipient_email", "other@example.test"),
        ("signing_key_fingerprint", "b" * 64),
    ],
)
@pytest.mark.asyncio
async def test_session_creation_rejects_ineligible_subject_before_stripe(
    field: str,
    value: Any,
) -> None:
    state = _State()
    assert state.eligibility is not None
    state.eligibility[field] = value
    provider = _Provider()

    with pytest.raises(EOMCardVaultNotFoundError):
        await EOMCardVaultService(pool=_Pool(state), provider=provider).start_session(
            token=_token(), public_base_url="https://eom.test/onboard"
        )

    assert provider.customer_calls == []
    assert provider.session_calls == []


@pytest.mark.asyncio
async def test_session_creation_rejects_missing_schema_and_untyped_token_before_stripe() -> (
    None
):
    state = _State()
    state.card_vault_schema_ready = False
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    with pytest.raises(EOMCardVaultNotFoundError):
        await service.start_session(  # type: ignore[arg-type]
            token=object(),
            public_base_url="https://eom.test/onboard",
        )
    with pytest.raises(EOMCardVaultUnavailableError):
        await service.start_session(
            token=_token(),
            public_base_url="https://eom.test/onboard",
        )

    assert provider.customer_calls == []
    assert provider.session_calls == []


@pytest.mark.asyncio
async def test_verified_completed_event_is_monotonic_and_replay_safe() -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")

    first = await service.confirm_checkout_session(event=_completed_event())

    async def unavailable(_setup_intent_id: str) -> dict[str, Any]:
        raise EOMCardVaultProviderError("injected provider failure")

    provider.retrieve_setup_intent = unavailable  # type: ignore[method-assign]
    second = await service.confirm_checkout_session(event=_completed_event())

    assert first == {
        "eventId": "evt_cardvault123",
        "enrollmentId": str(_ENROLLMENT_ID),
        "status": "ready",
        "idempotent": False,
    }
    assert second["idempotent"] is True
    assert state.enrollment is not None
    assert state.enrollment["status"] == "ready"
    assert state.enrollment["stripe_payment_method_id"] == "pm_cardvault123"
    assert len(state.events) == 1
    assert provider.retrieve_setup_intent_calls == ["seti_cardvault123"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("customer", "cus_someoneelse"),
        ("setup_intent", "seti_someoneelse"),
    ],
)
@pytest.mark.asyncio
async def test_stored_event_replay_requires_exact_signed_subject(
    field: str,
    value: str,
) -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")
    await service.confirm_checkout_session(event=_completed_event())
    replay = _completed_event()
    replay["data"]["object"][field] = value

    with pytest.raises(
        EOMCardVaultConflictError,
        match="Stripe event replay does not match",
    ):
        await service.confirm_checkout_session(event=replay)

    assert provider.retrieve_setup_intent_calls == ["seti_cardvault123"]


@pytest.mark.asyncio
async def test_confirmation_database_failure_logs_nonsecret_subject_context(
    caplog,
) -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")
    state.fail_event_store_once = True

    with caplog.at_level(logging.ERROR, logger="atlas_brain.services.eom_card_vault"):
        with pytest.raises(EOMCardVaultUnavailableError):
            await service.confirm_checkout_session(event=_completed_event())

    record = next(
        item
        for item in caplog.records
        if getattr(item, "eom_card_vault_operation", None) == "confirm_checkout_session"
    )
    assert record.eom_card_vault_context == {
        "event_id": "evt_cardvault123",
        "enrollment_id": str(_ENROLLMENT_ID),
        "session_id": str(_SESSION_ROW_ID),
    }
    assert _SECRET not in caplog.text
    assert "cus_cardvault123" not in caplog.text
    assert "pm_cardvault123" not in caplog.text
    assert state.events == []


@pytest.mark.asyncio
async def test_setup_intent_mismatch_cannot_mark_ready() -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")

    async def mismatched(_setup_intent_id: str) -> dict[str, Any]:
        return {
            "id": "seti_cardvault123",
            "customer": "cus_someoneelse",
            "status": "succeeded",
            "payment_method": "pm_cardvault123",
            "metadata": {
                "source": EOM_CARD_VAULT_SOURCE,
                "enrollment_id": str(_ENROLLMENT_ID),
            },
        }

    provider.retrieve_setup_intent = mismatched  # type: ignore[method-assign]
    with pytest.raises(EOMCardVaultConflictError):
        await service.confirm_checkout_session(event=_completed_event())
    assert state.enrollment is not None
    assert state.enrollment["status"] == "pending"
    assert state.events == []


@pytest.mark.asyncio
async def test_webhook_schema_gate_ignores_unrelated_events_but_precedes_provider_reads() -> (
    None
):
    state = _State()
    state.card_vault_schema_ready = False
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)

    ignored = await service.confirm_checkout_session(
        event={"id": "evt_unrelated123", "type": "customer.created"}
    )
    with pytest.raises(EOMCardVaultUnavailableError):
        await service.confirm_checkout_session(event=_completed_event())

    assert ignored == {
        "eventId": "evt_unrelated123",
        "status": "ignored",
        "idempotent": True,
    }
    assert provider.retrieve_setup_intent_calls == []


@pytest.mark.asyncio
async def test_commitment_schema_drift_blocks_issuance_but_not_open_confirmation() -> (
    None
):
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")

    state.commitment_schema_ready = False
    with pytest.raises(EOMCardVaultUnavailableError):
        await service.start_session(token=_token(), public_base_url="https://eom.test")
    confirmed = await service.confirm_checkout_session(event=_completed_event())

    assert confirmed["status"] == "ready"
    assert confirmed["idempotent"] is False
    assert provider.retrieve_setup_intent_calls == ["seti_cardvault123"]


@pytest.mark.asyncio
async def test_readiness_keeps_commercial_outside_the_card_requirement() -> None:
    state = _State()
    state.readiness = {
        "contact_id": _CONTACT_ID,
        "customer_type": "commercial",
        "business_context_id": "effingham_maids",
        "contact_type": "customer",
        "contact_status": "active",
        "candidate_id": None,
        "acceptance_id": None,
        "acceptance_audience": None,
        "enrollment_id": None,
        "enrollment_status": None,
        "ready_at": None,
        "later_material": False,
    }
    result = await EOMCardVaultService(
        pool=_Pool(state), provider=_Provider()
    ).get_readiness(contact_id=_CONTACT_ID)
    assert result["cardRequired"] is False
    assert result["cardReady"] is True
    assert result["reason"] == "not_required"


@pytest.mark.asyncio
async def test_one_time_readiness_skips_card_and_unclassified_fails_closed() -> None:
    one_time = _State()
    assert one_time.eligibility is not None
    one_time.eligibility["service_commitment"] = "one_time"
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(one_time), provider=provider)

    readiness = await service.get_readiness(contact_id=_CONTACT_ID)
    with pytest.raises(EOMCardVaultNotFoundError):
        await service.start_session(token=_token(), public_base_url="https://eom.test")

    assert readiness["cardRequired"] is False
    assert readiness["cardReady"] is True
    assert readiness["reason"] == "not_required"
    assert readiness["candidateId"] == str(_CANDIDATE_ID)
    assert provider.customer_calls == []
    assert provider.session_calls == []

    unclassified = _State()
    assert unclassified.eligibility is not None
    unclassified.eligibility["service_commitment"] = None
    unclassified_provider = _Provider()
    unclassified_service = EOMCardVaultService(
        pool=_Pool(unclassified), provider=unclassified_provider
    )

    unclassified_readiness = await unclassified_service.get_readiness(
        contact_id=_CONTACT_ID
    )
    with pytest.raises(EOMCardVaultNotFoundError):
        await unclassified_service.start_session(
            token=_token(), public_base_url="https://eom.test"
        )

    assert unclassified_readiness["cardRequired"] is True
    assert unclassified_readiness["cardReady"] is False
    assert unclassified_readiness["reason"] == "service_commitment_required"
    assert unclassified_provider.customer_calls == []
    assert unclassified_provider.session_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("updates", "enrollment_status", "expected"),
    [
        (
            {
                "customer_type": "commercial",
                "invitation_audience": "commercial",
                "acceptance_audience": "commercial",
                "candidate_id": None,
                "service_commitment": None,
            },
            None,
            {"cardRequired": False, "cardReady": True, "reason": "not_required"},
        ),
        (
            {"service_commitment": "one_time"},
            None,
            {"cardRequired": False, "cardReady": True, "reason": "not_required"},
        ),
        (
            {"acceptance_id": None, "acceptance_audience": None},
            None,
            {
                "cardRequired": True,
                "cardReady": False,
                "reason": "terms_not_ready",
            },
        ),
        (
            {"candidate_id": None, "service_commitment": None},
            None,
            {
                "cardRequired": True,
                "cardReady": False,
                "reason": "first_clean_not_confirmed",
            },
        ),
        (
            {"service_commitment": None},
            None,
            {
                "cardRequired": True,
                "cardReady": False,
                "reason": "service_commitment_required",
            },
        ),
        (
            {},
            None,
            {"cardRequired": True, "cardReady": False, "reason": "not_started"},
        ),
        (
            {},
            "pending",
            {"cardRequired": True, "cardReady": False, "reason": "pending"},
        ),
        (
            {},
            "ready",
            {"cardRequired": True, "cardReady": True, "reason": "ready"},
        ),
    ],
    ids=(
        "commercial",
        "one-time",
        "terms-not-ready",
        "first-clean-missing",
        "service-undecided",
        "not-started",
        "pending",
        "ready",
    ),
)
async def test_public_readiness_projects_the_existing_card_policy(
    updates: dict[str, Any],
    enrollment_status: str | None,
    expected: dict[str, Any],
) -> None:
    state = _State()
    assert state.eligibility is not None
    state.eligibility.update(updates)
    if enrollment_status is not None:
        state.enrollment = {
            "id": _ENROLLMENT_ID,
            "status": enrollment_status,
            "ready_at": _NOW if enrollment_status == "ready" else None,
        }

    result = await EOMCardVaultService(pool=_Pool(state)).get_public_readiness(
        token=_token()
    )

    assert result == expected
    assert state.transaction_count == 0


@pytest.mark.asyncio
async def test_public_readiness_guard_closes_token_container_and_subject_families() -> (
    None
):
    tokens = (
        ("signed", _token()),
        (
            "wrong-key",
            AuthenticatedEOMTermsToken(
                invitation_id=_INVITATION_ID,
                signing_key_fingerprint="b" * 64,
            ),
        ),
        ("raw", "eomt1.not-authenticated"),
    )
    containers = (
        ("scalar", lambda value: value),
        ("mapping", lambda value: {"token": value}),
        ("sequence", lambda value: [value]),
        ("nested", lambda value: {"token": [value]}),
    )
    families = (
        ("current", {}),
        ("missing", None),
        ("revoked", {"revoked_at": _NOW}),
        ("expired", {"is_expired": True}),
        ("wrong-business", {"business_context_id": "another_business"}),
        ("not-customer", {"contact_type": "lead"}),
        ("inactive", {"contact_status": "inactive"}),
        ("unknown-audience", {"customer_type": "unknown"}),
        ("audience-drift", {"invitation_audience": "commercial"}),
        ("name-drift", {"full_name": "Another Customer"}),
        ("email-drift", {"email": "another@example.test"}),
    )

    for (token_class, token), (container_class, wrap), (
        subject_class,
        updates,
    ) in product(tokens, containers, families):
        state = _State()
        if updates is None:
            state.eligibility = None
        else:
            assert state.eligibility is not None
            state.eligibility.update(updates)
        candidate = wrap(token)
        # Spec-derived oracle: only the signed scalar for the current subject
        # reaches the projection; every other class takes the not-found side.
        expected = (
            token_class == "signed"
            and container_class == "scalar"
            and subject_class == "current"
        )

        if expected:
            result = await EOMCardVaultService(pool=_Pool(state)).get_public_readiness(
                token=candidate
            )
            assert result["reason"] == "not_started"
        else:
            with pytest.raises(EOMCardVaultNotFoundError):
                await EOMCardVaultService(pool=_Pool(state)).get_public_readiness(
                    token=candidate
                )
        assert state.transaction_count == 0


@pytest.mark.asyncio
async def test_card_readiness_survives_a_later_current_terms_acceptance() -> None:
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")
    await service.confirm_checkout_session(event=_completed_event())
    assert state.eligibility is not None
    state.eligibility["acceptance_id"] = UUID("77777777-7777-4777-8777-777777777777")

    readiness = await service.get_readiness(contact_id=_CONTACT_ID)
    repeated = await service.start_session(
        token=_token(), public_base_url="https://eom.test"
    )
    state.eligibility["later_material"] = True
    after_material_change = await service.get_readiness(contact_id=_CONTACT_ID)

    assert readiness["cardReady"] is True
    assert readiness["reason"] == "ready"
    assert repeated["status"] == "ready"
    assert repeated["checkoutUrl"] is None
    assert after_material_change["cardReady"] is True


@pytest.mark.asyncio
async def test_stripe_adapter_passes_per_request_authority(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    configured_timeouts: list[int] = []

    class RecordingHTTPClient(stripe.HTTPClient):
        name = "recording"

        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def request_async(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
            post_data: str | None = None,
        ) -> tuple[bytes, int, dict[str, str]]:
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "post_data": post_data,
                }
            )
            return (
                json.dumps({"id": "cus_adapter123", "object": "customer"}).encode(),
                200,
                {"request-id": "req_adapter123"},
            )

        async def close_async(self) -> None:
            self.closed = True

    transport = RecordingHTTPClient()

    def httpx_client(*, timeout: int) -> RecordingHTTPClient:
        configured_timeouts.append(timeout)
        return transport

    monkeypatch.setattr(stripe, "HTTPXClient", httpx_client)
    provider = StripeEOMCardVaultProvider(
        secret_key="sk_" + "test_" + "placeholderxxxx",
        webhook_secret="whsec_" + "placeholderxxxx",
        timeout_seconds=7,
    )
    await provider.create_customer(
        email="customer@example.test",
        idempotency_key="eom-card-vault-customer:test",
    )
    await provider.close()

    assert configured_timeouts == [7]
    assert transport.closed is True
    assert len(calls) == 1
    assert calls[0]["method"] == "post"
    assert calls[0]["url"].endswith("/v1/customers")
    assert calls[0]["headers"]["Authorization"] == (
        "Bearer sk_" + "test_" + "placeholderxxxx"
    )
    assert calls[0]["headers"]["Stripe-Version"] == EOM_CARD_VAULT_STRIPE_API_VERSION
    assert calls[0]["headers"]["Idempotency-Key"] == ("eom-card-vault-customer:test")
    assert calls[0]["post_data"] == "email=customer%40example.test"
    assert "timeout" not in calls[0]["post_data"]


@pytest.mark.asyncio
async def test_stripe_adapter_verifies_the_raw_signed_payload() -> None:
    secret = "whsec_" + "placeholderxxxx"
    provider = StripeEOMCardVaultProvider(
        secret_key="sk_" + "test_" + "placeholderxxxx",
        webhook_secret=secret,
        timeout_seconds=7,
    )
    payload = (
        b'{"id":"evt_signature123","object":"event",'
        b'"type":"customer.created","data":{"object":{}}}'
    )
    timestamp = int(time.time())
    signed = f"{timestamp}.{payload.decode('utf-8')}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    header = f"t={timestamp},v1={signature}"

    event = provider.construct_event(payload, header)
    assert event["id"] == "evt_signature123"
    with pytest.raises(EOMCardVaultSignatureError):
        provider.construct_event(payload + b" ", header)
    await provider.close()


@pytest.mark.asyncio
async def test_provider_dependency_closes_its_stripe_transport(monkeypatch) -> None:
    created: list[Any] = []

    class Provider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(api_mod, "StripeEOMCardVaultProvider", Provider)
    dependency = api_mod._provider_dependency(
        config=auth_mod.EOMCardVaultProviderConfig(
            secret_key="sk_" + "test_" + "placeholderxxxx",
            webhook_secret="whsec_" + "placeholderxxxx",
            request_timeout_seconds=7,
        )
    )

    provider = await anext(dependency)
    assert provider.kwargs["timeout_seconds"] == 7
    assert provider.closed is False
    await dependency.aclose()

    assert created == [provider]
    assert provider.closed is True


class _RouteService:
    async def start_session(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "enrollmentId": str(_ENROLLMENT_ID),
            "contactId": str(_CONTACT_ID),
            "candidateId": str(_CANDIDATE_ID),
            "status": "pending",
            "checkoutUrl": "https://checkout.stripe.test/session",
            "checkoutExpiresAt": _NOW + timedelta(days=1),
            "providerConfirmedAt": None,
            "idempotent": False,
        }

    async def get_readiness(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "contactId": str(_CONTACT_ID),
            "audience": "residential",
            "cardRequired": True,
            "cardReady": True,
            "reason": "ready",
            "candidateId": str(_CANDIDATE_ID),
            "enrollmentId": str(_ENROLLMENT_ID),
            "providerConfirmedAt": _NOW,
        }

    async def confirm_checkout_session(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "eventId": "evt_ignored123",
            "status": "ignored",
            "idempotent": True,
        }


@pytest.mark.asyncio
async def test_real_eom_entrypoint_reaches_api_and_root_webhook() -> None:
    config = _config()
    state = _State()
    provider = _Provider()
    token = format_eom_terms_token(invitation_id=_INVITATION_ID, secret=_SECRET)
    authenticated = authenticate_eom_terms_token(token=token, secret=_SECRET)
    assert state.eligibility is not None
    state.eligibility["signing_key_fingerprint"] = authenticated.signing_key_fingerprint

    def completed_event(_payload: bytes, _signature: str) -> dict[str, Any]:
        return _completed_event()

    provider.construct_event = completed_event  # type: ignore[method-assign]
    original_pool_factory = main_eom.app.state.eom_funnel_card_vault_pool
    main_eom.app.state.eom_funnel_card_vault_pool = lambda: _Pool(state)
    main_eom.app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        config
    )
    main_eom.app.dependency_overrides[api_mod._provider_dependency] = lambda: provider
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app),
            base_url="http://test",
        ) as client:
            session = await client.post(
                "/api/v1/eom-funnel/card-vault/public/session",
                headers={"Authorization": f"Bearer {_SERVICE.token}"},
                json={"token": token},
            )
            webhook = await client.post(
                "/webhooks/eom-card-vault",
                headers={"Stripe-Signature": "valid-signature"},
                content=b"{}",
            )
            readiness = await client.get(
                f"/api/v1/eom-funnel/card-vault/readiness/{_CONTACT_ID}",
                headers={"Authorization": f"Bearer {_SERVICE.token}"},
            )
    finally:
        main_eom.app.dependency_overrides.clear()
        main_eom.app.state.eom_funnel_card_vault_pool = original_pool_factory

    assert session.status_code == 201
    assert session.json()["checkoutUrl"].startswith("https://checkout.stripe.test/")
    assert webhook.status_code == 200
    assert webhook.json()["status"] == "ready"
    assert webhook.json()["idempotent"] is False
    assert readiness.status_code == 200
    assert readiness.json()["cardReady"] is True
    assert readiness.json()["reason"] == "ready"


@pytest.mark.asyncio
async def test_readiness_does_not_require_provider_credentials() -> None:
    config = _config(
        card_vault_enabled=False,
        card_vault_stripe_secret_key="",
        card_vault_stripe_webhook_secret="",
    )
    state = _State()
    state.readiness = {
        "contact_id": _CONTACT_ID,
        "customer_type": "residential",
        "business_context_id": "effingham_maids",
        "contact_type": "customer",
        "contact_status": "active",
        "candidate_id": _CANDIDATE_ID,
        "service_commitment": "recurring",
        "acceptance_id": _ACCEPTANCE_ID,
        "acceptance_audience": "residential",
        "enrollment_id": _ENROLLMENT_ID,
        "enrollment_status": "ready",
        "ready_at": _NOW,
        "later_material": False,
    }
    original_pool_factory = main_eom.app.state.eom_funnel_card_vault_pool
    main_eom.app.state.eom_funnel_card_vault_pool = lambda: _Pool(state)
    main_eom.app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        config
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/v1/eom-funnel/card-vault/readiness/{_CONTACT_ID}",
                headers={"Authorization": f"Bearer {_SERVICE.token}"},
            )
    finally:
        main_eom.app.dependency_overrides.clear()
        main_eom.app.state.eom_funnel_card_vault_pool = original_pool_factory

    assert response.status_code == 200
    assert response.json()["cardReady"] is True
    assert response.json()["reason"] == "ready"


@pytest.mark.asyncio
async def test_public_readiness_route_is_minimal_and_provider_independent() -> None:
    config = _config(
        card_vault_enabled=False,
        card_vault_stripe_secret_key="",
        card_vault_stripe_webhook_secret="",
    )
    state = _State()
    token = format_eom_terms_token(invitation_id=_INVITATION_ID, secret=_SECRET)
    authenticated = authenticate_eom_terms_token(token=token, secret=_SECRET)
    assert state.eligibility is not None
    state.eligibility["signing_key_fingerprint"] = authenticated.signing_key_fingerprint
    original_pool_factory = main_eom.app.state.eom_funnel_card_vault_pool
    main_eom.app.state.eom_funnel_card_vault_pool = lambda: _Pool(state)
    main_eom.app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        config
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/eom-funnel/card-vault/public/readiness",
                headers={"Authorization": f"Bearer {_SERVICE.token}"},
                json={"token": token},
            )
            malformed = await client.post(
                "/api/v1/eom-funnel/card-vault/public/readiness",
                headers={"Authorization": f"Bearer {_SERVICE.token}"},
                json={"token": {"nested": token}},
            )
    finally:
        main_eom.app.dependency_overrides.clear()
        main_eom.app.state.eom_funnel_card_vault_pool = original_pool_factory

    assert response.status_code == 200
    assert response.json() == {
        "cardRequired": True,
        "cardReady": False,
        "reason": "not_started",
    }
    assert malformed.status_code == 404
    assert state.transaction_count == 0


@pytest.mark.asyncio
async def test_paused_issuance_keeps_signed_root_webhook_available() -> None:
    config = _config(card_vault_enabled=False)
    secret = config.card_vault_stripe_webhook_secret.get_secret_value()
    payload = (
        b'{"id":"evt_paused123","object":"event",'
        b'"type":"customer.created","data":{"object":{}}}'
    )
    timestamp = int(time.time())
    signed = f"{timestamp}.".encode("ascii") + payload
    signature = hmac.new(secret.encode("ascii"), signed, hashlib.sha256).hexdigest()
    header = f"t={timestamp},v1={signature}"
    main_eom.app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        config
    )
    main_eom.app.dependency_overrides[api_mod._service_dependency] = _RouteService
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app),
            base_url="http://test",
        ) as client:
            issuance = await client.post(
                "/api/v1/eom-funnel/card-vault/public/session",
                headers={"Authorization": f"Bearer {_SERVICE.token}"},
                json={"token": "unused-while-issuance-is-paused"},
            )
            webhook = await client.post(
                "/webhooks/eom-card-vault",
                headers={"Stripe-Signature": header},
                content=payload,
            )
    finally:
        main_eom.app.dependency_overrides.clear()

    assert issuance.status_code == 503
    assert webhook.status_code == 200
    assert webhook.json() == {
        "eventId": "evt_ignored123",
        "enrollmentId": None,
        "status": "ignored",
        "idempotent": True,
    }


@pytest.mark.asyncio
async def test_full_atlas_entrypoint_reaches_signed_root_webhook(monkeypatch) -> None:
    pytest.importorskip(
        "numpy",
        reason="full Atlas entrypoint uses the main requirements profile",
    )
    from atlas_brain import main

    config = _config()
    state = _State()
    provider = _Provider()
    service = EOMCardVaultService(pool=_Pool(state), provider=provider)
    await service.start_session(token=_token(), public_base_url="https://eom.test")
    original_pool_factory = main.app.state.eom_funnel_card_vault_pool

    def completed_event(payload: bytes, signature: str) -> dict[str, Any]:
        assert payload == b"{}"
        assert signature == "valid-signature"
        return _completed_event()

    provider.construct_event = completed_event  # type: ignore[method-assign]
    main.app.state.eom_funnel_card_vault_pool = lambda: _Pool(state)
    monkeypatch.setitem(
        main.app.dependency_overrides,
        auth_mod.get_eom_funnel_api_config,
        lambda: config,
    )
    monkeypatch.setitem(
        main.app.dependency_overrides,
        api_mod._provider_dependency,
        lambda: provider,
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/webhooks/eom-card-vault",
                headers={"Stripe-Signature": "valid-signature"},
                content=b"{}",
            )
    finally:
        main.app.state.eom_funnel_card_vault_pool = original_pool_factory

    assert response.status_code == 200
    assert response.json() == {
        "eventId": "evt_cardvault123",
        "enrollmentId": str(_ENROLLMENT_ID),
        "status": "ready",
        "idempotent": False,
    }
    assert state.enrollment is not None
    assert state.enrollment["status"] == "ready"


@pytest.mark.asyncio
async def test_webhook_rejects_missing_or_bad_signature_before_service() -> None:
    config = _config()
    provider = _Provider()
    main_eom.app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        config
    )
    main_eom.app.dependency_overrides[api_mod._provider_dependency] = lambda: provider
    main_eom.app.dependency_overrides[api_mod._service_dependency] = lambda: (
        SimpleNamespace(
            confirm_checkout_session=lambda **_kwargs: pytest.fail(
                "invalid signature reached service"
            )
        )
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app), base_url="http://test"
        ) as client:
            missing = await client.post(
                "/webhooks/eom-card-vault",
                content=b"{}",
            )
            invalid = await client.post(
                "/webhooks/eom-card-vault",
                headers={"Stripe-Signature": "invalid"},
                content=b"{}",
            )
    finally:
        main_eom.app.dependency_overrides.clear()
    assert missing.status_code == 422
    assert invalid.status_code == 400


@pytest.mark.asyncio
async def test_webhook_rejects_oversized_body_before_provider() -> None:
    config = _config()
    main_eom.app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: (
        config
    )
    main_eom.app.dependency_overrides[api_mod._provider_dependency] = _Provider
    main_eom.app.dependency_overrides[api_mod._service_dependency] = _RouteService
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhooks/eom-card-vault",
                headers={"Stripe-Signature": "valid-signature"},
                content=b"x" * (api_mod._MAX_WEBHOOK_BYTES + 1),
            )
    finally:
        main_eom.app.dependency_overrides.clear()
    assert response.status_code == 413


def test_migration_is_controlled_and_stores_no_raw_card_data() -> None:
    sql = _MIGRATION.read_text()
    assert sql.startswith("-- atlas: atomic-bookkeeping")
    assert "database administrator must run 398_eom_card_vault" in sql
    assert "397_eom_terms_acceptance" in sql
    assert "stripe_payment_method_id" in sql
    executable_sql = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    ).lower()
    for forbidden in ("card_number", "cvc", "routing_number", "account_number"):
        assert forbidden not in executable_sql
    assert "GRANT UPDATE (stripe_customer_id, status" in sql
    assert "checkout_success_url TEXT NOT NULL" in sql
    assert "checkout_cancel_url TEXT NOT NULL" in sql
    assert "provider_retry_until TIMESTAMPTZ NOT NULL" in sql
    assert "provider_retry_until <= created_at + INTERVAL '2 hours'" in sql
    assert (
        "GRANT INSERT (id, enrollment_id, acceptance_id, "
        "'\n        || 'checkout_success_url, checkout_cancel_url, provider_retry_until)"
        in sql
    )
    assert "GRANT DELETE" not in sql
