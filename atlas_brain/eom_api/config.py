"""Small settings surface for the slim EOM API profile."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values
from pydantic import Field, SecretStr
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")
RAW_RECEIVABLES_SERVICE_TOKEN_ENV = "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN"
_RAW_RECEIVABLES_SERVICE_TOKEN_ENV_KEY = RAW_RECEIVABLES_SERVICE_TOKEN_ENV.casefold()
RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV = "ATLAS_EOM_FUNNEL_SERVICE_TOKEN"
_RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV_KEY = RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV.casefold()


def _has_raw_service_token(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping_contains_raw_service_token(
    values: Mapping[str, object],
    *,
    raw_env_key: str,
) -> bool:
    return any(
        key.casefold() == raw_env_key
        and _has_raw_service_token(value)
        for key, value in values.items()
    )


def _raw_service_token_configured(
    *,
    raw_env_key: str,
    environ: Mapping[str, str] | None = None,
    env_files: Iterable[str | Path] = ENV_FILES,
) -> bool:
    """Return true when any admitted settings source carries raw token material."""
    source_environ = os.environ if environ is None else environ
    if _mapping_contains_raw_service_token(
        source_environ,
        raw_env_key=raw_env_key,
    ):
        return True
    for env_file in env_files:
        try:
            values = dotenv_values(env_file)
        except OSError:
            continue
        if _mapping_contains_raw_service_token(values, raw_env_key=raw_env_key):
            return True
    return False


def raw_receivables_service_token_configured(
    environ: Mapping[str, str] | None = None,
    env_files: Iterable[str | Path] = ENV_FILES,
) -> bool:
    """Return true when any admitted settings source carries raw token material."""
    return _raw_service_token_configured(
        raw_env_key=_RAW_RECEIVABLES_SERVICE_TOKEN_ENV_KEY,
        environ=environ,
        env_files=env_files,
    )


def raw_eom_funnel_service_token_configured(
    environ: Mapping[str, str] | None = None,
    env_files: Iterable[str | Path] = ENV_FILES,
) -> bool:
    """Return true when any admitted settings source carries raw funnel token."""
    return _raw_service_token_configured(
        raw_env_key=_RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV_KEY,
        environ=environ,
        env_files=env_files,
    )


class EOMRuntimeConfig(BaseSettings):
    """Runtime settings needed before the EOM app starts."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="text", description="Logging format: text or json")


class EOMProfileConfig(BaseSettings):
    """EOM-profile startup controls."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EOM_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    run_migrations: bool = Field(
        default=False,
        description="Run database migrations during the slim EOM profile startup",
    )
    canonical_crm_database_confirmed: bool = Field(
        default=False,
        description=(
            "Confirm the slim EOM funnel runtime is configured with the "
            "authoritative Atlas CRM database before it may serve funnel APIs"
        ),
    )


class EOMInvoicingConfig(BaseSettings):
    """Receivables auth settings used by the EOM service-to-service API."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_INVOICING_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    receivables_api_enabled: bool = Field(
        default=False,
        description="Enable the authenticated EOM receivables service API",
    )
    receivables_service_token_sha256: str = Field(
        default="",
        description=(
            "SHA-256 digest of the generated EOM receivables bearer token; "
            "the raw token must live only on the caller side"
        ),
    )

    @model_validator(mode="after")
    def reject_raw_receivables_service_token_env(self) -> "EOMInvoicingConfig":
        if raw_receivables_service_token_configured():
            raise ValueError(
                "Raw EOM receivables bearer token material must not be configured "
                f"in {RAW_RECEIVABLES_SERVICE_TOKEN_ENV}; provision only "
                "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256 on the Atlas "
                "API service and keep the raw token on the caller side."
            )
        return self


class EOMFunnelConfig(BaseSettings):
    """Private service boundary for EOM office lead conversion."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EOM_FUNNEL_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    api_enabled: bool = Field(
        default=False,
        description="Enable the service-authenticated EOM office conversion API",
    )
    service_token_sha256: str = Field(
        default="",
        description=(
            "SHA-256 digest of the generated bearer accepted from the EOM time "
            "tracker only; the raw bearer is never stored in Atlas"
        ),
    )
    db_connection_string: str = Field(
        default="",
        description=(
            "Authoritative Atlas CRM PostgreSQL connection string used only by "
            "the slim EOM funnel routes"
        ),
    )
    public_onboarding_enabled: bool = Field(
        default=False,
        description=(
            "Enable the configured tracker-only authority that validates and "
            "redeems public EOM onboarding links"
        ),
    )
    public_onboarding_issuance_enabled: bool | None = Field(
        default=None,
        description=(
            "Optional new-link issuance override; unset preserves the authority "
            "flag's legacy behavior, while false pauses issuance without "
            "disabling redemption of already-issued links"
        ),
    )
    public_onboarding_url: str = Field(
        default="",
        description=(
            "HTTPS Website onboarding page base URL; the bearer is appended only "
            "as a URL fragment at email-delivery time"
        ),
    )
    public_onboarding_hmac_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description=(
            "Atlas-only HMAC secret for public onboarding link signatures; never "
            "expose it to the tracker or browser"
        ),
    )
    public_onboarding_previous_hmac_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description=(
            "Optional immediately previous Atlas-only public-onboarding HMAC "
            "secret, retained only while issued links from one controlled key "
            "rotation are still active"
        ),
    )
    card_vault_enabled: bool = Field(
        default=False,
        description=(
            "Enable issuance of new Stripe-hosted card-on-file setup sessions "
            "for eligible residential post-clean onboarding; signed webhook "
            "confirmation remains available while dedicated credentials remain set"
        ),
    )
    card_vault_stripe_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description=(
            "Dedicated server-side Stripe key for EOM card-vault setup; never "
            "expose it to the tracker or browser"
        ),
    )
    card_vault_stripe_webhook_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description=(
            "Dedicated Stripe signing secret for the EOM card-vault webhook"
        ),
    )
    card_vault_request_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Bounded Stripe request timeout for EOM card-vault operations",
    )
    missed_call_recovery_enabled: bool = Field(
        default=False,
        description=(
            "Allow the EOM missed-call worker to deliver configured residential "
            "recovery emails. Disabled keeps recorded sequences blocked."
        ),
    )
    missed_call_booking_link: str = Field(
        default="",
        description=(
            "Authoritative HTTPS Google Calendar appointment-request link for "
            "EOM missed-call recovery emails; never a browser-supplied value."
        ),
    )
    missed_call_timezone: str = Field(
        default="America/Chicago",
        description=(
            "IANA time zone used for EOM missed-call business-day scheduling."
        ),
    )
    missed_call_poll_interval_seconds: int = Field(
        default=60,
        ge=30,
        le=900,
        description="Bounded polling interval for the EOM missed-call outbox worker.",
    )
    missed_call_max_delivery_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Maximum definite-pre-acceptance delivery attempts per missed-call "
            "sequence step."
        ),
    )
    missed_call_delivery_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Bounded Resend request timeout for a missed-call recovery step.",
    )

    @property
    def public_onboarding_issuance_is_enabled(self) -> bool:
        """Return the effective new-link issuance decision.

        An unset override preserves configured deployments' original behavior:
        enabling the public authority also enables issuance. Operators can set
        the override false to drain outstanding links without disabling the
        authority that validates and redeems them.
        """

        if self.public_onboarding_issuance_enabled is None:
            return self.public_onboarding_enabled
        return self.public_onboarding_issuance_enabled

    @property
    def missed_call_recovery_delivery_is_configured(self) -> bool:
        """Whether a worker may deliver recovery mail in this process.

        A blank booking link deliberately does not make the whole EOM API
        unavailable.  An operator's real no-answer evidence can still be
        recorded as a visible blocked sequence, but no message can render or
        send until a valid deploy-time destination is supplied.
        """

        return bool(
            self.missed_call_recovery_enabled and self.missed_call_booking_link.strip()
        )

    @property
    def missed_call_recovery_delivery_block_reason(self) -> str | None:
        """Return the honest operator-visible reason delivery cannot begin.

        ``disabled`` and ``missing link`` are intentionally distinct.  The
        former is a deployment/operator control; the latter is an incomplete
        customer-facing configuration.  Collapsing them would make a recovery
        action look like a bad Calendar configuration when an operator had
        deliberately paused the feature.
        """

        if not self.missed_call_recovery_enabled:
            return "recovery_disabled"
        if not self.missed_call_booking_link.strip():
            return "booking_link_unavailable"
        return None

    @model_validator(mode="after")
    def reject_raw_eom_funnel_service_token_env(self) -> "EOMFunnelConfig":
        if raw_eom_funnel_service_token_configured():
            raise ValueError(
                "Raw EOM funnel bearer token material must not be configured "
                f"in {RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV}; provision only "
                "ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256 on the Atlas API service "
                "and keep the raw token on the caller side."
            )
        return self

    @model_validator(mode="after")
    def validate_public_onboarding_configuration(self) -> "EOMFunnelConfig":
        """Make an explicitly configured bearer authority safe before use.

        The URL and primary secret may be staged while disabled, but a partial
        or unsafe tuple is never accepted. One validated previous secret is
        permitted only to bridge a controlled key rotation. That makes a later
        flag flip deterministic and keeps an operator from discovering malformed
        configuration only after a draft has already been claimed for sending.
        """

        raw_base_url = self.public_onboarding_url
        if any(
            character.isspace()
            or character == "\\"
            or ord(character) < 32
            or ord(character) == 127
            for character in raw_base_url
        ):
            raise ValueError(
                "public onboarding URL must not contain control characters, whitespace, "
                "or backslashes"
            )
        base_url = raw_base_url.strip()
        secret = self.public_onboarding_hmac_secret.get_secret_value().strip()
        previous_secret = (
            self.public_onboarding_previous_hmac_secret.get_secret_value().strip()
        )
        has_url = bool(base_url)
        has_secret = bool(secret)
        has_previous_secret = bool(previous_secret)
        if has_url != has_secret:
            raise ValueError(
                "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_URL and "
                "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_HMAC_SECRET must be set together"
            )
        if has_previous_secret and not has_secret:
            raise ValueError(
                "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_PREVIOUS_HMAC_SECRET requires "
                "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_HMAC_SECRET"
            )
        if has_previous_secret and previous_secret == secret:
            raise ValueError(
                "public onboarding previous HMAC secret must differ from the primary secret"
            )
        if self.public_onboarding_issuance_enabled and not self.public_onboarding_enabled:
            raise ValueError(
                "public onboarding issuance requires "
                "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_ENABLED=true"
            )
        if self.public_onboarding_enabled and not has_url:
            raise ValueError(
                "public onboarding requires ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_URL "
                "and ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_HMAC_SECRET"
            )
        if self.public_onboarding_enabled and not self.api_enabled:
            raise ValueError(
                "public onboarding requires ATLAS_EOM_FUNNEL_API_ENABLED=true"
            )
        if not has_url:
            return self
        try:
            parsed = urlsplit(base_url)
            # ``urlsplit`` defers an invalid numeric port until this property
            # is read, so force that validation while configuration is still
            # fail-closed rather than when an approved draft is sent.
            port = parsed.port
        except ValueError as exc:
            raise ValueError("public onboarding URL must be a valid HTTPS URL") from exc
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or port == 0
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "public onboarding URL must be an HTTPS URL without credentials, "
                "query, or fragment"
            )
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("public onboarding HMAC secret must be at least 32 bytes")
        if has_previous_secret and len(previous_secret.encode("utf-8")) < 32:
            raise ValueError(
                "public onboarding previous HMAC secret must be at least 32 bytes"
            )
        return self

    @model_validator(mode="after")
    def validate_card_vault_configuration(self) -> "EOMFunnelConfig":
        """Admit one complete, dedicated Stripe authority or keep it disabled."""

        secret_key = self.card_vault_stripe_secret_key.get_secret_value().strip()
        webhook_secret = (
            self.card_vault_stripe_webhook_secret.get_secret_value().strip()
        )
        has_secret_key = bool(secret_key)
        has_webhook_secret = bool(webhook_secret)
        if has_secret_key != has_webhook_secret:
            raise ValueError(
                "ATLAS_EOM_FUNNEL_CARD_VAULT_STRIPE_SECRET_KEY and "
                "ATLAS_EOM_FUNNEL_CARD_VAULT_STRIPE_WEBHOOK_SECRET must be set together"
            )
        if self.card_vault_enabled and not has_secret_key:
            raise ValueError(
                "card vault requires dedicated Stripe secret and webhook keys"
            )
        if self.card_vault_enabled and not self.public_onboarding_enabled:
            raise ValueError(
                "card vault requires ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_ENABLED=true"
            )
        if not has_secret_key:
            return self
        stripe_key_prefixes = ("sk_test_", "sk_live_", "rk_test_", "rk_live_")
        if (
            not secret_key.startswith(stripe_key_prefixes)
            or len(secret_key) < 16
            or not secret_key.isascii()
            or not all(character.isalnum() or character == "_" for character in secret_key)
        ):
            raise ValueError("card-vault Stripe secret key is invalid")
        if (
            not webhook_secret.startswith("whsec_")
            or len(webhook_secret) < 16
            or not webhook_secret.isascii()
            or not all(
                character.isalnum() or character == "_"
                for character in webhook_secret
            )
        ):
            raise ValueError("card-vault Stripe webhook secret is invalid")
        return self

    @model_validator(mode="after")
    def validate_missed_call_recovery_configuration(self) -> "EOMFunnelConfig":
        """Validate a supplied customer-facing booking URL before any send.

        A missing link is a supported, fail-closed rollout state. A nonblank
        malformed/non-Google link is not: accepting it would turn a config typo
        into a customer email with an unusable or unintended destination.
        """

        if self.missed_call_recovery_enabled and not self.api_enabled:
            raise ValueError(
                "missed-call recovery requires ATLAS_EOM_FUNNEL_API_ENABLED=true"
            )
        booking_link = self.missed_call_booking_link.strip()
        if booking_link:
            if any(
                character.isspace()
                or character == "\\"
                or ord(character) < 32
                or ord(character) == 127
                for character in booking_link
            ):
                raise ValueError(
                    "missed-call booking link must not contain control characters, "
                    "whitespace, or backslashes"
                )
            try:
                parsed = urlsplit(booking_link)
                port = parsed.port
            except ValueError as exc:
                raise ValueError(
                    "missed-call booking link must be a valid HTTPS Google Calendar URL"
                ) from exc
            allowed_hosts = {"calendar.google.com", "calendar.app.google"}
            hostname = (parsed.hostname or "").casefold()
            if (
                parsed.scheme != "https"
                or hostname not in allowed_hosts
                or parsed.username is not None
                or parsed.password is not None
                # Google Calendar hosts its public scheduler on the standard
                # HTTPS endpoint.  An explicit alternate port can never be a
                # supported public booking URL and is easy to mistype.
                or port is not None
                or not parsed.path.strip("/")
            ):
                raise ValueError(
                    "missed-call booking link must be an HTTPS Google Calendar URL"
                )
        try:
            ZoneInfo(self.missed_call_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("missed-call time zone must be a valid IANA zone") from exc
        return self


eom_settings = EOMRuntimeConfig()
eom_profile_settings = EOMProfileConfig()
invoicing_settings = EOMInvoicingConfig()
funnel_settings = EOMFunnelConfig()
