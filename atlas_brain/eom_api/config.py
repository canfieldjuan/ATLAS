"""Small settings surface for the slim EOM API profile."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")
RAW_RECEIVABLES_SERVICE_TOKEN_ENV = "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN"
_RAW_RECEIVABLES_SERVICE_TOKEN_ENV_KEY = RAW_RECEIVABLES_SERVICE_TOKEN_ENV.casefold()


def _has_raw_receivables_service_token(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping_contains_raw_receivables_service_token(
    values: Mapping[str, object],
) -> bool:
    return any(
        key.casefold() == _RAW_RECEIVABLES_SERVICE_TOKEN_ENV_KEY
        and _has_raw_receivables_service_token(value)
        for key, value in values.items()
    )


def raw_receivables_service_token_configured(
    environ: Mapping[str, str] | None = None,
    env_files: Iterable[str | Path] = ENV_FILES,
) -> bool:
    """Return true when any admitted settings source carries raw token material."""
    if _mapping_contains_raw_receivables_service_token(environ or os.environ):
        return True
    for env_file in env_files:
        try:
            values = dotenv_values(env_file)
        except OSError:
            continue
        if _mapping_contains_raw_receivables_service_token(values):
            return True
    return False


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
            "Confirm the slim EOM profile is connected to the canonical Atlas CRM "
            "database before it may serve funnel APIs"
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


eom_settings = EOMRuntimeConfig()
eom_profile_settings = EOMProfileConfig()
invoicing_settings = EOMInvoicingConfig()
funnel_settings = EOMFunnelConfig()
