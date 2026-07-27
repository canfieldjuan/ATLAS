"""Small settings surface for the slim EOM API profile."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")
RAW_RECEIVABLES_SERVICE_TOKEN_ENV = "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN"


def raw_receivables_service_token_env_value(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return raw receivables token material from process env, if configured."""
    value = (environ or os.environ).get(RAW_RECEIVABLES_SERVICE_TOKEN_ENV, "")
    return value.strip()


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
        if (
            self.receivables_api_enabled
            and raw_receivables_service_token_env_value()
        ):
            raise ValueError(
                "Raw EOM receivables bearer token material must not be configured "
                f"in {RAW_RECEIVABLES_SERVICE_TOKEN_ENV}; provision only "
                "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256 on the Atlas "
                "API service and keep the raw token on the caller side."
            )
        return self


eom_settings = EOMRuntimeConfig()
eom_profile_settings = EOMProfileConfig()
invoicing_settings = EOMInvoicingConfig()
