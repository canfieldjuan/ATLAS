"""Small settings surface for the slim EOM API profile."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")


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
    receivables_service_token: str = Field(
        default="",
        description="Bearer token accepted by the EOM receivables service API",
    )


eom_settings = EOMRuntimeConfig()
eom_profile_settings = EOMProfileConfig()
invoicing_settings = EOMInvoicingConfig()
