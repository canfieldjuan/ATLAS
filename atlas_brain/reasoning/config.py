"""Configuration for the Reasoning Agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReasoningConfig(BaseSettings):
    """Cross-domain reasoning agent configuration.

    Off by default. Set ATLAS_REASONING__ENABLED=true to activate.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REASONING_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable the reasoning agent event bus and consumer",
    )

    # LLM models
    model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model for deep reasoning",
    )
    max_tokens: int = Field(default=2048, description="Max tokens for reasoning calls")
    temperature: float = Field(default=0.3, description="Temperature for reasoning calls")

    triage_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Cheap model for event triage classification",
    )
    triage_max_tokens: int = Field(default=256, description="Max tokens for triage calls")

    # Entity locks
    lock_heartbeat_interval_s: int = Field(
        default=30, description="Heartbeat interval for entity locks (seconds)"
    )
    lock_expiry_s: int = Field(
        default=300, description="Expire stale locks after this many seconds"
    )

    # Event processing
    event_batch_size: int = Field(
        default=10, description="Max events to process per batch"
    )
    event_max_age_hours: int = Field(
        default=48, description="Discard unprocessed events older than this"
    )

    # Reflection schedule
    reflection_cron: str = Field(
        default="0 9,13,17,21 * * *",
        description="Cron expression for proactive reflection runs",
    )

    # Concurrency
    max_concurrent_reasoning: int = Field(
        default=1, description="Max concurrent reasoning graph invocations"
    )
