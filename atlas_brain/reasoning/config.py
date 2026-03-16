"""Configuration for the Reasoning Agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReasoningConfig(BaseSettings):
    """Cross-domain reasoning agent configuration.

    Off by default. Set ATLAS_REASONING__ENABLED=true to activate.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REASONING__",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable the reasoning agent event bus and consumer",
    )

    # LLM models (cross-domain reasoning agent)
    model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model for deep reasoning",
    )
    max_tokens: int = Field(default=16384, description="Max tokens for reasoning calls (includes thinking tokens for reasoning models)")
    temperature: float = Field(default=0.3, description="Temperature for reasoning calls (ignored by reasoning models like o4-mini)")

    # Stratified reasoning LLM backend (B2B churn pipeline)
    stratified_llm_workload: str = Field(
        default="openrouter",
        description="Pipeline LLM workload for stratified reasoning: 'openrouter', 'vllm', 'anthropic', or 'auto'",
    )
    stratified_openrouter_model: str = Field(
        default="openai/o4-mini",
        description="OpenRouter model ID for stratified reasoning (used when workload=openrouter)",
    )

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

    # Neo4j (episodic memory + knowledge graph)
    neo4j_bolt_url: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt protocol URL",
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password123", description="Neo4j password")

    # Multi-pass reasoning (classify -> challenge -> ground)
    multi_pass_enabled: bool = Field(
        default=False,
        description="Enable multi-pass reasoning (classify -> challenge -> ground)",
    )
    multi_pass_challenge_confidence_floor: float = Field(
        default=0.3,
        description="Skip challenge pass if Pass 1 confidence is at or below this",
    )
    multi_pass_ground_change_threshold: float = Field(
        default=0.05,
        description="Skip grounding pass if challenge changed conclusion by less than this delta",
    )
