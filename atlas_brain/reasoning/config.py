"""Configuration for the Reasoning Agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")


class ReasoningConfig(BaseSettings):
    """Cross-domain reasoning agent configuration.

    Off by default. Set ATLAS_REASONING__ENABLED=true to activate.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REASONING__",
        env_file=ENV_FILES,
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
    # Heavy model: archetype classification (Pass 1), pairwise battles
    # Light model: challenge/ground passes, reconstitute, category councils, asymmetry
    stratified_llm_workload: str = Field(
        default="openrouter",
        description="Pipeline LLM workload for stratified reasoning: 'openrouter', 'vllm', 'anthropic', or 'auto'",
    )
    stratified_openrouter_model: str = Field(
        default="openai/gpt-5.1",
        description="OpenRouter model for Tier 1 (heavy) reasoning: archetype classify, pairwise battles",
    )
    stratified_openrouter_model_light: str = Field(
        default="openai/o4-mini",
        description="OpenRouter model for Tier 2 (light) reasoning: challenge/ground passes, reconstitute, category councils, asymmetry",
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
        default=True,
        description="Enable multi-pass reasoning (classify -> challenge -> ground)",
    )
    multi_pass_challenge_confidence_floor: float = Field(
        default=0.3,
        description="Skip challenge pass if Pass 1 confidence is at or below this",
    )
    multi_pass_challenge_min_reviews: int = Field(
        default=8,
        description="Minimum review volume before low-confidence challenge escalation is considered evidence-backed",
    )
    multi_pass_challenge_mixed_polarity_min_share: float = Field(
        default=0.2,
        description="Minimum minority recommendation share that qualifies as mixed polarity for challenge escalation",
    )
    multi_pass_challenge_high_impact_churn_density: float = Field(
        default=20.0,
        description="Challenge low-confidence conclusions when churn density is at or above this threshold",
    )
    multi_pass_challenge_high_impact_avg_urgency: float = Field(
        default=6.0,
        description="Challenge low-confidence conclusions when avg urgency is at or above this threshold",
    )
    multi_pass_challenge_high_impact_displacement_mentions: int = Field(
        default=5,
        description="Challenge low-confidence conclusions when displacement mentions are at or above this threshold",
    )
    multi_pass_ground_always: bool = Field(
        default=True,
        description="Run the lightweight grounding pass even when challenge is skipped",
    )
    multi_pass_ground_change_threshold: float = Field(
        default=0.05,
        description="Minimum challenge confidence delta that counts as a materially changed conclusion",
    )
    reconstitute_threshold: float = Field(
        default=0.3,
        description="Maximum weighted evidence drift ratio that still permits reconstitution instead of full reasoning",
    )
    reconstitute_core_weight: float = Field(
        default=4.0,
        description="Weight for core churn metrics in evidence drift scoring",
    )
    reconstitute_thematic_weight: float = Field(
        default=3.0,
        description="Weight for pain, competitor, and feature-theme shifts in evidence drift scoring",
    )
    reconstitute_segment_weight: float = Field(
        default=3.0,
        description="Weight for buyer-role, use-case, and budget-context shifts in evidence drift scoring",
    )
    reconstitute_temporal_weight: float = Field(
        default=3.0,
        description="Weight for temporal and velocity shifts in evidence drift scoring",
    )
    reconstitute_quote_weight: float = Field(
        default=1.5,
        description="Weight for quote-level evidence changes in evidence drift scoring",
    )
    reconstitute_minor_weight: float = Field(
        default=1.0,
        description="Fallback weight for low-priority evidence drift signals",
    )
    reconstitute_strategic_component_threshold: float = Field(
        default=3.0,
        description="Minimum component drift score that forces full reasoning for strategic shifts like pain, role, competitive, or temporal changes",
    )
    reconstitute_contradiction_emergence_threshold: float = Field(
        default=4.0,
        description="Minimum contradiction-emergence score that forces full reasoning even when weighted drift stays below the reconstitute threshold",
    )
