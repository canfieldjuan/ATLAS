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
        default="claude-3-5-haiku-latest",
        description="Anthropic model for deep reasoning",
    )
    max_tokens: int = Field(default=16384, description="Max tokens for reasoning calls (includes thinking tokens for reasoning models)")
    temperature: float = Field(default=0.3, description="Temperature for reasoning calls (ignored by some reasoning models)")

    # Legacy stratified reasoning LLM backend (standalone tooling only)
    # Heavy model: archetype classification (Pass 1), pairwise battles
    # Light model: challenge/ground passes, reconstitute, category councils, asymmetry
    stratified_llm_workload: str = Field(
        default="openrouter",
        description="Legacy engine LLM workload for stratified reasoning: 'openrouter', 'vllm', 'anthropic', or 'auto'",
    )
    stratified_openrouter_model: str = Field(
        default="openai/gpt-oss-120b",
        description="OpenRouter model for Tier 1 (heavy) reasoning: archetype classify, pairwise battles",
    )
    stratified_openrouter_model_light: str = Field(
        default="",
        description=(
            "OpenRouter model for Tier 2 (light) reasoning: challenge/ground "
            "passes, reconstitute, category councils, asymmetry. "
            "Empty = reuse stratified_openrouter_model."
        ),
    )
    stratified_anthropic_model: str = Field(
        default="claude-3-5-haiku-latest",
        description="Anthropic model for the legacy stratified engine when workload is 'anthropic'",
    )

    triage_model: str = Field(
        default="claude-3-5-haiku-latest",
        description="Cheap model for event triage classification",
    )
    triage_max_tokens: int = Field(default=256, description="Max tokens for triage calls")
    graph_triage_workload: str = Field(
        default="triage",
        description=(
            "Pipeline LLM workload for reasoning-graph triage "
            "(triage, draft, synthesis, reasoning, openrouter, "
            "local_fast, vllm, anthropic)"
        ),
    )
    graph_reasoning_workload: str = Field(
        default="reasoning",
        description=(
            "Pipeline LLM workload for reasoning-graph deep analysis and reflection "
            "(triage, draft, synthesis, reasoning, openrouter, "
            "local_fast, vllm, anthropic)"
        ),
    )
    graph_synthesis_workload: str = Field(
        default="triage",
        description=(
            "Pipeline LLM workload for reasoning-graph notification synthesis "
            "(triage, draft, synthesis, reasoning, openrouter, "
            "local_fast, vllm, anthropic)"
        ),
    )
    graph_openrouter_model: str = Field(
        default="",
        description=(
            "Optional OpenRouter model override for reasoning-graph deep "
            "analysis node. Empty = use the workload's default model. "
            "Triage and synthesis always use their workload defaults."
        ),
    )

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
    similar_trace_limit: int = Field(
        default=3,
        description="Maximum number of similar episodic traces to include as prior context for the legacy stratified engine",
    )

    # Multi-pass reasoning (classify -> challenge -> ground)
    multi_pass_enabled: bool = Field(
        default=False,
        description="Enable multi-pass reasoning (classify -> challenge -> ground). "
        "Claude Sonnet 4 handles classify+challenge+ground in a single pass; "
        "enable for GPT-5.1 or models that need multi-pass self-checking.",
    )
    multi_pass_verify_enabled: bool = Field(
        default=True,
        description="Run a deterministic evidence sufficiency verifier after classify before challenge/ground",
    )
    multi_pass_verify_min_reviews: int = Field(
        default=12,
        description="Cap confidence and add uncertainty when classify evidence has fewer than this many reviews",
    )
    multi_pass_verify_min_snapshot_days: int = Field(
        default=14,
        description="Cap confidence and add uncertainty when temporal depth is below this many days",
    )
    multi_pass_verify_min_grounded_signals: int = Field(
        default=2,
        description="Minimum number of grounded key signals expected from classify output before verifier marks evidence as thin",
    )
    multi_pass_verify_confidence_cap: float = Field(
        default=0.58,
        description="Maximum confidence allowed when verifier finds thin evidence coverage",
    )
    multi_pass_light_max_tokens: int = Field(
        default=4096,
        description="Max completion tokens for light multi-pass calls like challenge and ground",
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
        default=False,
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
    reconstitute_max_tokens: int = Field(
        default=4096,
        description="Max completion tokens for reconstitute classify/ground calls",
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
    reconstitute_weighted_sequence_match_threshold: float = Field(
        default=0.2,
        description="Maximum weighted diff ratio for normalized list-of-dict evidence to still count as unchanged",
    )
    reconstitute_strategic_component_threshold: float = Field(
        default=3.0,
        description="Minimum component drift score that forces full reasoning for strategic shifts like pain, role, competitive, or temporal changes",
    )
    reconstitute_contradiction_emergence_threshold: float = Field(
        default=4.0,
        description="Minimum contradiction-emergence score that forces full reasoning even when weighted drift stays below the reconstitute threshold",
    )
