"""Standalone settings for extracted competitive intelligence."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class B2BChurnSubConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_CHURN_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    enabled: bool = True
    intelligence_enabled: bool = True
    intelligence_window_days: int = Field(default=365, ge=1, le=3650)
    intelligence_min_reviews: int = Field(default=3, ge=0, le=10000)
    intelligence_infra_blocked_sources: str = ""
    mcp_tool_groups: str = "all"
    competitive_set_preview_lookback_days: int = Field(default=90, ge=1, le=3650)
    reasoning_synthesis_max_stale_days: int = Field(default=3, ge=0, le=3650)
    reasoning_synthesis_rerun_if_missing_packet_artifacts: bool = True
    reasoning_synthesis_rerun_if_missing_reference_ids: bool = True

    openrouter_api_key: str = ""
    anthropic_batch_enabled: bool = False
    briefing_analyst_model: str = "anthropic/claude-sonnet-4-5"

    vendor_briefing_enabled: bool = True
    vendor_briefing_sender_name: str = "Atlas Intelligence"
    vendor_briefing_gate_base_url: str = ""
    vendor_briefing_gate_expiry_days: int = Field(default=7, ge=1, le=90)
    vendor_briefing_cooldown_days: int = Field(default=30, ge=0, le=365)
    vendor_briefing_max_per_batch: int = Field(default=25, ge=1, le=1000)
    vendor_briefing_account_cards_enabled: bool = True
    vendor_briefing_account_cards_max: int = Field(default=5, ge=0, le=100)
    vendor_briefing_account_cards_reasoning_depth: str = "standard"
    vendor_briefing_account_cards_adaptive_depth: bool = True
    vendor_briefing_scheduled_account_cards_reasoning_depth: str = "standard"
    vendor_briefing_scheduled_analyst_enrichment_enabled: bool = False

    battle_card_cache_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    battle_card_llm_attempts: int = Field(default=2, ge=0, le=10)
    battle_card_llm_concurrency: int = Field(default=4, ge=1, le=64)
    battle_card_llm_feedback_limit: int = Field(default=20, ge=1, le=1000)
    battle_card_llm_max_tokens: int = Field(default=4000, ge=1, le=200000)
    battle_card_llm_retry_delay_seconds: float = Field(default=2.0, ge=0.0, le=300.0)
    battle_card_llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    battle_card_llm_timeout_seconds: float = Field(default=120.0, ge=1.0, le=3600.0)
    feature_gap_min_mentions: int = Field(default=2, ge=1, le=1000)
    quotable_phrase_min_urgency: int = Field(default=60, ge=0, le=100)
    reasoning_witness_highlight_limit: int = Field(default=8, ge=0, le=100)

    challenger_brief_quote_candidate_limit: int = Field(default=30, ge=1, le=1000)
    challenger_brief_quote_fallback_limit: int = Field(default=10, ge=0, le=1000)
    challenger_brief_quote_similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    challenger_brief_report_fallback_days: int = Field(default=30, ge=1, le=3650)
    accounts_in_motion_invalid_alternative_terms: str = ""


class LLMSubConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_LLM_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    openrouter_reasoning_model: str = "anthropic/claude-sonnet-4-5"


class CampaignSequenceSubConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CAMPAIGN_SEQUENCE_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    resend_api_key: str = ""
    resend_from_email: str = ""


class SaasAuthSubConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_SAAS_AUTH_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    stripe_secret_key: str = ""
    stripe_vendor_standard_price_id: str = ""
    stripe_vendor_pro_price_id: str = ""


class B2BScrapeSubConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_SCRAPE_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    deferred_inventory_sources: str = ""


class MCPSubConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_MCP_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    host: str = "127.0.0.1"
    b2b_churn_port: int = Field(default=8062, ge=1, le=65535)


class CompIntelSettings(BaseModel):
    b2b_churn: B2BChurnSubConfig = Field(default_factory=B2BChurnSubConfig)
    b2b_scrape: B2BScrapeSubConfig = Field(default_factory=B2BScrapeSubConfig)
    campaign_sequence: CampaignSequenceSubConfig = Field(
        default_factory=CampaignSequenceSubConfig
    )
    llm: LLMSubConfig = Field(default_factory=LLMSubConfig)
    mcp: MCPSubConfig = Field(default_factory=MCPSubConfig)
    saas_auth: SaasAuthSubConfig = Field(default_factory=SaasAuthSubConfig)


settings = CompIntelSettings()
