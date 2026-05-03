"""Standalone settings for the LLM-infrastructure package.

A slim Pydantic Settings class carved out of ``atlas_brain.config`` that
exposes only the fields the scaffolded modules read at runtime.

Environment-variable layout mirrors atlas_brain so the same .env file
works in both modes:

  ATLAS_LLM_*               -> settings.llm.*
  ATLAS_B2B_CHURN_*         -> settings.b2b_churn.*
  (FTL tracing fields)      -> settings.ftl_tracing.*

Sub-configs:
  - ``LLMSubConfig`` (slim ``LLMConfig``)
  - ``B2BChurnSubConfig`` (only the openrouter + anthropic-batch fields)
  - ``ReasoningSubConfig`` (just ``model``)
  - ``ModelPricingConfig`` (per-model rates + ``cost_usd`` method)
  - ``FTLTracingSubConfig`` (FTL endpoint + pricing reference)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class ModelPricingConfig(BaseModel):
    """Per-model pricing in USD per 1M tokens.

    Mirrors ``atlas_brain.config.ModelPricingConfig`` exactly so cost
    calculations stay identical across delegate and standalone modes.
    """

    # Anthropic
    anthropic_sonnet_input: float = 3.00
    anthropic_sonnet_output: float = 15.00
    anthropic_sonnet_cache_read_input: float = 0.30
    anthropic_sonnet_cache_write_input: float = 3.75
    anthropic_haiku_input: float = 0.25
    anthropic_haiku_output: float = 1.25
    anthropic_haiku_cache_read_input: float = 0.03
    anthropic_haiku_cache_write_input: float = 0.30

    # Groq
    groq_llama70b_input: float = 0.59
    groq_llama70b_output: float = 0.79

    # OpenRouter
    openrouter_default_input: float = 1.10
    openrouter_default_output: float = 4.40

    # Together AI
    together_default_input: float = 0.88
    together_default_output: float = 0.88

    # Local models
    local_input: float = 0.0
    local_output: float = 0.0

    def cost_usd(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        billable_input_tokens: Optional[int] = None,
    ) -> float:
        p = (provider or "").lower()
        m = (model or "").lower()
        cache_read = max(int(cached_tokens or 0), 0)
        cache_write = max(int(cache_write_tokens or 0), 0)
        base_input = (
            max(int(billable_input_tokens), 0)
            if billable_input_tokens is not None
            else max(int(input_tokens or 0), 0)
        )
        if p in ("ollama", "vllm", "transformers-flash", "llama-cpp") or "local" in p:
            return 0.0
        if p == "anthropic" or "claude" in m:
            if "haiku" in m:
                return (
                    base_input * self.anthropic_haiku_input
                    + cache_read * self.anthropic_haiku_cache_read_input
                    + cache_write * self.anthropic_haiku_cache_write_input
                    + output_tokens * self.anthropic_haiku_output
                ) / 1_000_000
            return (
                base_input * self.anthropic_sonnet_input
                + cache_read * self.anthropic_sonnet_cache_read_input
                + cache_write * self.anthropic_sonnet_cache_write_input
                + output_tokens * self.anthropic_sonnet_output
            ) / 1_000_000
        if p == "groq":
            return (
                input_tokens * self.groq_llama70b_input
                + output_tokens * self.groq_llama70b_output
            ) / 1_000_000
        if p == "openrouter":
            return (
                input_tokens * self.openrouter_default_input
                + output_tokens * self.openrouter_default_output
            ) / 1_000_000
        if p in ("together", "cloud", "hybrid"):
            return (
                input_tokens * self.together_default_input
                + output_tokens * self.together_default_output
            ) / 1_000_000
        return 0.0


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


_DEFAULT_OPENROUTER_CLAUDE_SONNET = "anthropic/claude-sonnet-4-5"


class LLMSubConfig(BaseSettings):
    """Slim LLM configuration -- only the fields read by the
    extracted_llm_infrastructure scaffold's modules."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_LLM_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    # Ollama
    ollama_model: str = Field(default="qwen3:14b")
    ollama_url: str = Field(default="http://localhost:11434")
    ollama_timeout: int = Field(default=120)

    # vLLM
    vllm_model: str = Field(default="Qwen/Qwen3-14B-AWQ")
    vllm_url: str = Field(default="http://localhost:8082")
    vllm_guided_json_enabled: bool = Field(default=True)

    # Together AI
    together_model: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    together_api_key: Optional[str] = Field(default=None)

    # Groq
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_api_key: Optional[str] = Field(default=None)

    # OpenRouter (reasoning workloads)
    openrouter_reasoning_model: str = Field(default=_DEFAULT_OPENROUTER_CLAUDE_SONNET)
    openrouter_reasoning_strict: bool = Field(default=False)

    # Anthropic
    anthropic_model: str = Field(default="claude-haiku-4-5")
    anthropic_api_key: Optional[str] = Field(default=None)

    # Cloud (Ollama cloud-relay model)
    cloud_enabled: bool = Field(default=False)
    cloud_ollama_model: str = Field(default="minimax-m2:cloud")


class B2BChurnSubConfig(BaseSettings):
    """Slim B2B churn config -- only the LLM-batch and OpenRouter fields
    the scaffold reads at runtime. Atlas's full ``B2BChurnConfig`` has
    ~80 unrelated fields (billing, scrape, calibration, etc.) that the
    LLM-infra subsystem does not touch."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_CHURN_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    openrouter_api_key: str = Field(default="")
    anthropic_batch_enabled: bool = Field(default=False)
    anthropic_batch_poll_interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    anthropic_batch_timeout_seconds: float = Field(default=900.0, ge=30.0, le=86400.0)
    anthropic_batch_min_items: int = Field(default=2, ge=1, le=10000)
    llm_exact_cache_enabled: bool = Field(default=False)


class ReasoningSubConfig(BaseSettings):
    """Slim reasoning config -- just the model name read by
    services/llm_router.py."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REASONING_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    model: str = Field(default="claude-sonnet-4-5")


class FTLTracingSubConfig(BaseModel):
    """FTL tracing configuration. Fields match atlas_brain.config.FTLTracingConfig
    so the same env vars work in both modes."""

    enabled: bool = True
    base_url: str = "https://finetunelab.ai"
    api_key: str = ""
    user_id: str = ""
    capture_business_context: bool = True
    capture_reasoning_summaries: bool = True
    capture_raw_reasoning: bool = False
    max_reasoning_chars: int = Field(default=1200, ge=0, le=10000)
    pricing: ModelPricingConfig = Field(default_factory=ModelPricingConfig)


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class LLMInfraSettings(BaseModel):
    """Top-level settings exposed as the ``settings`` global.

    The shape mirrors atlas_brain's ``Settings`` for the fields this
    package reads, so the scaffold's call sites work without changes
    when the standalone toggle is on.
    """

    llm: LLMSubConfig = Field(default_factory=LLMSubConfig)
    b2b_churn: B2BChurnSubConfig = Field(default_factory=B2BChurnSubConfig)
    reasoning: ReasoningSubConfig = Field(default_factory=ReasoningSubConfig)
    ftl_tracing: FTLTracingSubConfig = Field(default_factory=FTLTracingSubConfig)


settings = LLMInfraSettings()
