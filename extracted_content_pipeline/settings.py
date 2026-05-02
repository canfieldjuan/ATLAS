from __future__ import annotations

import os
from types import SimpleNamespace


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def build_settings() -> SimpleNamespace:
    external_data = SimpleNamespace(
        enabled=_to_bool(os.getenv("EXTRACTED_EXTERNAL_ENABLED"), True),
        enrichment_enabled=_to_bool(os.getenv("EXTRACTED_ENRICHMENT_ENABLED"), True),
        complaint_mining_enabled=_to_bool(os.getenv("EXTRACTED_COMPLAINT_MINING_ENABLED"), True),
        blog_post_enabled=_to_bool(os.getenv("EXTRACTED_BLOG_POST_ENABLED"), True),
        complaint_content_enabled=_to_bool(os.getenv("EXTRACTED_COMPLAINT_CONTENT_ENABLED"), True),
        blog_post_max_per_run=_to_int(os.getenv("EXTRACTED_BLOG_POST_MAX_PER_RUN"), 1),
        blog_post_max_tokens=_to_int(os.getenv("EXTRACTED_BLOG_POST_MAX_TOKENS"), 2800),
        complaint_content_max_per_run=_to_int(os.getenv("EXTRACTED_COMPLAINT_CONTENT_MAX_PER_RUN"), 10),
        complaint_content_max_tokens=_to_int(os.getenv("EXTRACTED_COMPLAINT_CONTENT_MAX_TOKENS"), 1200),
        blog_base_url=os.getenv("EXTRACTED_CONSUMER_BLOG_BASE_URL")
        or os.getenv("EXTRACTED_BLOG_BASE_URL")
        or "https://example.com",
    )

    b2b_churn = SimpleNamespace(
        blog_post_enabled=_to_bool(os.getenv("EXTRACTED_B2B_BLOG_POST_ENABLED"), True),
        blog_post_max_per_run=_to_int(os.getenv("EXTRACTED_B2B_BLOG_POST_MAX_PER_RUN"), 1),
        blog_post_max_tokens=_to_int(os.getenv("EXTRACTED_B2B_BLOG_POST_MAX_TOKENS"), 3200),
        blog_post_temperature=_to_float(os.getenv("EXTRACTED_B2B_BLOG_POST_TEMPERATURE"), 0.2),
        blog_base_url=os.getenv("EXTRACTED_B2B_BLOG_BASE_URL")
        or os.getenv("EXTRACTED_BLOG_BASE_URL")
        or "https://example.com",
    )

    campaign_llm = SimpleNamespace(
        workload=os.getenv("EXTRACTED_CAMPAIGN_LLM_WORKLOAD") or "draft",
        prefer_cloud=_to_bool(os.getenv("EXTRACTED_CAMPAIGN_LLM_PREFER_CLOUD"), True),
        try_openrouter=_to_bool(os.getenv("EXTRACTED_CAMPAIGN_LLM_TRY_OPENROUTER"), True),
        auto_activate_ollama=_to_bool(
            os.getenv("EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA"),
            True,
        ),
        openrouter_model=os.getenv("EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL") or None,
    )

    return SimpleNamespace(
        external_data=external_data,
        b2b_churn=b2b_churn,
        campaign_llm=campaign_llm,
    )
