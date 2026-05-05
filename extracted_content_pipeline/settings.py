from __future__ import annotations

import os
import secrets
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
        intelligence_window_days=_to_int(os.getenv("EXTRACTED_B2B_INTELLIGENCE_WINDOW_DAYS"), 30),
        openrouter_api_key=os.getenv("EXTRACTED_OPENROUTER_API_KEY") or "",
        briefing_analyst_model=os.getenv("EXTRACTED_VENDOR_BRIEFING_ANALYST_MODEL")
        or "openai/gpt-4o-mini",
        vendor_briefing_enabled=_to_bool(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_ENABLED"),
            True,
        ),
        vendor_briefing_sender_name=os.getenv("EXTRACTED_VENDOR_BRIEFING_SENDER_NAME")
        or "Atlas",
        vendor_briefing_standard_churn_subject_template=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_STANDARD_CHURN_SUBJECT_TEMPLATE"
        )
        or "Churn Intelligence Briefing: {vendor_name}",
        vendor_briefing_standard_sales_subject_template=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_STANDARD_SALES_SUBJECT_TEMPLATE"
        )
        or "Sales Intelligence Briefing: {vendor_name}",
        vendor_briefing_prospect_churn_subject_template=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_PROSPECT_CHURN_SUBJECT_TEMPLATE"
        )
        or "{vendor_name} -- Churn Signals Detected",
        vendor_briefing_prospect_sales_subject_template=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_PROSPECT_SALES_SUBJECT_TEMPLATE"
        )
        or "{vendor_name} -- Accounts In Motion",
        vendor_briefing_gated_churn_subject_template=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_GATED_CHURN_SUBJECT_TEMPLATE"
        )
        or "Your {vendor_name} Churn Intelligence Report",
        vendor_briefing_gated_sales_subject_template=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_GATED_SALES_SUBJECT_TEMPLATE"
        )
        or "Your {vendor_name} Sales Intelligence Report",
        vendor_briefing_tag_type_name=os.getenv("EXTRACTED_VENDOR_BRIEFING_TAG_TYPE_NAME")
        or "type",
        vendor_briefing_tag_type_value=os.getenv("EXTRACTED_VENDOR_BRIEFING_TAG_TYPE_VALUE")
        or "vendor_briefing",
        vendor_briefing_tag_vendor_name=os.getenv(
            "EXTRACTED_VENDOR_BRIEFING_TAG_VENDOR_NAME"
        )
        or "vendor",
        vendor_briefing_gate_base_url=os.getenv("EXTRACTED_VENDOR_BRIEFING_GATE_BASE_URL")
        or "https://example.com/vendor-briefing",
        vendor_briefing_gate_expiry_days=_to_int(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_GATE_EXPIRY_DAYS"),
            14,
        ),
        vendor_briefing_max_per_batch=_to_int(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_MAX_PER_BATCH"),
            25,
        ),
        vendor_briefing_cooldown_days=_to_int(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_COOLDOWN_DAYS"),
            7,
        ),
        vendor_briefing_account_cards_enabled=_to_bool(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_ENABLED"),
            True,
        ),
        vendor_briefing_account_cards_max=_to_int(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_MAX"),
            3,
        ),
        vendor_briefing_account_cards_reasoning_depth=_to_int(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_REASONING_DEPTH"),
            2,
        ),
        vendor_briefing_account_cards_adaptive_depth=_to_bool(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_ADAPTIVE_DEPTH"),
            True,
        ),
        vendor_briefing_scheduled_analyst_enrichment_enabled=_to_bool(
            os.getenv("EXTRACTED_VENDOR_BRIEFING_ANALYST_ENRICHMENT_ENABLED"),
            False,
        ),
        vendor_briefing_scheduled_account_cards_reasoning_depth=_to_int(
            os.getenv(
                "EXTRACTED_VENDOR_BRIEFING_SCHEDULED_ACCOUNT_CARDS_REASONING_DEPTH"
            ),
            0,
        ),
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

    b2b_campaign = SimpleNamespace(
        max_tokens=_to_int(os.getenv("EXTRACTED_B2B_CAMPAIGN_MAX_TOKENS"), 1400),
        temperature=_to_float(os.getenv("EXTRACTED_B2B_CAMPAIGN_TEMPERATURE"), 0.2),
        llm_timeout_seconds=_to_float(
            os.getenv("EXTRACTED_B2B_CAMPAIGN_LLM_TIMEOUT_SECONDS"),
            120.0,
        ),
        anthropic_batch_detached_enabled=_to_bool(
            os.getenv("EXTRACTED_B2B_CAMPAIGN_ANTHROPIC_BATCH_DETACHED_ENABLED"),
            False,
        ),
        specificity_min_anchor_hits=_to_int(
            os.getenv("EXTRACTED_B2B_CAMPAIGN_SPECIFICITY_MIN_ANCHOR_HITS"),
            1,
        ),
        specificity_require_anchor_support=_to_bool(
            os.getenv("EXTRACTED_B2B_CAMPAIGN_SPECIFICITY_REQUIRE_ANCHOR_SUPPORT"),
            True,
        ),
        specificity_require_timing_or_numeric_when_available=_to_bool(
            os.getenv(
                "EXTRACTED_B2B_CAMPAIGN_SPECIFICITY_REQUIRE_TIMING_OR_NUMERIC"
            ),
            True,
        ),
        specificity_revision_term_limit=_to_int(
            os.getenv("EXTRACTED_B2B_CAMPAIGN_SPECIFICITY_REVISION_TERM_LIMIT"),
            3,
        ),
        word_limits={
            "default": {
                "email_cold": [50, 150],
                "email_followup": [75, 150],
                "linkedin": [0, 100],
            },
            "vendor_retention": {
                "email_cold": [50, 125],
                "email_followup": [75, 150],
            },
            "challenger_intel": {
                "email_cold": [50, 125],
                "email_followup": [75, 150],
            },
            "churning_company": {
                "email_cold": [75, 150],
                "email_followup": [75, 125],
                "linkedin": [0, 100],
            },
        },
    )

    campaign_sender_type = (
        os.getenv("EXTRACTED_CAMPAIGN_SEQUENCE_SENDER_TYPE")
        or os.getenv("EXTRACTED_CAMPAIGN_SENDER_TYPE")
        or "resend"
    )
    ses_from_email = os.getenv("EXTRACTED_SES_FROM_EMAIL") or ""
    resend_from_email = (
        os.getenv("EXTRACTED_RESEND_FROM_EMAIL")
        or os.getenv("EXTRACTED_CAMPAIGN_RESEND_FROM_EMAIL")
        or os.getenv("EXTRACTED_CAMPAIGN_SEQ_RESEND_FROM_EMAIL")
        or (ses_from_email if campaign_sender_type.lower() == "ses" else "")
        or ""
    )
    resend_api_key = (
        os.getenv("EXTRACTED_RESEND_API_KEY")
        or os.getenv("EXTRACTED_CAMPAIGN_RESEND_API_KEY")
        or os.getenv("EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY")
        or ("ses-configured" if campaign_sender_type.lower() == "ses" and ses_from_email else "")
        or ""
    )

    campaign_sequence = SimpleNamespace(
        enabled=_to_bool(os.getenv("EXTRACTED_CAMPAIGN_SEQUENCE_ENABLED"), True),
        sender_type=campaign_sender_type,
        resend_api_key=resend_api_key,
        resend_from_email=resend_from_email,
        resend_api_url=os.getenv("EXTRACTED_RESEND_API_URL") or "https://api.resend.com/emails",
        sender_timeout_seconds=_to_float(
            os.getenv("EXTRACTED_CAMPAIGN_SENDER_TIMEOUT_SECONDS")
            or os.getenv("EXTRACTED_CAMPAIGN_SEQ_SENDER_TIMEOUT_SECONDS"),
            30.0,
        ),
        ses_region=os.getenv("EXTRACTED_SES_REGION") or "us-east-1",
        ses_access_key_id=os.getenv("EXTRACTED_SES_ACCESS_KEY_ID") or "",
        ses_secret_access_key=os.getenv("EXTRACTED_SES_SECRET_ACCESS_KEY") or "",
        ses_configuration_set=os.getenv("EXTRACTED_SES_CONFIGURATION_SET") or "",
        ses_from_email=ses_from_email,
    )

    saas_auth = SimpleNamespace(
        jwt_secret=os.getenv("EXTRACTED_VENDOR_BRIEFING_JWT_SECRET")
        or secrets.token_urlsafe(32),
        jwt_algorithm=os.getenv("EXTRACTED_VENDOR_BRIEFING_JWT_ALGORITHM") or "HS256",
    )

    return SimpleNamespace(
        external_data=external_data,
        b2b_churn=b2b_churn,
        campaign_llm=campaign_llm,
        b2b_campaign=b2b_campaign,
        campaign_sequence=campaign_sequence,
        saas_auth=saas_auth,
    )
