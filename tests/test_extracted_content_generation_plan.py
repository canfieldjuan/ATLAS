import pytest

from extracted_content_pipeline.generation_plan import build_generation_plan_from_mapping
from extracted_content_pipeline.output_variations import VARIANT_ANGLES
from extracted_content_pipeline.reasoning_policy import (
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    packaged_reasoning_runtime_presets_for_output,
)


def test_plan_maps_email_campaign_to_campaign_generation_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["email_campaign"],
            "limit": 2,
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "channels": ["email_cold", "email_followup"],
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["target_mode"] == "vendor_retention"
    assert plan["limit"] == 2
    assert plan["steps"] == [
        {
            "output": "email_campaign",
            "runner": "CampaignGenerationService.generate",
            "status": "runnable",
            "config": {
                "skill_name": "digest/b2b_campaign_generation",
                "channels": ["email_cold", "email_followup"],
                "limit": 2,
                "max_tokens": 1200,
                "temperature": 0.4,
                "quality_revalidation_enabled": True,
                "quality_prompt_proof_term_limit": 5,
                "parse_retry_attempts": 1,
                "parse_retry_response_excerpt_chars": 800,
            },
            "reason": "",
        }
    ]


def test_plan_maps_report_to_report_generation_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["report"],
            "limit": 3,
            "inputs": {
                "opportunity_id": "opp_123",
                "report_type": "competitive_pressure",
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "ReportGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "skill_name": "digest/report_generation",
        "default_report_type": "competitive_pressure",
        "limit": 3,
        "max_tokens": 4096,
        "temperature": 0.3,
        "quality_gates_enabled": True,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
    }


def test_plan_threads_brand_voice_profile_id_to_supported_copy_outputs():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": [
                "email_campaign",
                "blog_post",
                "landing_page",
                "sales_brief",
                "social_post",
            ],
            "brand_voice_profile_id": "acme-main",
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "topic": "Churn pressure",
                "audience": "B2B SaaS founders",
                "source_material": [{"review_text": "Pricing pressure."}],
            },
        }
    )

    configs = {step["output"]: step["config"] for step in plan["steps"]}
    assert configs["email_campaign"]["brand_voice_profile_id"] == "acme-main"
    assert configs["blog_post"]["brand_voice_profile_id"] == "acme-main"
    assert configs["landing_page"]["brand_voice_profile_id"] == "acme-main"
    assert configs["sales_brief"]["brand_voice_profile_id"] == "acme-main"
    assert configs["social_post"]["brand_voice_profile_id"] == "acme-main"


def test_plan_threads_structured_reasoning_preset_to_report_and_sales_brief():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["report", "sales_brief"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {
                "opportunity_id": "opp_123",
                "target_account": "Acme",
            },
        }
    )

    for step in plan["steps"]:
        assert step["config"]["reasoning_preset"] == "multi_pass_structured"
        assert step["config"]["reasoning_multi_pass"] is True
        assert step["config"]["reasoning_narrative_planning"] is True
        assert step["config"]["reasoning_output_validation"] is True
        assert step["config"]["reasoning_blocking_validation"] is False
        assert step["config"]["reasoning_falsification"] is False


def test_plan_threads_strict_reasoning_preset_to_report_and_sales_brief():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["report", "sales_brief"],
            "reasoning_preset": "multi_pass_strict",
            "inputs": {
                "opportunity_id": "opp_123",
                "target_account": "Acme",
            },
        }
    )

    for step in plan["steps"]:
        assert step["config"]["reasoning_preset"] == "multi_pass_strict"
        assert step["config"]["reasoning_multi_pass"] is True
        assert step["config"]["reasoning_narrative_planning"] is True
        assert step["config"]["reasoning_output_validation"] is True
        assert step["config"]["reasoning_blocking_validation"] is True
        assert step["config"]["reasoning_falsification"] is True


def test_plan_threads_structured_reasoning_preset_to_blog_post():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["blog_post"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {
                "topic": "Churn pressure",
            },
        }
    )

    step = plan["steps"][0]
    assert step["config"]["reasoning_preset"] == "multi_pass_structured"
    assert step["config"]["reasoning_multi_pass"] is True
    assert step["config"]["reasoning_narrative_planning"] is True
    assert step["config"]["reasoning_output_validation"] is True
    assert step["config"]["reasoning_blocking_validation"] is False
    assert step["config"]["reasoning_falsification"] is False


@pytest.mark.parametrize("output", PACKAGED_REASONING_RUNTIME_OUTPUTS)
def test_plan_threads_all_packaged_runtime_reasoning_presets(output):
    inputs = {
        "blog_post": {"topic": "Churn pressure"},
        "email_campaign": {"target_account": "Acme", "offer": "Audit"},
        "report": {"opportunity_id": "opp_123"},
        "landing_page": {"offer": "Audit", "audience": "RevOps"},
        "sales_brief": {"target_account": "Acme"},
    }[output]

    for preset in packaged_reasoning_runtime_presets_for_output(output):
        plan = build_generation_plan_from_mapping(
            {
                "outputs": [output],
                "reasoning_preset": preset,
                "inputs": inputs,
            }
        )

        step = plan["steps"][0]
        assert step["config"]["reasoning_preset"] == preset
        assert step["config"]["reasoning_multi_pass"] is True


def test_plan_rejects_unknown_reasoning_preset_for_report():
    with pytest.raises(ValueError, match="unknown reasoning preset"):
        build_generation_plan_from_mapping(
            {
                "outputs": ["report"],
                "reasoning_preset": "unsupported",
                "inputs": {"opportunity_id": "opp_123"},
            }
        )


@pytest.mark.parametrize(
    ("output", "inputs", "preset", "expected_match"),
    (
        (
            "report",
            {"opportunity_id": "opp_123"},
            "single_pass",
            "multi_pass_structured or multi_pass_strict",
        ),
        (
            "report",
            {"opportunity_id": "opp_123"},
            "multi_pass_light",
            "multi_pass_structured or multi_pass_strict",
        ),
        (
            "email_campaign",
            {"target_account": "Acme", "offer": "Audit"},
            "multi_pass_light",
            "multi_pass_structured for email_campaign, blog_post, and landing_page",
        ),
        (
            "landing_page",
            {"offer": "Audit", "audience": "RevOps"},
            "multi_pass_light",
            "multi_pass_structured for email_campaign, blog_post, and landing_page",
        ),
        (
            "blog_post",
            {"topic": "Churn pressure"},
            "multi_pass_strict",
            "not supported",
        ),
    ),
)
def test_plan_rejects_runtime_unsupported_reasoning_preset(
    output,
    inputs,
    preset,
    expected_match,
):
    with pytest.raises(ValueError, match=expected_match):
        build_generation_plan_from_mapping(
            {
                "outputs": [output],
                "reasoning_preset": preset,
                "inputs": inputs,
            }
        )


def test_plan_rejects_runtime_reasoning_when_no_packaged_output_selected():
    with pytest.raises(
        ValueError,
        match="only to email_campaign, blog_post, report, landing_page, and sales_brief",
    ):
        build_generation_plan_from_mapping(
            {
                "outputs": ["signal_extraction"],
                "reasoning_preset": "single_pass",
                "inputs": {
                    "source_material": "Pricing pressure came up at renewal.",
                },
            }
        )


def test_plan_threads_structured_reasoning_to_email_and_report():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["email_campaign", "report"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {
                "target_account": "Acme",
                "offer": "Audit",
                "opportunity_id": "opp_123",
            },
        }
    )

    configs = {step["output"]: step["config"] for step in plan["steps"]}
    assert configs["email_campaign"]["reasoning_preset"] == "multi_pass_structured"
    assert configs["email_campaign"]["reasoning_multi_pass"] is True
    assert configs["report"]["reasoning_preset"] == "multi_pass_structured"


def test_plan_threads_structured_reasoning_preset_to_landing_page():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["landing_page"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {
                "offer": "Audit",
                "audience": "RevOps",
            },
        }
    )

    step = plan["steps"][0]
    assert step["config"]["reasoning_preset"] == "multi_pass_structured"
    assert step["config"]["reasoning_multi_pass"] is True
    assert step["config"]["reasoning_narrative_planning"] is True
    assert step["config"]["reasoning_output_validation"] is True
    assert step["config"]["reasoning_blocking_validation"] is False
    assert step["config"]["reasoning_falsification"] is False


def test_plan_maps_blog_to_blog_generation_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {
                "topic": "Churn pressure",
            },
        }
    )

    assert plan["preview"]["can_run"] is True
    assert plan["can_execute"] is True
    assert plan["preview"]["blocked_outputs"] == []
    assert plan["steps"][0]["runner"] == "BlogPostGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "skill_name": "digest/blog_post_generation",
        "limit": 1,
        "max_tokens": 4096,
        "temperature": 0.3,
        "quality_gates_enabled": True,
        "quality_repair_attempts": 2,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
        "topic": "Churn pressure",
    }


def test_plan_includes_blog_variant_angle_metadata_when_requested():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["blog_post"],
            "variant_count": 2,
            "inputs": {
                "topic": "Churn pressure",
            },
        }
    )

    config = plan["steps"][0]["config"]
    assert config["variant_count"] == 2
    assert config["variant_angles"] == [
        VARIANT_ANGLES[0].as_dict(),
        VARIANT_ANGLES[1].as_dict(),
    ]


def test_plan_includes_landing_page_variant_angle_metadata_when_requested():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["landing_page"],
            "variant_count": 2,
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        }
    )

    config = plan["steps"][0]["config"]
    assert config["variant_count"] == 2
    assert config["variant_angles"] == [
        VARIANT_ANGLES[0].as_dict(),
        VARIANT_ANGLES[1].as_dict(),
    ]


def test_plan_includes_sales_brief_variant_angle_metadata_when_requested():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["sales_brief"],
            "variant_count": 2,
            "inputs": {
                "target_account": "Acme",
                "brief_type": "renewal",
            },
        }
    )

    config = plan["steps"][0]["config"]
    assert config["variant_count"] == 2
    assert config["variant_angles"] == [
        VARIANT_ANGLES[0].as_dict(),
        VARIANT_ANGLES[1].as_dict(),
    ]


def test_plan_stays_non_executable_when_preview_fails_budget():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["email_campaign"],
            "max_cost_usd": 0.01,
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
        }
    )

    assert plan["preview"]["can_run"] is False
    assert plan["can_execute"] is False
    assert plan["steps"][0]["status"] == "runnable"


def test_plan_maps_landing_page_to_landing_page_generation_service():
    plan = build_generation_plan_from_mapping(
        {
            "preset": "lead_gen_campaign",
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        }
    )

    assert plan["can_execute"] is True
    assert [step["output"] for step in plan["steps"]] == [
        "email_campaign",
        "landing_page",
    ]
    assert plan["steps"][1]["runner"] == "LandingPageGenerationService.generate"
    assert plan["steps"][1]["status"] == "runnable"
    assert plan["steps"][1]["config"] == {
        "skill_name": "digest/landing_page_generation",
        "max_tokens": 4096,
        "temperature": 0.3,
        "quality_gates_enabled": True,
        "quality_repair_attempts": 1,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
    }


def test_plan_threads_landing_page_quality_repair_attempt_override_zero():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": 0,
            },
        }
    )

    assert plan["steps"][0]["config"]["quality_repair_attempts"] == 0


def test_plan_threads_landing_page_quality_repair_attempt_override_string():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": "2",
            },
        }
    )

    assert plan["steps"][0]["config"]["quality_repair_attempts"] == 2


def test_plan_threads_landing_page_quality_repair_attempt_override_max():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": 10,
            },
        }
    )

    assert plan["steps"][0]["config"]["quality_repair_attempts"] == 10


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, "landing_page_quality_repair_attempts must be at least 0"),
        (11, "landing_page_quality_repair_attempts must be at most 10"),
        (True, "landing_page_quality_repair_attempts must be an integer"),
        (1.5, "landing_page_quality_repair_attempts must be an integer"),
        ("many", "landing_page_quality_repair_attempts must be an integer"),
    ],
)
def test_plan_rejects_invalid_landing_page_quality_repair_attempt_override(value, message):
    with pytest.raises(ValueError, match=message):
        build_generation_plan_from_mapping(
            {
                "outputs": ["landing_page"],
                "inputs": {
                    "offer": "Churn audit",
                    "audience": "B2B SaaS founders",
                    "landing_page_quality_repair_attempts": value,
                },
            }
        )


def test_plan_maps_sales_brief_to_sales_brief_generation_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["sales_brief"],
            "limit": 2,
            "inputs": {
                "target_account": "Acme",
                "brief_type": "renewal",
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "SalesBriefGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "skill_name": "digest/sales_brief_generation",
        "default_brief_type": "renewal",
        "limit": 2,
        "max_tokens": 4096,
        "temperature": 0.3,
        "quality_gates_enabled": True,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
    }


def test_plan_maps_signal_extraction_to_signal_extraction_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "limit": 3,
            "inputs": {
                "source_material": [
                    {
                        "id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "Pricing pressure came up at renewal.",
                    }
                ],
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "SignalExtractionService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "limit": 3,
        "max_text_chars": 1200,
    }


def test_plan_maps_social_post_to_social_post_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["social_post"],
            "limit": 3,
            "inputs": {
                "source_material": [
                    {
                        "review_id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "Pricing pressure came up at renewal.",
                    }
                ],
                "source_max_text_chars": 300,
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "SocialPostGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "skill_name": "digest/social_post_generation",
        "channels": ["linkedin"],
        "limit": 3,
        "max_text_chars": 300,
        "max_tokens": 700,
        "temperature": 0.4,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
    }


def test_plan_threads_social_post_channels_to_social_post_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["social_post"],
            "limit": 2,
            "inputs": {
                "social_channels": ["linkedin", "twitter"],
                "source_material": [
                    {
                        "review_id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "Pricing pressure came up at renewal.",
                    }
                ],
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "SocialPostGenerationService.generate"
    assert plan["steps"][0]["config"]["channels"] == ["linkedin", "x"]


def test_plan_maps_ad_copy_to_ad_copy_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["ad_copy"],
            "limit": 3,
            "inputs": {
                "source_material": [
                    {
                        "review_id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "Pricing pressure came up at renewal.",
                    }
                ],
                "source_max_text_chars": 300,
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "AdCopyGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "limit": 3,
        "max_text_chars": 300,
    }


def test_plan_maps_quote_card_to_quote_card_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["quote_card"],
            "limit": 3,
            "inputs": {
                "source_material": [
                    {
                        "review_id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "Pricing pressure came up at renewal.",
                    }
                ],
                "source_max_text_chars": 300,
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "QuoteCardGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "limit": 3,
        "max_text_chars": 300,
    }


def test_plan_maps_stat_card_to_stat_card_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["stat_card"],
            "limit": 3,
            "inputs": {
                "source_material": [
                    {
                        "review_id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "NPS score is 42 after renewal.",
                        "nps_score": 42,
                    }
                ],
                "source_max_text_chars": 300,
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "StatCardGenerationService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "limit": 3,
        "max_text_chars": 300,
    }


def test_plan_maps_faq_markdown_to_ticket_faq_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["faq_markdown"],
            "limit": 4,
            "inputs": {
                "source_material": [{"source_type": "ticket", "text": "How do I change my email?"}],
                "faq_title": "Support FAQ",
                "faq_max_evidence_per_item": 2,
                "faq_source_types": "ticket, support-ticket",
                "source_max_text_chars": 80,
                "faq_window_days": 90,
                "faq_as_of_date": "2026-05-20",
                "faq_support_contact": "1-800-555-0100",
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "TicketFAQMarkdownService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "title": "Support FAQ",
        "max_items": 4,
        "max_evidence_per_item": 2,
        "source_types": ["ticket", "support-ticket"],
        "max_text_chars": 80,
        "window_days": 90,
        "as_of_date": "2026-05-20",
        "support_contact": "1-800-555-0100",
    }


def test_plan_maps_faq_deflection_report_to_report_service():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["faq_deflection_report"],
            "limit": 4,
            "inputs": {
                "source_material": [
                    {"source_type": "ticket", "text": "How do I export a report?"}
                ],
                "deflection_report_title": "Customer FAQ Deflection Report",
                "faq_title": "Source FAQ",
                "faq_documentation_terms": ["Download report"],
                "faq_vocabulary_gap_rules": [["export", "download"]],
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["runner"] == "FAQDeflectionReportService.generate"
    assert plan["steps"][0]["status"] == "runnable"
    assert plan["steps"][0]["config"] == {
        "title": "Source FAQ",
        "max_items": 4,
        "max_evidence_per_item": 3,
        "source_types": [
            "ticket",
            "support_ticket",
            "case",
            "chat",
            "chat_transcript",
            "conversation",
            "transcript",
            "sales_call",
            "meeting",
            "sales_objection",
            "objection",
            "complaint",
            "search_log",
            "search_query",
        ],
        "max_text_chars": 1200,
        "documentation_terms": ["Download report"],
        "vocabulary_gap_rules": [["export", "download"]],
        "report_title": "Customer FAQ Deflection Report",
    }


def test_plan_threads_custom_faq_intent_rules_to_deflection_report():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["faq_deflection_report"],
            "limit": 4,
            "inputs": {
                "source_material": [
                    {
                        "source_type": "ticket",
                        "text": "The warehouse sync is delayed.",
                    }
                ],
                "faq_intent_rules": [
                    "data freshness=warehouse sync,connector lag"
                ],
            },
        }
    )

    rules = plan["steps"][0]["config"]["intent_rules"]
    assert rules[0] == {
        "topic": "data freshness",
        "keywords": ["warehouse sync", "connector lag"],
    }
    assert any(rule["topic"] == "integration setup" for rule in rules)


def test_plan_rejects_faq_as_of_date_without_window_days():
    with pytest.raises(ValueError, match="faq_as_of_date requires faq_window_days"):
        build_generation_plan_from_mapping(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "source_material": [{"source_type": "ticket", "text": "How do I change my email?"}],
                    "faq_as_of_date": "2026-05-20",
                },
            }
        )


def test_plan_rejects_invalid_faq_as_of_date():
    with pytest.raises(ValueError, match="faq_as_of_date must use YYYY-MM-DD format"):
        build_generation_plan_from_mapping(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "source_material": [{"source_type": "ticket", "text": "How do I change my email?"}],
                    "faq_window_days": 90,
                    "faq_as_of_date": "2026-05-20T00:00:00",
                },
            }
        )


def test_plan_threads_signal_extraction_source_text_cap():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_material": "Pricing pressure came up at renewal.",
                "source_max_text_chars": 12,
            },
        }
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["config"]["max_text_chars"] == 12


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "source_max_text_chars must be an integer"),
        (False, "source_max_text_chars must be an integer"),
        ("abc", "source_max_text_chars must be an integer"),
        (3.7, "source_max_text_chars must be an integer"),
        (-1, "source_max_text_chars must be at least 1"),
        (0, "source_max_text_chars must be at least 1"),
    ],
)
def test_plan_rejects_invalid_signal_extraction_source_text_cap(value, message):
    with pytest.raises(ValueError, match=message):
        build_generation_plan_from_mapping(
            {
                "outputs": ["signal_extraction"],
                "inputs": {
                    "source_material": "Pricing pressure came up at renewal.",
                    "source_max_text_chars": value,
                },
            }
        )
