from extracted_content_pipeline.generation_plan import build_generation_plan_from_mapping


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
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
        "topic": "Churn pressure",
    }


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
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
    }


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
