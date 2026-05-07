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
    assert plan["target_mode"] == "b2b"
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
    }


def test_plan_marks_blog_as_planned_not_executable_until_service_adapter_exists():
    plan = build_generation_plan_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {
                "topic": "Churn pressure",
            },
        }
    )

    assert plan["preview"]["can_run"] is True
    assert plan["can_execute"] is False
    assert plan["steps"][0]["output"] == "blog_post"
    assert plan["steps"][0]["status"] == "planned"
    assert plan["steps"][0]["runner"] == "extracted_content_pipeline.autonomous.tasks.blog_post_generation"
    assert "does not yet expose the same service/port interface" in plan["steps"][0]["reason"]


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


def test_plan_omits_blocked_future_outputs_from_steps():
    plan = build_generation_plan_from_mapping(
        {
            "preset": "lead_gen_campaign",
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
        }
    )

    assert plan["preview"]["blocked_outputs"] == ["landing_page"]
    assert plan["steps"] == [
        {
            "output": "email_campaign",
            "runner": "CampaignGenerationService.generate",
            "status": "runnable",
            "config": {
                "skill_name": "digest/b2b_campaign_generation",
                "channels": ["email_cold", "email_followup"],
                "limit": 1,
                "max_tokens": 1200,
                "temperature": 0.4,
                "quality_revalidation_enabled": True,
                "quality_prompt_proof_term_limit": 5,
            },
            "reason": "",
        }
    ]
    assert plan["can_execute"] is False
