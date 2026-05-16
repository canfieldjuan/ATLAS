import pytest

from extracted_content_pipeline.generation_plan import build_generation_plan_from_mapping
from extracted_content_pipeline.reasoning_policy import (
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    PACKAGED_REASONING_RUNTIME_PRESETS,
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


@pytest.mark.parametrize("output", PACKAGED_REASONING_RUNTIME_OUTPUTS)
@pytest.mark.parametrize("preset", PACKAGED_REASONING_RUNTIME_PRESETS)
def test_plan_threads_all_packaged_runtime_reasoning_presets(output, preset):
    inputs = {
        "report": {"opportunity_id": "opp_123"},
        "sales_brief": {"target_account": "Acme"},
    }[output]

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


@pytest.mark.parametrize("preset", ("single_pass", "multi_pass_light"))
def test_plan_rejects_runtime_unsupported_reasoning_preset_for_report(preset):
    with pytest.raises(ValueError, match="multi_pass_structured or multi_pass_strict"):
        build_generation_plan_from_mapping(
            {
                "outputs": ["report"],
                "reasoning_preset": preset,
                "inputs": {"opportunity_id": "opp_123"},
            }
        )


@pytest.mark.parametrize(
    ("output", "inputs", "preset"),
    (
        ("blog_post", {"topic": "Churn pressure"}, "multi_pass_structured"),
        (
            "landing_page",
            {"offer": "Audit", "audience": "RevOps"},
            "multi_pass_structured",
        ),
        (
            "email_campaign",
            {"target_account": "Acme", "offer": "Audit"},
            "single_pass",
        ),
    ),
)
def test_plan_rejects_runtime_reasoning_when_no_packaged_output_selected(
    output,
    inputs,
    preset,
):
    with pytest.raises(ValueError, match="only to report and sales_brief"):
        build_generation_plan_from_mapping(
            {
                "outputs": [output],
                "reasoning_preset": preset,
                "inputs": inputs,
            }
        )


def test_plan_ignores_non_runtime_outputs_when_packaged_output_selected():
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
    assert "reasoning_preset" not in configs["email_campaign"]
    assert configs["report"]["reasoning_preset"] == "multi_pass_structured"


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
