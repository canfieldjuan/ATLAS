import pytest

from extracted_content_pipeline.control_surfaces import (
    PRESETS,
    evaluate_usage_budget,
    normalize_outputs,
    preview_from_mapping,
    request_from_mapping,
)


def test_normalize_outputs_accepts_csv_and_dedupes():
    assert normalize_outputs("email-campaign, blog_post, email_campaign") == (
        "email_campaign",
        "blog_post",
    )


def test_preview_defaults_to_email_only_and_reports_missing_inputs():
    preview = preview_from_mapping({})

    assert preview["can_run"] is False
    assert preview["outputs"] == ["email_campaign"]
    assert preview["missing_inputs"] == ["target_account", "offer"]
    assert preview["estimated_cost_usd"] == 0.36


def test_preview_allows_implemented_outputs_under_budget():
    preview = preview_from_mapping(
        {
            "outputs": ["email_campaign", "report"],
            "limit": 2,
            "max_cost_usd": 3.0,
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn intelligence audit",
                "opportunity_id": "opp_123",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["email_campaign", "report"]
    assert preview["estimated_cost_usd"] == 2.92
    assert preview["missing_inputs"] == []
    assert preview["blocked_outputs"] == []
    assert preview["normalized_request"]["account_usage_budget_usd"] is None
    assert preview["normalized_request"]["account_usage_budget_days"] == 7
    assert preview["normalized_request"]["content_ops_cache_policy"] is None


def test_request_normalizes_content_ops_cache_policy():
    request = request_from_mapping({
        "outputs": ["blog_post"],
        "content_ops_cache_policy": "exact-cache",
        "inputs": {"topic": "Support ticket questions"},
    })
    preview = preview_from_mapping({
        "outputs": ["blog_post"],
        "content_ops_cache_policy": "exact-cache",
        "inputs": {"topic": "Support ticket questions"},
    })

    assert request.content_ops_cache_policy == "exact"
    assert preview["normalized_request"]["content_ops_cache_policy"] == "exact"


def test_request_rejects_unsupported_content_ops_cache_policy():
    with pytest.raises(ValueError, match="unsupported content_ops_cache_policy"):
        request_from_mapping({
            "outputs": ["blog_post"],
            "content_ops_cache_policy": "semantic",
            "inputs": {"topic": "Support ticket questions"},
        })


def test_preview_blocks_unknown_preset_instead_of_falling_back_to_email():
    preview = preview_from_mapping(
        {
            "preset": "contmarket",
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn intelligence audit",
            },
        }
    )

    assert preview["can_run"] is False
    assert preview["outputs"] == []
    assert preview["blocked_outputs"] == ["contmarket"]
    assert "Unknown preset: contmarket" in preview["warnings"]


def test_preview_warns_when_outputs_override_preset():
    preview = preview_from_mapping(
        {
            "preset": "full_campaign",
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn intelligence audit",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["email_campaign"]
    assert (
        "Preset ignored because explicit outputs were provided: full_campaign"
        in preview["warnings"]
    )


def test_marketer_evidence_bundle_preset_selects_review_competitive_outputs():
    preset = PRESETS["marketer_evidence_bundle"]
    preview = preview_from_mapping(
        {
            "preset": preset.id,
            "inputs": {
                "audience": "SaaS operators",
                "offer": "Review-backed churn audit",
                "target_account": "Acme",
                "topic": "Competitor switching signals",
            },
        }
    )

    assert preset.outputs == ("landing_page", "blog_post", "sales_brief")
    assert preview["can_run"] is True
    assert preview["outputs"] == ["landing_page", "blog_post", "sales_brief"]
    assert preview["estimated_cost_usd"] == 4.2
    assert preview["missing_inputs"] == []


def test_missing_required_inputs_treats_empty_tuple_as_missing():
    preview = preview_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {
                "topic": (),
            },
        }
    )

    assert preview["can_run"] is False
    assert preview["missing_inputs"] == ["topic"]


def test_preview_allows_signal_extraction_when_source_material_present():
    preview = preview_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_material": "review export",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["signal_extraction"]
    assert preview["blocked_outputs"] == []
    assert preview["estimated_cost_usd"] == 0.0


def test_preview_allows_social_post_when_source_material_present():
    preview = preview_from_mapping(
        {
            "outputs": ["social_post"],
            "limit": 3,
            "inputs": {
                "source_material": [{"source_type": "review", "text": "Pricing pressure"}],
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["social_post"]
    assert preview["missing_inputs"] == []
    assert preview["blocked_outputs"] == []
    assert preview["estimated_cost_usd"] == 0.0


def test_preview_allows_ad_copy_when_source_material_present():
    preview = preview_from_mapping(
        {
            "outputs": ["ad_copy"],
            "limit": 3,
            "inputs": {
                "source_material": [{"source_type": "review", "text": "Pricing pressure"}],
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["ad_copy"]
    assert preview["missing_inputs"] == []
    assert preview["blocked_outputs"] == []
    assert preview["estimated_cost_usd"] == 0.0


def test_preview_allows_faq_markdown_when_source_material_present():
    preview = preview_from_mapping(
        {
            "outputs": ["faq_markdown"],
            "limit": 3,
            "inputs": {
                "source_material": [{"source_type": "ticket", "text": "How do I change my email?"}],
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["faq_markdown"]
    assert preview["blocked_outputs"] == []
    assert preview["estimated_cost_usd"] == 0.0


def test_preview_blocks_when_estimate_exceeds_budget():
    preview = preview_from_mapping(
            {
                "outputs": ["report"],
                "limit": 2,
                "max_cost_usd": 0.25,
                "inputs": {
                    "opportunity_id": "opp_123",
                },
            }
        )

    assert preview["can_run"] is False
    assert "Estimated cost exceeds max_cost_usd: 2.20 > 0.25" in preview["warnings"]


def test_preview_budget_gate_uses_retry_adjusted_cost():
    preview = preview_from_mapping(
        {
            "outputs": ["report"],
            "limit": 2,
            "max_cost_usd": 1.10,
            "inputs": {
                "opportunity_id": "opp_123",
            },
        }
    )

    assert preview["can_run"] is False
    assert preview["estimated_cost_usd"] == 2.2
    assert "Estimated cost exceeds max_cost_usd: 2.20 > 1.10" in preview["warnings"]


def test_preview_landing_page_cost_includes_default_quality_repair_attempt():
    preview = preview_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["estimated_cost_usd"] == 2.6


def test_preview_landing_page_cost_uses_quality_repair_attempt_override_zero():
    preview = preview_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": 0,
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["estimated_cost_usd"] == 1.3


def test_preview_landing_page_cost_uses_quality_repair_attempt_override_string():
    preview = preview_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": "3",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["estimated_cost_usd"] == 5.2


def test_preview_landing_page_cost_accepts_quality_repair_attempt_override_max():
    preview = preview_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": 10,
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["estimated_cost_usd"] == 14.3


def test_preview_validates_landing_page_repair_attempt_even_when_gates_disabled():
    with pytest.raises(
        ValueError,
        match="landing_page_quality_repair_attempts must be at most 10",
    ):
        preview_from_mapping(
            {
                "outputs": ["landing_page"],
                "require_quality_gates": False,
                "inputs": {
                    "offer": "Churn audit",
                    "audience": "B2B SaaS founders",
                    "landing_page_quality_repair_attempts": 11,
                },
            }
        )


def test_preview_landing_page_repair_cost_can_block_budget():
    preview = preview_from_mapping(
        {
            "outputs": ["landing_page"],
            "max_cost_usd": 2.5,
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        }
    )

    assert preview["can_run"] is False
    assert preview["estimated_cost_usd"] == 2.6
    assert "Estimated cost exceeds max_cost_usd: 2.60 > 2.50" in preview["warnings"]


def test_preview_landing_page_cost_ignores_quality_repair_when_gates_disabled():
    preview = preview_from_mapping(
        {
            "outputs": ["landing_page"],
            "require_quality_gates": False,
            "inputs": {
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "landing_page_quality_repair_attempts": 3,
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["estimated_cost_usd"] == 1.3


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
def test_preview_rejects_invalid_landing_page_quality_repair_attempt_override(
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        preview_from_mapping(
            {
                "outputs": ["landing_page"],
                "inputs": {
                    "offer": "Churn audit",
                    "audience": "B2B SaaS founders",
                    "landing_page_quality_repair_attempts": value,
                },
            }
        )


def test_preview_keeps_signal_extraction_blocked_without_source_material():
    preview = preview_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {},
        }
    )

    assert preview["can_run"] is False
    assert preview["outputs"] == ["signal_extraction"]
    assert preview["missing_inputs"] == ["source_material"]


def test_preview_keeps_social_post_blocked_without_source_material():
    preview = preview_from_mapping(
        {
            "outputs": ["social_post"],
            "inputs": {},
        }
    )

    assert preview["can_run"] is False
    assert preview["outputs"] == ["social_post"]
    assert preview["missing_inputs"] == ["source_material"]


def test_preview_keeps_ad_copy_blocked_without_source_material():
    preview = preview_from_mapping(
        {
            "outputs": ["ad_copy"],
            "inputs": {},
        }
    )

    assert preview["can_run"] is False
    assert preview["outputs"] == ["ad_copy"]
    assert preview["missing_inputs"] == ["source_material"]


def test_output_catalog_states_reasoning_requirement():
    from extracted_content_pipeline.control_surfaces import OUTPUT_CATALOG

    assert OUTPUT_CATALOG["email_campaign"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["report"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["landing_page"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["sales_brief"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["blog_post"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["social_post"].reasoning_requirement == "absent"
    assert OUTPUT_CATALOG["ad_copy"].reasoning_requirement == "absent"
    assert OUTPUT_CATALOG["signal_extraction"].reasoning_requirement == "absent"
    assert OUTPUT_CATALOG["faq_markdown"].reasoning_requirement == "absent"


def test_request_from_mapping_rejects_zero_limit():
    with pytest.raises(ValueError, match="limit must be at least 1; got 0"):
        request_from_mapping({"limit": 0})


def test_request_from_mapping_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="max_cost_usd must be positive; got -1"):
        request_from_mapping({"max_cost_usd": -1})


def test_request_from_mapping_rejects_non_positive_account_usage_budget():
    with pytest.raises(
        ValueError,
        match="account_usage_budget_usd must be positive; got 0",
    ):
        request_from_mapping({"account_usage_budget_usd": 0})


def test_request_from_mapping_rejects_account_usage_budget_days_out_of_range():
    with pytest.raises(
        ValueError,
        match="account_usage_budget_days must be between 1 and 90; got 91",
    ):
        request_from_mapping({"account_usage_budget_days": 91})


def test_evaluate_usage_budget_projects_current_plus_estimated_cost():
    evaluation = evaluate_usage_budget(
        budget_usd=2.0,
        period_days=7,
        current_cost_usd=1.2,
        estimated_cost_usd=0.9,
    )

    assert evaluation.exceeded is True
    assert evaluation.as_dict() == {
        "budget_usd": 2.0,
        "period_days": 7,
        "current_cost_usd": 1.2,
        "estimated_cost_usd": 0.9,
        "projected_cost_usd": 2.1,
        "exceeded": True,
    }


def test_request_from_mapping_rejects_non_object_inputs():
    with pytest.raises(ValueError, match="inputs must be an object"):
        request_from_mapping({"inputs": ["target_account", "Acme"]})


# -----------------------
# PR-Audit-MINOR-Batch-1: OUTPUT_CATALOG / PRESETS are immutable
# -----------------------


def test_output_catalog_rejects_assignment():
    """PR-Audit-MINOR-Batch-1 NIT: OUTPUT_CATALOG was a mutable
    module-level dict. Anything in the process could mutate it. Now
    wrapped in MappingProxyType -- assignment raises TypeError."""
    from extracted_content_pipeline.control_surfaces import OUTPUT_CATALOG

    # Reads still work.
    assert "email_campaign" in OUTPUT_CATALOG
    assert OUTPUT_CATALOG["email_campaign"].id == "email_campaign"

    # Assignment is rejected.
    with pytest.raises(TypeError):
        OUTPUT_CATALOG["new_output"] = OUTPUT_CATALOG["email_campaign"]  # type: ignore[index]


def test_presets_rejects_assignment():
    from extracted_content_pipeline.control_surfaces import PRESETS

    assert "email_only" in PRESETS
    assert PRESETS["email_only"].id == "email_only"

    with pytest.raises(TypeError):
        PRESETS["new_preset"] = PRESETS["email_only"]  # type: ignore[index]
