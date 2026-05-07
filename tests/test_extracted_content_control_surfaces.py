from extracted_content_pipeline.control_surfaces import (
    normalize_outputs,
    preview_from_mapping,
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
    assert preview["estimated_cost_usd"] == 0.18


def test_preview_allows_implemented_outputs_under_budget():
    preview = preview_from_mapping(
        {
            "outputs": ["email_campaign", "report"],
            "limit": 2,
            "max_cost_usd": 2.0,
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn intelligence audit",
                "opportunity_id": "opp_123",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["email_campaign", "report"]
    assert preview["estimated_cost_usd"] == 1.46
    assert preview["missing_inputs"] == []
    assert preview["blocked_outputs"] == []


def test_preview_blocks_unimplemented_outputs_by_default():
    preview = preview_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_material": "review export",
            },
        }
    )

    assert preview["can_run"] is False
    assert preview["outputs"] == []
    assert preview["blocked_outputs"] == ["signal_extraction"]
    assert "Output not implemented yet: signal_extraction" in preview["warnings"]


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
    assert "Estimated cost exceeds max_cost_usd: 1.10 > 0.25" in preview["warnings"]


def test_preview_can_include_future_outputs_when_explicitly_allowed():
    preview = preview_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "allow_unimplemented_outputs": True,
            "inputs": {
                "source_material": "review export",
            },
        }
    )

    assert preview["can_run"] is True
    assert preview["outputs"] == ["signal_extraction"]
    assert preview["blocked_outputs"] == []
