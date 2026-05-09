import pytest

from extracted_content_pipeline.control_surfaces import (
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


def test_output_catalog_states_reasoning_requirement():
    from extracted_content_pipeline.control_surfaces import OUTPUT_CATALOG

    assert OUTPUT_CATALOG["email_campaign"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["report"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["landing_page"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["sales_brief"].reasoning_requirement == "optional_host_context"
    assert OUTPUT_CATALOG["blog_post"].reasoning_requirement == "absent"
    assert OUTPUT_CATALOG["signal_extraction"].reasoning_requirement == "absent"


def test_request_from_mapping_rejects_zero_limit():
    with pytest.raises(ValueError, match="limit must be at least 1; got 0"):
        request_from_mapping({"limit": 0})


def test_request_from_mapping_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="max_cost_usd must be positive; got -1"):
        request_from_mapping({"max_cost_usd": -1})


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
