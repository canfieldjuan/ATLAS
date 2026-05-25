"""Tests for the AI Content Ops reasoning preset catalog."""

from __future__ import annotations

import pytest

from extracted_content_pipeline.reasoning_policy import (
    NOOP_REASONING_PRESETS,
    OUTPUT_REASONING_POLICIES,
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    REASONING_PRESETS,
    OutputReasoningPolicy,
    output_reasoning_policy,
    packaged_reasoning_runtime_presets_for_output,
    reasoning_preset_definition,
    resolve_reasoning_policy,
    supported_reasoning_presets,
)


def test_reasoning_preset_catalog_contains_expected_ids() -> None:
    assert tuple(REASONING_PRESETS) == (
        "none",
        "context_only",
        "single_pass",
        "multi_pass_light",
        "multi_pass_structured",
        "multi_pass_strict",
    )


def test_preset_capabilities_increase_by_depth() -> None:
    assert not REASONING_PRESETS["none"].generated_reasoning
    assert not REASONING_PRESETS["context_only"].generated_reasoning
    assert REASONING_PRESETS["single_pass"].generated_reasoning
    assert not REASONING_PRESETS["single_pass"].multi_pass
    assert REASONING_PRESETS["multi_pass_light"].multi_pass
    assert REASONING_PRESETS["multi_pass_structured"].narrative_planning
    assert REASONING_PRESETS["multi_pass_structured"].output_validation
    assert not REASONING_PRESETS["multi_pass_structured"].blocking_validation
    assert REASONING_PRESETS["multi_pass_strict"].falsification
    assert REASONING_PRESETS["multi_pass_strict"].blocking_validation


def test_output_policy_defaults_match_audit_recommendations() -> None:
    defaults = {
        output: policy.default_preset
        for output, policy in OUTPUT_REASONING_POLICIES.items()
    }
    assert defaults == {
        "signal_extraction": "none",
        "faq_markdown": "none",
        "email_campaign": "single_pass",
        "landing_page": "single_pass",
        "blog_post": "multi_pass_structured",
        "report": "multi_pass_structured",
        "sales_brief": "multi_pass_structured",
    }


def test_reasoning_policies_cover_every_output_catalog_entry() -> None:
    from extracted_content_pipeline.control_surfaces import OUTPUT_CATALOG

    assert set(OUTPUT_REASONING_POLICIES) == set(OUTPUT_CATALOG)


def test_packaged_runtime_reasoning_surface_is_catalog_supported() -> None:
    from extracted_content_pipeline.control_surfaces import OUTPUT_CATALOG

    assert set(PACKAGED_REASONING_RUNTIME_OUTPUTS) <= set(OUTPUT_CATALOG)
    for output in PACKAGED_REASONING_RUNTIME_OUTPUTS:
        assert OUTPUT_CATALOG[output].implemented is True
        policy = output_reasoning_policy(output)
        if output not in {"email_campaign", "landing_page"}:
            assert policy.default_preset in packaged_reasoning_runtime_presets_for_output(output)
        for preset in packaged_reasoning_runtime_presets_for_output(output):
            assert policy.supports(preset)


def test_noop_reasoning_presets_are_derived_from_catalog() -> None:
    assert NOOP_REASONING_PRESETS == ("none", "context_only")
    for preset in NOOP_REASONING_PRESETS:
        assert not REASONING_PRESETS[preset].generated_reasoning


def test_output_policy_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="supported_presets must not be empty"):
        OutputReasoningPolicy("report", "single_pass", ())
    with pytest.raises(ValueError, match="unknown reasoning preset"):
        OutputReasoningPolicy("report", "single_pass", ("multi_pass_lite",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_preset must be included"):
        OutputReasoningPolicy("report", "single_pass", ("multi_pass_light",))


def test_signal_extraction_only_supports_no_reasoning() -> None:
    assert supported_reasoning_presets("signal_extraction") == ("none",)
    with pytest.raises(ValueError, match="not supported"):
        resolve_reasoning_policy("signal_extraction", "single_pass")


def test_faq_markdown_only_supports_no_reasoning() -> None:
    assert supported_reasoning_presets("faq_markdown") == ("none",)
    with pytest.raises(ValueError, match="not supported"):
        resolve_reasoning_policy("faq_markdown", "single_pass")


def test_email_campaign_supports_structured_but_not_strict_preset() -> None:
    supported = supported_reasoning_presets("email_campaign")
    assert supported == (
        "none",
        "context_only",
        "single_pass",
        "multi_pass_light",
        "multi_pass_structured",
    )
    assert "multi_pass_strict" not in supported


def test_landing_page_supports_structured_but_not_strict_preset() -> None:
    supported = supported_reasoning_presets("landing_page")
    assert supported == (
        "none",
        "context_only",
        "single_pass",
        "multi_pass_light",
        "multi_pass_structured",
    )
    assert "multi_pass_strict" not in supported


def test_blog_post_supports_structured_but_not_strict_preset() -> None:
    assert supported_reasoning_presets("blog_post") == (
        "none",
        "context_only",
        "single_pass",
        "multi_pass_light",
        "multi_pass_structured",
    )
    assert "multi_pass_strict" not in supported_reasoning_presets("blog_post")


@pytest.mark.parametrize("output", ("report", "sales_brief"))
def test_report_and_sales_brief_support_structured_presets(output: str) -> None:
    assert supported_reasoning_presets(output) == (
        "none",
        "context_only",
        "single_pass",
        "multi_pass_light",
        "multi_pass_structured",
        "multi_pass_strict",
    )


def test_resolve_reasoning_policy_uses_output_default_when_preset_missing() -> None:
    assert resolve_reasoning_policy("report")[1].id == "multi_pass_structured"
    assert resolve_reasoning_policy("report", "")[1].id == "multi_pass_structured"
    assert resolve_reasoning_policy("report", "context_only")[1].id == "context_only"


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: output_reasoning_policy("podcast_repurpose"), "unknown content output"),
        (lambda: reasoning_preset_definition("deep_magic"), "unknown reasoning preset"),
    ),
)
def test_unknown_policy_inputs_raise_value_error(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_policy_catalog_is_immutable() -> None:
    with pytest.raises(TypeError):
        OUTPUT_REASONING_POLICIES["report"] = output_reasoning_policy("report")  # type: ignore[index]
