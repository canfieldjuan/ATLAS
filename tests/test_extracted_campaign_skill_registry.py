from __future__ import annotations

from extracted_content_pipeline.skills.registry import (
    LocalSkillRegistry,
    get_skill_registry,
)


def test_local_skill_registry_reads_packaged_campaign_prompt() -> None:
    registry = get_skill_registry()

    skill = registry.get("digest/b2b_campaign_generation")

    assert skill is not None
    assert skill.name == "digest/b2b_campaign_generation"
    assert skill.path is not None
    assert skill.path.name == "b2b_campaign_generation.md"
    assert registry.get_prompt("digest/b2b_campaign_generation") == skill.content


def test_packaged_report_prompt_frames_review_evidence_as_market_signal() -> None:
    prompt = get_skill_registry().get_prompt("digest/report_generation")

    assert prompt is not None
    assert 'source_type: "review"' in prompt
    assert "third-party market evidence" in prompt
    assert "Do not say the target account itself" in prompt


def test_packaged_campaign_prompt_frames_support_tickets_as_service_evidence() -> None:
    prompt = get_skill_registry().get_prompt("digest/b2b_campaign_generation")

    assert prompt is not None
    assert 'source_type: "support_ticket"' in prompt
    assert "service evidence, not buying-intent evidence" in prompt
    assert "is buying, is switching, is considering" in prompt


def test_packaged_report_prompt_frames_support_tickets_as_service_evidence() -> None:
    prompt = get_skill_registry().get_prompt("digest/report_generation")

    assert prompt is not None
    assert 'source_type: "support_ticket"' in prompt
    assert "service evidence, not buying-intent evidence" in prompt
    assert "complaint narratives describe" in prompt


def test_packaged_sales_brief_prompt_frames_review_evidence_as_market_signal() -> None:
    prompt = get_skill_registry().get_prompt("digest/sales_brief_generation")

    assert prompt is not None
    assert 'source_type: "review"' in prompt
    assert "third-party market evidence" in prompt
    assert "Do not say the target account itself" in prompt


def test_packaged_sales_brief_prompt_frames_support_tickets_as_service_evidence() -> None:
    prompt = get_skill_registry().get_prompt("digest/sales_brief_generation")

    assert prompt is not None
    assert 'source_type: "support_ticket"' in prompt
    assert "service evidence, not buying-intent evidence" in prompt
    assert "rep-safe framing" in prompt


def test_packaged_landing_page_prompt_frames_support_tickets_as_service_evidence() -> None:
    prompt = get_skill_registry().get_prompt("digest/landing_page_generation")

    assert prompt is not None
    assert 'source_type: "support_ticket"' in prompt
    assert "service evidence, not buying-intent evidence" in prompt
    assert "evaluating, buying, switching, or considering" in prompt


def test_packaged_landing_page_prompt_matches_readiness_gate_contract() -> None:
    prompt = get_skill_registry().get_prompt("digest/landing_page_generation")

    assert prompt is not None
    assert "Do not use generic slugs" in prompt
    assert "`meta.title_tag`: always include" in prompt
    assert "Do not output placeholder URLs" in prompt
    assert "Do not use generic headings" in prompt
    assert "first viewport must make it clear" in prompt
    assert "Include a clear problem section" in prompt
    assert "Include objection coverage" in prompt
    assert "Do not leave unresolved placeholders" in prompt


def test_packaged_landing_page_prompt_prevents_fake_proof() -> None:
    prompt = get_skill_registry().get_prompt("digest/landing_page_generation")

    assert prompt is not None
    assert "Do not invent reference ids" in prompt
    assert "If the campaign does not provide proof" in prompt
    assert "fake social proof" in prompt


def test_packaged_landing_page_prompt_requests_aeo_geo_section_metadata() -> None:
    prompt = get_skill_registry().get_prompt("digest/landing_page_generation")

    assert prompt is not None
    assert "sections.metadata.kind" in prompt
    assert "sections.metadata.primary_question" in prompt
    assert "sections.metadata.answer_summary" in prompt
    assert "same answer at the start of `body_markdown`" in prompt
    assert "Do not force a FAQ section" in prompt
    assert "supports answer extraction" in prompt


def test_local_skill_registry_accepts_host_root_override(tmp_path) -> None:
    skill_path = tmp_path / "digest" / "b2b_campaign_generation.md"
    skill_path.parent.mkdir()
    skill_path.write_text("Custom prompt: {opportunity_json}", encoding="utf-8")

    registry = get_skill_registry(root=tmp_path)

    skill = registry.get("digest/b2b_campaign_generation")
    assert skill is not None
    assert skill.content == "Custom prompt: {opportunity_json}"
    assert skill.path == skill_path


def test_local_skill_registry_can_disable_packaged_fallback(tmp_path) -> None:
    registry = get_skill_registry(root=tmp_path, include_package=False)

    assert registry.get_prompt("digest/b2b_campaign_generation") is None


def test_local_skill_registry_falls_back_to_packaged_prompt(tmp_path) -> None:
    registry = get_skill_registry(root=tmp_path)

    assert registry.get_prompt("digest/b2b_campaign_generation")


def test_local_skill_registry_rejects_traversal_names(tmp_path) -> None:
    registry = LocalSkillRegistry(tmp_path)

    assert registry.get("../secrets") is None
    assert registry.get("digest/../../secrets") is None


def test_local_skill_registry_accepts_md_suffix(tmp_path) -> None:
    skill_path = tmp_path / "digest" / "custom.md"
    skill_path.parent.mkdir()
    skill_path.write_text("Custom", encoding="utf-8")
    registry = LocalSkillRegistry(tmp_path)

    skill = registry.get("digest/custom.md")

    assert skill is not None
    assert skill.name == "digest/custom"
    assert skill.content == "Custom"
