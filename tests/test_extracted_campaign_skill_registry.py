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
