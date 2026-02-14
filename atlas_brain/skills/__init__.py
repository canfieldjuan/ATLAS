"""
Skills system for Atlas Brain.

Injectable markdown documents loaded from the filesystem that
provide domain-specific constraints and style rules for LLM prompts.

Usage:
    from atlas_brain.skills import get_skill_registry

    registry = get_skill_registry()
    skill = registry.get("email/cleaning_confirmation")
    print(skill.content)  # markdown body for prompt injection
"""

from .registry import Skill, SkillRegistry, get_skill_registry

__all__ = ["Skill", "SkillRegistry", "get_skill_registry"]
