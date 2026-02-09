"""
Skill registry and router for edge offline skills.
"""

import logging
import re
from typing import Optional

from .base import Skill, SkillResult

logger = logging.getLogger("atlas.edge.skills")


class SkillRegistry:
    """Registry of available edge skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill
        logger.info("Registered skill: %s (%d patterns)", skill.name, len(skill.patterns))

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())


class SkillRouter:
    """Routes queries to matching skills via regex patterns."""

    def __init__(self, registry: SkillRegistry):
        self._registry = registry

    def match(self, query: str) -> Optional[tuple[Skill, re.Match]]:
        """Try all skill patterns, return first match."""
        normalized = query.strip().lower()
        for skill in self._registry.all_skills():
            for pattern in skill.patterns:
                m = pattern.match(normalized)
                if m:
                    return skill, m
        return None

    async def execute(self, query: str) -> Optional[SkillResult]:
        """Match and execute in one step."""
        result = self.match(query)
        if result is None:
            return None
        skill, match = result
        try:
            return await skill.execute(query, match)
        except Exception as e:
            logger.error("Skill %s failed: %s", skill.name, e, exc_info=True)
            return SkillResult(
                success=False,
                skill_name=skill.name,
                error=str(e),
            )
