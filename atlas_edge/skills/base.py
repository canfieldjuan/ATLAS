"""
Base skill protocol and result type for edge offline skills.
"""

import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class SkillResult:
    """Result from a skill execution."""

    success: bool
    response_text: str = ""
    action_type: str = "skill"
    skill_name: str = ""
    error: Optional[str] = None


@runtime_checkable
class Skill(Protocol):
    """Protocol for edge skills."""

    name: str
    description: str
    patterns: list[re.Pattern]

    async def execute(self, query: str, match: re.Match) -> SkillResult: ...
