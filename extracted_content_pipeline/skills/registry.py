from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalSkill:
    name: str
    content: str


class LocalSkillRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, name: str):
        rel = Path(*name.split("/"))
        path = self.root / f"{rel}.md"
        if not path.exists():
            return None
        return LocalSkill(name=name, content=path.read_text(encoding="utf-8"))

    def get_prompt(self, name: str) -> str | None:
        skill = self.get(name)
        return skill.content if skill else None


def get_skill_registry():
    root = Path(__file__).resolve().parent
    return LocalSkillRegistry(root)
