from __future__ import annotations

import os
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
        return LocalSkill(name=name, content=path.read_text())


def get_skill_registry():
    if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
        root = Path(__file__).resolve().parent
        return LocalSkillRegistry(root)

    from atlas_brain.skills.registry import get_skill_registry as atlas_get_skill_registry

    return atlas_get_skill_registry()
