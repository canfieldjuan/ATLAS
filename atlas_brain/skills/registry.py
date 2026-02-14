"""
Skill registry for loading injectable markdown skill documents.

Skills are markdown files with optional YAML frontmatter that get
injected into LLM prompts at runtime. The registry discovers and
loads .md files from atlas_brain/skills/.

Usage:
    from atlas_brain.skills import get_skill_registry

    registry = get_skill_registry()
    skill = registry.get("email/cleaning_confirmation")
    skills = registry.get_by_domain("email")
    skills = registry.get_by_tag("sentiment")
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("atlas.skills")

# Skills root: atlas_brain/skills/
_SKILLS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Skill:
    """An injectable skill document parsed from a markdown file."""

    name: str
    domain: str
    content: str
    description: str
    tags: tuple[str, ...]
    version: int
    file_path: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str | list[str] | int], str]:
    """
    Parse optional YAML-like frontmatter from markdown text.

    Handles simple key: value and key: [a, b, c] syntax.
    No PyYAML dependency required.

    Returns (metadata_dict, body_text).
    """
    if not text.startswith("---"):
        return {}, text

    # Find closing ---
    end_match = re.search(r"\n---\s*\n", text[3:])
    if end_match is None:
        return {}, text

    frontmatter_str = text[3 : 3 + end_match.start()]
    body = text[3 + end_match.end() :]

    metadata: dict[str, str | list[str] | int] = {}
    for line in frontmatter_str.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        # Parse list: [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip() for item in value[1:-1].split(",")]
            metadata[key] = [item for item in items if item]
        # Parse integer
        elif value.isdigit():
            metadata[key] = int(value)
        else:
            metadata[key] = value

    return metadata, body


def _load_skill(file_path: Path, skills_dir: Path) -> Skill:
    """Load a single skill from a markdown file."""
    text = file_path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(text)

    # Derive defaults from file path
    relative = file_path.relative_to(skills_dir)
    default_name = str(relative.with_suffix(""))
    default_domain = relative.parts[0] if len(relative.parts) > 1 else ""

    name = str(metadata.get("name", default_name))
    domain = str(metadata.get("domain", default_domain))
    description = str(metadata.get("description", ""))
    version = metadata.get("version", 1)
    if not isinstance(version, int):
        version = 1

    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, list):
        tags = tuple(str(t) for t in raw_tags)
    else:
        tags = (str(raw_tags),)

    return Skill(
        name=name,
        domain=domain,
        content=body.strip(),
        description=description,
        tags=tags,
        version=version,
        file_path=str(file_path),
    )


class SkillRegistry:
    """
    Discovers and loads .md skill files from the skills directory.

    Lazy-loaded: skills are read from disk on first access.
    Call reload() to re-read from disk after changes.
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        self._skills_dir = skills_dir or _SKILLS_DIR
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def load(self) -> int:
        """
        Load all .md skill files from the skills directory.

        Returns the number of skills loaded.
        """
        self._skills.clear()
        count = 0

        for md_file in sorted(self._skills_dir.rglob("*.md")):
            try:
                skill = _load_skill(md_file, self._skills_dir)
                self._skills[skill.name] = skill
                count += 1
                logger.debug("Loaded skill: %s", skill.name)
            except Exception:
                logger.exception("Failed to load skill from %s", md_file)

        self._loaded = True
        logger.info("Loaded %d skills from %s", count, self._skills_dir)
        return count

    def _ensure_loaded(self) -> None:
        """Lazy-load skills on first access."""
        if not self._loaded:
            self.load()

    def reload(self) -> int:
        """Re-read all skills from disk. Returns count loaded."""
        self._loaded = False
        return self.load()

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name (e.g., 'email/cleaning_confirmation')."""
        self._ensure_loaded()
        return self._skills.get(name)

    def get_by_domain(self, domain: str) -> list[Skill]:
        """Get all skills in a domain (e.g., 'email')."""
        self._ensure_loaded()
        return [s for s in self._skills.values() if s.domain == domain]

    def get_by_tag(self, tag: str) -> list[Skill]:
        """Get all skills that have a given tag."""
        self._ensure_loaded()
        return [s for s in self._skills.values() if tag in s.tags]

    def list_skills(self) -> list[str]:
        """List all loaded skill names."""
        self._ensure_loaded()
        return sorted(self._skills.keys())


_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get or create the singleton SkillRegistry."""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry
