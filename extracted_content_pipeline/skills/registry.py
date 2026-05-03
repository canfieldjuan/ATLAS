from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PACKAGE_SKILL_ROOT = Path(__file__).resolve().parent


@dataclass
class LocalSkill:
    name: str
    content: str
    path: Path | None = None


class LocalSkillRegistry:
    """Markdown-backed skill registry for packaged and host-provided prompts."""

    def __init__(self, roots: Path | str | Sequence[Path | str]) -> None:
        self.roots = _normalize_roots(roots)

    def get(self, name: str):
        rel = _skill_relative_path(name)
        if rel is None:
            return None
        for root in self.roots:
            path = root / rel
            if path.exists() and path.is_file():
                return LocalSkill(
                    name=_skill_name_from_path(rel),
                    content=path.read_text(encoding="utf-8"),
                    path=path,
                )
        return None

    def get_prompt(self, name: str) -> str | None:
        skill = self.get(name)
        return skill.content if skill else None


def get_skill_registry(
    root: Path | str | None = None,
    *,
    roots: Sequence[Path | str] = (),
    include_package: bool = True,
):
    """Return a markdown skill registry.

    Host roots are searched before the packaged skills, so customer prompt
    contracts can override bundled prompts without replacing product code.
    """

    search_roots: list[Path | str] = []
    if root is not None:
        search_roots.append(root)
    search_roots.extend(roots)
    if include_package:
        search_roots.append(_PACKAGE_SKILL_ROOT)
    return LocalSkillRegistry(search_roots)


def _normalize_roots(roots: Path | str | Sequence[Path | str]) -> tuple[Path, ...]:
    if isinstance(roots, (str, Path)):
        values: Sequence[Path | str] = (roots,)
    else:
        values = roots
    normalized: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path not in normalized:
            normalized.append(path)
    return tuple(normalized)


def _skill_relative_path(name: Any) -> Path | None:
    raw = str(name or "").replace("\\", "/").strip().strip("/")
    if not raw:
        return None
    if raw.endswith(".md"):
        raw = raw[:-3]
    parts = [part for part in raw.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        return None
    path = Path(*parts)
    if path.is_absolute():
        return None
    return path.with_suffix(".md")


def _skill_name_from_path(path: Path) -> str:
    return path.with_suffix("").as_posix()
