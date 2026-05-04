"""Tests for the standalone skills substrate (PR-A6b).

Pins the contract that ``extracted_llm_infrastructure.skills`` works
in both modes:

- Default (``EXTRACTED_LLM_INFRA_STANDALONE`` unset/false) re-exports
  from ``atlas_brain.skills`` so the bridge sees Atlas's full skill
  catalog. Not exercised here -- atlas-mode is covered by existing
  ``atlas_brain.skills`` tests; this file focuses on the previously
  broken standalone branch.

- Standalone (``EXTRACTED_LLM_INFRA_STANDALONE=1``) uses the local
  ``.registry`` module's ``SkillRegistry``. Default skills directory
  is ``extracted_llm_infrastructure/skills/markdown/`` (empty by
  design); the ``EXTRACTED_LLM_INFRA_SKILLS_DIR`` env var lets
  callers point at an external markdown directory.

The end-to-end test asserts that ``services.b2b.llm_exact_cache.
build_skill_messages`` succeeds in standalone mode -- closing the
functional gap the bridge's NotImplementedError was producing
before this PR.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest


_STANDALONE_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"
_SKILLS_DIR_ENV_VAR = "EXTRACTED_LLM_INFRA_SKILLS_DIR"


def _reset_skills_modules() -> None:
    """Drop cached imports so the env var snapshot is re-read."""
    for mod_name in (
        "extracted_llm_infrastructure.skills",
        "extracted_llm_infrastructure.skills.registry",
        "extracted_llm_infrastructure.services.b2b.llm_exact_cache",
    ):
        sys.modules.pop(mod_name, None)


@pytest.fixture
def standalone_skills(monkeypatch) -> Iterator[None]:
    """Activate standalone mode and clean module caches afterward."""
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    _reset_skills_modules()
    try:
        yield
    finally:
        _reset_skills_modules()


@pytest.fixture
def standalone_skills_with_dir(monkeypatch, tmp_path: Path) -> Iterator[Path]:
    """Activate standalone mode pointed at a tmp markdown dir.

    Yields the dir so the test can drop fixture .md files into it.
    """
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    monkeypatch.setenv(_SKILLS_DIR_ENV_VAR, str(tmp_path))
    _reset_skills_modules()
    try:
        yield tmp_path
    finally:
        _reset_skills_modules()


# ---- Bridge no longer raises in standalone mode (gap #1 closure) ----


def test_standalone_get_skill_registry_returns_registry(standalone_skills):
    """Pre-PR-A6b this raised NotImplementedError. After the port,
    the bridge returns a real ``SkillRegistry``."""
    from extracted_llm_infrastructure.skills import get_skill_registry, SkillRegistry

    registry = get_skill_registry()
    assert isinstance(registry, SkillRegistry)


def test_standalone_default_skills_dir_is_empty(standalone_skills, monkeypatch):
    """The package's bundled default markdown dir is empty by design.
    A standalone caller without an override sees zero skills."""
    monkeypatch.delenv(_SKILLS_DIR_ENV_VAR, raising=False)
    _reset_skills_modules()
    from extracted_llm_infrastructure.skills import get_skill_registry

    registry = get_skill_registry()
    assert registry.list_skills() == []


def test_standalone_missing_skills_dir_does_not_raise(monkeypatch, tmp_path):
    """If the override points at a non-existent directory, ``load()``
    must return 0 instead of raising. Standalone callers that haven't
    set up a skills dir yet still get a working (empty) registry."""
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    monkeypatch.setenv(_SKILLS_DIR_ENV_VAR, str(tmp_path / "does_not_exist"))
    _reset_skills_modules()

    from extracted_llm_infrastructure.skills import get_skill_registry

    registry = get_skill_registry()
    assert registry.list_skills() == []


# ---- EXTRACTED_LLM_INFRA_SKILLS_DIR override ----


def test_standalone_override_dir_loads_markdown(standalone_skills_with_dir):
    """A markdown file in the override dir is discovered and parsed."""
    skill_file = standalone_skills_with_dir / "probe.md"
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            description: probe skill
            ---
            hello world
            """
        ),
        encoding="utf-8",
    )

    from extracted_llm_infrastructure.skills import get_skill_registry

    registry = get_skill_registry()
    assert "probe" in registry.list_skills()
    skill = registry.get("probe")
    assert skill is not None
    assert skill.content.strip() == "hello world"
    assert skill.description == "probe skill"


def test_standalone_override_dir_supports_subdirectory_naming(
    standalone_skills_with_dir,
):
    """Skill names match the path-relative naming the registry uses
    in atlas (``email/cleaning_confirmation``)."""
    nested_dir = standalone_skills_with_dir / "email"
    nested_dir.mkdir()
    (nested_dir / "cleaning_confirmation.md").write_text(
        "Body for cleaning confirmation.",
        encoding="utf-8",
    )

    from extracted_llm_infrastructure.skills import get_skill_registry

    registry = get_skill_registry()
    assert "email/cleaning_confirmation" in registry.list_skills()
    skill = registry.get("email/cleaning_confirmation")
    assert skill is not None
    assert skill.domain == "email"


# ---- llm_exact_cache end-to-end (gap #2 closure) ----


def test_llm_exact_cache_build_skill_messages_works_in_standalone_mode(
    standalone_skills_with_dir,
):
    """The actual functional gap PR-A6b closes: ``build_skill_messages``
    crashed in standalone before because the skills bridge raised
    ``NotImplementedError``. With the port in place, it loads the
    skill content and emits the system + user messages."""
    skill_file = standalone_skills_with_dir / "probe.md"
    skill_file.write_text(
        "Probe system content for the LLM.",
        encoding="utf-8",
    )

    from extracted_llm_infrastructure.services.b2b.llm_exact_cache import (
        build_skill_messages,
    )

    messages = build_skill_messages("probe", {"q": "hello"})

    system_messages = [m for m in messages if m.get("role") == "system"]
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert len(system_messages) == 1
    assert len(user_messages) == 1
    assert "Probe system content" in str(system_messages[0].get("content", ""))


# ---- Public API surface ----


def test_standalone_registry_exposes_atlas_compatible_api(standalone_skills):
    """Both bridge branches must expose the same public names so
    callers can import without conditional logic."""
    skills_module = importlib.import_module("extracted_llm_infrastructure.skills")
    for name in ("Skill", "SkillRegistry", "get_skill_registry"):
        assert hasattr(skills_module, name), f"missing {name}"
