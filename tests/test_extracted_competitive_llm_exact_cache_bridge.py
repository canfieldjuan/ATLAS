from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterator

import pytest


_COMP_INTEL_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"
_LLM_INFRA_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"
_SKILLS_DIR_ENV_VAR = "EXTRACTED_LLM_INFRA_SKILLS_DIR"
_BRIDGE_MODULE = "extracted_competitive_intelligence.services.b2b.llm_exact_cache"
_ATLAS_MODULE = "atlas_brain.services.b2b.llm_exact_cache"


def _reset_modules() -> None:
    for module_name in (
        _BRIDGE_MODULE,
        _ATLAS_MODULE,
        "extracted_llm_infrastructure.services.b2b.llm_exact_cache",
        "extracted_llm_infrastructure.skills",
        "extracted_llm_infrastructure.skills.registry",
    ):
        sys.modules.pop(module_name, None)


@pytest.fixture
def standalone_exact_cache(monkeypatch, tmp_path: Path) -> Iterator[Path]:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    monkeypatch.setenv(_SKILLS_DIR_ENV_VAR, str(tmp_path))
    _reset_modules()
    try:
        yield tmp_path
    finally:
        _reset_modules()


def test_competitive_llm_exact_cache_uses_extracted_llm_infrastructure(
    standalone_exact_cache: Path,
) -> None:
    skill_file = standalone_exact_cache / "digest"
    skill_file.mkdir()
    (skill_file / "battle_card_sales_copy.md").write_text(
        "Battle card sales copy system instructions.",
        encoding="utf-8",
    )

    module = importlib.import_module(_BRIDGE_MODULE)
    messages = module.build_skill_messages(
        "digest/battle_card_sales_copy",
        {"vendor": "Acme", "score": 91},
    )

    assert module.build_skill_messages.__module__ == (
        "extracted_llm_infrastructure.services.b2b.llm_exact_cache"
    )
    assert _ATLAS_MODULE not in sys.modules
    assert messages == [
        {
            "role": "system",
            "content": "Battle card sales copy system instructions.",
        },
        {
            "role": "user",
            "content": "{\"vendor\":\"Acme\",\"score\":91}",
        },
    ]
