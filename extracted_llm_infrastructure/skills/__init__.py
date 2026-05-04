"""Skills entry point for the LLM-infrastructure scaffold.

The exact LLM cache (``services/b2b/llm_exact_cache.py:160``) and the
LLM call pipeline (``pipelines/llm.py``) lazily import
``get_skill_registry`` from ``...skills`` to build skill-message
envelopes.

Default mode (``EXTRACTED_LLM_INFRA_STANDALONE`` unset/false): re-export
from ``atlas_brain.skills`` so the cache helpers run as a sibling of
Atlas and see Atlas's full skill catalog.

Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): use the local
``.registry`` module's ``SkillRegistry`` (PR-A6b carved this out as an
owned substrate file). Default skills directory is
``extracted_llm_infrastructure/skills/markdown/`` (empty by design --
the package ships no public skills per the 2026-05-04 strategy
decision); the ``EXTRACTED_LLM_INFRA_SKILLS_DIR`` env var lets callers
point at an external markdown directory.
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    from .registry import Skill, SkillRegistry, get_skill_registry  # noqa: F401
else:
    from atlas_brain.skills import *  # noqa: F401,F403
    from atlas_brain.skills import get_skill_registry  # noqa: F401
