"""Phase 1 bridge: re-exports atlas_brain.config so scaffolded modules with
top-level `from ..config import settings` style imports resolve cleanly.

Phase 2 carves a slim ``LLMInfraConfig`` Pydantic class out of
``atlas_brain.config`` and replaces this re-export with a standalone
implementation gated by ``EXTRACTED_LLM_INFRA_STANDALONE=1``.
"""
from atlas_brain.config import *  # noqa: F401,F403
from atlas_brain.config import settings  # noqa: F401
