"""Phase 1 bridge: re-exports atlas_brain.services.registry so LLM
provider modules with top-level `from ..registry import register_llm`
resolve cleanly. Importantly, the ``llm_registry`` global singleton is
shared across atlas_brain and the scaffold by virtue of this re-export.
"""
from atlas_brain.services.registry import *  # noqa: F401,F403
from atlas_brain.services.registry import llm_registry, register_llm  # noqa: F401
