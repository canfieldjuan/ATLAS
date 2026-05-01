"""Phase 1 bridge: re-exports atlas_brain.services.llm_router. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.llm_router import *  # noqa: F401,F403
from atlas_brain.services.llm_router import get_llm  # noqa: F401
