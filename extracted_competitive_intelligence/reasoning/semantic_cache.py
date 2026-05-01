"""Phase 1 bridge: re-exports atlas_brain.reasoning.semantic_cache. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.reasoning.semantic_cache import *  # noqa: F401,F403
