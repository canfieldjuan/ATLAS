"""Phase 1 bridge: re-exports atlas_brain.services.protocols. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.protocols import *  # noqa: F401,F403
from atlas_brain.services.protocols import Message, ModelInfo, LLMService  # noqa: F401
