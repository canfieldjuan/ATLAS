"""Phase 1 bridge: re-exports atlas_brain.config. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.config import *  # noqa: F401,F403
from atlas_brain.config import settings  # noqa: F401
