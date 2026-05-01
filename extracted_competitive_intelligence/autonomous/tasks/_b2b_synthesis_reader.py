"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks._b2b_synthesis_reader. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import *  # noqa: F401,F403
