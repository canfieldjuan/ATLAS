"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks._b2b_batch_utils. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks._b2b_batch_utils import *  # noqa: F401,F403
