"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks.campaign_suppression. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks.campaign_suppression import *  # noqa: F401,F403
from atlas_brain.autonomous.tasks.campaign_suppression import is_suppressed  # noqa: F401
