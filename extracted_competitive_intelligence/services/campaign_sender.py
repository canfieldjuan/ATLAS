"""Phase 1 bridge: re-exports atlas_brain.services.campaign_sender. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.campaign_sender import *  # noqa: F401,F403
from atlas_brain.services.campaign_sender import get_campaign_sender  # noqa: F401
