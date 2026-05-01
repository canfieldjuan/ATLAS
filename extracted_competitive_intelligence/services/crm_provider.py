"""Phase 1 bridge: re-exports atlas_brain.services.crm_provider. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.crm_provider import *  # noqa: F401,F403
from atlas_brain.services.crm_provider import get_crm_provider  # noqa: F401
