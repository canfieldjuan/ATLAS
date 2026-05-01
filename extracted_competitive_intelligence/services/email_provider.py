"""Phase 1 bridge: re-exports atlas_brain.services.email_provider. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.email_provider import *  # noqa: F401,F403
