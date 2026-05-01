"""Phase 1 bridge: re-exports atlas_brain.auth.dependencies. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.auth.dependencies import *  # noqa: F401,F403
from atlas_brain.auth.dependencies import AuthUser, require_auth  # noqa: F401
