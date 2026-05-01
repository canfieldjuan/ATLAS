"""Phase 1 bridge: re-exports atlas_brain.mcp.b2b.server. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.mcp.b2b.server import *  # noqa: F401,F403
from atlas_brain.mcp.b2b.server import mcp  # noqa: F401
