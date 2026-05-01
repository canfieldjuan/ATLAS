"""Phase 1 bridge: re-exports atlas_brain.mcp.b2b._shared. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.mcp.b2b._shared import *  # noqa: F401,F403
from atlas_brain.mcp.b2b._shared import logger, get_pool, _safe_json, _is_uuid, _suppress_predicate, VALID_REPORT_TYPES  # noqa: F401
