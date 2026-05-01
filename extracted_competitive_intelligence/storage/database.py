"""Phase 1 bridge: re-exports atlas_brain.storage.database. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.storage.database import *  # noqa: F401,F403
from atlas_brain.storage.database import get_db_pool  # noqa: F401
