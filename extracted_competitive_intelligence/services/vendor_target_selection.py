"""Phase 1 bridge: re-exports atlas_brain.services.vendor_target_selection. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.vendor_target_selection import *  # noqa: F401,F403
from atlas_brain.services.vendor_target_selection import dedupe_vendor_target_rows  # noqa: F401
