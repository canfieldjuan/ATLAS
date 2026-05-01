"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks._execution_progress. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks._execution_progress import *  # noqa: F401,F403
from atlas_brain.autonomous.tasks._execution_progress import _update_execution_progress  # noqa: F401
