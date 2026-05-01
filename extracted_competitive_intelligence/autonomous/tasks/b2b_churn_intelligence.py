"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks.b2b_churn_intelligence. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks.b2b_churn_intelligence import *  # noqa: F401,F403
