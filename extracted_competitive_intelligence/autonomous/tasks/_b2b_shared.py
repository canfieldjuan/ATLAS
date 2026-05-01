"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks._b2b_shared so scaffolded modules with
top-level relative imports targeting this path resolve cleanly when the
scaffold is imported alongside atlas_brain. Phase 2 work replaces this
with a standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks._b2b_shared import *  # noqa: F401,F403
