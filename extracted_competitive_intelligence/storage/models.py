"""Phase 1 bridge: re-exports atlas_brain.storage.models so scaffolded modules with
top-level relative imports targeting this path resolve cleanly when the
scaffold is imported alongside atlas_brain. Phase 2 work replaces this
with a standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.storage.models import *  # noqa: F401,F403
