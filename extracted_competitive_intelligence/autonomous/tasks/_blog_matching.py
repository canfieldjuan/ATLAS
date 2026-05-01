"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks._blog_matching. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks._blog_matching import *  # noqa: F401,F403
from atlas_brain.autonomous.tasks._blog_matching import fetch_relevant_blog_posts  # noqa: F401
