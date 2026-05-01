"""Phase 1 bridge: re-exports atlas_brain.services.b2b.pdf_renderer. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.b2b.pdf_renderer import *  # noqa: F401,F403
