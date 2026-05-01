"""Phase 1 bridge: re-exports atlas_brain.templates.email.vendor_checkout_confirmation. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.templates.email.vendor_checkout_confirmation import *  # noqa: F401,F403
