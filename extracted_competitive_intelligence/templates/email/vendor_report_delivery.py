"""Phase 1 bridge: re-exports atlas_brain.templates.email.vendor_report_delivery. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.templates.email.vendor_report_delivery import *  # noqa: F401,F403
