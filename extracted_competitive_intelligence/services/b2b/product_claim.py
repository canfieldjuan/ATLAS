"""Phase 1 bridge: re-exports atlas_brain.services.b2b.product_claim. Phase 2 replaces with
standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.services.b2b.product_claim import *  # noqa: F401,F403
from atlas_brain.services.b2b.product_claim import ProductClaim, SuppressionReason  # noqa: F401
