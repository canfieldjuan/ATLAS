"""Phase 1 bridge: re-exports atlas_brain.services.base so scaffolded LLM
provider modules with top-level `from ..base import BaseModelService`
imports resolve cleanly.

Phase 3 work decouples the LLM provider hierarchy from this base class.
"""
from atlas_brain.services.base import *  # noqa: F401,F403
