"""Phase 1 bridge: re-exports atlas_brain.services.protocols so LLM
provider modules and pipelines/llm.py with top-level
`from ..protocols import Message, ModelInfo` resolve cleanly.
"""
from atlas_brain.services.protocols import *  # noqa: F401,F403
