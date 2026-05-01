from __future__ import annotations

import os

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    class _StandaloneLLMRegistry:
        @staticmethod
        def get_active():
            return None

    llm_registry = _StandaloneLLMRegistry()
else:
    from atlas_brain.services import llm_registry

__all__ = ["llm_registry"]
