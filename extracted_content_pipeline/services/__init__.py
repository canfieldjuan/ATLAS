from __future__ import annotations


class _StandaloneLLMRegistry:
    @staticmethod
    def get_active():
        return None


llm_registry = _StandaloneLLMRegistry()

__all__ = ["llm_registry"]
