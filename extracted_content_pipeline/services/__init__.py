from __future__ import annotations

from .reasoning_provider_port import CampaignReasoningProviderPort


class _StandaloneLLMRegistry:
    @staticmethod
    def get_active():
        return None


llm_registry = _StandaloneLLMRegistry()

__all__ = ["llm_registry", "CampaignReasoningProviderPort"]
