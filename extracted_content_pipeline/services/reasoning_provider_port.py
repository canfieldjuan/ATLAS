"""Host-owned provider port for campaign reasoning context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ..campaign_ports import CampaignReasoningContext, TenantScope


@runtime_checkable
class CampaignReasoningProviderPort(Protocol):
    """Port for reading per-target reasoning context from a host provider."""

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | None:
        ...
