"""Standalone campaign analytics refresh orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .campaign_ports import AuditSink, CampaignRepository, VisibilitySink


@dataclass(frozen=True)
class CampaignAnalyticsRefreshResult:
    refreshed: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"refreshed": self.refreshed, "error": self.error}


class CampaignAnalyticsRefreshService:
    """Refresh campaign analytics through host-provided persistence ports."""

    def __init__(
        self,
        *,
        campaigns: CampaignRepository,
        audit: AuditSink | None = None,
        visibility: VisibilitySink | None = None,
    ):
        self._campaigns = campaigns
        self._audit = audit
        self._visibility = visibility

    async def refresh(self) -> CampaignAnalyticsRefreshResult:
        try:
            await self._campaigns.refresh_analytics()
        except Exception as exc:
            result = CampaignAnalyticsRefreshResult(refreshed=False, error=str(exc))
            await self._record("analytics_refresh_failed", result.as_dict())
            return result

        result = CampaignAnalyticsRefreshResult(refreshed=True)
        await self._record("analytics_refreshed", result.as_dict())
        return result

    async def _record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._audit:
            try:
                await self._audit.record(event_type, metadata=payload)
            except Exception:
                pass
        if self._visibility:
            try:
                await self._visibility.emit(event_type, payload)
            except Exception:
                pass
