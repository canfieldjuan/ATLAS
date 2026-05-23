"""Content Ops input provider for host-owned support-ticket sources."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .campaign_ports import TenantScope
from .content_ops_input_provider import ContentOpsInputPackage, RequestPayload
from .support_ticket_input_package import (
    DEFAULT_FAQ_REPORT_AUDIENCE,
    DEFAULT_FAQ_REPORT_CAMPAIGN_NAME,
    DEFAULT_FAQ_REPORT_CTA_LABEL,
    DEFAULT_FAQ_REPORT_CTA_URL,
    DEFAULT_FAQ_REPORT_OFFER,
    DEFAULT_FAQ_REPORT_TARGET_KEYWORD,
    DEFAULT_SUPPORT_TICKET_OUTPUTS,
    build_support_ticket_input_package,
)


SupportTicketSourceLoader = Callable[
    [TenantScope, RequestPayload | None],
    Any | Awaitable[Any],
]


@dataclass(frozen=True)
class SupportTicketInputProvider:
    """Provider adapter for support-ticket source material.

    Hosts may pass fixed ``source_material`` for simple in-memory wiring or a
    ``source_material_loader`` for tenant/request-aware lookup. The loader owns
    persistence and authorization; this adapter only turns already-loaded ticket
    material into a Content Ops input package.
    """

    source_material: Any = None
    source_material_loader: SupportTicketSourceLoader | None = None
    provider: str = "support_ticket_input_provider"
    outputs: Sequence[str] = DEFAULT_SUPPORT_TICKET_OUTPUTS
    window_days: int = 90
    max_rows: int = 1000
    campaign_name: str = DEFAULT_FAQ_REPORT_CAMPAIGN_NAME
    audience: str = DEFAULT_FAQ_REPORT_AUDIENCE
    offer: str = DEFAULT_FAQ_REPORT_OFFER
    target_keyword: str = DEFAULT_FAQ_REPORT_TARGET_KEYWORD
    cta_label: str = DEFAULT_FAQ_REPORT_CTA_LABEL
    cta_url: str = DEFAULT_FAQ_REPORT_CTA_URL
    secondary_keywords: Sequence[str] | None = None
    objections: Sequence[str] | None = None
    internal_links: Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_material is not None and self.source_material_loader is not None:
            raise ValueError(
                "provide either source_material or source_material_loader, not both"
            )
        if self.window_days < 1:
            raise ValueError("window_days must be at least 1")
        if self.max_rows < 1:
            raise ValueError("max_rows must be at least 1")

    def build_content_ops_input_package(
        self,
        *,
        scope: TenantScope,
        request: RequestPayload | None = None,
    ) -> ContentOpsInputPackage | Awaitable[ContentOpsInputPackage]:
        """Return a support-ticket input package for one scoped request."""

        source_material = self._source_material(scope=scope, request=request)
        if hasattr(source_material, "__await__"):
            return self._build_from_awaitable(source_material)
        return self._build_package(source_material)

    def _source_material(
        self,
        *,
        scope: TenantScope,
        request: RequestPayload | None,
    ) -> Any | Awaitable[Any]:
        if self.source_material_loader is None:
            return self.source_material
        return self.source_material_loader(scope, request)

    async def _build_from_awaitable(self, source_material: Awaitable[Any]) -> ContentOpsInputPackage:
        return self._build_package(await source_material)

    def _build_package(self, source_material: Any) -> ContentOpsInputPackage:
        package = build_support_ticket_input_package(
            source_material,
            provider=self.provider,
            outputs=self.outputs,
            window_days=self.window_days,
            max_rows=self.max_rows,
            campaign_name=self.campaign_name,
            audience=self.audience,
            offer=self.offer,
            target_keyword=self.target_keyword,
            cta_label=self.cta_label,
            cta_url=self.cta_url,
            secondary_keywords=self.secondary_keywords,
            objections=self.objections,
            internal_links=self.internal_links,
        )
        if not self.metadata:
            return package
        return replace(
            package,
            metadata={**dict(self.metadata), **dict(package.metadata)},
        )


__all__ = [
    "SupportTicketInputProvider",
    "SupportTicketSourceLoader",
]
