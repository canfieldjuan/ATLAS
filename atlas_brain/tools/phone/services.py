"""
Get Services tool - Returns business services and pricing info.

Retrieves service information from the BusinessContext
for the receptionist to answer caller questions.
"""

import logging
from typing import Any, Optional

from ..base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.phone.services")


class GetServicesTool:
    """Tool to retrieve business services and pricing."""

    def __init__(self) -> None:
        self._context_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "get_services"

    @property
    def description(self) -> str:
        return "Get business services, pricing, and service area information"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="context_id",
                param_type="string",
                description="Business context ID",
                required=False,
            ),
            ToolParameter(
                name="query",
                param_type="string",
                description="Caller's question for context",
                required=False,
            ),
        ]

    def _get_business_context(self, context_id: Optional[str] = None) -> Optional[Any]:
        """Get business context by ID."""
        from ...comms.context import get_context_router

        router = get_context_router()

        if context_id:
            return router.get_context(context_id)

        # Return default context
        contexts = router.list_contexts()
        if contexts:
            return contexts[0]

        return None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the get_services tool."""
        context_id = params.get("context_id")
        query = params.get("query", "")

        try:
            ctx = self._get_business_context(context_id)

            if ctx is None:
                return ToolResult(
                    success=False,
                    error="NO_CONTEXT",
                    message="No business context available",
                )

            # Build service data
            data = {
                "business_name": ctx.name,
                "business_type": getattr(ctx, "business_type", ""),
                "service_area": getattr(ctx, "service_area", ""),
            }

            # Add services list
            services = getattr(ctx, "services", [])
            if services:
                data["services"] = services

            # Add pricing info
            pricing = getattr(ctx, "pricing_info", "")
            if pricing:
                data["pricing"] = pricing

            # Build message for LLM
            message_parts = [f"Business: {ctx.name}"]

            if services:
                message_parts.append(f"Services: {'; '.join(services)}")

            if data.get("service_area"):
                message_parts.append(f"Service area: {data['service_area']}")

            if pricing:
                message_parts.append(f"Pricing: {pricing}")

            return ToolResult(
                success=True,
                data=data,
                message=" | ".join(message_parts),
            )

        except Exception as e:
            logger.exception("Error getting services")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )


# Module instance
get_services_tool = GetServicesTool()
