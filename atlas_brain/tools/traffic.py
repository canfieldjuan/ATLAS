"""
Traffic tool using TomTom Traffic API.

Provides real-time traffic flow and route traffic information.
Requires TomTom API key (free tier: 2,500 requests/day).
"""

import logging
from typing import Any

import httpx

from ..config import settings
from .base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.traffic")

# TomTom API base URLs
FLOW_SEGMENT_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
ROUTING_URL = "https://api.tomtom.com/routing/1/calculateRoute"


class TrafficTool:
    """Traffic information tool using TomTom API."""

    def __init__(self) -> None:
        self._config = settings.tools
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "get_traffic"

    @property
    def description(self) -> str:
        return "Get real-time traffic conditions for a location or route"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="latitude",
                param_type="float",
                description="Latitude of location",
                required=False,
            ),
            ToolParameter(
                name="longitude",
                param_type="float",
                description="Longitude of location",
                required=False,
            ),
            ToolParameter(
                name="destination_lat",
                param_type="float",
                description="Destination latitude (for route traffic)",
                required=False,
            ),
            ToolParameter(
                name="destination_lon",
                param_type="float",
                description="Destination longitude (for route traffic)",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["traffic", "commute", "road conditions", "drive time"]

    @property
    def category(self) -> str:
        return "utility"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute traffic query."""
        if not self._config.traffic_enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Traffic tool is disabled",
            )

        api_key = self._config.traffic_api_key
        if not api_key:
            return ToolResult(
                success=False,
                error="NO_API_KEY",
                message="TomTom API key not configured",
            )

        lat = params.get("latitude", self._config.weather_default_lat)
        lon = params.get("longitude", self._config.weather_default_lon)
        dest_lat = params.get("destination_lat")
        dest_lon = params.get("destination_lon")

        try:
            if dest_lat is not None and dest_lon is not None:
                if lat is None or lon is None:
                    return ToolResult(
                        success=False,
                        error="MISSING_ORIGIN",
                        message="Origin coordinates required for route traffic",
                    )
                traffic_data = await self._fetch_route_traffic(
                    lat, lon, dest_lat, dest_lon, api_key
                )
            elif lat is not None and lon is not None:
                traffic_data = await self._fetch_flow_segment(lat, lon, api_key)
            else:
                return ToolResult(
                    success=False,
                    error="MISSING_COORDINATES",
                    message="No location available for traffic lookup",
                )

            return ToolResult(
                success=True,
                data=traffic_data,
                message=self._format_message(traffic_data),
            )
        except httpx.HTTPStatusError as e:
            logger.error("Traffic API HTTP error: %s", e)
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Traffic API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Traffic tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def _fetch_flow_segment(
        self, lat: float, lon: float, api_key: str
    ) -> dict[str, Any]:
        """Fetch traffic flow data for a road segment near coordinates."""
        client = await self._ensure_client()

        url = f"{FLOW_SEGMENT_URL}/absolute/10/json"
        params = {
            "key": api_key,
            "point": f"{lat},{lon}",
        }

        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        flow = data.get("flowSegmentData", {})
        current_speed = flow.get("currentSpeed", 0)
        free_flow_speed = flow.get("freeFlowSpeed", 0)
        confidence = flow.get("confidence", 0)

        if free_flow_speed > 0:
            congestion_ratio = 1 - (current_speed / free_flow_speed)
        else:
            congestion_ratio = 0

        return {
            "type": "flow_segment",
            "current_speed_mph": current_speed,
            "free_flow_speed_mph": free_flow_speed,
            "congestion_ratio": round(congestion_ratio, 2),
            "confidence": confidence,
            "road_closure": flow.get("roadClosure", False),
            "latitude": lat,
            "longitude": lon,
        }

    async def _fetch_route_traffic(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        api_key: str,
    ) -> dict[str, Any]:
        """Fetch traffic data for a route between two points."""
        client = await self._ensure_client()

        coords = f"{origin_lat},{origin_lon}:{dest_lat},{dest_lon}"
        url = f"{ROUTING_URL}/{coords}/json"
        params = {
            "key": api_key,
            "traffic": "true",
            "travelMode": "car",
        }

        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        routes = data.get("routes", [])
        if not routes:
            return {
                "type": "route",
                "error": "No route found",
            }

        summary = routes[0].get("summary", {})
        travel_time_sec = summary.get("travelTimeInSeconds", 0)
        traffic_delay_sec = summary.get("trafficDelayInSeconds", 0)
        distance_m = summary.get("lengthInMeters", 0)

        travel_time_min = travel_time_sec // 60
        traffic_delay_min = traffic_delay_sec // 60
        distance_miles = round(distance_m / 1609.34, 1)

        return {
            "type": "route",
            "travel_time_minutes": travel_time_min,
            "traffic_delay_minutes": traffic_delay_min,
            "distance_miles": distance_miles,
            "origin": {"latitude": origin_lat, "longitude": origin_lon},
            "destination": {"latitude": dest_lat, "longitude": dest_lon},
        }

    def _format_message(self, data: dict[str, Any]) -> str:
        """Format traffic data as human-readable message."""
        traffic_type = data.get("type", "")

        if traffic_type == "flow_segment":
            current = data.get("current_speed_mph", 0)
            free_flow = data.get("free_flow_speed_mph", 0)
            congestion = data.get("congestion_ratio", 0)

            if data.get("road_closure"):
                return "Road is currently closed."

            if congestion < 0.1:
                status = "Traffic is flowing freely"
            elif congestion < 0.3:
                status = "Traffic is light"
            elif congestion < 0.5:
                status = "Traffic is moderate"
            else:
                status = "Traffic is heavy"

            return f"{status}. Current speed: {current} mph (normally {free_flow} mph)."

        elif traffic_type == "route":
            if data.get("error"):
                return data["error"]

            travel_min = data.get("travel_time_minutes", 0)
            delay_min = data.get("traffic_delay_minutes", 0)
            distance = data.get("distance_miles", 0)

            msg = f"Route is {distance} miles, {travel_min} minutes"
            if delay_min > 0:
                msg += f" ({delay_min} min delay due to traffic)"
            else:
                msg += " with no traffic delays"
            return msg + "."

        return "Traffic data unavailable."


# Module-level instance
traffic_tool = TrafficTool()
