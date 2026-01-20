"""
Weather tool using Open-Meteo API.

Provides current weather and forecast data.
No API key required.
"""

import logging
from typing import Any

import httpx

from ..config import settings
from .base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.weather")

# Open-Meteo API base URL
API_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather codes mapping
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherTool:
    """Weather information tool using Open-Meteo API."""

    def __init__(self) -> None:
        self._config = settings.tools
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get current weather conditions and forecast for a location"

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
        ]

    @property
    def aliases(self) -> list[str]:
        return ["weather", "forecast", "temperature", "outside"]

    @property
    def category(self) -> str:
        return "utility"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute weather query."""
        if not self._config.weather_enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Weather tool is disabled",
            )

        lat = params.get("latitude", self._config.weather_default_lat)
        lon = params.get("longitude", self._config.weather_default_lon)

        try:
            weather_data = await self._fetch_weather(lat, lon)
            return ToolResult(
                success=True,
                data=weather_data,
                message=self._format_message(weather_data),
            )
        except httpx.HTTPStatusError as e:
            logger.error("Weather API HTTP error: %s", e)
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Weather API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Weather tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def _fetch_weather(self, lat: float, lon: float) -> dict[str, Any]:
        """Fetch weather data from Open-Meteo API."""
        client = await self._ensure_client()

        temp_unit = self._config.weather_units
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "temperature_unit": temp_unit,
            "windspeed_unit": "mph",
        }

        response = await client.get(API_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        current = data.get("current_weather", {})
        weather_code = current.get("weathercode", 0)

        return {
            "temperature": current.get("temperature"),
            "unit": "F" if temp_unit == "fahrenheit" else "C",
            "windspeed": current.get("windspeed"),
            "wind_unit": "mph",
            "condition": WMO_CODES.get(weather_code, "Unknown"),
            "condition_code": weather_code,
            "is_day": current.get("is_day", 1) == 1,
            "latitude": lat,
            "longitude": lon,
        }

    def _format_message(self, data: dict[str, Any]) -> str:
        """Format weather data as human-readable message."""
        temp = data.get("temperature", "N/A")
        unit = data.get("unit", "F")
        condition = data.get("condition", "Unknown")
        wind = data.get("windspeed", "N/A")
        wind_unit = data.get("wind_unit", "mph")

        return (
            f"Currently {temp}{unit} and {condition.lower()}. "
            f"Wind: {wind} {wind_unit}."
        )


# Module-level instance
weather_tool = WeatherTool()
