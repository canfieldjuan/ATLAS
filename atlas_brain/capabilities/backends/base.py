"""
Base protocol for communication backends.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """
    Protocol for communication backends.

    Backends handle the actual communication with devices/services
    (MQTT, HTTP, Home Assistant API, GPIO, etc.)
    """

    @property
    def backend_type(self) -> str:
        """Identifier for this backend type (e.g., 'mqtt', 'homeassistant')."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if the backend is currently connected."""
        ...

    async def connect(self) -> None:
        """Establish connection to the backend."""
        ...

    async def disconnect(self) -> None:
        """Clean up backend connection."""
        ...

    async def send_command(
        self,
        target: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a command and return the response.

        Args:
            target: Topic, path, or endpoint for the command
            payload: Command data

        Returns:
            Response from the backend
        """
        ...

    async def get_state(self, target: str) -> dict[str, Any]:
        """
        Query current state from the backend.

        Args:
            target: Topic, path, or entity ID to query

        Returns:
            State data from the backend
        """
        ...
