"""
Phone-specific tools for the Receptionist Agent.

These tools are separate from the main Atlas tools and are used
exclusively for business phone call handling.
"""

import logging
from typing import Optional

from ..base import Tool, ToolResult
from ..registry import ToolRegistry

logger = logging.getLogger("atlas.tools.phone")


class PhoneToolRegistry(ToolRegistry):
    """Registry for phone-specific tools."""

    _instance: Optional["PhoneToolRegistry"] = None

    @classmethod
    def get_instance(cls) -> "PhoneToolRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Singleton instance
phone_tool_registry = PhoneToolRegistry.get_instance()


def get_phone_tool_registry() -> PhoneToolRegistry:
    """Get the phone tool registry."""
    return phone_tool_registry


# Import and register tools
from .services import GetServicesTool, get_services_tool
from .availability import CheckAvailabilityTool, check_availability_tool
from .booking import BookAppointmentTool, book_appointment_tool
from .message import TakeMessageTool, take_message_tool

# Register tools
phone_tool_registry.register(get_services_tool)
phone_tool_registry.register(check_availability_tool)
phone_tool_registry.register(book_appointment_tool)
phone_tool_registry.register(take_message_tool)

__all__ = [
    "PhoneToolRegistry",
    "phone_tool_registry",
    "get_phone_tool_registry",
    "GetServicesTool",
    "get_services_tool",
    "CheckAvailabilityTool",
    "check_availability_tool",
    "BookAppointmentTool",
    "book_appointment_tool",
    "TakeMessageTool",
    "take_message_tool",
]
