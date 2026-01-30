"""
Response generation for edge devices.

Includes templates for device responses and offline fallback messages.
"""

from .templates import ResponseTemplates, OFFLINE_FALLBACK_MESSAGE

__all__ = [
    "ResponseTemplates",
    "OFFLINE_FALLBACK_MESSAGE",
]
