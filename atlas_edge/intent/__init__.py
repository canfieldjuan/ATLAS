"""
Intent classification and parsing for edge devices.

Includes DistilBERT classifier and pattern-based device parsing.
"""

from .classifier import EdgeIntentClassifier, ClassificationResult
from .patterns import DevicePatternParser, DEVICE_PATTERNS
from .actions import ActionIntent

__all__ = [
    "EdgeIntentClassifier",
    "ClassificationResult",
    "DevicePatternParser",
    "DEVICE_PATTERNS",
    "ActionIntent",
]
