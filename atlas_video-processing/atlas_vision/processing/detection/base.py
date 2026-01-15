"""
Base detector protocol.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.protocols import Detection


@runtime_checkable
class BaseDetector(Protocol):
    """Protocol for all detectors."""

    @property
    def name(self) -> str:
        """Detector name."""
        ...

    def detect(self, frame: np.ndarray, source_id: str) -> list[Detection]:
        """
        Detect objects/motion in a frame.

        Args:
            frame: Input frame as numpy array (BGR format)
            source_id: Camera/source identifier

        Returns:
            List of Detection objects
        """
        ...

    def reset(self) -> None:
        """Reset detector state (e.g., background model)."""
        ...
