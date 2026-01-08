"""
Base class providing shared utilities for AI model services.

Services can inherit from this class for common functionality like
metrics collection, logging setup, and device detection.
"""

import logging
import time
from abc import ABC
from pathlib import Path
from typing import Any, Optional

import torch

from .protocols import InferenceMetrics


class BaseModelService(ABC):
    """
    Optional base class providing common utilities for model services.

    Services can inherit from this or implement the Protocol directly.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        cache_path: Optional[Path] = None,
        log_file: Optional[Path] = None,
    ):
        self.name = name
        self.model_id = model_id
        self.cache_path = cache_path or Path("/app/models") / name
        self._model: Any = None
        self._device: Optional[str] = None

        # Setup logging
        self.logger = logging.getLogger(f"atlas.{name}")
        if log_file and not self.logger.handlers:
            self._setup_logging(log_file)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get the device (cuda/cpu) for model inference."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _setup_logging(self, log_file: Path) -> None:
        """Configure file logging for this service."""
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def gather_metrics(self, duration: float) -> InferenceMetrics:
        """Collect inference timing and GPU memory stats."""
        metrics = InferenceMetrics(
            duration_ms=round(duration * 1000, 2),
            device=self.device,
        )

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            metrics.memory_allocated_mb = round(
                torch.cuda.memory_allocated(idx) / (1024 ** 2), 2
            )
            metrics.memory_reserved_mb = round(
                torch.cuda.memory_reserved(idx) / (1024 ** 2), 2
            )
            metrics.memory_total_mb = round(props.total_memory / (1024 ** 2), 2)

        return metrics

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache after unloading a model."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared")


class InferenceTimer:
    """Context manager for timing inference operations."""

    def __init__(self):
        self.start_time: float = 0
        self.duration: float = 0

    def __enter__(self) -> "InferenceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.duration = time.perf_counter() - self.start_time
