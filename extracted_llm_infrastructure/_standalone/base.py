"""Standalone, torch-free ``BaseModelService`` for the LLM-infrastructure
package.

The atlas_brain version (``atlas_brain.services.base``) imports torch at
module load and exposes GPU memory diagnostics via the ``InferenceMetrics``
helper. The LLM-infra subsystem only needs the lifecycle hooks (logger,
``is_loaded`` flag, model_id, cache_path) for the cloud LLM providers --
they do not actually use the GPU memory accounting. This standalone copy
strips the torch dependency so the package is importable without it.

If a future caller needs GPU memory accounting, they can subclass this
and override ``gather_metrics`` / ``device``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from pathlib import Path
from typing import Any, Optional

from .protocols import InferenceMetrics


class BaseModelService(ABC):
    """Optional base class providing common utilities for model services.

    Services can inherit from this or implement the ``LLMService``
    Protocol directly.
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

        self.logger = logging.getLogger(f"atlas.{name}")
        if log_file and not self.logger.handlers:
            self._setup_logging(log_file)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Return the device string for inference.

        Without torch we cannot probe CUDA availability, so the
        standalone default is ``cpu``. Subclasses that integrate with
        local GPU runtimes (transformers, vLLM) should override this.
        """
        if self._device is None:
            self._device = "cpu"
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
        """Collect inference timing. GPU memory stats stay zero in
        standalone mode.
        """
        return InferenceMetrics(
            duration_ms=round(duration * 1000, 2),
            device=self.device,
        )

    def _clear_gpu_memory(self) -> None:
        """No-op in standalone mode (no torch)."""
        return None


class InferenceTimer:
    """Context manager for timing inference operations."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.duration: float = 0

    def __enter__(self) -> "InferenceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.duration = time.perf_counter() - self.start_time
