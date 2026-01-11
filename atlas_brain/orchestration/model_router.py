"""
Model Router for intelligent LLM selection.

Routes queries to appropriate model tier based on complexity.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .complexity_analyzer import ComplexityAnalyzer, QueryComplexity, get_complexity_analyzer

logger = logging.getLogger("atlas.model_router")


@dataclass
class SelectedTier:
    """Selected model tier information."""
    name: str
    model_path: str
    complexity: QueryComplexity
    score: float


class ModelRouter:
    """Routes queries to appropriate LLM based on complexity."""

    def __init__(self, config: Any):
        self.config = config
        self._analyzer = get_complexity_analyzer()
        self._current_model: Optional[str] = None
        self._last_swap: Optional[datetime] = None
        self._swap_count: int = 0

    def select_tier(self, query: str, context: Any = None) -> SelectedTier:
        """
        Select the appropriate model tier for the query.

        Args:
            query: User query text
            context: Optional pipeline context

        Returns:
            SelectedTier with model info
        """
        if not self.config.enabled:
            return SelectedTier(
                name=self.config.medium_model_name,
                model_path=self.config.medium_model_path,
                complexity=QueryComplexity.MEDIUM,
                score=0.5,
            )

        result = self._analyzer.analyze(query, context)

        if result.level == QueryComplexity.SIMPLE:
            return SelectedTier(
                name=self.config.simple_model_name,
                model_path=self.config.simple_model_path,
                complexity=result.level,
                score=result.score,
            )
        elif result.level == QueryComplexity.COMPLEX:
            return SelectedTier(
                name=self.config.complex_model_name,
                model_path=self.config.complex_model_path,
                complexity=result.level,
                score=result.score,
            )
        else:
            return SelectedTier(
                name=self.config.medium_model_name,
                model_path=self.config.medium_model_path,
                complexity=result.level,
                score=result.score,
            )

    def ensure_model_loaded(self, tier: SelectedTier) -> bool:
        """
        Ensure the selected model is loaded.

        Args:
            tier: Selected tier info

        Returns:
            True if model was swapped, False if already loaded
        """
        from ..services import llm_registry

        active = llm_registry.get_active()
        if active is None:
            self._load_model(tier)
            return True

        current_name = getattr(active, "model_id", None)
        if current_name == tier.name:
            return False

        logger.info("Swapping model: %s -> %s", current_name, tier.name)
        self._load_model(tier)
        self._swap_count += 1
        self._last_swap = datetime.now()
        return True

    def _load_model(self, tier: SelectedTier) -> None:
        """Load a model tier."""
        from ..services import llm_registry

        model_path = Path(tier.model_path)
        if not model_path.exists():
            logger.warning("Model not found: %s, using default", tier.model_path)
            return

        llm_registry.activate(
            "llama-cpp",
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
        )
        self._current_model = tier.name
        logger.info("Loaded model: %s", tier.name)

    def get_stats(self) -> dict:
        """Get routing statistics."""
        return {
            "enabled": self.config.enabled,
            "current_model": self._current_model,
            "swap_count": self._swap_count,
            "last_swap": self._last_swap.isoformat() if self._last_swap else None,
        }


_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Get or create the model router singleton."""
    global _router
    if _router is None:
        from ..config import settings
        _router = ModelRouter(settings.routing)
    return _router


def reset_model_router() -> None:
    """Reset the model router singleton."""
    global _router
    _router = None
