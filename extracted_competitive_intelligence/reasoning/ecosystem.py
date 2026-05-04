"""Host integration port for competitive ecosystem analysis."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class EcosystemAnalyzerPortNotConfigured(RuntimeError):
    """Raised when a host has not registered ecosystem analysis."""


class EcosystemAnalyzerClient(Protocol):
    async def analyze_all_categories(self) -> dict[str, Any]:
        """Return category-keyed ecosystem evidence."""


EcosystemAnalyzerFactory = Callable[[Any], EcosystemAnalyzerClient]


_ecosystem_analyzer_factory: EcosystemAnalyzerFactory | None = None


def configure_ecosystem_analyzer_factory(
    factory: EcosystemAnalyzerFactory | None,
) -> None:
    """Register the host adapter used by battle-card ecosystem enrichment."""
    global _ecosystem_analyzer_factory
    _ecosystem_analyzer_factory = factory


class EcosystemAnalyzer:
    """Compatibility facade for the host-provided ecosystem analyzer."""

    def __init__(self, pool: Any):
        self._pool = pool

    async def analyze_all_categories(self) -> dict[str, Any]:
        if _ecosystem_analyzer_factory is None:
            raise EcosystemAnalyzerPortNotConfigured(
                "No ecosystem analyzer factory has been configured"
            )
        analyzer = _ecosystem_analyzer_factory(self._pool)
        return await analyzer.analyze_all_categories()


__all__ = [
    "EcosystemAnalyzer",
    "EcosystemAnalyzerClient",
    "EcosystemAnalyzerFactory",
    "EcosystemAnalyzerPortNotConfigured",
    "configure_ecosystem_analyzer_factory",
]
