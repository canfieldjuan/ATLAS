"""Compatibility wrapper for the shared extracted reasoning evidence engine.

PR-C1i replaced the prior ~338-line local fork with this thin wrapper.
The canonical implementation lives in
`extracted_reasoning_core.evidence_engine` (slim core split per PR-C1d /
PR #104). The content-pipeline-specific rule catalog (consumer-review
pipeline conclusions like `pricing_crisis`, `losing_market_share`,
`active_churn_wave`, `support_quality_risk`) lives in
`evidence_map.yaml` next to this wrapper.

The exported `EvidenceEngine` subclass defaults to the content-pipeline
YAML when called with no arguments and exposes the same `"builtin"`
sentinel for `map_path` that the prior fork did, so existing
content_pipeline callers and the
`tests/test_extracted_reasoning_evidence_engine.py` test suite keep
working unchanged.

Public contract types `ConclusionResult` / `SuppressionResult` are
re-exported from `extracted_reasoning_core.types` (promoted in PR-C1c).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from extracted_reasoning_core.evidence_engine import (
    EvidenceEngine as _CoreEvidenceEngine,
)
from extracted_reasoning_core.types import ConclusionResult, SuppressionResult


_DEFAULT_MAP_PATH = Path(__file__).parent / "evidence_map.yaml"


class EvidenceEngine(_CoreEvidenceEngine):
    """Content-pipeline-tuned evidence engine.

    Same evaluation logic as the core engine; differs only in default
    rule catalog (the consumer-review YAML next to this wrapper) and
    in carrying the prior fork's ``"builtin"`` ``map_path`` sentinel
    when no path is passed explicitly.
    """

    def __init__(self, map_path: str | Path | None = None) -> None:
        super().__init__(map_path or _DEFAULT_MAP_PATH)
        if map_path is None:
            self.map_path = "builtin"


_engine: EvidenceEngine | None = None
_engine_path: str | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    """Return a cached `EvidenceEngine` instance for the content pipeline.

    Defaults to the content-pipeline-specific `evidence_map.yaml`
    shipped next to this wrapper. Pass an explicit `map_path` to load a
    different rule catalog (e.g. tests injecting a tmp_path JSON file).
    """
    global _engine, _engine_path

    resolved = str(map_path) if map_path else "builtin"
    if _engine is None or _engine_path != resolved:
        _engine = EvidenceEngine(map_path)
        _engine_path = resolved
    return _engine


def reload_evidence_engine() -> EvidenceEngine:
    """Force-reload the cached engine (e.g. after on-disk YAML edits)."""
    global _engine, _engine_path
    _engine = None
    _engine_path = None
    return get_evidence_engine()


__all__ = [
    "ConclusionResult",
    "EvidenceEngine",
    "SuppressionResult",
    "get_evidence_engine",
    "reload_evidence_engine",
]
