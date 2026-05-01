from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ConclusionResult:
    conclusion_id: str
    met: bool
    confidence: str
    fallback_label: str | None = None
    fallback_action: str | None = None


@dataclass(frozen=True, slots=True)
class SuppressionResult:
    suppress: bool = False
    degrade: bool = False
    disclaimer: str | None = None
    fallback_label: str | None = None


class EvidenceEngine:
    def __init__(self, map_path: str | Path | None = None) -> None:
        self.map_path = str(map_path or "")
        self.map_hash = "standalone"

    def evaluate_conclusions(
        self,
        vendor_evidence: dict[str, Any],
    ) -> list[ConclusionResult]:
        return []

    def evaluate_suppression(
        self,
        section: str,
        evidence: dict[str, Any],
    ) -> SuppressionResult:
        return SuppressionResult()


_engine: EvidenceEngine | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    global _engine
    if _engine is None or map_path is not None:
        _engine = EvidenceEngine(map_path)
    return _engine
