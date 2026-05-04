"""Local re-export for evidence hash helpers used by copied campaign tasks."""

from __future__ import annotations

from extracted_llm_infrastructure.reasoning.semantic_cache import compute_evidence_hash

__all__ = ["compute_evidence_hash"]
