"""Ports for product-specific quality-gate integrations.

The core package defines protocols only. Atlas, customer deployments,
and extracted products bind these to their own stores, logs, clocks,
and model/embedding infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence

from .types import QualityPolicy


class Clock(Protocol):
    def now(self) -> datetime:
        """Return the current timestamp for deterministic gate events."""


class AuditLog(Protocol):
    async def record_gate_event(self, event: Mapping[str, Any]) -> None:
        """Persist or emit a quality-gate audit event."""


class ApprovalStore(Protocol):
    async def create_request(self, request: Mapping[str, Any]) -> str:
        """Create an approval request and return its identifier."""

    async def get_status(self, request_id: str) -> Mapping[str, Any] | None:
        """Return approval status, or None when the request is unknown."""


class EvidenceClaimStore(Protocol):
    async def fetch_claims(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        """Fetch evidence claims for an artifact."""


class EmbeddingSimilarity(Protocol):
    async def similarity(self, left: str, right: str) -> float:
        """Return a similarity score in the range expected by the caller."""


class PolicyProvider(Protocol):
    def get_policy(self, name: str, *, version: str | None = None) -> QualityPolicy:
        """Resolve a named quality policy."""


@dataclass(frozen=True)
class QualityPorts:
    clock: Clock | None = None
    audit_log: AuditLog | None = None
    approval_store: ApprovalStore | None = None
    evidence_claim_store: EvidenceClaimStore | None = None
    embedding_similarity: EmbeddingSimilarity | None = None
    policy_provider: PolicyProvider | None = None


__all__ = [
    "ApprovalStore",
    "AuditLog",
    "Clock",
    "EmbeddingSimilarity",
    "EvidenceClaimStore",
    "PolicyProvider",
    "QualityPorts",
]
