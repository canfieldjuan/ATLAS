"""Standalone suppression policy for the campaign generation product."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .campaign_ports import SuppressionRepository


def normalize_email(email: str | None) -> str | None:
    """Normalize an email address for suppression matching."""
    if email is None:
        return None
    value = str(email).strip().lower()
    return value or None


def normalize_domain(domain: str | None) -> str | None:
    """Normalize a domain for suppression matching."""
    if domain is None:
        return None
    value = str(domain).strip().lower()
    if value.startswith("@"):
        value = value[1:]
    value = value.strip().strip(".")
    return value or None


def domain_from_email(email: str | None) -> str | None:
    """Extract a suppressible domain from a normalized email address."""
    value = normalize_email(email)
    if not value:
        return None
    local, sep, domain = value.rpartition("@")
    if not sep or not local or not domain:
        return None
    return normalize_domain(domain)


@dataclass(frozen=True)
class SuppressionInput:
    """Normalized suppression write payload."""

    reason: str
    email: str | None = None
    domain: str | None = None
    source: str = "system"
    campaign_id: str | None = None
    notes: str | None = None
    expires_at: datetime | None = None
    metadata: Mapping[str, Any] | None = None


class CampaignSuppressionService:
    """Campaign suppression checks decoupled from Atlas persistence."""

    def __init__(self, repository: SuppressionRepository):
        self._repository = repository

    async def is_suppressed(
        self,
        *,
        email: str | None,
        domain: str | None = None,
    ) -> bool:
        """Return whether an email or domain is suppressed.

        Exact email suppressions win first, matching Atlas' production helper.
        If no exact email match exists, the explicit domain or email domain is
        checked next.
        """
        normalized_email = normalize_email(email)
        if normalized_email:
            if await self._repository.is_suppressed(email=normalized_email, domain=None):
                return True

        normalized_domain = normalize_domain(domain) or domain_from_email(normalized_email)
        if normalized_domain:
            return bool(
                await self._repository.is_suppressed(
                    email=None,
                    domain=normalized_domain,
                )
            )
        return False

    async def add_suppression(
        self,
        *,
        reason: str,
        email: str | None = None,
        domain: str | None = None,
        source: str = "system",
        campaign_id: str | None = None,
        notes: str | None = None,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Persist a normalized suppression if it has an email or domain."""
        payload = build_suppression_input(
            reason=reason,
            email=email,
            domain=domain,
            source=source,
            campaign_id=campaign_id,
            notes=notes,
            expires_at=expires_at,
            metadata=metadata,
        )
        if payload is None:
            return False
        await self._repository.add_suppression(
            reason=payload.reason,
            email=payload.email,
            domain=payload.domain,
            source=payload.source,
            campaign_id=payload.campaign_id,
            notes=payload.notes,
            expires_at=payload.expires_at,
            metadata=payload.metadata,
        )
        return True


def build_suppression_input(
    *,
    reason: str,
    email: str | None = None,
    domain: str | None = None,
    source: str = "system",
    campaign_id: str | None = None,
    notes: str | None = None,
    expires_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SuppressionInput | None:
    """Normalize and validate a suppression write payload."""
    normalized_email = normalize_email(email)
    normalized_domain = normalize_domain(domain)
    normalized_reason = str(reason or "").strip()
    normalized_source = str(source or "system").strip() or "system"
    if not normalized_reason:
        raise ValueError("reason is required")
    if not normalized_email and not normalized_domain:
        return None
    return SuppressionInput(
        reason=normalized_reason,
        email=normalized_email,
        domain=normalized_domain,
        source=normalized_source,
        campaign_id=str(campaign_id).strip() if campaign_id else None,
        notes=str(notes).strip() if notes else None,
        expires_at=expires_at,
        metadata=metadata,
    )
