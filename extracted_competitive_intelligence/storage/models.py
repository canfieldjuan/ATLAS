"""Storage models used by extracted_competitive_intelligence.

Only the two dataclasses this package actually consumes are defined
here: ``ScheduledTask`` (used by ``b2b_battle_cards`` and
``b2b_vendor_briefing`` for autonomous-runner records) and
``CompetitiveSet`` (used by ``services/b2b_competitive_sets`` for
operator-defined vendor scopes).

The 20+ unrelated dataclasses in the atlas peer module
(User, Session, ConversationTurn, Memory, Alert, etc.) are NOT
exposed here -- they describe home-assistant / RAG / notification
domains that are not part of competitive intelligence and would
leak unrelated product surface.

Drift protection: ``tests/test_extracted_competitive_storage_models_parity.py``
asserts field-name parity against the atlas peer. If atlas adds or
renames a field on ``ScheduledTask`` or ``CompetitiveSet``, CI fails
loudly and the maintainer must update both copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID


def _utcnow() -> datetime:
    """Timezone-aware UTC now for dataclass defaults."""
    return datetime.now(timezone.utc)


@dataclass
class ScheduledTask:
    """A scheduled task for autonomous execution."""

    id: UUID
    name: str
    task_type: str  # "agent_prompt", "builtin", "hook"
    schedule_type: str  # "cron", "interval", "once"
    description: Optional[str] = None
    prompt: Optional[str] = None
    agent_type: str = "atlas"
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    run_at: Optional[datetime] = None
    timezone: str = "America/Chicago"
    enabled: bool = True
    max_retries: int = 0
    retry_delay_seconds: int = 60
    timeout_seconds: int = 120
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if isinstance(self.metadata, str):
            import json as _json
            self.metadata = _json.loads(self.metadata)
        elif self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "agent_type": self.agent_type,
            "schedule_type": self.schedule_type,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "timezone": self.timezone,
            "enabled": self.enabled,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
        }


@dataclass
class CompetitiveSet:
    """Operator-defined vendor scope for competitive synthesis."""

    id: UUID
    account_id: UUID
    name: str
    focal_vendor_name: str
    competitor_vendor_names: list[str] = field(default_factory=list)
    active: bool = True
    refresh_mode: str = "manual"
    refresh_interval_hours: Optional[int] = None
    vendor_synthesis_enabled: bool = True
    pairwise_enabled: bool = True
    category_council_enabled: bool = False
    asymmetry_enabled: bool = False
    last_run_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_run_status: Optional[str] = None
    last_run_summary: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "account_id": str(self.account_id),
            "name": self.name,
            "focal_vendor_name": self.focal_vendor_name,
            "competitor_vendor_names": self.competitor_vendor_names,
            "active": self.active,
            "refresh_mode": self.refresh_mode,
            "refresh_interval_hours": self.refresh_interval_hours,
            "vendor_synthesis_enabled": self.vendor_synthesis_enabled,
            "pairwise_enabled": self.pairwise_enabled,
            "category_council_enabled": self.category_council_enabled,
            "asymmetry_enabled": self.asymmetry_enabled,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_run_status": self.last_run_status,
            "last_run_summary": self.last_run_summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["CompetitiveSet", "ScheduledTask"]
