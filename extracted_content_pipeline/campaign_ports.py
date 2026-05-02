"""Standalone ports for the campaign generation product.

These interfaces define what the sellable campaign module is allowed to ask
from its host application. Product code should depend on these ports, not on
Atlas runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TenantScope:
    """Host-provided tenant/account context for scoped campaign reads."""

    account_id: str | None = None
    user_id: str | None = None
    allowed_vendors: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: Any | None = None


@dataclass(frozen=True)
class CampaignDraft:
    target_id: str
    target_mode: str
    channel: str
    subject: str
    body: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignReasoningContext:
    """Normalized host-provided reasoning context for campaign generation."""

    anchor_examples: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    witness_highlights: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    reference_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)
    account_signals: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    timing_windows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    proof_points: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    coverage_limits: Sequence[str] = field(default_factory=tuple)
    scope_summary: Mapping[str, Any] = field(default_factory=dict)
    delta_summary: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            key: value
            for key, value in {
                "anchor_examples": {
                    str(label): [dict(row) for row in rows]
                    for label, rows in self.anchor_examples.items()
                },
                "witness_highlights": [dict(row) for row in self.witness_highlights],
                "reference_ids": {
                    str(key): [str(item) for item in values]
                    for key, values in self.reference_ids.items()
                },
                "account_signals": [dict(row) for row in self.account_signals],
                "timing_windows": [dict(row) for row in self.timing_windows],
                "proof_points": [dict(row) for row in self.proof_points],
                "coverage_limits": [str(item) for item in self.coverage_limits],
                "scope_summary": dict(self.scope_summary),
                "delta_summary": dict(self.delta_summary),
            }.items()
            if value not in ({}, [], (), None)
        }

    def has_content(self) -> bool:
        return bool(self.as_dict())


@dataclass(frozen=True)
class SendRequest:
    campaign_id: str
    to_email: str
    subject: str
    html_body: str
    text_body: str | None = None
    from_email: str | None = None
    reply_to: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    tags: Sequence[Mapping[str, str]] = field(default_factory=tuple)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class SendResult:
    provider: str
    message_id: str
    raw: Any | None = None


@dataclass(frozen=True)
class WebhookEvent:
    provider: str
    event_type: str
    message_id: str | None = None
    email: str | None = None
    occurred_at: datetime | None = None
    payload: JsonDict = field(default_factory=dict)


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a completion for a campaign prompt."""


class SkillStore(Protocol):
    def get_prompt(self, name: str) -> str | None:
        """Return a prompt contract by name."""


class IntelligenceRepository(Protocol):
    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[JsonDict]:
        """Return source intelligence for campaign generation."""

    async def read_vendor_targets(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        vendor_name: str | None = None,
    ) -> Sequence[JsonDict]:
        """Return configured vendor/account targets."""


class CampaignReasoningContextProvider(Protocol):
    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | Mapping[str, Any] | None:
        """Return pre-compressed reasoning/witness context for one opportunity."""


class CampaignRepository(Protocol):
    async def save_drafts(
        self,
        drafts: Sequence[CampaignDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist generated drafts and return campaign ids."""

    async def list_due_sends(
        self,
        *,
        limit: int,
        now: datetime,
    ) -> Sequence[JsonDict]:
        """Return queued campaigns ready for send evaluation."""

    async def mark_sent(
        self,
        *,
        campaign_id: str,
        result: SendResult,
        sent_at: datetime,
    ) -> None:
        """Persist provider send result."""

    async def mark_cancelled(
        self,
        *,
        campaign_id: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a terminal cancellation before send."""

    async def mark_send_failed(
        self,
        *,
        campaign_id: str,
        error: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a failed send attempt."""

    async def record_webhook_event(self, event: WebhookEvent) -> None:
        """Persist provider engagement or delivery event."""

    async def refresh_analytics(self) -> None:
        """Refresh campaign analytics aggregates."""


class CampaignSequenceRepository(Protocol):
    async def list_due_sequences(
        self,
        *,
        limit: int,
        now: datetime,
    ) -> Sequence[JsonDict]:
        """Return active sequences ready for follow-up generation."""

    async def list_previous_campaigns(
        self,
        *,
        sequence_id: str,
        limit: int,
    ) -> Sequence[JsonDict]:
        """Return prior campaigns for a sequence in step order."""

    async def queue_sequence_step(
        self,
        *,
        sequence: JsonDict,
        content: JsonDict,
        from_email: str,
        queued_at: datetime,
    ) -> str:
        """Persist a generated follow-up campaign and return its id."""

    async def mark_sequence_step(
        self,
        *,
        sequence_id: str,
        current_step: int,
        updated_at: datetime,
    ) -> None:
        """Persist sequence progression after a follow-up is queued."""


class SuppressionRepository(Protocol):
    async def is_suppressed(
        self,
        *,
        email: str | None = None,
        domain: str | None = None,
    ) -> bool:
        """Return whether the recipient or domain is suppressed."""

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
    ) -> None:
        """Persist an email/domain suppression."""


class CampaignSender(Protocol):
    async def send(self, request: SendRequest) -> SendResult:
        """Send one campaign email through an ESP."""


class WebhookVerifier(Protocol):
    def verify_and_parse(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
    ) -> WebhookEvent:
        """Verify provider signature and normalize the webhook payload."""


class AuditSink(Protocol):
    async def record(
        self,
        event_type: str,
        *,
        campaign_id: str | None = None,
        sequence_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an immutable campaign lifecycle event."""


class VisibilitySink(Protocol):
    async def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Emit host-visible task progress without coupling to a host runtime."""


class Clock(Protocol):
    def now(self) -> datetime:
        """Return the current time for scheduling and tests."""
