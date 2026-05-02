"""Runnable offline campaign generation example over product ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from .campaign_generation import CampaignGenerationConfig, CampaignGenerationService
from .campaign_ports import (
    CampaignDraft,
    LLMClient,
    LLMMessage,
    LLMResponse,
    SkillStore,
    TenantScope,
)


_EXAMPLE_PROMPT = (
    "You are generating one outbound campaign draft from normalized customer "
    "opportunity data. Mode={target_mode}; opportunity={opportunity_json}"
)


class InMemoryIntelligenceRepository:
    """Tiny host repository for examples and customer-data smoke tests."""

    def __init__(self, opportunities: Sequence[Mapping[str, Any]]) -> None:
        self.opportunities = [dict(row) for row in opportunities]
        self.calls: list[dict[str, Any]] = []

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        self.calls.append(
            {
                "scope": scope,
                "target_mode": target_mode,
                "limit": limit,
                "filters": dict(filters or {}),
            }
        )
        return self.opportunities[:limit]

    async def read_vendor_targets(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        vendor_name: str | None = None,
    ) -> Sequence[dict[str, Any]]:  # pragma: no cover - protocol filler
        del scope
        del target_mode
        del vendor_name
        return []


class InMemoryCampaignRepository:
    """Capture generated drafts without requiring a database."""

    def __init__(self) -> None:
        self.drafts: list[CampaignDraft] = []

    async def save_drafts(
        self,
        drafts: Sequence[CampaignDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        del scope
        start = len(self.drafts) + 1
        self.drafts.extend(drafts)
        return [f"draft-{index}" for index in range(start, start + len(drafts))]

    async def list_due_sends(self, *, limit, now):  # pragma: no cover - protocol filler
        del limit
        del now
        return []

    async def mark_sent(self, *, campaign_id, result, sent_at):  # pragma: no cover
        del campaign_id
        del result
        del sent_at

    async def mark_cancelled(self, *, campaign_id, reason, metadata=None):  # pragma: no cover
        del campaign_id
        del reason
        del metadata

    async def mark_send_failed(self, *, campaign_id, error, metadata=None):  # pragma: no cover
        del campaign_id
        del error
        del metadata

    async def record_webhook_event(self, event):  # pragma: no cover
        del event

    async def refresh_analytics(self):  # pragma: no cover
        return None


class StaticCampaignSkillStore:
    """Host skill store used by the offline example."""

    def __init__(self, prompt: str = _EXAMPLE_PROMPT) -> None:
        self.prompt = prompt

    def get_prompt(self, name: str) -> str | None:
        del name
        return self.prompt


class DeterministicCampaignLLM:
    """Offline LLM stand-in that proves the product wiring end-to-end."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "metadata": dict(metadata or {}),
            }
        )
        opportunity = _extract_opportunity_from_prompt(messages)
        company = str(opportunity.get("company_name") or "your team")
        vendor = str(opportunity.get("vendor_name") or "your current platform")
        pains = opportunity.get("pain_points") or []
        pain = str(pains[0]) if pains else "workflow friction"
        subject = f"{company}: {pain}"[:80]
        body = (
            f"<p>{company} appears to be weighing {vendor} because of {pain}.</p>"
            "<p>We can turn that signal into a focused account sequence using "
            "your own data and approval rules.</p>"
        )
        return LLMResponse(
            content=json.dumps(
                {
                    "subject": subject,
                    "body": body,
                    "cta": "Review the generated sequence",
                    "angle_reasoning": "Offline deterministic draft from normalized opportunity data.",
                },
                separators=(",", ":"),
            ),
            model="offline-deterministic",
            usage={"input_tokens": 0, "output_tokens": 0},
        )


def _extract_opportunity_from_prompt(messages: Sequence[LLMMessage]) -> dict[str, Any]:
    content = "\n".join(str(message.content or "") for message in messages)
    marker = "opportunity="
    start = content.find(marker)
    if start < 0:
        return {}
    payload_start = content.find("{", start + len(marker))
    if payload_start < 0:
        return {}
    depth = 0
    payload_end = -1
    for index, char in enumerate(content[payload_start:], start=payload_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload_end = index + 1
                break
    if payload_end < 0:
        return {}
    raw = content[payload_start:payload_end]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scope_from_payload(payload: Mapping[str, Any]) -> TenantScope:
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        return TenantScope()
    return TenantScope(
        account_id=str(scope.get("account_id") or "") or None,
        user_id=str(scope.get("user_id") or "") or None,
        allowed_vendors=tuple(str(item) for item in scope.get("allowed_vendors") or ()),
        roles=tuple(str(item) for item in scope.get("roles") or ()),
    )


def _draft_to_dict(draft: CampaignDraft, campaign_id: str) -> dict[str, Any]:
    return {
        "id": campaign_id,
        "target_id": draft.target_id,
        "target_mode": draft.target_mode,
        "channel": draft.channel,
        "subject": draft.subject,
        "body": draft.body,
        "metadata": dict(draft.metadata),
    }


def _llm_model_label(llm: LLMClient, drafts: Sequence[CampaignDraft]) -> str:
    for draft in drafts:
        metadata = draft.metadata
        model = str(metadata.get("generation_model") or "").strip()
        if model:
            return model
    for attr in ("model", "model_id", "name", "openrouter_model", "workload"):
        model = str(getattr(llm, attr, "") or "").strip()
        if model:
            return model
    return llm.__class__.__name__


async def generate_campaign_drafts_from_payload(
    payload: Mapping[str, Any],
    *,
    llm: LLMClient | None = None,
    skills: SkillStore | None = None,
) -> dict[str, Any]:
    """Run campaign generation from a portable JSON-compatible payload."""
    opportunities = payload.get("opportunities")
    if not isinstance(opportunities, Sequence) or isinstance(opportunities, (str, bytes, bytearray)):
        raise ValueError("payload must include an opportunities array")
    target_mode = str(payload.get("target_mode") or "vendor_retention")
    channel = str(payload.get("channel") or "email")
    limit = int(payload.get("limit") or len(opportunities) or 20)

    intelligence = InMemoryIntelligenceRepository(
        [row for row in opportunities if isinstance(row, Mapping)]
    )
    campaigns = InMemoryCampaignRepository()
    llm_client = llm or DeterministicCampaignLLM()
    skill_store = skills or StaticCampaignSkillStore()
    service = CampaignGenerationService(
        intelligence=intelligence,
        campaigns=campaigns,
        llm=llm_client,
        skills=skill_store,
        config=CampaignGenerationConfig(channel=channel, limit=limit),
    )

    result = await service.generate(
        scope=_scope_from_payload(payload),
        target_mode=target_mode,
        limit=limit,
        filters=payload.get("filters") if isinstance(payload.get("filters"), Mapping) else None,
    )
    return {
        "result": result.as_dict(),
        "drafts": [
            _draft_to_dict(draft, campaign_id)
            for draft, campaign_id in zip(campaigns.drafts, result.saved_ids, strict=False)
        ],
        "llm_model": _llm_model_label(llm_client, campaigns.drafts),
    }


__all__ = [
    "DeterministicCampaignLLM",
    "InMemoryCampaignRepository",
    "InMemoryIntelligenceRepository",
    "StaticCampaignSkillStore",
    "generate_campaign_drafts_from_payload",
]
