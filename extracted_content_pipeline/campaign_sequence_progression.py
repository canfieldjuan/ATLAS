"""Standalone campaign sequence progression orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Mapping, Sequence

from .campaign_ports import (
    AuditSink,
    CampaignSequenceRepository,
    Clock,
    LLMClient,
    LLMMessage,
    SkillStore,
)
from .campaign_sequence_context import (
    DEFAULT_LIMITS,
    SequenceContextLimits,
    plain_text_preview,
    prepare_sequence_prompt_contexts,
    prompt_max_tokens,
)


@dataclass(frozen=True)
class CampaignSequenceProgressionConfig:
    enabled: bool = True
    batch_limit: int = 20
    max_steps: int = 5
    from_email: str = ""
    onboarding_product_name: str = ""
    temperature: float = 0.7
    context_limits: SequenceContextLimits = DEFAULT_LIMITS


@dataclass(frozen=True)
class CampaignSequenceProgressionResult:
    due_sequences: int = 0
    progressed: int = 0
    skipped: int = 0
    disabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "due_sequences": self.due_sequences,
            "progressed": self.progressed,
            "skipped": self.skipped,
            "disabled": self.disabled,
        }


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def sequence_max_steps(
    sequence: Mapping[str, Any],
    *,
    config: CampaignSequenceProgressionConfig | None = None,
) -> int:
    configured = config or CampaignSequenceProgressionConfig()
    return int(sequence.get("max_steps") or configured.max_steps)


def build_engagement_summary(
    sequence: Mapping[str, Any],
    previous_campaigns: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    parts: list[str] = []
    opens = int(sequence.get("open_count") or 0)
    clicks = int(sequence.get("click_count") or 0)

    parts.append(f"Opened {opens} time(s)" if opens > 0 else "No opens recorded")
    parts.append(f"Clicked {clicks} time(s)" if clicks > 0 else "No clicks recorded")

    if sequence.get("reply_received_at"):
        intent = str(sequence.get("reply_intent") or "unknown")
        summary = str(sequence.get("reply_summary") or "")
        parts.append(f"Reply received ({intent}): {summary[:200]}")

    if previous_campaigns:
        parts.append("")
        parts.append("Per-step breakdown:")
        for campaign in previous_campaigns:
            step = campaign.get("step_number", "?")
            opened = "Opened" if campaign.get("opened_at") else "No opens"
            clicked = "Clicked" if campaign.get("clicked_at") else "No clicks"
            parts.append(f"- Step {step}: {opened}, {clicked}")
    return "\n".join(parts)


def build_previous_emails(
    campaigns: Sequence[Mapping[str, Any]],
    *,
    limits: SequenceContextLimits = DEFAULT_LIMITS,
) -> str:
    if not campaigns:
        return "No previous emails sent."

    parts: list[str] = []
    for campaign in campaigns:
        step = campaign.get("step_number", "?")
        subject = campaign.get("subject") or "(no subject)"
        body = plain_text_preview(str(campaign.get("body") or ""), limits=limits)
        status = campaign.get("status") or ""

        engagement_parts: list[str] = []
        if campaign.get("opened_at"):
            engagement_parts.append("Opened")
        if campaign.get("clicked_at"):
            engagement_parts.append("Clicked")
        engagement_line = (
            f"Engagement: {' | '.join(engagement_parts)}"
            if engagement_parts
            else "Engagement: No opens or clicks recorded"
        )

        parts.append(
            f"--- Step {step} (status: {status}) ---\n"
            f"Subject: {subject}\n"
            f"{engagement_line}\n"
            f"Preview: {body or '(no body)'}\n"
        )
    return "\n".join(parts)


def sequence_skill_name(recipient_type: str | None) -> str:
    if recipient_type == "onboarding":
        return "digest/b2b_onboarding_sequence"
    if recipient_type == "amazon_seller":
        return "digest/amazon_seller_campaign_sequence"
    if recipient_type == "vendor_retention":
        return "digest/b2b_vendor_sequence"
    if recipient_type == "challenger_intel":
        return "digest/b2b_challenger_sequence"
    return "digest/b2b_campaign_sequence"


def target_mode_for_recipient_type(recipient_type: str | None) -> str:
    if recipient_type == "amazon_seller":
        return "amazon_seller"
    if recipient_type == "vendor_retention":
        return "vendor_retention"
    if recipient_type == "challenger_intel":
        return "challenger_intel"
    return "churning_company"


def parse_generated_sequence_step(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("subject"):
        return parsed

    depth = 0
    start = -1
    for index, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("subject"):
                    return parsed
                start = -1
    return None


class CampaignSequenceProgressionService:
    """Generate and queue due follow-up campaign steps through product ports."""

    def __init__(
        self,
        *,
        sequences: CampaignSequenceRepository,
        llm: LLMClient,
        skills: SkillStore,
        audit: AuditSink | None = None,
        clock: Clock | None = None,
        config: CampaignSequenceProgressionConfig | None = None,
    ):
        self._sequences = sequences
        self._llm = llm
        self._skills = skills
        self._audit = audit
        self._clock = clock or SystemClock()
        self._config = config or CampaignSequenceProgressionConfig()

    async def progress_due(self) -> CampaignSequenceProgressionResult:
        if not self._config.enabled:
            return CampaignSequenceProgressionResult(disabled=True)

        now = self._clock.now()
        due = [
            dict(row)
            for row in await self._sequences.list_due_sequences(
                limit=self._config.batch_limit,
                now=now,
            )
        ]
        if not due:
            return CampaignSequenceProgressionResult()

        progressed = 0
        skipped = 0
        for sequence in due:
            sequence_id = str(sequence.get("id") or "").strip()
            if not sequence_id:
                skipped += 1
                continue

            max_steps = sequence_max_steps(sequence, config=self._config)
            previous = [
                dict(row)
                for row in await self._sequences.list_previous_campaigns(
                    sequence_id=sequence_id,
                    limit=max_steps,
                )
            ]
            content = await self.generate_next_step(sequence, previous)
            if not content:
                skipped += 1
                continue

            next_step = int(sequence.get("current_step") or 0) + 1
            content = {
                **content,
                "step_number": next_step,
                "target_mode": target_mode_for_recipient_type(
                    content.get("_recipient_type")
                ),
                "product_category": (
                    content.get("_category")
                    if content.get("_recipient_type") == "amazon_seller"
                    else None
                ),
            }
            campaign_id = await self._sequences.queue_sequence_step(
                sequence=sequence,
                content=content,
                from_email=self._config.from_email,
                queued_at=now,
            )
            await self._sequences.mark_sequence_step(
                sequence_id=sequence_id,
                current_step=next_step,
                updated_at=now,
            )
            await self._record_audit(
                "generated",
                campaign_id=campaign_id,
                sequence_id=sequence_id,
                step_number=next_step,
                sequence=sequence,
                content=content,
            )
            await self._record_audit(
                "queued",
                campaign_id=campaign_id,
                sequence_id=sequence_id,
                step_number=next_step,
                sequence=sequence,
                content=content,
            )
            progressed += 1

        return CampaignSequenceProgressionResult(
            due_sequences=len(due),
            progressed=progressed,
            skipped=skipped,
        )

    async def generate_next_step(
        self,
        sequence: Mapping[str, Any],
        previous_campaigns: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        company_context, selling_context = prepare_sequence_prompt_contexts(
            dict(sequence),
            limits=self._config.context_limits,
        )
        recipient_type = str(company_context.get("recipient_type") or "").strip() or None
        skill = self._skills.get_prompt(sequence_skill_name(recipient_type))
        if not skill:
            return None

        replacements = self._template_replacements(
            sequence,
            previous_campaigns,
            company_context=company_context,
            selling_context=selling_context,
        )
        system_prompt = skill
        for placeholder, value in replacements.items():
            system_prompt = system_prompt.replace(placeholder, value)

        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content="Generate the next email in this sequence.",
                ),
            ],
            max_tokens=prompt_max_tokens(self._config.context_limits),
            temperature=self._config.temperature,
            metadata={
                "sequence_id": sequence.get("id"),
                "company_name": sequence.get("company_name"),
                "recipient_type": recipient_type,
            },
        )
        parsed = parse_generated_sequence_step(response.content)
        if not parsed:
            return None
        return {
            **parsed,
            "_recipient_type": recipient_type,
            "_category": company_context.get("category"),
        }

    def _template_replacements(
        self,
        sequence: Mapping[str, Any],
        previous_campaigns: Sequence[Mapping[str, Any]],
        *,
        company_context: Mapping[str, Any],
        selling_context: Mapping[str, Any],
    ) -> dict[str, str]:
        days_since = "N/A"
        last_sent_at = sequence.get("last_sent_at")
        if isinstance(last_sent_at, datetime):
            days_since = str((self._clock.now() - last_sent_at).days)

        max_steps = sequence_max_steps(sequence, config=self._config)
        replacements = {
            "{company_name}": str(sequence.get("company_name") or ""),
            "{company_context}": json.dumps(
                dict(company_context),
                separators=(",", ":"),
                default=str,
            ),
            "{selling_context}": json.dumps(
                dict(selling_context),
                separators=(",", ":"),
                default=str,
            ),
            "{current_step}": str(int(sequence.get("current_step") or 0) + 1),
            "{max_steps}": str(max_steps),
            "{days_since_last}": days_since,
            "{engagement_summary}": build_engagement_summary(
                sequence,
                previous_campaigns,
            ),
            "{previous_emails}": build_previous_emails(
                previous_campaigns,
                limits=self._config.context_limits,
            ),
            "{product_name}": str(
                company_context.get("product_name")
                or self._config.onboarding_product_name
            ),
        }
        if company_context.get("recipient_type") == "amazon_seller":
            category_intel = company_context.get("category_intelligence")
            replacements.update({
                "{recipient_name}": str(company_context.get("seller_name") or ""),
                "{recipient_company}": str(company_context.get("seller_name") or ""),
                "{recipient_type}": "amazon_seller",
                "{category}": str(company_context.get("category") or ""),
                "{category_intelligence}": json.dumps(
                    category_intel if isinstance(category_intel, dict) else {},
                    separators=(",", ":"),
                    default=str,
                ),
            })
        return replacements

    async def _record_audit(
        self,
        event_type: str,
        *,
        campaign_id: str,
        sequence_id: str,
        step_number: int,
        sequence: Mapping[str, Any],
        content: Mapping[str, Any],
    ) -> None:
        if not self._audit:
            return
        await self._audit.record(
            event_type,
            campaign_id=campaign_id,
            sequence_id=sequence_id,
            metadata={
                "step_number": step_number,
                "subject": content.get("subject"),
                "recipient_email": sequence.get("recipient_email"),
                "angle_reasoning": content.get("angle_reasoning"),
            },
        )
