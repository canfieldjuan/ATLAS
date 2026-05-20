"""Build grounded FAQ Markdown from support-ticket source evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

from .campaign_ports import TenantScope
from .campaign_source_adapters import source_rows_to_campaign_opportunities
from .ticket_faq_ports import TicketFAQDraft, TicketFAQRepository


DEFAULT_TICKET_SOURCE_TYPES = (
    "ticket",
    "support_ticket",
    "case",
    "conversation",
    "complaint",
)
DEFAULT_TITLE = "Customer Ticket FAQ"
_WHITESPACE_RE = re.compile(r"\s+")
_KEY_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_ACTION_RULES = (
    (("export", "report", "dashboard", "attribution"), (
        "Check whether your plan and role include the needed export or reporting access.",
        "If the option is missing, ask support or an admin to enable access or provide the export.",
    )),
    (("handoff", "follow-up", "workflow", "automation", "manual"), (
        "Check the workflow or automation rule that should handle this step.",
        "If the handoff still needs manual cleanup, document the exact step and escalate it.",
    )),
    (("login", "email", "profile", "password", "account"), (
        "Confirm the account details you are trying to change or access.",
        "If self-service does not work, contact support with the affected email or account id.",
    )),
)


@dataclass(frozen=True)
class TicketFAQMarkdownResult:
    """FAQ Markdown plus metadata useful for CLI summaries and tests."""

    markdown: str
    items: tuple[dict[str, Any], ...]
    source_count: int
    ticket_source_count: int
    output_checks: dict[str, bool]
    warnings: tuple[dict[str, Any], ...] = ()
    saved_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": len(self.items),
            "markdown": self.markdown,
            "items": [dict(item) for item in self.items],
            "source_count": self.source_count,
            "ticket_source_count": self.ticket_source_count,
            "output_checks": dict(self.output_checks),
            "warnings": [dict(warning) for warning in self.warnings],
            "saved_ids": list(self.saved_ids),
        }


@dataclass(frozen=True)
class TicketFAQMarkdownConfig:
    """Config for service-shaped FAQ Markdown generation."""

    title: str = DEFAULT_TITLE
    max_items: int = 8
    max_evidence_per_item: int = 3
    source_types: tuple[str, ...] = DEFAULT_TICKET_SOURCE_TYPES
    max_text_chars: int = 1200


class TicketFAQMarkdownService:
    """Build FAQ Markdown from inline source material."""

    def __init__(
        self,
        config: TicketFAQMarkdownConfig | None = None,
        *,
        ticket_faqs: TicketFAQRepository | None = None,
    ) -> None:
        self.config = config or TicketFAQMarkdownConfig()
        self._ticket_faqs = ticket_faqs

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        title: str | None = None,
        max_items: int | None = None,
        max_evidence_per_item: int | None = None,
        source_types: Sequence[str] | None = None,
        max_text_chars: int | None = None,
        **kwargs: Any,
    ) -> TicketFAQMarkdownResult:
        del kwargs
        resolved_max_items = int(max_items) if max_items is not None else self.config.max_items
        resolved_max_evidence = (
            int(max_evidence_per_item)
            if max_evidence_per_item is not None
            else self.config.max_evidence_per_item
        )
        resolved_max_text_chars = (
            int(max_text_chars)
            if max_text_chars is not None
            else self.config.max_text_chars
        )
        if resolved_max_text_chars < 1:
            raise ValueError("max_text_chars must be positive")
        normalized = source_rows_to_campaign_opportunities(
            _rows_from_source_material(source_material),
            target_mode=target_mode,
            max_text_chars=resolved_max_text_chars,
        )
        resolved_source_types = (
            tuple(source_types)
            if source_types is not None
            else self.config.source_types
        )
        title_text = title or self.config.title
        result = build_ticket_faq_markdown(
            normalized.opportunities,
            title=title_text,
            max_items=resolved_max_items,
            max_evidence_per_item=resolved_max_evidence,
            source_types=resolved_source_types,
        )
        result = replace(
            result,
            warnings=tuple(warning.as_dict() for warning in normalized.warnings),
        )
        if self._ticket_faqs is None or not result.items:
            return result
        saved_ids = await self._ticket_faqs.save_drafts(
            [
                TicketFAQDraft(
                    target_id=_target_id(normalized.opportunities),
                    target_mode=target_mode,
                    title=title_text,
                    markdown=result.markdown,
                    items=result.items,
                    source_count=result.source_count,
                    ticket_source_count=result.ticket_source_count,
                    output_checks=result.output_checks,
                    warnings=result.warnings,
                    metadata={"source_types": list(resolved_source_types)},
                )
            ],
            scope=scope,
        )
        return replace(result, saved_ids=tuple(str(item) for item in saved_ids))


def build_ticket_faq_markdown(
    opportunities: Sequence[Mapping[str, Any]],
    *,
    title: str = DEFAULT_TITLE,
    max_items: int = 8,
    max_evidence_per_item: int = 3,
    source_types: Sequence[str] = DEFAULT_TICKET_SOURCE_TYPES,
) -> TicketFAQMarkdownResult:
    """Render an extractive FAQ from normalized source-row opportunities."""

    if max_items < 1:
        raise ValueError("max_items must be positive")
    if max_evidence_per_item < 1:
        raise ValueError("max_evidence_per_item must be positive")

    allowed = {_source_type_key(item) for item in source_types if _source_type_key(item)}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for opportunity_index, opportunity in enumerate(opportunities, start=1):
        for evidence_index, evidence in enumerate(_evidence_rows(opportunity), start=1):
            source_type = _source_type_key(evidence.get("source_type") or opportunity.get("source_type"))
            # Empty allowed set means "no filter" rather than "reject all".
            if allowed and source_type not in allowed:
                continue
            text = _compact(evidence.get("text"))
            if not text:
                continue
            source_id = _clean(
                evidence.get("source_id")
                or opportunity.get("source_id")
                or opportunity.get("target_id")
                or opportunity.get("id")
            )
            dedupe_id = source_id or f"row:{opportunity_index}:evidence:{evidence_index}"
            key = (dedupe_id, text)
            if key in seen:
                continue
            seen.add(key)
            groups[_topic(opportunity, evidence)].append({
                "text": text,
                "source_id": source_id or "unknown",
                "source_key": source_id or dedupe_id,
                "source_title": _clean(evidence.get("source_title") or opportunity.get("source_title")),
            })

    items = tuple(
        _item(topic, rows[:max_evidence_per_item])
        for topic, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0].lower()))[:max_items]
    )
    return TicketFAQMarkdownResult(
        markdown=_render(title=title, items=items, source_count=len(opportunities), ticket_source_count=len(seen)),
        items=items,
        source_count=len(opportunities),
        ticket_source_count=len(seen),
        output_checks=_output_checks(items=items, ticket_source_count=len(seen)),
    )


def _evidence_rows(opportunity: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    evidence = opportunity.get("evidence")
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        rows = tuple(row for row in evidence if isinstance(row, Mapping))
        if rows:
            return rows
    return ({
        "text": opportunity.get("text") or opportunity.get("description") or "",
        "source_id": opportunity.get("source_id") or opportunity.get("target_id") or "",
        "source_type": opportunity.get("source_type") or "",
        "source_title": opportunity.get("source_title") or "",
    },)


def _topic(opportunity: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    pain_points = opportunity.get("pain_points")
    if isinstance(pain_points, Sequence) and not isinstance(pain_points, (str, bytes, bytearray)):
        for value in pain_points:
            text = _compact(value)
            if text:
                return text.lower()
    return (_compact(evidence.get("source_title") or opportunity.get("source_title")) or "customer support issues").lower()


def _item(topic: str, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    sources = tuple(_source_label(row) for row in rows)
    source_keys = (row.get("source_key") or row.get("source_id", "") for row in rows)
    source_ids = tuple(dict.fromkeys(value for value in source_keys if value))
    snippets = " / ".join(_quote(row.get("text", "")) for row in rows)
    return {
        "topic": topic,
        "question": f"What are customers asking about {topic}?",
        "answer": f"Customers mention: {snippets} Evidence comes from {len(source_ids)} ticket source(s).",
        "action_items": _action_items(topic, snippets),
        "source_ids": source_ids,
        "source_labels": sources,
        "evidence_count": len(rows),
    }


def _render(
    *,
    title: str,
    items: Sequence[Mapping[str, Any]],
    source_count: int,
    ticket_source_count: int,
) -> str:
    lines = [
        f"# {_md(title) or DEFAULT_TITLE}",
        "",
        f"_Source rows analyzed: {source_count}. Ticket evidence rows used: {ticket_source_count}._",
        "",
    ]
    if not items:
        lines.extend([
            "No ticket FAQ items were generated.",
            "",
            "Provide source rows with support-ticket, case, conversation, or complaint evidence.",
            "",
        ])
        return "\n".join(lines)
    for index, item in enumerate(items, start=1):
        lines.extend([
            f"## {index}. {_md(item['question'])}",
            "",
            _md(item["answer"]),
            "",
            "**What to do next:**",
            *[f"- {_md(step)}" for step in item["action_items"]],
            "",
            "**Sources:**",
            *[f"- {_md(label)}" for label in item["source_labels"]],
            "",
        ])
    return "\n".join(lines)


def _source_label(row: Mapping[str, str]) -> str:
    source_id = _clean(row.get("source_id"))
    title = _clean(row.get("source_title"))
    if source_id and title:
        return f"`{source_id}` - {title}"
    return f"`{source_id or 'unknown'}`"


def _action_items(topic: str, evidence_text: str) -> tuple[str, ...]:
    text = f"{topic} {evidence_text}".lower()
    for terms, steps in _ACTION_RULES:
        if any(term in text for term in terms):
            return steps
    return (
        "Check whether this issue affects your current account or workflow.",
        "Contact support with the cited ticket details if the answer does not resolve it.",
    )


def _output_checks(
    *,
    items: Sequence[Mapping[str, Any]],
    ticket_source_count: int,
) -> dict[str, bool]:
    return {
        "uses_user_vocabulary": all(_clean(item.get("topic")) for item in items),
        "condensed": len(items) <= ticket_source_count,
        "has_action_items": all(bool(item.get("action_items")) for item in items),
    }


def _quote(value: Any, *, limit: int = 220) -> str:
    text = _compact(value)
    if len(text) > limit:
        text = f"{text[:limit].rstrip()}..."
    return f'"{_md(text)}"'


def _compact(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _target_id(opportunities: Sequence[Mapping[str, Any]]) -> str:
    for opportunity in opportunities:
        for key in ("target_id", "source_id", "id", "company_name", "vendor_name"):
            value = _clean(opportunity.get(key))
            if value:
                return value
    return "ticket_faq_markdown"


def _source_type_key(value: Any) -> str:
    return _KEY_SEPARATOR_RE.sub("_", _clean(value).lower()).strip("_")


def _md(value: Any) -> str:
    return _clean(value).replace("|", "\\|")


def _rows_from_source_material(source_material: Any) -> list[Any]:
    if isinstance(source_material, str):
        text = source_material.strip()
        return [{"text": text, "source_type": "support_ticket"}] if text else []
    if isinstance(source_material, Mapping):
        for key in (
            "support_tickets",
            "tickets",
            "cases",
            "conversations",
            "complaints",
            "sources",
            "rows",
            "data",
            "opportunities",
        ):
            value = source_material.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                rows = list(value)
                if rows:
                    return rows
        return [dict(source_material)]
    if isinstance(source_material, Sequence) and not isinstance(source_material, (bytes, bytearray)):
        return list(source_material)
    return []


__all__ = [
    "DEFAULT_TICKET_SOURCE_TYPES",
    "TicketFAQMarkdownConfig",
    "TicketFAQMarkdownResult",
    "TicketFAQMarkdownService",
    "build_ticket_faq_markdown",
]
