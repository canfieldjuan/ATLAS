"""Build grounded FAQ Markdown from support-ticket source evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
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
MAX_EXTRACTED_QUESTION_CHARS = 140
_WHITESPACE_RE = re.compile(r"\s+")
_KEY_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPEAKER_LABEL_RE = re.compile(
    r"(?:^|\n|[.!?]\s+)\s*(customer|user|requester|client|agent|support|representative|rep|admin)\s*:",
    re.IGNORECASE,
)
_CUSTOMER_SPEAKERS = {"customer", "user", "requester", "client"}
DEFAULT_INTENT_RULES = (
    ("reporting friction", ("export", "report", "dashboard", "attribution")),
    ("manual follow-up", ("handoff", "follow-up", "workflow", "automation", "manual")),
    ("login reset", ("login reset", "password reset", "reset password", "reset my password")),
    ("email and profile updates", ("change my email", "update the email", "email address", "profile")),
    ("login and account access", ("login", "account access")),
    ("billing and payments", ("billing", "invoice", "payment", "receipt", "charge")),
    ("integration setup", ("integration", "api", "webhook", "sync", "connection")),
    ("renewal and cancellation", ("renewal", "cancel", "cancellation", "contract")),
)
_DATE_KEYS = (
    "created_at",
    "created",
    "ticket_created_at",
    "case_created_at",
    "submitted_at",
    "opened_at",
    "received_at",
    "updated_at",
    "closed_at",
    "date",
    "timestamp",
)
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
    window_days: int | None = None
    as_of_date: str | None = None
    intent_rules: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_INTENT_RULES


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
        window_days: int | None = None,
        as_of_date: Any = None,
        intent_rules: Sequence[tuple[str, Sequence[str]]] | None = None,
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
        resolved_window_days = (
            int(window_days)
            if window_days is not None
            else self.config.window_days
        )
        resolved_as_of_date = as_of_date if as_of_date is not None else self.config.as_of_date
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
        resolved_intent_rules = (
            tuple((topic, tuple(keywords)) for topic, keywords in intent_rules)
            if intent_rules is not None
            else self.config.intent_rules
        )
        title_text = title or self.config.title
        result = build_ticket_faq_markdown(
            normalized.opportunities,
            title=title_text,
            max_items=resolved_max_items,
            max_evidence_per_item=resolved_max_evidence,
            source_types=resolved_source_types,
            window_days=resolved_window_days,
            as_of_date=resolved_as_of_date,
            intent_rules=resolved_intent_rules,
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
                    metadata={
                        "source_types": list(resolved_source_types),
                        **_date_window_metadata(
                            window_days=resolved_window_days,
                            as_of_date=resolved_as_of_date,
                        ),
                    },
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
    window_days: int | None = None,
    as_of_date: Any = None,
    intent_rules: Sequence[tuple[str, Sequence[str]]] = DEFAULT_INTENT_RULES,
) -> TicketFAQMarkdownResult:
    """Render an extractive FAQ from normalized source-row opportunities."""

    if max_items < 1:
        raise ValueError("max_items must be positive")
    if max_evidence_per_item < 1:
        raise ValueError("max_evidence_per_item must be positive")

    allowed = {_source_type_key(item) for item in source_types if _source_type_key(item)}
    date_window = _date_window(window_days=window_days, as_of_date=as_of_date)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for opportunity_index, opportunity in enumerate(opportunities, start=1):
        for evidence_index, evidence in enumerate(_evidence_rows(opportunity), start=1):
            if date_window is not None and not _inside_date_window(
                opportunity,
                evidence,
                date_window=date_window,
            ):
                continue
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
            groups[_topic(opportunity, evidence, intent_rules=intent_rules)].append({
                "text": text,
                "source_id": source_id or "unknown",
                "source_key": source_id or dedupe_id,
                "source_title": _clean(evidence.get("source_title") or opportunity.get("source_title")),
            })

    items = tuple(
        _item(topic, rows, max_evidence_per_item=max_evidence_per_item)
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


def _topic(
    opportunity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    intent_rules: Sequence[tuple[str, Sequence[str]]],
) -> str:
    intent = _intent_topic(opportunity, evidence, intent_rules=intent_rules)
    if intent:
        return intent
    pain_points = opportunity.get("pain_points")
    if isinstance(pain_points, Sequence) and not isinstance(pain_points, (str, bytes, bytearray)):
        for value in pain_points:
            text = _compact(value)
            if text:
                return text.lower()
    return (_compact(evidence.get("source_title") or opportunity.get("source_title")) or "customer support issues").lower()


def _intent_topic(
    opportunity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    intent_rules: Sequence[tuple[str, Sequence[str]]],
) -> str:
    text = " ".join((
        _compact(evidence.get("source_title") or opportunity.get("source_title")),
        _compact(evidence.get("text")),
        _compact(opportunity.get("text") or opportunity.get("description")),
        _pain_text(opportunity),
    )).lower()
    if not text:
        return ""
    for topic, keywords in intent_rules:
        if any(_keyword_matches(text, keyword) for keyword in keywords):
            return _clean(topic).lower()
    return ""


def _keyword_matches(text: str, keyword: Any) -> bool:
    value = _clean(keyword).lower()
    if not value:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _pain_text(opportunity: Mapping[str, Any]) -> str:
    pain_points = opportunity.get("pain_points")
    if isinstance(pain_points, Sequence) and not isinstance(pain_points, (str, bytes, bytearray)):
        return " ".join(_compact(value) for value in pain_points if _compact(value))
    return _compact(pain_points)


def _item(
    topic: str,
    rows: Sequence[Mapping[str, str]],
    *,
    max_evidence_per_item: int,
) -> dict[str, Any]:
    display_rows = rows[:max_evidence_per_item]
    sources = tuple(_source_label(row) for row in display_rows)
    source_keys = (row.get("source_key") or row.get("source_id", "") for row in rows)
    source_ids = tuple(dict.fromkeys(value for value in source_keys if value))
    snippets = " / ".join(_quote(row.get("text", "")) for row in display_rows)
    question, question_source = _question(topic, display_rows)
    return {
        "topic": topic,
        "question": question,
        "question_source": question_source,
        "answer": f"Customers mention: {snippets} Evidence comes from {len(source_ids)} ticket source(s).",
        "action_items": _action_items(topic, snippets),
        "source_ids": source_ids,
        "source_labels": sources,
        "evidence_count": len(display_rows),
        "displayed_evidence_count": len(display_rows),
        "ticket_count": len(source_ids),
    }


def _question(topic: str, rows: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    for row in rows:
        text = _question_text(row.get("text", ""))
        if text:
            return (text, "customer_wording")
    return (f"What are customers asking about {topic}?", "topic_fallback")


def _question_text(value: Any) -> str:
    for text in _question_candidate_texts(value):
        if "?" in text:
            prefix = text.split("?", 1)[0].strip()
            sentence_parts = [part.strip() for part in re.split(r"[.!:;]+", prefix) if part.strip()]
            candidate = sentence_parts[-1] if sentence_parts else prefix
            normalized = _normalize_question_text(candidate)
            if _usable_question(normalized):
                return normalized
            continue
        normalized = _question_start_text(text)
        if normalized:
            return normalized
    return ""


def _question_candidate_texts(value: Any) -> tuple[str, ...]:
    text = _clean(value)
    if not text:
        return ()
    matches = list(_SPEAKER_LABEL_RE.finditer(text))
    if not matches:
        return (_compact(_URL_RE.sub("", text)),)

    candidates = []
    leading = _question_segment(text[:matches[0].start()])
    if leading:
        candidates.append(leading)
    for index, match in enumerate(matches):
        speaker = match.group(1).lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if speaker in _CUSTOMER_SPEAKERS:
            segment = _question_segment(text[match.end():end])
            if segment:
                candidates.append(segment)
    return tuple(candidates)


def _question_segment(value: str) -> str:
    return _compact(_URL_RE.sub("", value))


def _question_start_text(text: str) -> str:
    question_starts = ("how ", "what ", "where ", "when ", "why ", "can ", "could ", "do ", "does ", "is ")
    lowered = text.lower()
    if lowered.startswith(question_starts):
        normalized = _normalize_question_text(text)
        if _usable_question(normalized):
            return normalized
    return ""


def _usable_question(value: str) -> bool:
    return bool(value) and len(value) <= MAX_EXTRACTED_QUESTION_CHARS


def _normalize_question_text(value: str) -> str:
    candidate = _compact(value).rstrip("?.!,;: ")
    if not candidate:
        return ""
    return f"{candidate}?"


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
            "",
            *[f"- {_md(step)}" for step in item["action_items"]],
            "",
            "**Sources:**",
            "",
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


def _date_window(*, window_days: int | None, as_of_date: Any) -> tuple[date, date] | None:
    if window_days is None:
        if _clean(as_of_date):
            raise ValueError("as_of_date requires window_days")
        return None
    days = int(window_days)
    if days < 1:
        raise ValueError("window_days must be positive")
    as_of = _parse_as_of_date(as_of_date) or date.today()
    return (as_of - timedelta(days=days), as_of)


def _inside_date_window(
    opportunity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    date_window: tuple[date, date],
) -> bool:
    source_date = _source_date(evidence) or _source_date(opportunity)
    if source_date is None:
        return False
    start, end = date_window
    return start <= source_date <= end


def _source_date(row: Mapping[str, Any]) -> date | None:
    for key in _DATE_KEYS:
        value = _field_value(row, key)
        parsed = _parse_source_date(value)
        if parsed is not None:
            return parsed
    return None


def _field_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    target = _source_type_key(key)
    compact_target = _compact_key(key)
    for raw_key, value in row.items():
        if _source_type_key(raw_key) == target or _compact_key(raw_key) == compact_target:
            return value
    return None


def _compact_key(value: Any) -> str:
    return _KEY_SEPARATOR_RE.sub("", _clean(value).lower())


def _parse_source_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_as_of_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if not text:
        return None
    parsed = _parse_source_date(value)
    try:
        strict = date.fromisoformat(text)
    except ValueError:
        raise ValueError("as_of_date must be a valid ISO date")
    if parsed != strict:
        raise ValueError("as_of_date must be a valid ISO date")
    return strict


def _date_window_metadata(*, window_days: int | None, as_of_date: Any) -> dict[str, Any]:
    if window_days is None:
        return {}
    metadata: dict[str, Any] = {"window_days": int(window_days)}
    parsed = _parse_as_of_date(as_of_date)
    if parsed is not None:
        metadata["as_of_date"] = parsed.isoformat()
    return metadata


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
    has_items = bool(items)
    return {
        "uses_user_vocabulary": has_items
        and all(item.get("question_source") == "customer_wording" for item in items),
        "condensed": has_items and (ticket_source_count <= 1 or len(items) < ticket_source_count),
        "has_action_items": has_items and all(bool(item.get("action_items")) for item in items),
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
    "DEFAULT_INTENT_RULES",
    "DEFAULT_TICKET_SOURCE_TYPES",
    "TicketFAQMarkdownConfig",
    "TicketFAQMarkdownResult",
    "TicketFAQMarkdownService",
    "build_ticket_faq_markdown",
]
