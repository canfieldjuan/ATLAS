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
_SUPPORTED_QUESTION_SOURCES = {"customer_wording", "source_policy"}
_GENERIC_QUESTION_TEXTS = {
    "help?",
    "need help?",
    "please help?",
    "can you help?",
    "what should i do?",
    "what can i do?",
}
_GENERIC_QUESTION_PHRASES = (
    "submitted several complaints",
    "filed several complaints",
    "submitted a complaint",
    "filing this complaint",
    "this complaint is about",
)
_REDACTION_TOKEN_RE = re.compile(r"\bX{2,}\b", re.IGNORECASE)
_REDACTED_DATE_RE = re.compile(r"\bX{2}/X{2}/(?:X{2,4}|\d{2,4})\b", re.IGNORECASE)
_REDACTED_MONEY_RE = re.compile(r"\{\$|\$\s*X{2,}", re.IGNORECASE)
_TRAILING_TITLE_RE = re.compile(r"\b(?:mr|mrs|ms|dr)\.?$", re.IGNORECASE)
_QUOTE_ARTIFACT_RE = re.compile(r"(?:''|``)")
_MORTGAGE_INTENT_TERMS = (
    "mortgage",
    "home loan",
    "foreclosure",
    "reverse mortgage",
    "mortgage servicer",
)
_MORTGAGE_ACTION_STEPS = (
    "Gather the mortgage statement, payment history, escrow record, payoff quote, or loss-mitigation notice tied to the issue.",
    "Send the servicer a written request or dispute with the dates, amounts, account number, and copies of the records you want reviewed.",
)
DEFAULT_INTENT_RULES = (
    ("credit report disputes", (
        "credit report",
        "credit file",
        "credit bureau",
        "credit bureaus",
        "credit reporting",
        "incorrect information",
    )),
    ("debt collection disputes", (
        "debt not owed",
        "not owe",
        "do not owe",
        "collect debt",
        "debt collection",
        "collector",
        "debt collector",
        "collection letter",
        "collection agency",
        "debt validation",
        "validation letter",
        "medical debt",
        "settled the debt",
    )),
    ("mortgage servicing issues", _MORTGAGE_INTENT_TERMS),
    ("reporting friction", (
        "export",
        "report export",
        "dashboard",
        "analytics report",
        "attribution",
    )),
    ("opening an account", ("opening an account", "open an account", "opened an account")),
    ("closing an account", ("closing an account", "close an account", "closed an account")),
    ("getting a credit card", ("getting a credit card", "applied for a credit card")),
    ("advertising", ("advertising", "promotion offer", "promotional offer")),
    ("other transaction problem", ("transaction problem", "failed scheduled transaction")),
    ("manual follow-up", ("handoff", "follow-up", "workflow", "automation", "manual")),
    ("login reset", ("login reset", "password reset", "reset password", "reset my password")),
    ("email and profile updates", ("change my email", "update the email", "email address", "profile")),
    ("login and account access", ("login", "account access")),
    ("billing and payments", (
        "billing",
        "invoice",
        "payment",
        "receipt",
        "charge",
        "charged",
        "fee",
        "fees",
        "interest",
        "loan",
        "lease",
        "statement",
        "dispute",
    )),
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
_SOURCE_CONTEXT_KEYS = (
    "product",
    "category",
    "sub_product",
    "issue",
    "sub_issue",
)
_ACTION_RULES = (
    (("credit report", "credit file", "credit bureau", "credit bureaus", "credit reporting", "incorrect information"), (
        "Get your latest credit reports and mark the account, date, balance, or status that looks wrong.",
        "File a dispute with the credit bureau and the company that supplied the information, then keep the confirmation numbers and copies of your records.",
    )),
    (("debt not owed", "not owe", "do not owe", "collect debt", "debt collection", "collector", "debt collector", "collection letter", "collection agency", "debt validation", "validation letter", "medical debt", "settled the debt"), (
        "Ask the collector in writing to identify the original creditor, the amount, the account, and why they say you owe it.",
        "Compare the notice with your payment, settlement, insurance, or provider records before you pay or share more information.",
    )),
    (_MORTGAGE_INTENT_TERMS, _MORTGAGE_ACTION_STEPS),
    (("opening an account", "open an account", "opened an account", "early warning services", "identity theft"), (
        "Gather the application, account-opening notice, denial reason, identity-theft report, or bank message tied to the issue.",
        "Ask the bank or card issuer in writing to explain the account decision, fraud record, bonus term, or access restriction and keep copies of the response.",
    )),
    (("getting a credit card", "applied for a credit card", "credit card application", "activate card"), (
        "Gather the card application, offer, denial notice, account terms, or activation message tied to the issue.",
        "Ask the issuer to explain the decision, promotion, account status, or card record in writing before you reapply or activate anything.",
    )),
    (("export", "dashboard", "attribution", "analytics report", "download report", "report export"), (
        "Open the reporting or analytics area and choose the date range you need.",
        "Look for an Export or Download option, then ask an admin to check your role and plan access if it is missing.",
    )),
    (("handoff", "follow-up", "workflow", "automation", "manual"), (
        "Find the workflow or automation rule that should handle this step.",
        "Check the last failed handoff, note which step stopped, and send that detail to support if it still needs manual cleanup.",
    )),
    (("billing", "invoice", "payment", "receipt", "charge", "fee", "fees", "interest", "loan", "lease", "statement", "dispute"), (
        "Open the bill, statement, payment history, or dispute record connected to the issue.",
        "Compare the charge, fee, payment, or balance against your receipt, contract, or written confirmation.",
    )),
    (("login", "email", "profile", "password", "account"), (
        "Open your profile, account settings, or login settings and find the email, password, or account field you need to change.",
        "Save the change, then check the old and new inboxes for a confirmation message.",
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
    support_contact: str | None = None
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
        support_contact: str | None = None,
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
        resolved_support_contact = (
            support_contact
            if support_contact is not None
            else self.config.support_contact
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
            support_contact=resolved_support_contact,
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
                        **_support_contact_metadata(resolved_support_contact),
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
    support_contact: str | None = None,
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
    source_keys: set[str] = set()

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
            source_key = source_id or f"row:{opportunity_index}"
            dedupe_id = source_id or f"{source_key}:evidence:{evidence_index}"
            key = (dedupe_id, text)
            if key in seen:
                continue
            seen.add(key)
            source_keys.add(source_key)
            groups[_topic(opportunity, evidence, intent_rules=intent_rules)].append({
                "text": text,
                "source_id": source_id or "unknown",
                "source_key": source_key,
                "source_title": _clean(evidence.get("source_title") or opportunity.get("source_title")),
            })

    sorted_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0].lower()))
    if len(sorted_groups) > max_items:
        visible_groups = tuple(sorted_groups[: max(1, max_items - 1)])
        overflow_rows: list[dict[str, str]] = []
        for _topic_name, rows in sorted_groups[len(visible_groups):]:
            overflow_rows.extend(rows)
        if len(visible_groups) == max_items:
            topic, rows = visible_groups[0]
            selected_groups = ((topic, [*rows, *overflow_rows]),)
        else:
            selected_groups = (*visible_groups, ("other support issues", overflow_rows))
    else:
        selected_groups = tuple(sorted_groups)

    items = tuple(
        _item(
            topic,
            rows,
            max_evidence_per_item=max_evidence_per_item,
            support_contact=support_contact,
        )
        for topic, rows in selected_groups
    )
    return TicketFAQMarkdownResult(
        markdown=_render(title=title, items=items, source_count=len(opportunities), ticket_source_count=len(source_keys)),
        items=items,
        source_count=len(opportunities),
        ticket_source_count=len(source_keys),
        output_checks=_output_checks(
            items=items,
            ticket_source_count=len(source_keys),
            rendered_ticket_source_count=_rendered_ticket_source_count(items),
        ),
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
        _source_context_text(opportunity, evidence),
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


def _source_context_text(*rows: Mapping[str, Any]) -> str:
    values: list[str] = []
    for row in rows:
        for key in _SOURCE_CONTEXT_KEYS:
            text = _compact(_field_value(row, key))
            if text:
                values.append(text)
    return " ".join(values)


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
    support_contact: str | None,
) -> dict[str, Any]:
    display_rows = rows[:max_evidence_per_item]
    sources = tuple(_source_label(row) for row in display_rows)
    source_keys = (row.get("source_key") or row.get("source_id", "") for row in rows)
    source_ids = tuple(dict.fromkeys(value for value in source_keys if value))
    snippets = " / ".join(_quote(row.get("text", "")) for row in display_rows)
    action_context = " / ".join((
        snippets,
        " / ".join(_clean(row.get("source_title")) for row in display_rows if _clean(row.get("source_title"))),
    ))
    question, question_source = _question(topic, display_rows)
    summary = _summary(topic=topic, rows=display_rows, source_count=len(source_ids))
    steps = _article_steps(topic, action_context, support_contact=support_contact)
    escalation = _escalation_guidance(topic, action_context, support_contact=support_contact)
    evidence_quotes = tuple(_evidence_quote(row) for row in display_rows)
    return {
        "topic": topic,
        "question": question,
        "question_source": question_source,
        "answer": f"Customers mention: {snippets} Evidence comes from {len(source_ids)} ticket source(s).",
        "action_items": _action_items(topic, action_context),
        "summary": summary,
        "steps": steps,
        "when_to_contact_support": escalation,
        "evidence_quotes": evidence_quotes,
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
    policy_question = _policy_question(topic)
    if policy_question:
        return (policy_question, "source_policy")
    return (f"What are customers asking about {topic}?", "topic_fallback")


def _question_text(value: Any) -> str:
    for text in _question_candidate_texts(value):
        if "?" in text:
            prefix, remainder = text.split("?", 1)
            prefix = prefix.strip()
            sentence_parts = [part.strip() for part in re.split(r"[.!:;]+", prefix) if part.strip()]
            candidate = sentence_parts[-1] if sentence_parts else prefix
            normalized = _normalize_question_text(candidate)
            if _usable_question(normalized):
                return normalized
            tail = _compact(remainder)
            if tail:
                normalized = _question_start_text(tail)
                if normalized:
                    return normalized
                normalized = _first_person_issue_question_text(tail)
                if normalized:
                    return normalized
            continue
        normalized = _question_start_text(text)
        if normalized:
            return normalized
        normalized = _first_person_issue_question_text(text)
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


def _first_person_issue_question_text(text: str) -> str:
    sentence = _first_sentence(text)
    lowered = sentence.lower()
    patterns = (
        ("i cannot ", "How do I "),
        ("i can't ", "How do I "),
        ("i can not ", "How do I "),
        ("i need to ", "How do I "),
        ("i want to ", "How do I "),
        ("i have to ", "How do I "),
        ("i am trying to ", "How do I "),
        ("i'm trying to ", "How do I "),
        ("we cannot ", "How do we "),
        ("we can't ", "How do we "),
        ("we can not ", "How do we "),
        ("we need to ", "How do we "),
        ("we want to ", "How do we "),
        ("we have to ", "How do we "),
        ("we are trying to ", "How do we "),
        ("we're trying to ", "How do we "),
    )
    for prefix, question_prefix in patterns:
        if lowered.startswith(prefix):
            remainder = sentence[len(prefix):].strip()
            normalized = _normalize_question_text(f"{question_prefix}{remainder}")
            if _usable_question(normalized):
                return normalized
    complaint_patterns = (
        ("i was ", "What should I do if I was "),
        ("i got ", "What should I do if I got "),
        ("i received ", "What should I do if I received "),
        ("i paid ", "What should I do if I paid "),
        ("my ", "What should I do if my "),
        ("we were ", "What should we do if we were "),
        ("we got ", "What should we do if we got "),
        ("we received ", "What should we do if we received "),
        ("we paid ", "What should we do if we paid "),
        ("our ", "What should we do if our "),
    )
    for prefix, question_prefix in complaint_patterns:
        if lowered.startswith(prefix):
            remainder = sentence[len(prefix):].strip()
            normalized = _normalize_question_text(f"{question_prefix}{remainder}")
            if _usable_question(normalized):
                return normalized
    return ""


def _first_sentence(text: str) -> str:
    parts = [part.strip() for part in re.split(r"[.!?;:\n]+", text, maxsplit=1) if part.strip()]
    return parts[0] if parts else ""


def _usable_question(value: str) -> bool:
    lowered = _compact(value).lower()
    return (
        bool(value)
        and len(value) <= MAX_EXTRACTED_QUESTION_CHARS
        and lowered not in _GENERIC_QUESTION_TEXTS
        and not any(phrase in lowered for phrase in _GENERIC_QUESTION_PHRASES)
        and not _looks_malformed_question(value)
    )


def _normalize_question_text(value: str) -> str:
    candidate = _compact(value).rstrip("?.!,;: ")
    if not candidate:
        return ""
    return f"{candidate}?"


def _looks_malformed_question(value: str) -> bool:
    text = _compact(value)
    if not text:
        return True
    if (
        _REDACTED_DATE_RE.search(text)
        or _REDACTED_MONEY_RE.search(text)
        or _QUOTE_ARTIFACT_RE.search(text)
        or _TRAILING_TITLE_RE.search(text.rstrip("?.!,;: "))
    ):
        return True
    if _REDACTION_TOKEN_RE.search(text):
        return True
    if text.count('"') % 2:
        return True
    return False


def _policy_question(topic: str) -> str:
    normalized = _topic_label(topic).lower()
    if normalized == "credit report disputes":
        return "What should I do if information on my credit report is wrong?"
    if normalized == "debt collection disputes":
        return "What should I do if a collector says I owe a debt I do not recognize?"
    if normalized == "mortgage servicing issues":
        return "What should I do if my mortgage servicer will not fix a payment, payoff, foreclosure, or modification issue?"
    if normalized == "opening an account":
        return "What should I do if a bank will not open an account or says my account was opened incorrectly?"
    if normalized == "closing an account":
        return "What should I do if I cannot close an account or recover the remaining funds?"
    if normalized == "getting a credit card":
        return "What should I do if my card application, offer, or activation does not look right?"
    if normalized == "advertising":
        return "What should I do if a financial product advertisement or offer seems wrong?"
    if normalized == "other transaction problem":
        return "What should I do if a transaction was scheduled, blocked, or processed incorrectly?"
    return f"What should I do about {normalized}?"


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
        f"_Source rows analyzed: {source_count}. Ticket sources used: {ticket_source_count}._",
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
            _md(item.get("summary") or item["answer"]),
            "",
            "**What to do next:**",
            "",
            *[f"{step_index}. {_md(step)}" for step_index, step in enumerate(_list(item.get("steps") or item["action_items"]), start=1)],
            "",
            "**When to contact support:**",
            "",
            _md(item.get("when_to_contact_support") or "Contact support with the cited ticket details if the answer does not resolve it."),
            "",
            "**Sources:**",
            "",
            *[f"- {_md(quote)}" for quote in _list(item.get("evidence_quotes"))],
            "",
        ])
    return "\n".join(lines)


def _source_label(row: Mapping[str, str]) -> str:
    source_id = _clean(row.get("source_id"))
    title = _clean(row.get("source_title"))
    if source_id and title:
        return f"`{source_id}` - {title}"
    return f"`{source_id or 'unknown'}`"


def _summary(*, topic: str, rows: Sequence[Mapping[str, str]], source_count: int) -> str:
    issue = _topic_label(topic)
    example = _quote(rows[0].get("text", ""), limit=180) if rows else "the cited ticket evidence"
    if source_count > 1:
        return (
            f"Customers are asking about {issue} across {source_count} ticket sources. "
            f"The clearest customer wording is {example}, so this FAQ should answer "
            "that request directly and tell users exactly what to try next."
        )
    return (
        f"A customer asked about {issue}: {example}. This FAQ should answer the "
        "request directly and tell the user exactly what to try next."
    )


def _article_steps(topic: str, evidence_text: str, *, support_contact: str | None) -> tuple[str, ...]:
    steps = _action_items(topic, evidence_text)
    return (
        steps[0],
        steps[1],
        _support_step(support_contact),
    )


def _escalation_guidance(topic: str, evidence_text: str, *, support_contact: str | None) -> str:
    text = f"{topic} {evidence_text}".lower()
    if any(term in text for term in ("credit report", "credit file", "credit bureau", "credit bureaus", "credit reporting", "incorrect information")):
        return (
            f"{_support_sentence(support_contact)} if the account still appears "
            "incorrect after you dispute it or if the furnisher does not explain the record."
        )
    if any(term in text for term in ("debt not owed", "not owe", "do not owe", "collect debt", "debt collection", "collector", "debt collector", "collection letter", "collection agency", "debt validation", "validation letter", "medical debt", "settled the debt")):
        return (
            f"{_support_sentence(support_contact)} if the collector will not validate "
            "the debt, keeps contacting you about a debt you do not recognize, or reports it again."
        )
    if topic == "mortgage servicing issues" or any(term in text for term in _MORTGAGE_INTENT_TERMS):
        return (
            f"{_support_sentence(support_contact)} if the servicer will not explain "
            "the payment, payoff, foreclosure, modification, escrow, or insurance issue in writing."
        )
    if topic in {"opening an account", "closing an account", "getting a credit card"} or any(
        term in text
        for term in (
            "opening an account",
            "open an account",
            "opened an account",
            "closing an account",
            "close an account",
            "closed an account",
            "getting a credit card",
            "applied for a credit card",
            "credit card application",
            "early warning services",
            "identity theft",
        )
    ):
        return (
            f"{_support_sentence(support_contact)} if the bank or card issuer "
            "will not explain the decision, account status, fraud record, or card issue in writing."
        )
    if any(term in text for term in ("export", "dashboard", "attribution", "analytics report", "download report", "report export")):
        return (
            f"{_support_sentence(support_contact)} if the export is missing, "
            "locked by plan or role, or still unavailable after an admin checks permissions."
        )
    if any(term in text for term in ("billing", "invoice", "payment", "receipt", "charge", "fee", "fees", "interest", "loan", "lease", "statement", "dispute")):
        return (
            f"{_support_sentence(support_contact)} if the charge, fee, payment, "
            "balance, or dispute still looks wrong after you compare it with your records."
        )
    if any(term in text for term in ("login", "email", "profile", "password", "account")):
        return (
            f"{_support_sentence(support_contact)} if you cannot access the account, "
            "the field is locked, or the confirmation message never arrives."
        )
    return (
        f"{_support_sentence(support_contact)} if the issue still affects the workflow "
        "after you try the steps above."
    )


def _support_step(support_contact: str | None) -> str:
    contact = _clean(support_contact)
    if contact:
        return (
            f"If it still does not work, contact support at {contact} and include "
            "the cited ticket details."
        )
    return "If it still does not work, contact support and include the cited ticket details."


def _support_sentence(support_contact: str | None) -> str:
    contact = _clean(support_contact)
    if contact:
        return f"Contact support at {contact}"
    return "Contact support"


def _evidence_quote(row: Mapping[str, str]) -> str:
    label = _source_label(row)
    text = _quote(row.get("text", ""), limit=220)
    return f"{label}: {text}"


def _topic_label(topic: str) -> str:
    text = _clean(topic).replace("_", " ")
    return text or "this support issue"


def _list(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


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


def _support_contact_metadata(support_contact: str | None) -> dict[str, str]:
    contact = _clean(support_contact)
    if not contact:
        return {}
    return {"support_contact": contact}


def _action_items(topic: str, evidence_text: str) -> tuple[str, ...]:
    if topic == "mortgage servicing issues":
        return _MORTGAGE_ACTION_STEPS
    text = f"{topic} {evidence_text}".lower()
    for terms, steps in _ACTION_RULES:
        if any(term in text for term in terms):
            return steps
    return (
        "Review the account, page, or workflow named in the ticket.",
        "Write down what you tried and the exact message or behavior you saw.",
    )


def _output_checks(
    *,
    items: Sequence[Mapping[str, Any]],
    ticket_source_count: int,
    rendered_ticket_source_count: int,
) -> dict[str, bool]:
    has_items = bool(items)
    covers_all_sources = rendered_ticket_source_count == ticket_source_count
    return {
        "uses_user_vocabulary": has_items
        and all(item.get("question_source") in _SUPPORTED_QUESTION_SOURCES for item in items),
        "condensed": has_items
        and covers_all_sources
        and (ticket_source_count <= 1 or len(items) < ticket_source_count),
        "has_action_items": has_items and all(bool(item.get("action_items")) for item in items),
    }


def _rendered_ticket_source_count(items: Sequence[Mapping[str, Any]]) -> int:
    source_ids: set[str] = set()
    for item in items:
        values = item.get("source_ids")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            source_ids.update(_clean(value) for value in values if _clean(value))
    return len(source_ids)


def _quote(value: Any, *, limit: int = 220) -> str:
    text = _compact(value)
    if len(text) > limit:
        text = f"{text[:limit].rstrip()}..."
    return f'"{text}"'


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
