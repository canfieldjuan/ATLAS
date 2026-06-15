"""Build grounded FAQ Markdown from support-ticket source evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import math
import re
from typing import Any

from .campaign_ports import TenantScope
from .campaign_source_adapters import (
    source_material_to_source_rows,
    source_rows_to_campaign_opportunities,
)
from .embedding_port import EmbeddingPort, cosine_similarity
from .support_ticket_clustering import (
    support_ticket_plain_text,
    support_ticket_tokens,
)
from .support_ticket_dates import parse_support_ticket_source_date
from .support_ticket_input_package import (
    _normalize_status_state as _normalize_ticket_status_state,
    _parse_csat_score as _parse_ticket_csat_score,
)
from .ticket_faq_ports import TicketFAQDraft, TicketFAQRepository


DEFAULT_TICKET_SOURCE_TYPES = (
    "ticket",
    "support_ticket",
    "case",
    "chat",
    "chat_transcript",
    "conversation",
    "transcript",
    "sales_call",
    "meeting",
    "sales_objection",
    "objection",
    "complaint",
    "search_log",
    "search_query",
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
_REPRESENTATIVE_EMAIL_RE = re.compile(r"\b\S+@\S+\b")
_REPRESENTATIVE_LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
_CUSTOMER_HEADING_PHONE_RE = re.compile(
    r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"
)
_CUSTOMER_IDENTIFIER_NUMBER_RE = re.compile(
    r"\b(?:account|acct|case|claim|confirmation|customer|id|invoice|member|"
    r"number|order|ref|reference|ticket)\s*(?:id|number|no\.?|#)?\s*[:#-]?\s*\d{4,}\b"
    r"|\b\d{4,}\s*(?:account|acct|case|claim|customer|invoice|member|order|ref|reference|ticket)\b",
    re.IGNORECASE,
)
_PUBLISHED_TEXT_PRIVACY_PLACEHOLDER = "Customer-provided details omitted for privacy."
_REPRESENTATIVE_LABEL_SOURCE_TYPES = {
    "ticket",
    "support_ticket",
    "case",
    "chat",
    "chat_transcript",
    "conversation",
    "transcript",
    "sales_call",
    "meeting",
    "sales_objection",
    "objection",
}
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
    "my complaint is about",
    "our complaint is about",
    "this complaint is about",
)
_MIXED_EVIDENCE_REVIEW_QUESTION = "Which remaining support questions need manual review?"
_REDACTION_TOKEN_RE = re.compile(r"\bX{2,}\b", re.IGNORECASE)
_REDACTED_DATE_RE = re.compile(r"\bX{2}/X{2}/(?:X{2,4}|\d{2,4})\b", re.IGNORECASE)
_REDACTED_MONEY_RE = re.compile(r"\{\$|\$\s*X{2,}", re.IGNORECASE)
_TRAILING_TITLE_RE = re.compile(r"\b(?:mr|mrs|ms|dr)\.?$", re.IGNORECASE)
_QUOTE_ARTIFACT_RE = re.compile(r"(?:''|``)")
_LEADING_BRACKETED_METADATA_RE = re.compile(r"^(?:\[[^\]\n]{1,80}\]\s*)+")
_METADATA_QUESTION_START_RE = re.compile(
    r"\b(?:how|what|where|when|why|can|could|do|does|como)\s+",
    re.IGNORECASE,
)
_MORTGAGE_INTENT_TERMS = (
    "mortgage",
    "home loan",
    "foreclosure",
    "reverse mortgage",
    "mortgage servicer",
)
_FAILURE_RISK_RULES = (
    ("blocked_access", ("cannot", "can't", "can not", "unable", "locked", "blocked", "denied")),
    ("failed_workflow", ("failed", "fails", "not working", "does not work", "keeps timing out", "timed out", "missing")),
    ("incorrect_record", ("wrong", "incorrect", "inaccurate", "error", "mistake", "does not recognize")),
    ("money_or_account_risk", ("charged", "fee", "fees", "payment", "balance", "debt", "foreclosure", "fraud")),
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
    ("communication and contact issues", (
        "call at all hours",
        "calling at all hours",
        "various numbers",
        "telephone calls",
        "harass",
        "harassed",
        "harassment",
    )),
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
    "source_date",
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
    "product_name",
    "category",
    "sub_product",
    "sub_product_name",
    "issue",
    "sub_issue",
    "sub_issue_type",
)
_REPRESENTATIVE_SAFE_CONTEXT_KEYS = (
    "product",
    "product_name",
    "sub_product",
    "sub_product_name",
    "issue",
    "issue_type",
    "sub_issue",
    "sub_issue_type",
)
_DOCUMENTATION_SOURCE_TYPES = {
    "article",
    "document",
    "documentation",
    "docs",
    "help_article",
    "kb_article",
    "knowledge_base",
}
_DOCUMENTATION_TERM_KEYS = (
    "source_title",
    "title",
    "name",
    "heading",
    "topic",
    "category",
    "text",
    "description",
)
_SOURCE_WEIGHT_KEYS = (
    "source_weight",
    "search_count",
    "query_count",
    "request_count",
    "occurrences",
    "occurrence_count",
    "volume",
    "search_volume",
    "impressions",
    "weight",
    # Some exports call source volume "frequency"; keep it after explicit
    # aggregate fields so round-tripped FAQ output does not mask search_count.
    "frequency",
)
_RESOLUTION_TEXT_KEYS = (
    "resolution_text",
    "resolved_text",
    "resolution_summary",
    "support_resolution",
    "support_response",
    "support_reply",
    "agent_resolution",
    "agent_response",
    "agent_reply",
    "admin_reply",
    "latest_agent_reply",
    "last_agent_reply",
    "public_agent_reply",
    "staff_reply",
    "answer_text",
)
_RESOLUTION_ACTION_TERMS = {
    "add",
    "approve",
    "ask",
    "change",
    "check",
    "choose",
    "clear",
    "click",
    "compare",
    "configure",
    "confirm",
    "connect",
    "contact",
    "create",
    "deploy",
    "disable",
    "download",
    "enable",
    "export",
    "grant",
    "import",
    "map",
    "open",
    "paste",
    "refresh",
    "remove",
    "rerun",
    "reset",
    "review",
    "revoke",
    "run",
    "save",
    "select",
    "send",
    "set",
    "start",
    "update",
    "verify",
}
_RESOLUTION_TOPIC_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "can",
    "cannot",
    "could",
    "customer",
    "does",
    "for",
    "from",
    "help",
    "how",
    "into",
    "issue",
    "need",
    "not",
    "please",
    "request",
    "support",
    "that",
    "the",
    "then",
    "this",
    "ticket",
    "user",
    "what",
    "when",
    "where",
    "why",
    "with",
}
_RESOLUTION_CLOSURE_BOILERPLATE_RE = re.compile(
    r"\b(?:customer|user|requester|client)\s+"
    r"(?:did not|didn't|does not|doesn't)\s+respond\b"
    r"|\bno response from (?:the )?(?:customer|user|requester|client)\b"
    r"|\bclosed due to no response\b"
    r"|\bclosing due to no response\b"
    r"|\bclosing this out\b"
    r"|\bthanks[,. ]+(?:closing|closed)\b",
    re.IGNORECASE,
)
_RESOLUTION_INTERNAL_NOTE_RE = re.compile(
    r"\b(?:assigned|escalated|routed)\s+to\s+(?:t[0-9]+|tier\s*[0-9]+|l[0-9]+)\b"
    r"|\bpolicy\s+\d+(?:\.\d+)+\b"
    r"|\binternal\s+(?:note|only)\b",
    re.IGNORECASE,
)
_RESOLUTION_DISPOSITION_ONLY_ACTION_TERMS = {
    "ask",
    "check",
    "confirm",
    "contact",
    "review",
    "send",
    "start",
    "update",
}
_RESOLUTION_DISPOSITION_ONLY_RE = re.compile(
    r"\b(?:replied|responded)\s+to\s+(?:the\s+)?(?:customer|client|user|requester)\b"
    r"|\b(?:sent|provided)\s+(?:the\s+)?(?:customer|client|user|requester)\s+"
    r"(?:an?\s+)?(?:update|reply|response)\b"
    r"|\b(?:sent|provided)\s+(?:an?\s+)?(?:update|reply|response)\s+"
    r"to\s+(?:the\s+)?(?:customer|client|user|requester)\b"
    r"|\b(?:customer|client|user|requester)\s+(?:was\s+)?"
    r"(?:updated|notified|informed)\b",
    re.IGNORECASE,
)
_RESOLUTION_TOPIC_EQUIVALENCE_GROUPS = (
    frozenset({
        "auth",
        "authentication",
        "code",
        "credential",
        "login",
        "log",
        "password",
        "reset",
        "sign",
    }),
    frozenset({
        "bill",
        "billing",
        "charge",
        "invoice",
        "payment",
        "receipt",
        "statement",
    }),
    frozenset({
        "connect",
        "connection",
        "integration",
        "sync",
    }),
    frozenset({
        "cancel",
        "cancellation",
        "renewal",
        "subscription",
    }),
)
_RESOLUTION_TOPIC_EQUIVALENCE: dict[str, frozenset[str]] = {
    token: group
    for group in _RESOLUTION_TOPIC_EQUIVALENCE_GROUPS
    for token in group
}
_VOCABULARY_GAP_RULES = (
    ("export", "download", "download report", "report download"),
    ("dashboard", "analytics", "reporting", "reports"),
    ("bill", "billing", "invoice", "statement"),
    ("login", "log in", "sign in", "authentication"),
    ("password", "credentials", "authentication"),
    ("connect", "connection", "integration", "sync"),
    ("cancel", "cancellation", "renewal"),
)
_NEGATIVE_TEXTUAL_CSAT = frozenset({
    "bad",
    "negative",
    "poor",
    "unhappy",
    "unsatisfied",
    "dissatisfied",
})
_OUTCOME_STATUS_KEYS = (
    "ticket_status",
    "issue_status",
    "case_status",
    "status",
    "ticket_state",
    "state",
)
_OUTCOME_CSAT_KEYS = (
    "csat",
    "satisfaction_score",
    "satisfaction_rating",
    "customer_satisfaction_rating",
    "customer_satisfaction_score",
    "customer_satisfaction",
    "satisfaction",
    "rating",
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
    non_repeat_ticket_count: int = 0
    non_repeat_question_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": len(self.items),
            "markdown": self.markdown,
            "items": [dict(item) for item in self.items],
            "source_count": self.source_count,
            "ticket_source_count": self.ticket_source_count,
            "non_repeat_ticket_count": self.non_repeat_ticket_count,
            "non_repeat_question_count": self.non_repeat_question_count,
            "output_checks": dict(self.output_checks),
            "warnings": [dict(warning) for warning in self.warnings],
            "saved_ids": list(self.saved_ids),
        }


@dataclass(frozen=True)
class TicketFAQMarkdownConfig:
    """Config for service-shaped FAQ Markdown generation."""

    title: str = DEFAULT_TITLE
    max_items: int | None = 8
    max_evidence_per_item: int = 3
    source_types: tuple[str, ...] = DEFAULT_TICKET_SOURCE_TYPES
    max_text_chars: int = 1200
    window_days: int | None = None
    as_of_date: str | None = None
    support_contact: str | None = None
    intent_rules: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_INTENT_RULES
    documentation_terms: tuple[str, ...] = ()
    representative_taxonomy_terms: tuple[str, ...] = ()
    vocabulary_gap_rules: tuple[tuple[str, ...], ...] = ()
    embedding_port: EmbeddingPort | None = None


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
        documentation_terms: Sequence[str] | None = None,
        representative_taxonomy_terms: Sequence[str] | None = None,
        vocabulary_gap_rules: Sequence[Sequence[str]] | None = None,
        embedding_port: EmbeddingPort | None = None,
        **kwargs: Any,
    ) -> TicketFAQMarkdownResult:
        del kwargs
        resolved_max_items = _normalize_max_items(
            max_items if max_items is not None else self.config.max_items
        )
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
            normalize_intent_rules(intent_rules)
            if intent_rules is not None
            else self.config.intent_rules
        )
        resolved_documentation_terms = (
            tuple(_clean(term) for term in documentation_terms if _clean(term))
            if documentation_terms is not None
            else self.config.documentation_terms
        )
        resolved_representative_taxonomy_terms = (
            tuple(_clean(term) for term in representative_taxonomy_terms if _clean(term))
            if representative_taxonomy_terms is not None
            else self.config.representative_taxonomy_terms
        )
        resolved_vocabulary_gap_rules = (
            vocabulary_gap_rules
            if vocabulary_gap_rules is not None
            else self.config.vocabulary_gap_rules
        )
        resolved_embedding_port = (
            embedding_port
            if embedding_port is not None
            else self.config.embedding_port
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
            documentation_terms=resolved_documentation_terms,
            representative_taxonomy_terms=resolved_representative_taxonomy_terms,
            vocabulary_gap_rules=resolved_vocabulary_gap_rules,
            embedding_port=resolved_embedding_port,
        )
        result = replace(
            result,
            warnings=(
                *(warning.as_dict() for warning in normalized.warnings),
                *result.warnings,
            ),
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
                        **_documentation_term_metadata(resolved_documentation_terms),
                        **_representative_taxonomy_term_metadata(
                            resolved_representative_taxonomy_terms
                        ),
                        **_vocabulary_gap_rule_metadata(resolved_vocabulary_gap_rules),
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
    max_items: int | None = 8,
    max_evidence_per_item: int = 3,
    source_types: Sequence[str] = DEFAULT_TICKET_SOURCE_TYPES,
    window_days: int | None = None,
    as_of_date: Any = None,
    support_contact: str | None = None,
    intent_rules: Sequence[tuple[str, Sequence[str]]] = DEFAULT_INTENT_RULES,
    documentation_terms: Sequence[str] = (),
    representative_taxonomy_terms: Sequence[str] = (),
    vocabulary_gap_rules: Sequence[Sequence[str]] = (),
    embedding_port: EmbeddingPort | None = None,
    embedding_merge_recorder: Callable[[Mapping[str, Any]], None] | None = None,
) -> TicketFAQMarkdownResult:
    """Render an extractive FAQ from normalized source-row opportunities."""

    resolved_max_items = _normalize_max_items(max_items)
    if max_evidence_per_item < 1:
        raise ValueError("max_evidence_per_item must be positive")

    allowed = {_source_type_key(item) for item in source_types if _source_type_key(item)}
    date_window = _date_window(window_days=window_days, as_of_date=as_of_date)
    resolved_documentation_terms = _documentation_terms(opportunities, documentation_terms)
    resolved_representative_taxonomy_terms = _clean_terms(representative_taxonomy_terms)
    resolved_vocabulary_gap_rules = _vocabulary_gap_rules(vocabulary_gap_rules)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
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
            text = support_ticket_plain_text(evidence.get("text"))
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
            topic = _topic(opportunity, evidence, intent_rules=intent_rules)
            resolution_context = " / ".join(
                value
                for value in (
                    text,
                    topic,
                    _clean(evidence.get("source_title") or opportunity.get("source_title")),
                    _clean(evidence.get("pain_category") or opportunity.get("pain_category")),
                    _clean(evidence.get("tags") or opportunity.get("tags")),
                )
                if value
            )
            resolution_text = _resolution_text(
                evidence,
                opportunity,
                question_text=resolution_context,
            )
            evidence_group_key = _evidence_group_key(resolution_text)
            group_key = (topic, evidence_group_key or f"topic:{_compact_key(topic)}")
            source_date = _source_date(evidence) or _source_date(opportunity)
            groups[group_key].append({
                "text": text,
                "source_id": source_id or "unknown",
                "source_key": source_key,
                "source_type": source_type,
                "source_title": support_ticket_plain_text(
                    evidence.get("source_title") or opportunity.get("source_title")
                ),
                "support_ticket_cluster": support_ticket_plain_text(
                    evidence.get("support_ticket_cluster")
                    or opportunity.get("support_ticket_cluster")
                ),
                "safe_label_context": _representative_safe_context_text(
                    opportunity,
                    evidence,
                    representative_taxonomy_terms=resolved_representative_taxonomy_terms,
                ),
                "source_date": source_date.isoformat() if source_date is not None else "",
                "evidence_group_key": evidence_group_key,
                "results_count": _first_present(evidence, opportunity, key="results_count"),
                "result_count": _first_present(evidence, opportunity, key="result_count"),
                "zero_results": _first_present(evidence, opportunity, key="zero_results"),
                "zero_result": _first_present(evidence, opportunity, key="zero_result"),
                "source_weight": _source_weight(evidence, opportunity),
                "resolution_text": resolution_text,
                "ticket_status_state": _first_present(
                    evidence, opportunity, key="ticket_status_state"
                ),
                "ticket_status": _first_available(
                    evidence,
                    opportunity,
                    keys=_OUTCOME_STATUS_KEYS,
                ),
                "csat": _first_present(evidence, opportunity, key="csat"),
                "csat_score": _first_present(evidence, opportunity, key="csat_score"),
                "csat_raw": _first_available(evidence, opportunity, keys=_OUTCOME_CSAT_KEYS),
            })

    # #1460: topic-degraded groups (scope "topic:*") measure bucket
    # membership, not question repetition. Split them into question
    # sub-clusters; only clusters asked by >= 2 DISTINCT source tickets stay
    # as FAQ groups (one ticket contributing several evidence rows is not a
    # repeat), and excluded singletons are counted and surfaced instead of
    # silently rendering as "repeat" work. Resolution-scoped groups are
    # untouched.
    subclustered_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    excluded_singleton_keys: set[str] = set()
    non_repeat_question_count = 0
    for (topic, scope), rows in groups.items():
        if not scope.startswith("topic:"):
            subclustered_groups[(topic, scope)] = rows
            continue
        for cluster_index, cluster_rows in enumerate(
            _question_subclusters(
                rows,
                embedding_port=embedding_port,
                embedding_merge_recorder=embedding_merge_recorder,
            )
        ):
            cluster_keys = _row_source_keys(cluster_rows)
            if len(cluster_keys) < 2:
                excluded_singleton_keys.update(cluster_keys)
                non_repeat_question_count += 1
                continue
            subclustered_groups[(topic, f"{scope}:question:{cluster_index}")] = list(
                cluster_rows
            )
    # A ticket only counts as non-repeat if none of its evidence rows landed
    # in any kept group; this keeps the condensed-coverage accounting exact
    # (rendered distinct tickets + non-repeat distinct tickets == sources).
    kept_keys: set[str] = set()
    for rows in subclustered_groups.values():
        kept_keys.update(_row_source_keys(rows))
    non_repeat_ticket_count = len(excluded_singleton_keys - kept_keys)
    warnings: list[dict[str, Any]] = []
    if non_repeat_ticket_count:
        warnings.append({
            "code": "non_repeat_tickets_excluded",
            "message": (
                f"Excluded {non_repeat_ticket_count} tickets whose question "
                "appeared only once; they are counted separately and not "
                "billed as repeat work."
            ),
            "ticket_count": non_repeat_ticket_count,
            "question_count": non_repeat_question_count,
        })

    sorted_groups = sorted(
        ((topic, rows) for (topic, _scope), rows in subclustered_groups.items()),
        key=_group_sort_key,
    )
    if resolved_max_items is not None and len(sorted_groups) > resolved_max_items:
        visible_groups = tuple(sorted_groups[: max(1, resolved_max_items - 1)])
        overflow_rows: list[dict[str, str]] = []
        for _topic_name, rows in sorted_groups[len(visible_groups):]:
            overflow_rows.extend(rows)
        if len(visible_groups) == resolved_max_items:
            topic, rows = visible_groups[0]
            selected_groups = ((topic, [*rows, *overflow_rows]),)
        else:
            selected_groups = (*visible_groups, ("other support issues", overflow_rows))
    else:
        selected_groups = tuple(sorted_groups)

    item_records = tuple(
        (
            rows,
            _item(
                topic,
                rows,
                max_evidence_per_item=max_evidence_per_item,
                support_contact=support_contact,
                documentation_terms=resolved_documentation_terms,
                vocabulary_gap_rules=resolved_vocabulary_gap_rules,
            ),
        )
        for topic, rows in selected_groups
    )
    items, disambiguation_warnings = _disambiguate_source_policy_question_collisions(
        item_records,
        documentation_terms=resolved_documentation_terms,
    )
    warnings.extend(disambiguation_warnings)
    return TicketFAQMarkdownResult(
        markdown=_render(title=title, items=items, source_count=len(opportunities), ticket_source_count=len(source_keys)),
        items=items,
        source_count=len(opportunities),
        ticket_source_count=len(source_keys),
        non_repeat_ticket_count=non_repeat_ticket_count,
        non_repeat_question_count=non_repeat_question_count,
        output_checks=_output_checks(
            items=items,
            ticket_source_count=len(source_keys),
            rendered_ticket_source_count=_rendered_ticket_source_count(items),
            non_repeat_ticket_count=non_repeat_ticket_count,
        ),
        warnings=tuple(warnings),
    )


def _normalize_max_items(max_items: int | None) -> int | None:
    if max_items is None:
        return None
    value = int(max_items)
    if value < 0:
        raise ValueError("max_items must be positive or 0 for unlimited")
    if value == 0:
        return None
    return value


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


# --- Question sub-clustering (#1460) -------------------------------------
#
# Topic/intent buckets are a coarse pre-partition: when rows carry no
# resolution evidence the group key degrades to "topic:<topic>" and every
# ticket in the bucket would render as ONE FAQ whose ticket_count measures
# bucket membership, not question repetition. These helpers split each
# degraded bucket into question-similarity sub-clusters with deterministic
# prefix-filtered exact-Jaccard verification, so only questions customers
# actually asked more than once count as repeats.

_SUBCLUSTER_JACCARD_THRESHOLD = 1 / 3
_SUBCLUSTER_GIST_TOKEN_LIMIT = 30
_EMBEDDING_MNN_COSINE_FLOOR = 0.78
_EMBEDDING_MNN_MARGIN = 0.05


def _question_gist_text(text: str) -> str:
    question = _question_text(text)
    return question if question else " ".join(text.split()[: _SUBCLUSTER_GIST_TOKEN_LIMIT])


def _question_gist_tokens(text: str) -> frozenset[str]:
    """Tokens of the row's question sentence, or its opening gist."""

    tokens = support_ticket_tokens(_question_gist_text(text))
    return frozenset(
        token for token in tokens if not _REDACTION_TOKEN_RE.fullmatch(token)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _jaccard_prefix_tokens(
    tokens: frozenset[str],
    token_counts: Mapping[str, int],
) -> tuple[str, ...]:
    ordered = sorted(tokens, key=lambda token: (token_counts[token], token))
    prefix_size = len(ordered) - math.ceil(_SUBCLUSTER_JACCARD_THRESHOLD * len(ordered)) + 1
    return tuple(ordered[:max(0, prefix_size)])


def _jaccard_lengths_can_match(left_size: int, right_size: int) -> bool:
    if left_size <= 0 or right_size <= 0:
        return False
    return min(left_size, right_size) / max(left_size, right_size) >= _SUBCLUSTER_JACCARD_THRESHOLD


def _row_source_keys(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {
        key
        for row in rows
        if (key := str(row.get("source_key") or "").strip())
    }


def _question_subclusters(
    rows: Sequence[Mapping[str, str]],
    *,
    embedding_port: EmbeddingPort | None = None,
    embedding_merge_recorder: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[list[Mapping[str, str]]]:
    """Split one degraded topic bucket into question-similarity clusters.

    Deterministic: exact duplicate matching plus prefix-filtered candidate
    nomination, followed by exact-Jaccard verification. Rows with an empty
    gist never merge.
    """

    gists = [_question_gist_tokens(str(row.get("text") or "")) for row in rows]
    token_counts: dict[str, int] = defaultdict(int)
    for gist in gists:
        for token in gist:
            token_counts[token] += 1

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    exact: dict[frozenset[str], int] = {}
    prefix_index: dict[str, list[int]] = defaultdict(list)
    for index, gist in enumerate(gists):
        if not gist:
            continue
        first_exact = exact.setdefault(gist, index)
        if first_exact != index:
            union(first_exact, index)
            continue
        candidates: set[int] = set()
        prefix = _jaccard_prefix_tokens(gist, token_counts)
        for token in prefix:
            candidates.update(prefix_index[token])
        for candidate in sorted(candidates):
            if find(candidate) == find(index):
                continue
            candidate_gist = gists[candidate]
            if not _jaccard_lengths_can_match(len(candidate_gist), len(gist)):
                continue
            if _jaccard(candidate_gist, gist) >= _SUBCLUSTER_JACCARD_THRESHOLD:
                union(candidate, index)
        for token in prefix:
            prefix_index[token].append(index)

    _apply_embedding_booster(
        rows,
        gists=gists,
        find=find,
        union=union,
        embedding_port=embedding_port,
        embedding_merge_recorder=embedding_merge_recorder,
    )

    clusters: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[find(index)].append(row)
    return [clusters[root] for root in sorted(clusters)]


def _apply_embedding_booster(
    rows: Sequence[Mapping[str, str]],
    *,
    gists: Sequence[frozenset[str]],
    find: Callable[[int], int],
    union: Callable[[int, int], None],
    embedding_port: EmbeddingPort | None,
    embedding_merge_recorder: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    if embedding_port is None or len(rows) < 2:
        return
    texts = [
        _question_gist_text(str(row.get("text") or "")).strip()
        for row in rows
    ]
    component_sizes: dict[int, int] = defaultdict(int)
    for index in range(len(rows)):
        component_sizes[find(index)] += 1
    active_indexes = [
        index
        for index, text in enumerate(texts)
        if text and gists[index] and component_sizes[find(index)] == 1
    ]
    if len(active_indexes) < 2:
        return
    try:
        embedded = embedding_port.embed_texts([texts[index] for index in active_indexes])
    except Exception:
        return
    if (
        not isinstance(embedded, Sequence)
        or isinstance(embedded, (str, bytes, bytearray))
        or len(embedded) != len(active_indexes)
    ):
        return
    vectors: dict[int, Sequence[float]] = {
        index: embedded[position]
        for position, index in enumerate(active_indexes)
    }
    best_by_index: dict[int, tuple[int, float, float]] = {}
    for index in active_indexes:
        scored: list[tuple[float, int]] = []
        for candidate in active_indexes:
            if candidate == index or find(candidate) == find(index):
                continue
            score = cosine_similarity(vectors[index], vectors[candidate])
            if score is None:
                continue
            scored.append((score, candidate))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_index = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        best_by_index[index] = (best_index, best_score, runner_up)
    pairs: set[tuple[int, int]] = set()
    for index, (candidate, score, runner_up) in best_by_index.items():
        candidate_best = best_by_index.get(candidate)
        if candidate_best is None or candidate_best[0] != index:
            continue
        if score < _EMBEDDING_MNN_COSINE_FLOOR:
            continue
        if score - runner_up < _EMBEDDING_MNN_MARGIN:
            continue
        if candidate_best[1] - candidate_best[2] < _EMBEDDING_MNN_MARGIN:
            continue
        pairs.add(tuple(sorted((index, candidate))))
    for left, right in sorted(pairs):
        if find(left) != find(right):
            if embedding_merge_recorder is not None:
                left_best, left_score, left_runner_up = best_by_index[left]
                right_best, right_score, right_runner_up = best_by_index[right]
                score = left_score if left_best == right else right_score
                embedding_merge_recorder({
                    "left_index": left,
                    "right_index": right,
                    "left_source_id": _embedding_merge_source_id(rows[left]),
                    "right_source_id": _embedding_merge_source_id(rows[right]),
                    "left_text": texts[left],
                    "right_text": texts[right],
                    "cosine": score,
                    "left_runner_up": left_runner_up,
                    "right_runner_up": right_runner_up,
                    "left_margin": left_score - left_runner_up,
                    "right_margin": right_score - right_runner_up,
                    "token_jaccard": _jaccard(gists[left], gists[right]),
                })
            union(left, right)


def _embedding_merge_source_id(row: Mapping[str, str]) -> str:
    source_id = str(row.get("source_id") or "").strip()
    if source_id and source_id.lower() != "unknown":
        return source_id
    return str(row.get("source_key") or "").strip()


def _topic(
    opportunity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    intent_rules: Sequence[tuple[str, Sequence[str]]],
) -> str:
    provided_cluster = _provided_support_ticket_cluster_topic(opportunity, evidence)
    if provided_cluster:
        return provided_cluster.lower()
    intent = _intent_topic(opportunity, evidence, intent_rules=intent_rules)
    if intent:
        return intent
    pain_points = opportunity.get("pain_points")
    if isinstance(pain_points, Sequence) and not isinstance(pain_points, (str, bytes, bytearray)):
        for value in pain_points:
            text = _compact(value)
            if text:
                return text.lower()
    fallback_topic = _published_customer_text(
        evidence.get("source_title") or opportunity.get("source_title")
    )
    return (fallback_topic or "customer support issues").lower()


def _provided_support_ticket_cluster_topic(
    *rows: Mapping[str, Any],
) -> str:
    for row in rows:
        value = support_ticket_plain_text(row.get("support_ticket_cluster"))
        if value:
            return value
    return ""


def _evidence_group_key(resolution_text: Any) -> str:
    text = _compact(resolution_text)
    return f"resolution:{_compact_key(text)}" if text else ""


def _has_mixed_evidence_scopes(rows: Sequence[Mapping[str, Any]]) -> bool:
    scopes = {
        _clean(row.get("evidence_group_key"))
        for row in rows
        if _clean(row.get("evidence_group_key"))
    }
    has_unscoped_rows = any(not _clean(row.get("evidence_group_key")) for row in rows)
    return len(scopes) > 1 or (bool(scopes) and has_unscoped_rows)


def _group_sort_key(item: tuple[str, Sequence[Mapping[str, str]]]) -> tuple[int, int, int, str]:
    topic, rows = item
    score = _opportunity_score(topic, rows)
    return (
        -score["opportunity_score"],
        -score["frequency"],
        -score["failure_risk_score"],
        topic.lower(),
    )


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
    return _context_text(_SOURCE_CONTEXT_KEYS, *rows)


def _representative_safe_context_text(
    *rows: Mapping[str, Any],
    representative_taxonomy_terms: Sequence[str],
) -> str:
    taxonomy_keys = _representative_taxonomy_keys(representative_taxonomy_terms)
    if not taxonomy_keys:
        return ""
    values: list[str] = []
    for row in rows:
        for key in _REPRESENTATIVE_SAFE_CONTEXT_KEYS:
            text = _compact(_field_value(row, key))
            if text and _compact_key(text) in taxonomy_keys:
                values.append(text)
    return " ".join(values)


def _representative_taxonomy_keys(terms: Sequence[str]) -> frozenset[str]:
    return frozenset(
        _compact_key(term)
        for term in terms
        if _compact_key(term)
    )


def _context_text(keys: Sequence[str], *rows: Mapping[str, Any]) -> str:
    values: list[str] = []
    for row in rows:
        for key in keys:
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
    documentation_terms: Sequence[str],
    vocabulary_gap_rules: Sequence[Sequence[str]],
) -> dict[str, Any]:
    display_rows = rows[:max_evidence_per_item]
    sources = tuple(_source_label(row) for row in display_rows)
    source_keys = (row.get("source_key") or row.get("source_id", "") for row in rows)
    source_ids = tuple(dict.fromkeys(value for value in source_keys if value))
    snippets = " / ".join(_published_customer_quote(row.get("text", "")) for row in display_rows)
    action_context = " / ".join((
        snippets,
        " / ".join(
            title
            for title in (
                _published_customer_text(row.get("source_title"))
                for row in display_rows
            )
            if title
        ),
    ))
    question, question_source, question_row = _resolve_question_label(
        topic,
        rows,
        max_evidence_per_item=max_evidence_per_item,
        documentation_terms=documentation_terms,
    )
    has_mixed_evidence_scopes = _has_mixed_evidence_scopes(rows)
    summary_rows = _rows_with_question_source_first(display_rows, question_row)
    summary = _summary(
        topic=topic,
        rows=summary_rows,
        source_count=len(source_ids),
        question_source=question_source,
    )
    resolution_rows = (
        ()
        if has_mixed_evidence_scopes
        else _resolution_rows_for_question(rows, question_row)
    )
    resolution_texts = _resolution_texts(resolution_rows)
    resolution_evidence_scope = _resolution_evidence_scope_status(
        has_mixed_evidence_scopes=has_mixed_evidence_scopes,
        question_row=question_row,
        resolution_rows=resolution_rows,
        resolution_texts=resolution_texts,
    )
    scoped_resolution_texts = (
        resolution_texts if resolution_evidence_scope == "scoped" else ()
    )
    answer_evidence_status = (
        "resolution_evidence" if scoped_resolution_texts else "draft_needs_review"
    )
    steps = _article_steps(
        topic,
        action_context,
        support_contact=support_contact,
        resolution_texts=scoped_resolution_texts,
    )
    escalation = _escalation_guidance(topic, action_context, support_contact=support_contact)
    evidence_quotes = tuple(_evidence_quote(row) for row in display_rows)
    opportunity = _opportunity_score(topic, rows)
    term_mappings = _term_mappings(rows, documentation_terms, vocabulary_gap_rules)
    item = {
        "topic": topic,
        "question": question,
        "question_source": question_source,
        "frequency": opportunity["frequency"],
        "weighted_frequency": opportunity["weighted_frequency"],
        "failure_risk_score": opportunity["failure_risk_score"],
        "failure_risk_signals": opportunity["failure_risk_signals"],
        "opportunity_score": opportunity["opportunity_score"],
        "answer": _answer_summary(
            question=question,
            source_count=len(source_ids),
            answer_evidence_status=answer_evidence_status,
            resolution_texts=scoped_resolution_texts,
        ),
        "action_items": steps,
        "summary": summary,
        "steps": steps,
        "answer_evidence_status": answer_evidence_status,
        "resolution_evidence_scope": resolution_evidence_scope,
        "resolution_source_count": _resolution_source_count(resolution_rows),
        "when_to_contact_support": escalation,
        "evidence_quotes": evidence_quotes,
        "term_mappings": term_mappings,
        "source_ids": source_ids,
        "source_labels": sources,
        "source_type_counts": _item_source_type_counts(rows),
        "weighted_source_volume_by_type": _item_weighted_source_volume_by_type(rows),
        "evidence_count": len(display_rows),
        "displayed_evidence_count": len(display_rows),
        "ticket_count": len(source_ids),
    }
    outcome_diagnostics = _outcome_diagnostics(rows)
    if outcome_diagnostics:
        item["outcome_diagnostics"] = outcome_diagnostics
    source_date_span = _source_date_span(rows)
    if source_date_span is not None:
        item["source_date_span"] = source_date_span
    return item


def _source_date_span(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    date_by_source: dict[str, date] = {}
    missing_sources: set[str] = set()
    for index, row in enumerate(rows, start=1):
        source_key = _clean(row.get("source_key") or row.get("source_id")) or f"row:{index}"
        parsed = _source_date(row)
        if parsed is None:
            missing_sources.add(source_key)
            continue
        date_by_source[source_key] = parsed
    missing_sources.difference_update(date_by_source)
    if not date_by_source:
        return None
    start = min(date_by_source.values())
    end = max(date_by_source.values())
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "window_days": (end - start).days + 1,
        "dated_source_count": len(date_by_source),
        "missing_source_count": len(missing_sources),
    }


def _opportunity_score(topic: str, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    frequency = _weighted_frequency(rows)
    failure_risk_signals = _failure_risk_signals(topic, rows)
    failure_risk_score = len(failure_risk_signals)
    return {
        "frequency": frequency,
        "weighted_frequency": frequency,
        "failure_risk_score": failure_risk_score,
        "failure_risk_signals": failure_risk_signals,
        "opportunity_score": frequency * (1 + failure_risk_score),
    }


def _outcome_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tickets: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        source_key = _clean(row.get("source_key") or row.get("source_id")) or f"row:{index}"
        ticket = tickets.setdefault(source_key, {
            "status": "",
            "csat_present": False,
            "csat_score": None,
            "negative_csat": False,
        })
        status = _outcome_status(row)
        if status:
            ticket["status"] = _stronger_outcome_status(
                _clean(ticket.get("status")),
                status,
            )
        csat_text, csat_score = _outcome_csat(row)
        if csat_text or csat_score is not None:
            ticket["csat_present"] = True
        if csat_score is not None and ticket.get("csat_score") is None:
            ticket["csat_score"] = csat_score
        if _negative_csat(csat_text, csat_score):
            ticket["negative_csat"] = True

    status_summary: dict[str, int] = {}
    csat_scores: list[float] = []
    diagnostic_keys: set[str] = set()
    risk_keys: set[str] = set()
    reopened_count = 0
    csat_present_count = 0
    negative_csat_count = 0
    for source_key, ticket in tickets.items():
        status = _clean(ticket.get("status")).lower()
        if status:
            diagnostic_keys.add(source_key)
            status_summary[status] = status_summary.get(status, 0) + 1
            if status == "reopened":
                reopened_count += 1
                risk_keys.add(source_key)
        csat_score = ticket.get("csat_score")
        if ticket.get("csat_present"):
            diagnostic_keys.add(source_key)
            csat_present_count += 1
        if isinstance(csat_score, (int, float)):
            csat_scores.append(csat_score)
        if ticket.get("negative_csat"):
            negative_csat_count += 1
            risk_keys.add(source_key)
    if not diagnostic_keys:
        return {}
    out: dict[str, Any] = {
        "diagnostic_ticket_count": len(diagnostic_keys),
        "outcome_risk_ticket_count": len(risk_keys),
    }
    if status_summary:
        out["ticket_status_summary"] = dict(sorted(status_summary.items()))
    if reopened_count:
        out["reopened_ticket_count"] = reopened_count
    if csat_present_count:
        out["csat_present_count"] = csat_present_count
    if negative_csat_count:
        out["negative_csat_ticket_count"] = negative_csat_count
    if csat_scores:
        out["csat_score_average"] = round(sum(csat_scores) / len(csat_scores), 2)
    return out


def _outcome_status(row: Mapping[str, Any]) -> str:
    state = _clean(row.get("ticket_status_state")).lower()
    if state:
        return state
    raw_status = _first_available(row, keys=_OUTCOME_STATUS_KEYS)
    return _normalize_ticket_status_state(raw_status)


def _stronger_outcome_status(current: str, candidate: str) -> str:
    priority = {
        "reopened": 5,
        "open": 4,
        "other": 3,
        "cancelled": 2,
        "resolved": 1,
    }
    if priority.get(candidate, 0) > priority.get(current, 0):
        return candidate
    return current


def _outcome_csat(row: Mapping[str, Any]) -> tuple[str, float | None]:
    raw_text = _first_available(row, keys=("csat_raw", *_OUTCOME_CSAT_KEYS))
    raw_score = _first_present(row, key="csat_score")
    score = _parse_ticket_csat_score(raw_score)
    if score is None:
        score = _parse_ticket_csat_score(raw_text)
    text = _clean(raw_text if raw_text not in (None, "", [], {}) else raw_score).lower()
    return text, score


def _negative_csat(csat_text: str, csat_score: float | None) -> bool:
    if csat_score is not None:
        return csat_score <= 2
    return csat_text in _NEGATIVE_TEXTUAL_CSAT


def _distinct_source_keys(rows: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    values = (row.get("source_key") or row.get("source_id", "") for row in rows)
    return tuple(dict.fromkeys(value for value in values if value))


def _item_source_type_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    source_keys_by_type: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=1):
        source_type = _source_type_key(row.get("source_type")) or "unknown"
        source_key = _clean(row.get("source_key") or row.get("source_id")) or f"row:{index}"
        source_keys_by_type.setdefault(source_type, set()).add(source_key)
    return {
        source_type: len(source_keys)
        for source_type, source_keys in sorted(source_keys_by_type.items())
    }


def _item_weighted_source_volume_by_type(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    # Breakdown is by type, so a rare multi-type source key is counted once per type.
    return weighted_source_volume_by_group(
        rows,
        group_key=lambda row: _source_type_key(row.get("source_type")) or "unknown",
    )


def _weighted_frequency(rows: Sequence[Mapping[str, Any]]) -> int:
    return weighted_source_volume_by_group(rows, group_key=lambda _row: "all").get("all", 0)


def weighted_source_volume_by_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: Callable[[Mapping[str, Any]], str],
    source_key: Callable[[Mapping[str, Any], int], str] | None = None,
    row_weight: Callable[[Mapping[str, Any]], int] | None = None,
) -> dict[str, int]:
    """Sum max represented source weight by group.

    Default weighting honors all source-weight aliases. In the ranking path,
    rows are already normalized with source_weight before this helper runs.
    """

    weights: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows, start=1):
        group = group_key(row) or "unknown"
        key = (
            source_key(row, index)
            if source_key is not None
            else _clean(row.get("source_key") or row.get("source_id")) or f"row:{index}"
        )
        weight = row_weight(row) if row_weight is not None else source_row_weight(row)
        weights[(group, key)] = max(weights.get((group, key), 0), max(weight, 1))
    counts: dict[str, int] = {}
    for (group, _source_key), weight in weights.items():
        counts[group] = counts.get(group, 0) + weight
    return dict(sorted(counts.items()))


def source_row_weight(*rows: Mapping[str, Any]) -> int:
    """Return the represented source volume for normalized source rows."""

    for row in rows:
        for key in _SOURCE_WEIGHT_KEYS:
            weight = _integer_or_none(_field_value(row, key))
            if weight is not None and weight > 0:
                return weight
    return 1


def _source_weight(*rows: Mapping[str, Any]) -> int:
    return source_row_weight(*rows)


def _resolution_text(*rows: Mapping[str, Any], question_text: str = "") -> str:
    for row in rows:
        for key in _RESOLUTION_TEXT_KEYS:
            text = support_ticket_plain_text(_field_value(row, key))
            if text and _resolution_text_is_publishable(text, question_text=question_text):
                return text
    return ""


# Imperative-shape detection for the publishable gate (#1466 Option 1). A
# resolution is publishable when it survives the reject filters AND looks like
# an instruction, rather than when it contains a token from a fixed action-term
# list. This recognizes unknown verbs (schedule, pin, narrow, forward, ...) by
# the "<verb> the/your/a <noun>" imperative shape, numbered steps, or a UI
# navigation path -- the enumerated lists could never keep up (four rounds).
# Split on sentence terminators but KEEP them attached (lookbehind), so a
# trailing "?" survives and an interrogative sentence ("Did the lights change?")
# can be told apart from a step. The lead word and object shapes are anchored to
# the start of each sentence, so a retained terminator does not affect them. The
# `(?<![0-9].)` guard keeps a list-number period ("1. Open ...") from splitting
# the number off its step, so the per-sentence reject filters still apply to the
# whole numbered sentence (a numbered question is rejected, not accepted).
_RESOLUTION_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?<![0-9].)\s+|\n+")
_RESOLUTION_UI_PATH_RE = re.compile(
    r"\b[a-z][\w-]*\s+then\s+[a-z][\w-]*\b", re.IGNORECASE
)
# A leading clause that precedes the real instruction ("To fix this, reset ...").
# A list-number prefix ("1.", "2)", "step 3:") is stripped here too, so a
# numbered step is evaluated by the same lead/object logic as any other sentence
# -- the numbering is not a standalone "this is a step" signal that could bypass
# the question/declarative/redirect rejects (round-7 review BLOCKER).
_RESOLUTION_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(?:please\s+|kindly\s+|first[,]?\s+|then[,]?\s+|next[,]?\s+|now[,]?\s+"
    r"|(?:step\s+)?\d+[.):]\s+"
    r"|to\s+[a-z]+\s+(?:this|it|that|the\s+(?:issue|problem|error))\s*[,:]\s*)+",
    re.IGNORECASE,
)
# Sentence-lead tokens that are not imperative verbs, so a leading "<word> the"
# match (e.g. "About the export") is not mistaken for an instruction.
_RESOLUTION_NON_VERB_LEADS = frozenset({
    "about", "after", "also", "an", "and", "any", "as", "because", "before",
    "but", "for", "he", "her", "here", "his", "how", "however", "i", "in",
    "into", "it", "its", "my", "no", "not", "once", "or", "our", "per", "she",
    "since", "so", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "thanks", "thank", "to", "unfortunately", "we",
    "what", "when", "where", "which", "while", "who", "why", "with", "you",
    "your",
})
# The object of an imperative is an article/possessive-introduced noun phrase or
# an imperative preposition. Demonstratives/pronouns (this/that/it/...) are
# excluded: they are clause subjects, so "<adverb> this is a known issue" must
# not parse as "<verb> <object>".
_RESOLUTION_INSTRUCTION_OBJECT_RE = re.compile(
    r"^(?:the|your|a|an|its|their|to|into|onto|on|off|from)\b",
    re.IGNORECASE,
)
# Linking verbs never lead an imperative; a sentence opening with one is a
# question or declarative ("Is the account active?"), not a step.
_RESOLUTION_COPULA_LEADS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "seem", "seems", "seemed", "look", "looks", "looked",
    "remain", "remains", "remained", "appear", "appears", "appeared",
    "become", "becomes", "became",
})
# "<article> <noun ...> is/are ..." is a declarative subject + copula, not an
# instruction object ("the issue is annoying", "the export is a known limit").
_RESOLUTION_OBJECT_DECLARATIVE_RE = re.compile(
    r"^(?:the|your|a|an|its|their)\s+[a-z'-]+(?:\s+[a-z'-]+){0,2}\s+"
    r"(?:is|are|was|were|be|been|seems?|looks?|remains?|appears?|becomes?)\b",
    re.IGNORECASE,
)
_RESOLUTION_LEAD_WORD_RE = re.compile(r"([a-z][a-z'-]*)", re.IGNORECASE)
# A first-person subject preceding the action ("I enabled ...", "We reset ...").
_RESOLUTION_FIRST_PERSON_LEAD_RE = re.compile(
    r"^(?:i|we)(?:\s+(?:have|had|then|just|already|also))?\s+", re.IGNORECASE
)
# Contact-channel redirection: directing the requester to reach a human over a
# private channel ("send us a DM", "message us", "shoot me a private message")
# is a hand-off, not a self-serve resolution. This is the imperative-shaped
# sibling of the disposition reject (a "Please DM us" reply passes the
# instruction-shape test but answers nothing), surfaced by running the gate over
# real support replies. Narrow on purpose -- it matches "<verb> us/me" and the
# "DM/PM/private message" redirect nouns, not legitimate steps that merely
# contain "send"/"message" ("Send the report to your team", "Message the channel
# owner"). Applied per sentence, so a genuine step followed by a "...then DM us"
# fallback still publishes on the real step.
_RESOLUTION_CONTACT_REDIRECT_RE = re.compile(
    r"\b(?:send|shoot|drop|message|dm|pm|ping)\s+(?:us|me)\b"
    r"|\b(?:dm|pm|message|ping|contact)\s+us\b"
    r"|\bsend\s+(?:us|me)\s+a\s+(?:dm\b|pm\b|message\b|private\s+message|direct\s+message)"
    r"|\b(?:private|direct)\s+message\s+(?:us|me)\b"
    r"|\breach\s+out\s+to\s+us\b"
    r"|\bprivate\s+message\b",
    re.IGNORECASE,
)


def _resolution_text_is_publishable(value: Any, *, question_text: str) -> bool:
    text = _compact(support_ticket_plain_text(value))
    if not text:
        return False
    # Reject filters (honesty floor -- kept verbatim from #1456).
    if _RESOLUTION_CLOSURE_BOILERPLATE_RE.search(text):
        return False
    if _RESOLUTION_INTERNAL_NOTE_RE.search(text):
        return False
    resolution_tokens = _resolution_signal_tokens(text)
    if len(resolution_tokens) < 3:
        return False
    if _resolution_text_is_disposition_only(text, resolution_tokens):
        return False
    # Positive gate: structural instruction shape, not list membership.
    if not _resolution_text_looks_instructional(text):
        return False
    # Action-term membership and question-topic overlap are demoted (#1466
    # Option 1) from hard rejects to advisory signals: computed here for a
    # future confidence surface, but they never block a structurally-valid,
    # non-boilerplate instruction.
    _resolution_advisory_signals(resolution_tokens, question_text)
    return True


def _resolution_text_looks_instructional(text: str) -> bool:
    """True when the resolution reads like a step a user can follow.

    Every positive signal -- a UI navigation path ("Settings then Phone"), a
    known action verb, or the "<verb> the/your/a <noun>" imperative shape for
    verbs not in the list -- is evaluated per sentence and only AFTER that
    sentence clears the question / disposition / contact-redirect /
    declarative-lead rejects. A numbered/UI-path/declarative diagnostic ("1. Did
    the lights change?", "The issue is in Billing then Plan ...") therefore can
    no longer short-circuit those rejects (round-7 review BLOCKER); list numbers
    are stripped as a prefix, not treated as a standalone "this is a step" flag.
    """

    for sentence in _RESOLUTION_SENTENCE_SPLIT_RE.split(text):
        stripped = sentence.strip()
        if not stripped:
            continue
        # A question is not a step. An agent reply that ends in "?" ("Did the
        # lights change on the router?") is a diagnostic prompt back to the
        # requester, not a resolution they can act on.
        if stripped.endswith("?"):
            continue
        if _RESOLUTION_DISPOSITION_ONLY_RE.search(stripped):
            continue
        prefix = _RESOLUTION_INSTRUCTION_PREFIX_RE.match(stripped)
        remainder = stripped[prefix.end():] if prefix else stripped
        first_person = _RESOLUTION_FIRST_PERSON_LEAD_RE.match(remainder)
        if first_person:
            remainder = remainder[first_person.end():]
        # A contact redirect only disqualifies the sentence when it is the lead
        # clause ("Send us a DM ...", "Have your friend message us"). A redirect
        # trailing a real step ("Reset the cache, then DM us if it persists")
        # leaves the step intact, so check only up to the first clause break.
        lead_clause = re.split(r"[,;]", remainder, maxsplit=1)[0]
        if _RESOLUTION_CONTACT_REDIRECT_RE.search(lead_clause):
            continue
        lead_match = _RESOLUTION_LEAD_WORD_RE.match(remainder)
        if not lead_match:
            continue
        lead = lead_match.group(1).lower()
        if lead in _RESOLUTION_NON_VERB_LEADS or lead in _RESOLUTION_COPULA_LEADS:
            continue
        # A UI navigation path ("Settings then Phone") is a step -- but only now
        # that the sentence has cleared the question / disposition / redirect /
        # non-verb-lead rejects above, so a declarative ("The issue is in Billing
        # then Plan") or question ("Did Settings then Phone show ...?") cannot
        # qualify through the path shape.
        if _RESOLUTION_UI_PATH_RE.search(remainder):
            return True
        # Stem the lead so past-tense / gerund action verbs ("Enabled the SSO",
        # "Configured ...", "Updating ...") still register as instructions.
        if _resolution_signal_token(lead) in _RESOLUTION_ACTION_TERMS:
            return True
        # The generic "<verb> the/your <noun>" shape qualifies unknown verbs
        # ("Schedule the sync", "Pin the view") only in imperative position. A
        # first-person subject makes it narration ("We received your request",
        # "We closed your ticket"), so it must use a known action verb instead.
        if first_person:
            continue
        rest = remainder[lead_match.end():].lstrip()
        # The object must be a genuine instruction object, not a clause subject
        # that a copula turns into a declarative ("Honestly the issue is ...").
        if _RESOLUTION_INSTRUCTION_OBJECT_RE.match(rest) and not (
            _RESOLUTION_OBJECT_DECLARATIVE_RE.match(rest)
        ):
            return True
    return False


def _resolution_advisory_signals(
    resolution_tokens: set[str], question_text: str
) -> dict[str, bool]:
    """Demoted confidence signals (#1466 Option 1): computed, never gating.

    The action-term-membership and question-topic-overlap checks were hard
    rejects through four enumeration rounds and over-rejected real answers.
    They are retained here as advisory diagnostics so the maps stay live for a
    future confidence surface; they do not influence publishability.
    """

    question_tokens = _resolution_signal_tokens(question_text)
    return {
        "has_known_action_term": not resolution_tokens.isdisjoint(
            _RESOLUTION_ACTION_TERMS
        ),
        "topic_aligned": (
            not question_tokens
            or not _resolution_overlap_tokens(resolution_tokens).isdisjoint(
                _resolution_overlap_tokens(question_tokens)
            )
        ),
    }


def _resolution_signal_tokens(value: Any) -> set[str]:
    return {
        _resolution_signal_token(token)
        for token in re.findall(r"[a-z0-9]+", _compact(value).lower())
        if len(token) > 2 and token not in _RESOLUTION_TOPIC_STOPWORDS
    }


def _resolution_text_is_disposition_only(text: str, resolution_tokens: set[str]) -> bool:
    action_tokens = resolution_tokens & _RESOLUTION_ACTION_TERMS
    return bool(
        action_tokens
        and action_tokens <= _RESOLUTION_DISPOSITION_ONLY_ACTION_TERMS
        and _RESOLUTION_DISPOSITION_ONLY_RE.search(text)
    )


def _resolution_overlap_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_RESOLUTION_TOPIC_EQUIVALENCE.get(token, ()))
    return expanded


def _resolution_signal_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if base in _RESOLUTION_ACTION_TERMS:
            return base
        if f"{base}e" in _RESOLUTION_ACTION_TERMS:
            return f"{base}e"
        if len(base) > 3 and base[-1:] == base[-2:-1] and base[:-1] in _RESOLUTION_ACTION_TERMS:
            return base[:-1]
        return base
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        if base in _RESOLUTION_ACTION_TERMS:
            return base
        if f"{base}e" in _RESOLUTION_ACTION_TERMS:
            return f"{base}e"
        if base.endswith("i") and f"{base[:-1]}y" in _RESOLUTION_ACTION_TERMS:
            return f"{base[:-1]}y"
        if len(base) > 3 and base[-1:] == base[-2:-1] and base[:-1] in _RESOLUTION_ACTION_TERMS:
            return base[:-1]
        return base
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _resolution_texts(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values = (
        support_ticket_plain_text(row.get("resolution_text"))
        for row in rows
        if support_ticket_plain_text(row.get("resolution_text"))
    )
    return tuple(dict.fromkeys(values))


def _resolution_rows_for_question(
    rows: Sequence[Mapping[str, Any]],
    question_row: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    evidence_group_key = _clean(
        question_row.get("evidence_group_key") if question_row is not None else ""
    )
    if not evidence_group_key:
        return tuple(rows)
    scoped_rows = tuple(
        row for row in rows if _clean(row.get("evidence_group_key")) == evidence_group_key
    )
    return scoped_rows or tuple(rows)


def _resolution_source_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(_distinct_source_keys([
        row for row in rows if _compact(row.get("resolution_text"))
    ]))


def _resolution_evidence_scope_status(
    *,
    has_mixed_evidence_scopes: bool,
    question_row: Mapping[str, Any] | None,
    resolution_rows: Sequence[Mapping[str, Any]],
    resolution_texts: Sequence[str],
) -> str:
    if not resolution_texts:
        return "not_applicable"
    if has_mixed_evidence_scopes:
        return "mixed_evidence_scope"
    resolution_scopes = {
        _clean(row.get("evidence_group_key"))
        for row in resolution_rows
        if _compact(row.get("resolution_text")) and _clean(row.get("evidence_group_key"))
    }
    if len(resolution_scopes) != 1:
        return "missing_resolution_scope"
    question_scope = _clean(
        question_row.get("evidence_group_key") if question_row is not None else ""
    )
    if not question_scope:
        return "missing_question_scope"
    if question_scope and resolution_scopes != {question_scope}:
        return "scope_mismatch"
    return "scoped"


def _resolution_excerpt(value: Any, *, limit: int = 180) -> str:
    sentence = _first_sentence(_compact(value))
    text = sentence or _compact(value)
    if len(text) <= limit:
        return text
    excerpt = text[:limit].rstrip()
    if " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{excerpt or text[:limit].rstrip()}..."


def _answer_summary(
    *,
    question: str,
    source_count: int,
    answer_evidence_status: str,
    resolution_texts: Sequence[str] = (),
) -> str:
    source_label = _source_count_label(source_count)
    if answer_evidence_status == "resolution_evidence":
        return _resolution_answer_summary(resolution_texts, source_label=source_label)
    return (
        f"No verified resolution evidence was found in {source_label}; keep "
        f"this FAQ in review before answering: {question}"
    )


def _resolution_answer_summary(
    resolution_texts: Sequence[str],
    *,
    source_label: str,
) -> str:
    steps = _resolution_help_center_steps(resolution_texts)
    if not steps:
        return f"This draft answer is backed by uploaded resolution evidence from {source_label}."
    first = _clause_text(steps[0])
    if len(steps) == 1:
        return f"To resolve this, {first}."
    second = _clause_text(steps[1])
    return f"To resolve this, {first}. Then {second}."


def _source_count_label(source_count: int) -> str:
    if source_count == 1:
        return "1 ticket source"
    return f"{source_count} ticket sources"


def _term_mappings(
    rows: Sequence[Mapping[str, str]],
    documentation_terms: Sequence[str],
    vocabulary_gap_rules: Sequence[Sequence[str]],
) -> tuple[dict[str, Any], ...]:
    doc_terms = _clean_terms(documentation_terms)
    if not rows or not doc_terms:
        return ()
    customer_text = " ".join((
        " ".join(_clean(row.get("source_title")) for row in rows),
        " ".join(_clean(row.get("text")) for row in rows),
    )).lower()
    if not customer_text:
        return ()
    source_ids = _distinct_source_keys(rows)
    # Vocabulary-gap impact should reflect the evidence rows, not the FAQ topic label.
    opportunity = _opportunity_score("", rows)
    zero_result_source_count = _zero_result_source_count(rows)
    mappings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for aliases in vocabulary_gap_rules:
        for customer_term in aliases:
            if not _keyword_matches(customer_text, customer_term):
                continue
            if any(_keyword_matches(term.lower(), customer_term) for term in doc_terms):
                continue
            documentation_term = _matching_documentation_term(
                doc_terms,
                aliases=aliases,
                customer_term=customer_term,
            )
            if not documentation_term:
                continue
            key = (customer_term, documentation_term.lower())
            if key in seen:
                continue
            seen.add(key)
            mappings.append({
                "customer_term": customer_term,
                "documentation_term": documentation_term,
                "suggestion": (
                    f'Add "{customer_term}" as alternate phrasing for '
                    f'"{documentation_term}" in FAQ headings and answer text.'
                ),
                "source_id_count": len(source_ids),
                "zero_result_source_count": zero_result_source_count,
                "failure_risk_score": opportunity["failure_risk_score"],
                "failure_risk_signals": opportunity["failure_risk_signals"],
                "opportunity_score": opportunity["opportunity_score"],
                "first_source_id": source_ids[0] if source_ids else None,
            })
            break
        if len(mappings) >= 3:
            break
    return tuple(mappings)


def _vocabulary_gap_rules(custom_rules: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return (*_custom_vocabulary_gap_rules(custom_rules), *_VOCABULARY_GAP_RULES)


def normalize_vocabulary_gap_rules(
    custom_rules: Sequence[Sequence[str]],
    *,
    label: str = "vocabulary_gap_rules",
) -> tuple[tuple[str, ...], ...]:
    """Normalize caller-supplied vocabulary-gap alias groups."""
    if not isinstance(custom_rules, Sequence) or isinstance(
        custom_rules,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{label} must be an array of string arrays")
    rules: list[tuple[str, ...]] = []
    for index, aliases in enumerate(custom_rules, start=1):
        if isinstance(aliases, (str, bytes, bytearray)):
            raise ValueError(f"{label} entries must include at least two terms")
        if not isinstance(aliases, Sequence):
            raise ValueError(f"{label}[{index}] must be a string array")
        cleaned = _clean_terms(aliases)
        if len(cleaned) < 2:
            raise ValueError(f"{label} entries must include at least two terms")
        rules.append(cleaned)
    return tuple(rules)


def _custom_vocabulary_gap_rules(custom_rules: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return normalize_vocabulary_gap_rules(custom_rules)


def normalize_intent_rules(
    custom_rules: Sequence[Any],
    *,
    label: str = "intent_rules",
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Normalize caller-supplied FAQ intent mapping rules."""

    if not isinstance(custom_rules, Sequence) or isinstance(
        custom_rules,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{label} must be an array of intent rules")

    rules: list[tuple[str, tuple[str, ...]]] = []
    for index, raw_rule in enumerate(custom_rules, start=1):
        topic: Any
        keywords: Any
        if isinstance(raw_rule, str):
            topic, separator, raw_keywords = raw_rule.partition("=")
            if not separator:
                raise ValueError(f"{label}[{index}] must use topic=keyword,keyword")
            keywords = raw_keywords.split(",")
        elif isinstance(raw_rule, Mapping):
            topic = raw_rule.get("topic")
            keywords = raw_rule.get("keywords")
        elif isinstance(raw_rule, Sequence) and not isinstance(
            raw_rule,
            (bytes, bytearray),
        ):
            values = tuple(raw_rule)
            if len(values) != 2:
                raise ValueError(f"{label}[{index}] must include topic and keywords")
            topic, keywords = values
        else:
            raise ValueError(f"{label}[{index}] must be an intent rule")

        cleaned_topic = _clean(topic)
        cleaned_keywords = _clean_terms(_list(keywords))
        if not cleaned_topic or not cleaned_keywords:
            raise ValueError(
                f"{label}[{index}] must include topic and at least one keyword"
            )
        rules.append((cleaned_topic, cleaned_keywords))
    return tuple(rules)


def _zero_result_source_count(rows: Sequence[Mapping[str, Any]]) -> int:
    source_ids = _distinct_source_keys([
        row for row in rows if _is_zero_result_search_row(row)
    ])
    return len(source_ids)


def _matching_documentation_term(
    documentation_terms: Sequence[str],
    *,
    aliases: Sequence[str],
    customer_term: str,
) -> str:
    for term in documentation_terms:
        lowered = term.lower()
        if _keyword_matches(lowered, customer_term):
            continue
        if any(_keyword_matches(lowered, alias) for alias in aliases if alias != customer_term):
            return term
    return ""


def _documentation_terms(
    opportunities: Sequence[Mapping[str, Any]],
    explicit_terms: Sequence[str],
) -> tuple[str, ...]:
    terms: list[str] = []
    terms.extend(_clean(term) for term in explicit_terms if _clean(term))
    for opportunity in opportunities:
        opportunity_type = _source_type_key(opportunity.get("source_type"))
        if opportunity_type in _DOCUMENTATION_SOURCE_TYPES:
            terms.extend(_documentation_terms_from_row(opportunity))
        for evidence in _evidence_rows(opportunity):
            evidence_type = _source_type_key(evidence.get("source_type") or opportunity_type)
            if evidence_type in _DOCUMENTATION_SOURCE_TYPES:
                terms.extend(_documentation_terms_from_row(evidence))
    return _clean_terms(terms)


def _documentation_terms_from_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    terms: list[str] = []
    for key in _DOCUMENTATION_TERM_KEYS:
        text = _compact(_field_value(row, key))
        if text:
            terms.append(_term_excerpt(text))
    return tuple(terms)


def _clean_terms(terms: Sequence[str]) -> tuple[str, ...]:
    out: dict[str, str] = {}
    for term in terms:
        text = _compact(term)
        if not text:
            continue
        out.setdefault(text.lower(), text)
    return tuple(out.values())


def _term_excerpt(value: str, *, limit: int = 90) -> str:
    text = _compact(value)
    if len(text) <= limit:
        return text
    sentence = _first_sentence(text)
    if sentence and len(sentence) <= limit:
        return sentence
    return f"{text[:limit].rstrip()}..."


def _failure_risk_signals(topic: str, rows: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    text = " ".join((
        topic,
        " ".join(_clean(row.get("source_title")) for row in rows),
        " ".join(_clean(row.get("text")) for row in rows),
    )).lower()
    if not text:
        return ()
    signals = []
    for signal, terms in _FAILURE_RISK_RULES:
        if any(_keyword_matches(text, term) for term in terms):
            signals.append(signal)
    if any(_is_zero_result_search_row(row) for row in rows):
        signals.append("zero_result_search")
    return tuple(signals)


def is_zero_result_search_row(row: Mapping[str, Any]) -> bool:
    """Return whether a normalized source row represents a zero-result search."""

    source_type = _source_type_key(row.get("source_type"))
    if source_type not in {"search_log", "search_query"}:
        return False
    for key in ("zero_results", "zero_result"):
        value = row.get(key)
        if _truthy(value):
            return True
    for key in ("results_count", "result_count"):
        count = _integer_or_none(row.get(key))
        if count == 0:
            return True
    return False


def _is_zero_result_search_row(row: Mapping[str, Any]) -> bool:
    return is_zero_result_search_row(row)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean(value).lower()
    return text in {"1", "true", "yes", "y"}


def _integer_or_none(value: Any) -> int | None:
    if value in (None, "", [], {}):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(*rows: Mapping[str, Any], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def _first_available(*rows: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _first_present(*rows, key=key)
        if value not in (None, "", [], {}):
            return value
    return ""


def _question(
    topic: str,
    rows: Sequence[Mapping[str, str]],
    documentation_terms: Sequence[str],
) -> tuple[str, str, Mapping[str, str] | None]:
    for row in rows:
        text = _publishable_customer_question_text(row.get("text", ""))
        if text:
            return (text, "customer_wording", row)
    representative_question = _representative_source_question(
        topic,
        rows,
        documentation_terms=documentation_terms,
    )
    if representative_question:
        return (representative_question, "source_policy", None)
    policy_question = _policy_question(topic)
    if policy_question:
        return (policy_question, "source_policy", None)
    return (f"What are customers asking about {topic}?", "topic_fallback", None)


def _representative_source_question(
    topic: str,
    rows: Sequence[Mapping[str, str]],
    *,
    documentation_terms: Sequence[str],
) -> str:
    if not any(_clean(row.get("support_ticket_cluster")) for row in rows):
        return ""
    if not any(
        _source_type_key(row.get("source_type")) in _REPRESENTATIVE_LABEL_SOURCE_TYPES
        for row in rows
    ):
        return ""
    label = _safe_vocabulary_representative_label(
        topic,
        rows,
        documentation_terms=documentation_terms,
    )
    if not label:
        return ""
    question = _normalize_question_text(f"What should I do about {label.lower()}")
    return question if _usable_question(question) else ""


def _safe_vocabulary_representative_label(
    topic: str,
    rows: Sequence[Mapping[str, str]],
    *,
    documentation_terms: Sequence[str],
) -> str:
    safe_tokens = _safe_representative_tokens(
        (
            *documentation_terms,
            *(
                _clean(row.get("safe_label_context"))
                for row in rows
                if _clean(row.get("safe_label_context"))
            ),
        )
    )
    if not safe_tokens:
        return ""
    topic_tokens = support_ticket_tokens(topic)
    token_counts: dict[str, int] = {}
    for row in rows:
        text = str(row.get("text") or "")
        for token in _question_gist_tokens(text):
            if (
                token not in safe_tokens
                or token in topic_tokens
                or _REDACTION_TOKEN_RE.fullmatch(token)
            ):
                continue
            token_counts[token] = token_counts.get(token, 0) + 1
    repeated_tokens = {
        token: count
        for token, count in token_counts.items()
        if count >= 2
    }
    if len(repeated_tokens) < 2:
        return ""
    tokens = sorted(
        repeated_tokens,
        key=lambda token: (-repeated_tokens[token], safe_tokens[token][0], token),
    )[:5]
    return " ".join(safe_tokens[token][1] for token in tokens)


def _disambiguate_source_policy_question_collisions(
    item_records: Sequence[tuple[Sequence[Mapping[str, str]], Mapping[str, Any]]],
    *,
    documentation_terms: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    items = [dict(item) for _rows, item in item_records]
    indexes_by_question: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        if item.get("question_source") != "source_policy":
            continue
        question = _clean(item.get("question"))
        if question:
            indexes_by_question[question].append(index)

    warnings: list[dict[str, Any]] = []
    for question, indexes in indexes_by_question.items():
        if len(indexes) < 2:
            continue
        suffixes: list[str] = []
        seen_suffixes: set[str] = set()
        for index in indexes:
            rows = item_records[index][0]
            suffix = _source_policy_disambiguation_suffix(
                question,
                rows,
                documentation_terms=documentation_terms,
            )
            if not suffix or suffix in seen_suffixes:
                suffixes = []
                break
            suffixes.append(suffix)
            seen_suffixes.add(suffix)
        if len(suffixes) != len(indexes):
            warnings.append(_duplicate_source_policy_question_warning(question, items, indexes))
            continue
        disambiguated_questions = tuple(
            _disambiguated_question(question, suffix) for suffix in suffixes
        )
        if not all(disambiguated_questions):
            warnings.append(_duplicate_source_policy_question_warning(question, items, indexes))
            continue
        for index, disambiguated in zip(indexes, disambiguated_questions):
            rows = item_records[index][0]
            items[index]["question"] = disambiguated
            _promote_disambiguated_resolution_evidence(items[index], rows, disambiguated)
            items[index]["answer"] = _answer_for_disambiguated_question(
                items[index],
                disambiguated,
            )
    return (tuple(items), tuple(warnings))


def _promote_disambiguated_resolution_evidence(
    item: dict[str, Any],
    rows: Sequence[Mapping[str, str]],
    question: str,
) -> None:
    if item.get("resolution_evidence_scope") != "missing_question_scope":
        return
    resolution_texts = _resolution_texts(rows)
    if not resolution_texts:
        return
    item["answer_evidence_status"] = "resolution_evidence"
    item["resolution_evidence_scope"] = "scoped"
    item["steps"] = _article_steps(
        _clean(item.get("topic")),
        question,
        support_contact=None,
        resolution_texts=resolution_texts,
    )
    item["action_items"] = item["steps"]


def _answer_for_disambiguated_question(
    item: Mapping[str, Any],
    question: str,
) -> str:
    if item.get("answer_evidence_status") == "resolution_evidence":
        return str(item.get("answer") or "")
    return _answer_summary(
        question=question,
        source_count=len(_list(item.get("source_ids"))),
        answer_evidence_status=str(item.get("answer_evidence_status") or ""),
    )


def _source_policy_disambiguation_suffix(
    question: str,
    rows: Sequence[Mapping[str, str]],
    *,
    documentation_terms: Sequence[str],
) -> str:
    safe_tokens = _safe_representative_tokens(
        (
            *documentation_terms,
            *(
                _clean(row.get("safe_label_context"))
                for row in rows
                if _clean(row.get("safe_label_context"))
            ),
        )
    )
    if not safe_tokens:
        return ""
    question_tokens = support_ticket_tokens(question)
    token_counts: dict[str, int] = {}
    for row in rows:
        for token in _question_gist_tokens(str(row.get("text") or "")):
            if (
                token not in safe_tokens
                or token in question_tokens
                or _REDACTION_TOKEN_RE.fullmatch(token)
            ):
                continue
            token_counts[token] = token_counts.get(token, 0) + 1
    if not token_counts:
        return ""
    tokens = sorted(
        token_counts,
        key=lambda token: (-token_counts[token], safe_tokens[token][0], token),
    )[:3]
    return " ".join(safe_tokens[token][1] for token in tokens)


def _disambiguated_question(question: str, suffix: str) -> str:
    stem = _clean(question).rstrip("?.!,;: ")
    suffix_text = _clean(suffix).lower()
    if not stem or not suffix_text:
        return ""
    disambiguated = _normalize_question_text(f"{stem} - {suffix_text}")
    return disambiguated if _usable_question(disambiguated) else ""


def _duplicate_source_policy_question_warning(
    question: str,
    items: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
) -> dict[str, Any]:
    source_ids: list[str] = []
    for index in indexes:
        source_ids.extend(str(source_id) for source_id in _list(items[index].get("source_ids")))
    return {
        "code": "duplicate_source_policy_questions",
        "message": (
            "Some source-policy FAQ headings remain duplicated because no safe "
            "controlled-vocabulary disambiguator was available."
        ),
        "question": question,
        "source_ids": source_ids,
    }


def _safe_representative_tokens(
    documentation_terms: Sequence[str],
) -> dict[str, tuple[int, str]]:
    ordered: dict[str, tuple[int, str]] = {}
    for term in documentation_terms:
        if _has_representative_label_pii(term):
            continue
        for token, display in _ordered_support_ticket_tokens(term):
            if _REDACTION_TOKEN_RE.fullmatch(token):
                continue
            ordered.setdefault(token, (len(ordered), display))
    return ordered


def _ordered_support_ticket_tokens(value: Any) -> tuple[tuple[str, str], ...]:
    tokens = support_ticket_tokens(value)
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", support_ticket_plain_text(value).lower()):
        folded = support_ticket_tokens(raw)
        for token in sorted(folded):
            if token in tokens and token not in seen:
                ordered.append((token, raw))
                seen.add(token)
    return tuple(ordered)


def _has_representative_label_pii(value: Any) -> bool:
    text = support_ticket_plain_text(value)
    return bool(
        _REPRESENTATIVE_EMAIL_RE.search(text)
        or _REPRESENTATIVE_LONG_NUMBER_RE.search(text)
    )


def _rows_with_question_source_first(
    rows: Sequence[Mapping[str, str]],
    question_row: Mapping[str, str] | None,
) -> tuple[Mapping[str, str], ...]:
    if question_row is None or not rows or rows[0] is question_row:
        return tuple(rows)
    ordered = [question_row]
    ordered.extend(row for row in rows if row is not question_row)
    return tuple(ordered)


def _question_text(value: Any) -> str:
    return _question_text_matching(value, _usable_question)


def _publishable_customer_question_text(value: Any) -> str:
    return _question_text_matching(
        value,
        _usable_customer_heading_question,
        context_predicate=lambda text: not _has_customer_heading_pii(text),
    )


def _question_text_matching(
    value: Any,
    predicate: Callable[[str], bool],
    *,
    context_predicate: Callable[[str], bool] | None = None,
) -> str:
    for text in _question_candidate_texts(value):
        if context_predicate is not None and not context_predicate(text):
            continue
        if "?" in text:
            prefix, remainder = text.split("?", 1)
            prefix = prefix.strip()
            sentence_parts = [part.strip() for part in re.split(r"[.!:;]+", prefix) if part.strip()]
            candidate = sentence_parts[-1] if sentence_parts else prefix
            normalized = _question_start_text(candidate, predicate=predicate)
            if normalized:
                return normalized
            normalized = _normalize_question_text(candidate)
            if predicate(normalized):
                return normalized
            tail = _compact(remainder)
            if tail:
                normalized = _question_start_text(tail, predicate=predicate)
                if normalized:
                    return normalized
                normalized = _first_person_issue_question_text(tail, predicate=predicate)
                if normalized:
                    return normalized
            continue
        normalized = _question_start_text(text, predicate=predicate)
        if normalized:
            return normalized
        normalized = _first_person_issue_question_text(text, predicate=predicate)
        if normalized:
            return normalized
    return ""


def _question_candidate_texts(value: Any) -> tuple[str, ...]:
    text = _clean(value)
    if not text:
        return ()
    matches = list(_SPEAKER_LABEL_RE.finditer(text))
    if not matches:
        return _question_candidate_variants(text)

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
    variants = _question_candidate_variants(value)
    return variants[0] if variants else ""


def _question_candidate_variants(value: str) -> tuple[str, ...]:
    text = _compact(_URL_RE.sub("", value))
    if not text:
        return ()
    stripped = _strip_leading_question_metadata(text)
    variants = []
    if stripped != text:
        tail = _question_after_metadata_prefix(stripped)
        if tail:
            variants.append(tail)
        variants.append(stripped)
        variants.append(text)
    else:
        variants.append(text)
    return tuple(dict.fromkeys(candidate for candidate in variants if candidate))


def _strip_leading_question_metadata(value: str) -> str:
    return _compact(_LEADING_BRACKETED_METADATA_RE.sub("", value))


def _question_after_metadata_prefix(value: str) -> str:
    for match in _METADATA_QUESTION_START_RE.finditer(value):
        if match.start() <= 0:
            continue
        return _compact(value[match.start():])
    return ""


def _question_start_text(
    text: str,
    *,
    predicate: Callable[[str], bool] | None = None,
) -> str:
    resolved_predicate = predicate or _usable_question
    question_starts = (
        "how ",
        "what ",
        "where ",
        "when ",
        "why ",
        "can ",
        "could ",
        "do ",
        "does ",
        "is ",
        "como ",
    )
    lowered = text.lower()
    if lowered.startswith(question_starts):
        normalized = _normalize_question_text(text)
        if resolved_predicate(normalized):
            return normalized
    return ""


def _first_person_issue_question_text(
    text: str,
    *,
    predicate: Callable[[str], bool] | None = None,
) -> str:
    resolved_predicate = predicate or _usable_question
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
            if resolved_predicate(normalized):
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
            if resolved_predicate(normalized):
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


def _usable_customer_heading_question(value: str) -> bool:
    return _usable_question(value) and not _has_customer_heading_pii(value)


def _has_customer_heading_pii(value: Any) -> bool:
    return _has_published_customer_text_pii(value)


def _published_customer_text(value: Any) -> str:
    text = support_ticket_plain_text(value)
    if not text or _has_published_customer_text_pii(text):
        return ""
    return text


def _published_customer_quote(value: Any, *, limit: int = 220) -> str:
    text = _published_customer_text(value)
    if not text:
        return _PUBLISHED_TEXT_PRIVACY_PLACEHOLDER
    return _quote(text, limit=limit)


def _has_published_customer_text_pii(value: Any) -> bool:
    text = support_ticket_plain_text(value)
    return bool(
        _REPRESENTATIVE_EMAIL_RE.search(text)
        or _CUSTOMER_IDENTIFIER_NUMBER_RE.search(text)
        or _CUSTOMER_HEADING_PHONE_RE.search(text)
        or _REDACTION_TOKEN_RE.search(text)
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
    letters = [character for character in text if character.isalpha()]
    if len(letters) >= 5 and all(character.isupper() for character in letters):
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
            (
                "Provide source rows with support-ticket, case, conversation, "
                "chat, transcript, objection, search, or complaint evidence."
            ),
            "",
        ])
        return "\n".join(lines)
    for index, item in enumerate(items, start=1):
        lines.extend([
            f"## {index}. {_md(item['question'])}",
            "",
            _md(item.get("summary") or item["answer"]),
            "",
        ])
        term_mappings = _list(item.get("term_mappings"))
        if term_mappings:
            lines.extend([
                "**Vocabulary gaps:**",
                "",
                *[f"- {_md(_term_mapping_line(mapping))}" for mapping in term_mappings],
                "",
            ])
        lines.extend([
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
    title = _published_customer_text(row.get("source_title"))
    if source_id and title:
        return f"`{source_id}` - {title}"
    return f"`{source_id or 'unknown'}`"


def _term_mapping_line(mapping: Any) -> str:
    if not isinstance(mapping, Mapping):
        return ""
    customer_term = _clean(mapping.get("customer_term"))
    documentation_term = _clean(mapping.get("documentation_term"))
    suggestion = _clean(mapping.get("suggestion"))
    if customer_term and documentation_term and suggestion:
        line = (
            f'Customers say "{customer_term}"; documentation says '
            f'"{documentation_term}". {suggestion}'
        )
        impact = _term_mapping_impact_line(mapping)
        return f"{line} {impact}" if impact else line
    return suggestion or ""


def _term_mapping_impact_line(mapping: Mapping[str, Any]) -> str:
    source_count = _integer_or_none(mapping.get("source_id_count")) or 0
    zero_result_count = _integer_or_none(mapping.get("zero_result_source_count")) or 0
    opportunity_score = _integer_or_none(mapping.get("opportunity_score")) or 0
    if source_count < 1 and zero_result_count < 1 and opportunity_score < 1:
        return ""
    parts = []
    if source_count:
        parts.append(f"Seen in {source_count} source(s)")
    if zero_result_count:
        parts.append(f"{zero_result_count} zero-result search source(s)")
    if opportunity_score:
        parts.append(f"mapping score {opportunity_score}")
    return f"({'; '.join(parts)}.)"


def _resolve_question_label(
    topic: str,
    rows: Sequence[Mapping[str, str]],
    *,
    max_evidence_per_item: int,
    documentation_terms: Sequence[str],
) -> tuple[str, str, Mapping[str, str] | None]:
    if _has_mixed_evidence_scopes(rows):
        return (
            _MIXED_EVIDENCE_REVIEW_QUESTION,
            "source_policy",
            None,
        )
    display_rows = rows[:max_evidence_per_item]
    return _question(topic, display_rows, documentation_terms)


def _summary(
    *,
    topic: str,
    rows: Sequence[Mapping[str, str]],
    source_count: int,
    question_source: str,
) -> str:
    issue = _topic_label(topic)
    example = (
        _published_customer_quote(rows[0].get("text", ""), limit=180)
        if rows
        else "the cited ticket evidence"
    )
    if source_count > 1:
        if question_source != "customer_wording":
            return (
                f"Customers are asking about {issue} across {source_count} ticket sources. "
                f"Representative evidence includes {example}, so this FAQ should answer "
                "the issue directly and tell users exactly what to try next."
            )
        return (
            f"Customers are asking about {issue} across {source_count} ticket sources. "
            f"The clearest customer wording is {example}, so this FAQ should answer "
            "that request directly and tell users exactly what to try next."
        )
    if question_source != "customer_wording":
        return (
            f"A customer asked about {issue}: {example}. This FAQ should answer the "
            "issue directly and tell the user exactly what to try next."
        )
    return (
        f"A customer asked about {issue}: {example}. This FAQ should answer the "
        "request directly and tell the user exactly what to try next."
    )


def _article_steps(
    _topic: str,
    _evidence_text: str,
    *,
    support_contact: str | None,
    resolution_texts: Sequence[str] = (),
) -> tuple[str, ...]:
    if resolution_texts:
        return _resolution_article_steps(resolution_texts, support_contact=support_contact)
    return _draft_review_steps(support_contact)


def _resolution_article_steps(
    resolution_texts: Sequence[str],
    *,
    support_contact: str | None,
) -> tuple[str, ...]:
    steps = _resolution_help_center_steps(resolution_texts)
    if len(steps) >= 2:
        return (*steps, _support_step(support_contact))
    if len(steps) == 1:
        return (
            steps[0],
            _support_step(support_contact),
        )
    return _draft_review_steps(support_contact)


def _resolution_help_center_steps(resolution_texts: Sequence[str]) -> tuple[str, ...]:
    excerpts = tuple(
        dict.fromkeys(
            _complete_sentence(excerpt)
            for text in resolution_texts
            for excerpt in _resolution_sentence_excerpts(text)
        )
    )
    return tuple(excerpts[:2])


def _resolution_sentence_excerpts(value: Any, *, limit: int = 180) -> tuple[str, ...]:
    text = _compact(value)
    if not text:
        return ()
    parts = tuple(
        part.strip()
        for part in re.split(r"[.!?;\n]+", text)
        if part.strip()
    )
    if not parts:
        return ()
    return tuple(_resolution_excerpt(part, limit=limit) for part in parts if part)


def _complete_sentence(value: Any) -> str:
    text = _compact(value).rstrip(" ,;:")
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _clause_text(value: Any) -> str:
    text = _compact(value).rstrip(".!?")
    if not text:
        return "follow the verified resolution steps"
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return f"{text[0].lower()}{text[1:]}"
    return text


def _draft_review_steps(support_contact: str | None) -> tuple[str, ...]:
    return (
        "Review the cited ticket evidence and confirm the policy-approved answer before publishing.",
        "Draft the customer-facing steps from a verified help article, runbook, macro, or resolved ticket.",
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
    if any(term in text for term in ("communication and contact issues", "call at all hours", "calling at all hours", "various numbers", "telephone calls", "harass", "harassed", "harassment")):
        return (
            f"{_support_sentence(support_contact)} if the company keeps contacting "
            "you after you ask for an explanation or correction in writing."
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
    text = _published_customer_quote(row.get("text", ""), limit=220)
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
    return parse_support_ticket_source_date(value)


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


def _documentation_term_metadata(documentation_terms: Sequence[str]) -> dict[str, Any]:
    terms = _clean_terms(documentation_terms)
    if not terms:
        return {}
    return {"documentation_terms": list(terms)}


def _representative_taxonomy_term_metadata(
    representative_taxonomy_terms: Sequence[str],
) -> dict[str, Any]:
    terms = _clean_terms(representative_taxonomy_terms)
    if not terms:
        return {}
    return {"representative_taxonomy_terms": list(terms)}


def _vocabulary_gap_rule_metadata(rules: Sequence[Sequence[str]]) -> dict[str, Any]:
    cleaned = _custom_vocabulary_gap_rules(rules)
    if not cleaned:
        return {}
    return {"vocabulary_gap_rules": [list(rule) for rule in cleaned]}


def _output_checks(
    *,
    items: Sequence[Mapping[str, Any]],
    ticket_source_count: int,
    rendered_ticket_source_count: int,
    non_repeat_ticket_count: int = 0,
) -> dict[str, bool]:
    has_items = bool(items)
    # Excluded one-off tickets (#1460/#1481) are intentionally absent from
    # rendered items but counted and stated in the report, so they still
    # count as covered sources.
    covers_all_sources = (
        rendered_ticket_source_count + non_repeat_ticket_count
        == ticket_source_count
    )
    return {
        "uses_user_vocabulary": has_items
        and all(item.get("question_source") in _SUPPORTED_QUESTION_SOURCES for item in items),
        "condensed": has_items
        and covers_all_sources
        and (ticket_source_count <= 1 or len(items) < ticket_source_count),
        "has_action_items": has_items and all(bool(item.get("action_items")) for item in items),
        "resolution_evidence_scoped": has_items
        and all(_resolution_evidence_is_scoped(item) for item in items),
    }


def _resolution_evidence_is_scoped(item: Mapping[str, Any]) -> bool:
    if item.get("answer_evidence_status") != "resolution_evidence":
        return True
    return item.get("resolution_evidence_scope") == "scoped"


def _rendered_ticket_source_count(items: Sequence[Mapping[str, Any]]) -> int:
    source_ids: set[str] = set()
    for item in items:
        values = item.get("source_ids")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            source_ids.update(_clean(value) for value in values if _clean(value))
    return len(source_ids)


def _quote(value: Any, *, limit: int = 220) -> str:
    text = support_ticket_plain_text(value)
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
    return source_material_to_source_rows(source_material)


__all__ = [
    "DEFAULT_INTENT_RULES",
    "DEFAULT_TICKET_SOURCE_TYPES",
    "TicketFAQMarkdownConfig",
    "TicketFAQMarkdownResult",
    "TicketFAQMarkdownService",
    "build_ticket_faq_markdown",
    "normalize_intent_rules",
    "normalize_vocabulary_gap_rules",
]
