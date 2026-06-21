"""Customer-facing support-ticket deflection report renderer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

from .campaign_ports import TenantScope
from .ticket_faq_markdown import TicketFAQMarkdownResult, TicketFAQMarkdownService


_RESOLUTION_EVIDENCE_STATUS = "resolution_evidence"
_RESOLUTION_EVIDENCE_SCOPE_SCOPED = "scoped"
_DRAFT_NEEDS_REVIEW_STATUS = "draft_needs_review"
DEFAULT_DEFLECTION_SNAPSHOT_TOP_N = 5
DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT = 3
DEFAULT_DEFLECTION_SEO_TARGET_LIMIT = 50
DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION = "deflection_evidence.v1"
DEFLECTION_REPORT_SCHEMA_VERSION = "deflection.v1"
DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION = "deflection_full_report_qa_scorecard.v1"
DEFLECTION_REPORT_MODEL_FIELDS = (
    "schema_version",
    "title",
    "summary",
    "sections",
)
DEFLECTION_REPORT_SECTION_FIELDS = (
    "id",
    "title",
    "priority",
    "surfaces",
    "default_limit",
    "required_data",
    "snapshot_safe_fields",  # Privacy-boundary field; keep frontend docs in sync.
    "data",
)
DEFLECTION_FULL_REPORT_QA_REQUIRED_SURFACES = (
    "email",
    "result_page",
    "pdf",
    "evidence_export",
)
_UNCAPPED_REPORT_MAX_ITEMS = 0
_ASSISTED_CONTACT_COST = 13.50
_ASSISTED_CONTACT_COST_LABEL = "$13.50"
_ACTION_RESULT_PAGE_LIMIT = 3
_ACTION_PDF_LIMIT = 10
_ACTION_BACKLOG_LIMIT = 25
_ACTION_STATUS_PRIORITY_WEIGHTS = MappingProxyType({
    "Needs answer": 50,
    "Already covered but still recurring": 45,
    "Needs review": 35,
    "Draft ready": 25,
    "Low confidence": 0,
})
_SOURCE_EXAMPLE_LIMIT = 3
_DEFLECTION_BOUNDARY_LEFT = r"(?<![A-Za-z0-9])"
_DEFLECTION_BOUNDARY_RIGHT = r"(?![A-Za-z0-9])"
_DEFLECTION_EMAIL_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}"
)
_DEFLECTION_PHONE_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}"
)
_DEFLECTION_SSN_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"\d{3}-\d{2}-\d{4}"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}"
)
_DEFLECTION_PAYMENT_CARD_RE = re.compile(
    r"(?<![A-Za-z0-9@])"
    r"\d(?:[\s-]?\d){12,63}"
    r"(?![A-Za-z0-9@])"
)
_DEFLECTION_IDENTIFIER_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"(?:account|accounts|acct|case|cases|claim|claims|confirmation|"
    r"confirmations|customer|customers|id|ids|invoice|invoices|member|"
    r"members|order|orders|ref|refs|reference|references|ticket|tickets)"
    r"\s*(?:#|number|no\.?|is)?\s*[:#-]?\s*\d{4,}"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}"
    r"|"
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"\d{4,}\s*(?:account|accounts|acct|case|cases|claim|claims|customer|"
    r"customers|invoice|invoices|member|members|order|orders|ref|refs|"
    r"reference|references|ticket|tickets)"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}",
    re.IGNORECASE,
)
_DEFLECTION_REDACTION_ARTIFACT_RE = re.compile(
    r"\[(?:redacted(?!-(?:address|email|identifier|name|phone|text)\])|removed|hidden)[^\]]*\]"
    r"|\bX{4,}\b",
    re.IGNORECASE,
)
_DEFLECTION_PERSON_NAME_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"(?P<prefix>"
    r"(?i:(?:customer|requester|contact|client|user|agent|member|person)"
    r"(?:\s+name)?(?:\s*(?:is|:|=|-)\s*|\s+)"
    r"|name\s*(?:is|:|=|-)\s*)"
    r")"
    r"(?P<name>"
    r"[A-Za-z][A-Za-z'-]+"
    r"\s+[A-Za-z][A-Za-z'-]+"
    r"(?:\s+[A-Z][A-Za-z'-]+)?"
    r")"
    r"(?=$|[\s,.;:!?)]|</)",
)
_DEFLECTION_PERSON_NAME_REJECT_TOKENS = frozenset(
    {
        "account",
        "analytics",
        "api",
        "billing",
        "case",
        "claim",
        "client",
        "contact",
        "customer",
        "dashboard",
        "desk",
        "export",
        "faq",
        "help",
        "id",
        "identifier",
        "invoice",
        "is",
        "member",
        "name",
        "number",
        "password",
        "person",
        "portal",
        "report",
        "reporting",
        "requester",
        "reset",
        "source",
        "sso",
        "support",
        "ticket",
        "token",
        "user",
        "was",
        "were",
        "will",
    }
)
_DEFLECTION_ROLE_PHRASE_TOKENS = frozenset(
    {
        "annual",
        "benefit",
        "benefits",
        "cycle",
        "experience",
        "feature",
        "manager",
        "plan",
        "platinum",
        "premium",
        "program",
        "research",
        "subscription",
        "success",
        "team",
    }
)
_DEFLECTION_TRAILING_LABEL_CONTEXT_TOKENS = frozenset(
    {
        "account",
        "license",
        "membership",
        "package",
        "plan",
        "seat",
        "status",
        "subscription",
        "tier",
    }
)
_DEFLECTION_STREET_ADDRESS_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"\d{1,6}\s+"
    r"(?P<street_body>"
    r"(?:[NSEW]\.?\s+)?"
    r"(?:[A-Z0-9][A-Za-z0-9'.-]*\s+){0,6}"
    r")"
    r"(?P<street_suffix>"
    r"Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
    r"Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Way|Place|Pl\.?|Terrace|Ter\.?|"
    r"Circle|Cir\.?|Highway|Hwy\.?"
    r")"
    r"(?:\s+(?:Apt|Apartment|Suite|Ste\.?|Unit|#)\s*[A-Za-z0-9-]+)?"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}",
    re.IGNORECASE,
)
_DEFLECTION_STREET_ADDRESS_REJECT_TOKENS = frozenset(
    {
        "are",
        "case",
        "cases",
        "claim",
        "claims",
        "in",
        "invoice",
        "invoices",
        "is",
        "one",
        "on",
        "the",
        "ticket",
        "tickets",
    }
)
_DEFLECTION_CONTEXTUAL_OPAQUE_IDENTIFIER_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"(?P<prefix>"
    r"(?i:(?:account|case|claim|contact|customer|member|order|profile|ref|"
    r"reference|session|token|user)(?:\s+(?:code|id|identifier|token))?"
    r"|(?:code|id|identifier|token))"
    r"\s*(?:is|:|=|-)?\s*)"
    r"(?P<identifier>"
    r"(?:[A-Z0-9]{3,}(?:[-_][A-Z0-9]{3,})+|[A-Z0-9]{8,}|\d{4,})"
    r")"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}",
    re.IGNORECASE,
)
_DEFLECTION_STANDARD_PREFIXED_CONTEXTUAL_IDENTIFIER_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"(?P<prefix>"
    r"(?i:(?:account|case|claim|contact|customer|member|order|profile|ref|"
    r"reference|session|token|user)(?:\s+(?:code|id|identifier|token))?"
    r"|(?:code|id|identifier|token))"
    r"\s*(?:is|:|=|-)?\s*)"
    r"(?P<identifier>"
    r"(?:ISO|HIPAA)[-_ ]+"
    r"(?:[A-Z0-9]{3,}(?:[-_][A-Z0-9]{3,})+|[A-Z0-9]{6,}|\d{4,})"
    r")"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}",
    re.IGNORECASE,
)
_DEFLECTION_OPAQUE_IDENTIFIER_RE = re.compile(
    rf"{_DEFLECTION_BOUNDARY_LEFT}"
    r"(?:[A-Z0-9]{8,}|[A-Z0-9]{3,}(?:[-_][A-Z0-9]{3,})+)"
    rf"{_DEFLECTION_BOUNDARY_RIGHT}",
    re.IGNORECASE,
)
_DEFLECTION_OPAQUE_IDENTIFIER_PREFIXES = (
    "acct",
    "account",
    "case",
    "claim",
    "contact",
    "cust",
    "customer",
    "member",
    "ord",
    "order",
    "profile",
    "ref",
    "session",
    "user",
    "usr",
)
_DEFLECTION_TECHNICAL_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"CVE-\d{4}-\d{4,}"
    r"|CWE-\d{1,6}"
    r"|GHSA-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}"
    r"|ISO[-_ ]?\d{3,5}"
    r"|HIPAA[-_ ]?\d{2,4}"
    r"|SKU[-_ ]?\d{4,}"
    r")",
    re.IGNORECASE,
)
_DEFLECTION_STANDARD_NUMBER_LABEL_RE = re.compile(
    r"\s*\d{3,5}(?:\s+(?:certification|certifications|reference|references|standard|standards))?\s*",
    re.IGNORECASE,
)
_DEFLECTION_IDENTIFIER_KEYS = frozenset(
    {
        "account",
        "account_id",
        "account_ids",
        "acct",
        "acct_id",
        "acct_ids",
        "case",
        "case_id",
        "case_ids",
        "claim",
        "claim_id",
        "claim_ids",
        "confirmation",
        "confirmation_id",
        "confirmation_ids",
        "customer",
        "customer_id",
        "customer_ids",
        "invoice",
        "invoice_id",
        "invoice_ids",
        "member",
        "member_id",
        "member_ids",
        "order",
        "order_id",
        "order_ids",
        "ref",
        "ref_id",
        "ref_ids",
        "reference",
        "reference_id",
        "reference_ids",
        "ticket",
    }
)
_DEFLECTION_SOURCE_LINK_KEYS = frozenset(
    {
        "source_id",
        "source_ids",
        "ticket_id",
        "ticket_ids",
    }
)
_DEFLECTION_IDENTIFIER_KEY_SUFFIXES = (
    "_account_id",
    "_account_ids",
    "_acct_id",
    "_acct_ids",
    "_case_id",
    "_case_ids",
    "_claim_id",
    "_claim_ids",
    "_confirmation_id",
    "_confirmation_ids",
    "_customer_id",
    "_customer_ids",
    "_invoice_id",
    "_invoice_ids",
    "_member_id",
    "_member_ids",
    "_order_id",
    "_order_ids",
    "_ref_id",
    "_ref_ids",
    "_reference_id",
    "_reference_ids",
)
_DEFLECTION_SOURCE_LINK_KEY_SUFFIXES = (
    "_source_id",
    "_source_ids",
    "_ticket_id",
    "_ticket_ids",
)
_DEFLECTION_IDENTIFIER_TOKEN_PREFIX = "deflection-ref"


@dataclass(frozen=True)
class DeflectionSnapshot:
    """Free preview projection that excludes paid answer/evidence fields."""

    summary: dict[str, Any]
    top_questions: tuple[dict[str, Any], ...]
    locked_questions: tuple[dict[str, int], ...]
    teaser: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "top_questions": [dict(question) for question in self.top_questions],
            "locked_questions": [
                dict(question) for question in self.locked_questions
            ],
            "teaser": {
                "full_answer": (
                    dict(self.teaser["full_answer"])
                    if isinstance(self.teaser.get("full_answer"), Mapping)
                    else None
                ),
                "previews": [
                    dict(preview)
                    for preview in self.teaser.get("previews", ())
                    if isinstance(preview, Mapping)
                ],
            },
        }


@dataclass(frozen=True)
class DeflectionReportSectionDefinition:
    """Registry metadata for one paid deflection report section."""

    id: str
    title: str
    priority: int
    surfaces: tuple[str, ...]
    default_limit: int | None
    required_data: tuple[str, ...]
    snapshot_safe_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeflectionReportSection:
    """Structured report section plus interim Markdown rendering lines."""

    id: str
    title: str
    priority: int
    surfaces: tuple[str, ...]
    default_limit: int | None
    required_data: tuple[str, ...]
    snapshot_safe_fields: tuple[str, ...]
    data: dict[str, Any]
    markdown_lines: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority,
            "surfaces": list(self.surfaces),
            "default_limit": self.default_limit,
            "required_data": list(self.required_data),
            "snapshot_safe_fields": list(self.snapshot_safe_fields),
            "data": _json_ready(self.data),
        }


@dataclass(frozen=True)
class DeflectionStructuredReport:
    """Surface-neutral paid report model for future web/PDF/export renderers."""

    schema_version: str
    title: str
    summary: dict[str, Any]
    sections: tuple[DeflectionReportSection, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "summary": dict(self.summary),
            "sections": [section.as_dict() for section in self.sections],
        }


@dataclass(frozen=True)
class DeflectionReportArtifact:
    """Rendered deflection report plus compact proof metadata."""

    markdown: str
    summary: dict[str, Any]
    faq_result: TicketFAQMarkdownResult
    report_model: DeflectionStructuredReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "summary": dict(self.summary),
            "faq_result": self.faq_result.as_dict(),
            "report_model": self.report_model.as_dict(),
            "evidence_export": build_deflection_evidence_export(self),
        }

    def snapshot(self, *, top_n: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N) -> DeflectionSnapshot:
        return build_deflection_snapshot(self, top_n=top_n)


def scrub_deflection_report_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON payload with supported customer PII classes redacted."""

    scrubber = _DeflectionPayloadScrubber(payload)
    return scrubber.scrub_payload(payload)


_DEFLECTION_REPORT_SECTION_DEFINITIONS = (
    DeflectionReportSectionDefinition(
        id="support_tax",
        title="Support Tax Confirmation",
        priority=10,
        surfaces=("web", "pdf", "email_summary", "markdown"),
        default_limit=None,
        required_data=(
            "repeat_ticket_count",
            "non_repeat_ticket_count",
            "generated_question_count",
            "assisted_contact_cost",
            "estimated_support_cost",
            "source_date_window",
            "drafted_answer_count",
            "no_proven_answer_count",
            "ticket_source_count",
        ),
        snapshot_safe_fields=(
            "repeat_ticket_count",
            "non_repeat_ticket_count",
            "generated_question_count",
            "drafted_answer_count",
            "no_proven_answer_count",
            "ticket_source_count",
            "source_date_window",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="source_file",
        title="Source file",
        priority=15,
        surfaces=("web", "pdf", "markdown"),
        default_limit=None,
        required_data=("source_label",),
    ),
    DeflectionReportSectionDefinition(
        id="seo_targets",
        title="Your Help-Desk SEO Targeting List",
        priority=20,
        surfaces=("web", "pdf", "markdown"),
        default_limit=DEFAULT_DEFLECTION_SEO_TARGET_LIMIT,
        required_data=(
            "phrases",
            "total_phrase_count",
            "displayed_phrase_count",
            "omitted_phrase_count",
            "limit",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="ranked_questions",
        title="Ranked Question Opportunities",
        priority=30,
        surfaces=("web", "pdf", "markdown"),
        default_limit=None,
        required_data=("rows",),
        snapshot_safe_fields=(
            "rows.rank",
            "rows.question",
            "rows.ticket_count",
            "rows.weighted_frequency",
            "rows.customer_wording",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="priority_fix_queue",
        title="Priority Fix Queue",
        priority=35,
        surfaces=("web", "pdf", "email_summary"),
        default_limit=3,
        required_data=(
            "items",
            "status_counts",
            "result_page_limit",
            "pdf_limit",
            "backlog_limit",
            "support_cost_basis",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="top_unresolved_repeats",
        title="Top Unresolved Repeats",
        priority=36,
        surfaces=("web", "pdf"),
        default_limit=3,
        required_data=(
            "items",
            "top_item_count",
            "support_cost_basis",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="drafted_resolutions",
        title="Drafted Resolutions Ready to Publish",
        priority=37,
        surfaces=("web", "pdf", "email_summary"),
        default_limit=3,
        required_data=(
            "items",
            "top_item_count",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="already_covered_still_recurring",
        title="Already Covered but Still Recurring",
        priority=38,
        surfaces=("web", "pdf"),
        default_limit=3,
        required_data=(
            "items",
            "top_item_count",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="backlog_table",
        title="Backlog Table",
        priority=39,
        surfaces=("web", "pdf", "export"),
        default_limit=25,
        required_data=(
            "items",
            "total_item_count",
            "default_limit",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="outcome_diagnostics",
        title="Resolution Outcome Diagnostics",
        priority=40,
        surfaces=("web", "pdf", "markdown"),
        default_limit=None,
        required_data=(
            "outcome_diagnostic_ticket_count",
            "outcome_risk_ticket_count",
            "reopened_ticket_count",
            "negative_csat_ticket_count",
            "rows",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="question_details",
        title="Question Details and Evidence",
        priority=50,
        surfaces=("web", "pdf", "markdown"),
        default_limit=None,
        required_data=("rows",),
        snapshot_safe_fields=(
            "rows.rank",
            "rows.question",
            "rows.answer_evidence_status",
            "rows.resolution_evidence_scope",
            "rows.weighted_frequency",
            "rows.source_count",
        ),
    ),
    DeflectionReportSectionDefinition(
        id="complete_evidence",
        title="Complete Evidence",
        priority=90,
        surfaces=("export",),
        default_limit=None,
        required_data=(
            "question_count",
            "evidence_row_count",
            "source_id_count",
            "surfaces",
        ),
    ),
)

DEFLECTION_REPORT_SECTION_REGISTRY: Mapping[str, DeflectionReportSectionDefinition] = (
    MappingProxyType({
        definition.id: definition
        for definition in _DEFLECTION_REPORT_SECTION_DEFINITIONS
    })
)
_SNAPSHOT_REQUIRED_SECTION_IDS = frozenset({
    "support_tax",
    "ranked_questions",
    "question_details",
})
_SNAPSHOT_REQUIRED_SUPPORT_TAX_FIELDS = frozenset({
    "repeat_ticket_count",
    "non_repeat_ticket_count",
    "generated_question_count",
    "drafted_answer_count",
    "no_proven_answer_count",
})
_SNAPSHOT_REQUIRED_RANKED_ROW_FIELDS = frozenset({
    "rank",
    "question",
    "ticket_count",
    "weighted_frequency",
    "customer_wording",
})
_SNAPSHOT_REQUIRED_DETAIL_PROJECTION_FIELDS = frozenset({
    "rank",
    "question",
    "answer_evidence_status",
    "resolution_evidence_scope",
    "weighted_frequency",
    "source_count",
})
_SNAPSHOT_REQUIRED_DETAIL_TEASER_FIELDS = frozenset({
    *_SNAPSHOT_REQUIRED_DETAIL_PROJECTION_FIELDS,
    "answer",
    "steps",
})


def deflection_report_model_contract_shape() -> dict[str, Any]:
    """Return the structural contract bound to DEFLECTION_REPORT_SCHEMA_VERSION."""

    return {
        "schema_version": DEFLECTION_REPORT_SCHEMA_VERSION,
        "model_fields": list(DEFLECTION_REPORT_MODEL_FIELDS),
        "section_fields": list(DEFLECTION_REPORT_SECTION_FIELDS),
        "sections": [
            {
                "id": definition.id,
                "title": definition.title,
                "priority": definition.priority,
                "surfaces": list(definition.surfaces),
                "default_limit": definition.default_limit,
                "required_data": list(definition.required_data),
                "snapshot_safe_fields": list(definition.snapshot_safe_fields),
            }
            for definition in _DEFLECTION_REPORT_SECTION_DEFINITIONS
        ],
    }


DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS: Mapping[str, Mapping[str, int]] = (
    MappingProxyType({
        "result_page": MappingProxyType({
            "ranked_questions": 25,
            "question_details": 10,
            "seo_targets": 20,
            "outcome_diagnostics": 25,
        }),
        "pdf": MappingProxyType({
            "ranked_questions": 25,
            "question_details": 10,
        }),
    })
)
_FULL_REPORT_QA_SURFACE_COUNT_KEYS: Mapping[str, tuple[str, ...]] = (
    MappingProxyType({
        "email": (
            "repeat_ticket_count",
            "generated_question_count",
            "drafted_answer_count",
            "no_proven_answer_count",
            "ticket_source_count",
            "estimated_support_cost",
        ),
        "result_page": (
            "repeat_ticket_count",
            "generated_question_count",
            "ranked_question_count",
            "drafted_answer_count",
            "no_proven_answer_count",
            "ticket_source_count",
            "estimated_support_cost",
            "evidence_row_count",
            "source_id_count",
        ),
        "pdf": (
            "repeat_ticket_count",
            "generated_question_count",
            "ranked_question_count",
            "drafted_answer_count",
            "no_proven_answer_count",
            "ticket_source_count",
            "estimated_support_cost",
        ),
        "evidence_export": (
            "evidence_question_count",
            "evidence_row_count",
            "source_id_count",
            "drafted_answer_count",
            "no_proven_answer_count",
        ),
    })
)
_SCORECARD_SAFE_STRINGS = frozenset({
    DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
    DEFLECTION_REPORT_SCHEMA_VERSION,
    DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION,
    "mapping",
    "dict",
    "list",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "NoneType",
})
_SCORECARD_COUNT_KEYS = frozenset({
    "repeat_ticket_count",
    "generated_question_count",
    "ranked_question_count",
    "drafted_answer_count",
    "no_proven_answer_count",
    "ticket_source_count",
    "estimated_support_cost",
    "evidence_question_count",
    "evidence_row_count",
    "source_id_count",
    "seo_total_phrase_count",
    "seo_displayed_phrase_count",
    "seo_omitted_phrase_count",
    "outcome_diagnostic_row_count",
    "question_detail_count",
})
_SCORECARD_SURFACE_KEYS = frozenset({
    "email",
    "email_summary",
    "evidence_export",
    "export",
    "pdf",
    "result_page",
    "web",
})


class FAQDeflectionReportService:
    """Service-shaped generator for customer-facing FAQ deflection reports."""

    def __init__(self, faq_markdown: TicketFAQMarkdownService | None = None) -> None:
        self._faq_markdown = faq_markdown or TicketFAQMarkdownService()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        report_title: str | None = None,
        faq_title: str | None = None,
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
        **kwargs: Any,
    ) -> DeflectionReportArtifact:
        del kwargs, max_items
        faq_result = await self._faq_markdown.generate(
            scope=scope,
            target_mode=target_mode,
            source_material=source_material,
            title=faq_title,
            max_items=_UNCAPPED_REPORT_MAX_ITEMS,
            max_evidence_per_item=max_evidence_per_item,
            source_types=source_types,
            max_text_chars=max_text_chars,
            window_days=window_days,
            as_of_date=as_of_date,
            support_contact=support_contact,
            intent_rules=intent_rules,
            documentation_terms=documentation_terms,
            representative_taxonomy_terms=representative_taxonomy_terms,
            vocabulary_gap_rules=vocabulary_gap_rules,
        )
        return build_deflection_report_artifact(
            faq_result,
            title=report_title or "Support Ticket Deflection Report",
        )


def build_deflection_report_artifact(
    faq_result: TicketFAQMarkdownResult,
    *,
    title: str = "Support Ticket Deflection Report",
    source_label: str | None = None,
) -> DeflectionReportArtifact:
    """Render a customer-facing report from a generated FAQ result."""

    summary = deflection_report_summary(faq_result)
    report_model = build_deflection_report_model(
        faq_result,
        title=title,
        source_label=source_label,
        summary=summary,
    )
    markdown = render_deflection_report_model(report_model)
    return DeflectionReportArtifact(
        markdown=markdown,
        summary=summary,
        faq_result=faq_result,
        report_model=report_model,
    )


def build_deflection_evidence_export(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the uncapped structured evidence export for a paid report."""

    summary = dict(_artifact_summary(artifact))
    items = _artifact_items(artifact)
    questions = tuple(
        _evidence_export_question(rank, item)
        for rank, item in enumerate(items, start=1)
    )
    evidence_rows = tuple(
        row
        for rank, item in enumerate(items, start=1)
        for row in _evidence_export_rows(rank, item)
    )
    source_ids = sorted({
        source_id
        for item in items
        for source_id in _texts(item.get("source_ids"))
    })
    return {
        "schema_version": DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
        "summary": {
            "question_count": len(questions),
            "evidence_row_count": len(evidence_rows),
            "source_id_count": len(source_ids),
            "drafted_answer_count": _int(summary.get("drafted_answer_count")),
            "no_proven_answer_count": _int(summary.get("no_proven_answer_count")),
        },
        "report_summary": summary,
        "questions": [dict(question) for question in questions],
        "evidence_rows": [dict(row) for row in evidence_rows],
    }


def build_deflection_full_report_qa_scorecard(
    report_model: DeflectionStructuredReport | Mapping[str, Any],
    *,
    evidence_export: Mapping[str, Any] | None = None,
    surface_observations: Mapping[str, Mapping[str, Any]] | None = None,
    surface_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Return a sanitized model-anchored QA scorecard for full-report surfaces."""

    model = (
        report_model.as_dict()
        if isinstance(report_model, DeflectionStructuredReport)
        else report_model
    )
    if not isinstance(model, Mapping):
        model = {}
    sections = _scorecard_sections(model)
    counts = _scorecard_counts(sections)
    assertions: list[dict[str, Any]] = []

    def add(assertion_id: str, ok: bool, *, expected: Any, actual: Any) -> None:
        assertions.append({
            "id": assertion_id,
            "ok": bool(ok),
            "expected": _scorecard_safe_value(expected),
            "actual": _scorecard_safe_value(actual),
        })

    add(
        "model.schema_version",
        _text(model.get("schema_version")) == DEFLECTION_REPORT_SCHEMA_VERSION,
        expected=DEFLECTION_REPORT_SCHEMA_VERSION,
        actual=_text(model.get("schema_version")),
    )
    for section_id in (
        "support_tax",
        "seo_targets",
        "ranked_questions",
        "question_details",
        "complete_evidence",
    ):
        add(
            f"model.section.{section_id}.present",
            section_id in sections,
            expected=True,
            actual=section_id in sections,
        )
    for section_id, section in sections.items():
        definition = DEFLECTION_REPORT_SECTION_REGISTRY.get(section_id)
        if definition is None:
            continue
        data = section.get("data") if isinstance(section.get("data"), Mapping) else {}
        for key in definition.required_data:
            add(
                f"model.section.{section_id}.data.{key}.present",
                key in data,
                expected=True,
                actual=key in data,
            )

    _add_evidence_export_assertions(add, counts, evidence_export)
    _add_surface_observation_assertions(
        add,
        counts,
        surface_observations or {},
        surface_caps or DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS,
    )
    return {
        "schema_version": DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION,
        "ok": all(assertion["ok"] for assertion in assertions),
        "counts": _json_ready(counts),
        "assertions": assertions,
    }


def build_deflection_full_report_qa_deterministic_harness(
    report_model: DeflectionStructuredReport | Mapping[str, Any],
    *,
    evidence_export: Mapping[str, Any] | None = None,
    surface_observations: Mapping[str, Mapping[str, Any]] | None = None,
    surface_caps: Mapping[str, Mapping[str, int]] | None = None,
    required_surfaces: Sequence[str] = DEFLECTION_FULL_REPORT_QA_REQUIRED_SURFACES,
) -> dict[str, Any]:
    """Return a deterministic all-surface QA scorecard for CI harnesses."""

    model = (
        report_model.as_dict()
        if isinstance(report_model, DeflectionStructuredReport)
        else report_model
    )
    if not isinstance(model, Mapping):
        model = {}
    caps = surface_caps or DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS
    counts = _scorecard_counts(_scorecard_sections(model))
    observations = (
        surface_observations
        if surface_observations is not None
        else _full_report_qa_surface_observations(counts, caps)
    )
    observations = observations if isinstance(observations, Mapping) else {}

    scorecard = build_deflection_full_report_qa_scorecard(
        model,
        evidence_export=evidence_export,
        surface_observations=observations,
        surface_caps=caps,
    )
    assertions = [dict(assertion) for assertion in scorecard["assertions"]]
    observed_surface_keys = {str(surface) for surface in observations}
    for index, surface in enumerate(required_surfaces, start=1):
        surface_text = str(surface)
        surface_id = _scorecard_id_segment(
            surface_text,
            allowed=_SCORECARD_SURFACE_KEYS,
            fallback=f"required_surface_{index}",
        )
        assertions.append({
            "id": f"harness.surface.{surface_id}.present",
            "ok": surface_text in observed_surface_keys,
            "expected": True,
            "actual": surface_text in observed_surface_keys,
        })
        observation = observations.get(surface_text)
        if isinstance(observation, Mapping):
            _add_full_report_qa_required_metric_assertions(
                assertions,
                surface=surface_text,
                surface_id=surface_id,
                observation=observation,
                surface_caps=caps,
            )

    result = dict(scorecard)
    result["ok"] = all(assertion["ok"] for assertion in assertions)
    result["assertions"] = assertions
    result["surfaces"] = {
        "required": [
            _scorecard_id_segment(
                surface,
                allowed=_SCORECARD_SURFACE_KEYS,
                fallback=f"required_surface_{index}",
            )
            for index, surface in enumerate(required_surfaces, start=1)
        ],
        "observed": [
            _scorecard_id_segment(
                surface,
                allowed=_SCORECARD_SURFACE_KEYS,
                fallback=f"surface_{index}",
            )
            for index, surface in enumerate(sorted(observed_surface_keys), start=1)
        ],
    }
    return result


def _add_full_report_qa_required_metric_assertions(
    assertions: list[dict[str, Any]],
    *,
    surface: str,
    surface_id: str,
    observation: Mapping[str, Any],
    surface_caps: Mapping[str, Mapping[str, int]],
) -> None:
    observed_counts = (
        observation.get("counts")
        if isinstance(observation.get("counts"), Mapping)
        else {}
    )
    for key_index, key in enumerate(
        _FULL_REPORT_QA_SURFACE_COUNT_KEYS.get(surface, ()),
        start=1,
    ):
        key_id = _scorecard_id_segment(
            key,
            allowed=_SCORECARD_COUNT_KEYS,
            fallback=f"count_{key_index}",
        )
        assertions.append({
            "id": f"harness.surface.{surface_id}.count.{key_id}.present",
            "ok": key in observed_counts,
            "expected": True,
            "actual": key in observed_counts,
        })

    displayed_rows = (
        observation.get("displayed_rows")
        if isinstance(observation.get("displayed_rows"), Mapping)
        else {}
    )
    caps = surface_caps.get(surface, {})
    caps = caps if isinstance(caps, Mapping) else {}
    for section_index, section_id in enumerate(sorted(caps), start=1):
        section_safe_id = _scorecard_id_segment(
            section_id,
            allowed=frozenset(DEFLECTION_REPORT_SECTION_REGISTRY),
            fallback=f"section_{section_index}",
        )
        assertions.append({
            "id": (
                f"harness.surface.{surface_id}.displayed_rows."
                f"{section_safe_id}.present"
            ),
            "ok": section_id in displayed_rows,
            "expected": True,
            "actual": section_id in displayed_rows,
        })


def _full_report_qa_surface_observations(
    counts: Mapping[str, Any],
    surface_caps: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for surface, keys in _FULL_REPORT_QA_SURFACE_COUNT_KEYS.items():
        observation: dict[str, Any] = {
            "counts": {key: counts.get(key) for key in keys},
        }
        caps = surface_caps.get(surface)
        if isinstance(caps, Mapping) and caps:
            observation["displayed_rows"] = {
                section_id: min(
                    _scorecard_section_total(section_id, counts),
                    max(0, _int(cap)),
                )
                for section_id, cap in sorted(caps.items())
            }
        observations[surface] = observation
    return observations


def _scorecard_sections(model: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_sections = model.get("sections")
    if not isinstance(raw_sections, Sequence) or isinstance(
        raw_sections,
        (str, bytes, bytearray),
    ):
        return {}
    sections: dict[str, Mapping[str, Any]] = {}
    for raw in raw_sections:
        if not isinstance(raw, Mapping):
            continue
        section_id = _text(raw.get("id"))
        if section_id:
            sections[section_id] = raw
    return sections


def _scorecard_section_data(
    sections: Mapping[str, Mapping[str, Any]],
    section_id: str,
) -> Mapping[str, Any]:
    section = sections.get(section_id)
    if not isinstance(section, Mapping):
        return {}
    data = section.get("data")
    return data if isinstance(data, Mapping) else {}


def _scorecard_rows_count(data: Mapping[str, Any]) -> int:
    rows = data.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return 0
    return len([row for row in rows if isinstance(row, Mapping)])


def _scorecard_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scorecard_counts(sections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    support_tax = _scorecard_section_data(sections, "support_tax")
    seo_targets = _scorecard_section_data(sections, "seo_targets")
    ranked_questions = _scorecard_section_data(sections, "ranked_questions")
    diagnostics = _scorecard_section_data(sections, "outcome_diagnostics")
    question_details = _scorecard_section_data(sections, "question_details")
    complete_evidence = _scorecard_section_data(sections, "complete_evidence")
    return {
        "repeat_ticket_count": _int(support_tax.get("repeat_ticket_count")),
        "generated_question_count": _int(
            support_tax.get("generated_question_count")
        ),
        "ranked_question_count": _scorecard_rows_count(ranked_questions),
        "drafted_answer_count": _int(support_tax.get("drafted_answer_count")),
        "no_proven_answer_count": _int(support_tax.get("no_proven_answer_count")),
        "ticket_source_count": _int(support_tax.get("ticket_source_count")),
        "estimated_support_cost": _scorecard_float(
            support_tax.get("estimated_support_cost")
        ),
        "evidence_question_count": _int(complete_evidence.get("question_count")),
        "evidence_row_count": _int(complete_evidence.get("evidence_row_count")),
        "source_id_count": _int(complete_evidence.get("source_id_count")),
        "seo_total_phrase_count": _int(seo_targets.get("total_phrase_count")),
        "seo_displayed_phrase_count": _int(
            seo_targets.get("displayed_phrase_count")
        ),
        "seo_omitted_phrase_count": _int(seo_targets.get("omitted_phrase_count")),
        "outcome_diagnostic_row_count": _scorecard_rows_count(diagnostics),
        "question_detail_count": _scorecard_rows_count(question_details),
    }


def _add_evidence_export_assertions(
    add: Any,
    counts: Mapping[str, Any],
    evidence_export: Mapping[str, Any] | None,
) -> None:
    export = evidence_export if isinstance(evidence_export, Mapping) else {}
    summary = export.get("summary") if isinstance(export.get("summary"), Mapping) else {}
    questions = export.get("questions")
    rows = export.get("evidence_rows")
    questions_valid = isinstance(questions, Sequence) and not isinstance(
        questions,
        (str, bytes, bytearray),
    )
    rows_valid = isinstance(rows, Sequence) and not isinstance(
        rows,
        (str, bytes, bytearray),
    )
    question_count = len(questions) if questions_valid else 0
    row_count = len(rows) if rows_valid else 0

    add(
        "evidence_export.present",
        bool(export),
        expected=True,
        actual=bool(export),
    )
    add(
        "evidence_export.schema_version",
        _text(export.get("schema_version"))
        == DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
        expected=DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
        actual=_text(export.get("schema_version")),
    )
    for key, expected_key in (
        ("question_count", "evidence_question_count"),
        ("evidence_row_count", "evidence_row_count"),
        ("source_id_count", "source_id_count"),
        ("drafted_answer_count", "drafted_answer_count"),
        ("no_proven_answer_count", "no_proven_answer_count"),
    ):
        key_present = key in summary
        add(
            f"evidence_export.summary.{key}",
            key_present and _int(summary.get(key)) == _int(counts.get(expected_key)),
            expected=_int(counts.get(expected_key)),
            actual=_int(summary.get(key)) if key_present else None,
        )
    add(
        "evidence_export.questions.present",
        questions_valid,
        expected=True,
        actual=questions_valid,
    )
    add(
        "evidence_export.questions.length",
        questions_valid and question_count == _int(counts.get("evidence_question_count")),
        expected=_int(counts.get("evidence_question_count")),
        actual=question_count,
    )
    add(
        "evidence_export.evidence_rows.present",
        rows_valid,
        expected=True,
        actual=rows_valid,
    )
    add(
        "evidence_export.evidence_rows.length",
        rows_valid and row_count == _int(counts.get("evidence_row_count")),
        expected=_int(counts.get("evidence_row_count")),
        actual=row_count,
    )


def _scorecard_section_total(section_id: str, counts: Mapping[str, Any]) -> int:
    return {
        "seo_targets": _int(counts.get("seo_total_phrase_count")),
        "ranked_questions": _int(counts.get("ranked_question_count")),
        "outcome_diagnostics": _int(counts.get("outcome_diagnostic_row_count")),
        "question_details": _int(counts.get("question_detail_count")),
        "complete_evidence": _int(counts.get("evidence_row_count")),
    }.get(section_id, 0)


def _add_surface_observation_assertions(
    add: Any,
    counts: Mapping[str, Any],
    surface_observations: Mapping[str, Mapping[str, Any]],
    surface_caps: Mapping[str, Mapping[str, int]],
) -> None:
    for surface_index, (surface, observation) in enumerate(
        sorted(surface_observations.items()),
        start=1,
    ):
        surface_id = _scorecard_id_segment(
            surface,
            allowed=_SCORECARD_SURFACE_KEYS,
            fallback=f"surface_{surface_index}",
        )
        if not isinstance(observation, Mapping):
            add(
                f"surface.{surface_id}.observation_shape",
                False,
                expected="mapping",
                actual=type(observation).__name__,
            )
            continue
        observed_counts = (
            observation.get("counts")
            if isinstance(observation.get("counts"), Mapping)
            else {}
        )
        displayed_rows = (
            observation.get("displayed_rows")
            if isinstance(observation.get("displayed_rows"), Mapping)
            else {}
        )
        add(
            f"surface.{surface_id}.observation_has_data",
            bool(observed_counts) or bool(displayed_rows),
            expected="counts or displayed_rows",
            actual={
                "counts": bool(observed_counts),
                "displayed_rows": bool(displayed_rows),
            },
        )
        for key_index, (key, actual) in enumerate(
            sorted(observed_counts.items()),
            start=1,
        ):
            key_text = str(key)
            key_id = _scorecard_id_segment(
                key_text,
                allowed=_SCORECARD_COUNT_KEYS,
                fallback=f"count_{key_index}",
            )
            expected = counts.get(key_text)
            add(
                f"surface.{surface_id}.count.{key_id}",
                expected is not None and _scorecard_observed_number_equals(
                    actual,
                    expected,
                ),
                expected=expected,
                actual=actual,
            )

        caps = surface_caps.get(surface, {})
        caps = caps if isinstance(caps, Mapping) else {}
        for section_index, (section_id, actual) in enumerate(
            sorted(displayed_rows.items()),
            start=1,
        ):
            section_text = str(section_id)
            section_safe_id = _scorecard_id_segment(
                section_text,
                allowed=frozenset(DEFLECTION_REPORT_SECTION_REGISTRY),
                fallback=f"section_{section_index}",
            )
            total = _scorecard_section_total(section_text, counts)
            cap = (
                max(0, _int(caps[section_text]))
                if section_text in caps
                else total
            )
            expected = min(total, cap)
            observed = _scorecard_observed_int(actual)
            add(
                f"surface.{surface_id}.displayed_rows.{section_safe_id}",
                observed is not None and observed == expected,
                expected=expected,
                actual=observed if observed is not None else actual,
            )


def _scorecard_id_segment(value: Any, *, allowed: frozenset[str], fallback: str) -> str:
    text = _text(value)
    if text in allowed:
        return text
    return fallback


def _scorecard_observed_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _scorecard_observed_number_equals(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    if not isinstance(actual, (int, float)):
        return False
    return actual == expected


def _scorecard_safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return value if value in _SCORECARD_SAFE_STRINGS else "<redacted-string>"
    if isinstance(value, Mapping):
        return {str(key): _scorecard_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_scorecard_safe_value(item) for item in value]
    return _json_ready(value)


def build_deflection_snapshot(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
    *,
    top_n: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
    teaser_preview_count: int = DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT,
) -> DeflectionSnapshot:
    """Project a report into the free snapshot shape.

    The snapshot intentionally omits Markdown, answers, steps, evidence,
    source IDs, term mappings, and nested FAQ item payloads.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if teaser_preview_count < 0:
        raise ValueError("teaser_preview_count must be non-negative")
    report_model = _artifact_report_model(artifact)
    if report_model is not None and _report_model_supports_snapshot_projection(
        report_model
    ):
        return _build_deflection_snapshot_from_report_model(
            report_model,
            top_n=top_n,
            teaser_preview_count=teaser_preview_count,
        )
    summary = _artifact_summary(artifact)
    items = _artifact_items(artifact)
    snapshot_summary: dict[str, Any] = {
        "generated": _int(summary.get("generated")),
        "drafted_answer_count": _int(summary.get("drafted_answer_count")),
        "no_proven_answer_count": _int(summary.get("no_proven_answer_count")),
        "support_ticket_resolution_evidence_present": _resolution_evidence_present(
            summary
        ),
        "support_ticket_resolution_evidence_count": _resolution_evidence_count(
            summary
        ),
        "repeat_ticket_count": _repeat_ticket_count(items),
        "non_repeat_ticket_count": _non_repeat_ticket_count(summary, items),
    }
    source_date_window = _complete_source_date_window(summary, items)
    if source_date_window:
        snapshot_summary.update(source_date_window)
    top_questions: list[dict[str, Any]] = []
    for rank, item in enumerate(items[:top_n], start=1):
        question = _text(item.get("question"))
        top_questions.append({
            "rank": rank,
            "question": question,
            "ticket_count": _ticket_count(item),
            "weighted_frequency": _int(
                item.get("weighted_frequency") or item.get("frequency")
            ),
            "customer_wording": _snapshot_customer_wording(item, question),
        })
    teaser = _snapshot_teaser(items, preview_count=teaser_preview_count)
    teaser_full_rank = _teaser_full_answer_rank(teaser)
    locked_questions = tuple(
        {
            "rank": rank,
            "ticket_count": _ticket_count(item),
        }
        for rank, item in enumerate(items[top_n:], start=top_n + 1)
        if rank != teaser_full_rank
    )
    return DeflectionSnapshot(
        summary=snapshot_summary,
        top_questions=tuple(top_questions),
        locked_questions=locked_questions,
        teaser=teaser,
    )


def _build_deflection_snapshot_from_report_model(
    report_model: Mapping[str, Any],
    *,
    top_n: int,
    teaser_preview_count: int,
) -> DeflectionSnapshot:
    sections = _snapshot_report_model_projection(report_model)
    support_tax = sections.get("support_tax", {})
    ranked_rows = _snapshot_rows(sections.get("ranked_questions"))
    detail_rows = _report_model_section_rows(report_model, "question_details")
    snapshot_summary: dict[str, Any] = {
        "generated": _int(support_tax.get("generated_question_count")),
        "drafted_answer_count": _int(support_tax.get("drafted_answer_count")),
        "no_proven_answer_count": _int(support_tax.get("no_proven_answer_count")),
        "support_ticket_resolution_evidence_present": _int(
            support_tax.get("drafted_answer_count")
        ) > 0,
        "support_ticket_resolution_evidence_count": _int(
            support_tax.get("drafted_answer_count")
        ),
        "repeat_ticket_count": _int(support_tax.get("repeat_ticket_count")),
        "non_repeat_ticket_count": _int(support_tax.get("non_repeat_ticket_count")),
    }
    source_date_window = support_tax.get("source_date_window")
    if isinstance(source_date_window, Mapping):
        snapshot_summary.update(_summary_source_date_window(source_date_window))

    top_questions = [
        {
            "rank": _int(row.get("rank")) or index,
            "question": _text(row.get("question")),
            "ticket_count": _int(row.get("ticket_count")),
            "weighted_frequency": _int(row.get("weighted_frequency")),
            "customer_wording": _text(row.get("customer_wording")),
        }
        for index, row in enumerate(ranked_rows[:top_n], start=1)
    ]
    teaser = _snapshot_teaser(detail_rows, preview_count=teaser_preview_count)
    teaser_full_rank = _teaser_full_answer_rank(teaser)
    locked_questions = tuple(
        {
            "rank": _int(row.get("rank")) or rank,
            "ticket_count": _int(row.get("ticket_count")),
        }
        for rank, row in enumerate(ranked_rows[top_n:], start=top_n + 1)
        if (_int(row.get("rank")) or rank) != teaser_full_rank
    )
    return DeflectionSnapshot(
        summary=snapshot_summary,
        top_questions=tuple(top_questions),
        locked_questions=locked_questions,
        teaser=teaser,
    )


def _artifact_report_model(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if isinstance(artifact, DeflectionReportArtifact):
        return artifact.report_model.as_dict()
    value = artifact.get("report_model")
    return value if isinstance(value, Mapping) else None


def _report_model_supports_snapshot_projection(report_model: Mapping[str, Any]) -> bool:
    sections = _snapshot_report_model_projection(report_model)
    if not _SNAPSHOT_REQUIRED_SECTION_IDS <= set(sections):
        return False
    support_tax = sections.get("support_tax", {})
    if not _mapping_has_fields(support_tax, _SNAPSHOT_REQUIRED_SUPPORT_TAX_FIELDS):
        return False
    ranked_rows = _snapshot_rows(sections.get("ranked_questions"))
    detail_projection_rows = _snapshot_rows(sections.get("question_details"))
    detail_rows = _report_model_section_rows(report_model, "question_details")
    generated_count = _int(support_tax.get("generated_question_count"))
    if generated_count > 0 and not ranked_rows:
        return False
    if len(detail_rows) < len(ranked_rows):
        return False
    return (
        _rows_have_fields(ranked_rows, _SNAPSHOT_REQUIRED_RANKED_ROW_FIELDS)
        and _rows_have_fields(
            detail_projection_rows,
            _SNAPSHOT_REQUIRED_DETAIL_PROJECTION_FIELDS,
        )
        and _rows_have_fields(
            detail_rows,
            _SNAPSHOT_REQUIRED_DETAIL_TEASER_FIELDS,
        )
    )


def _snapshot_report_model_projection(
    report_model: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_sections = report_model.get("sections")
    if not isinstance(raw_sections, Sequence) or isinstance(
        raw_sections,
        (str, bytes, bytearray),
    ):
        return {}
    sections: dict[str, dict[str, Any]] = {}
    for raw in raw_sections:
        if not isinstance(raw, Mapping):
            continue
        section_id = _text(raw.get("id"))
        definition = DEFLECTION_REPORT_SECTION_REGISTRY.get(section_id)
        if definition is None or not definition.snapshot_safe_fields:
            continue
        data = raw.get("data")
        if not isinstance(data, Mapping):
            continue
        projected = _snapshot_safe_section_data(
            data,
            definition.snapshot_safe_fields,
        )
        if projected:
            sections[section_id] = projected
    return sections


def _report_model_section_rows(
    report_model: Mapping[str, Any],
    section_id: str,
) -> tuple[Mapping[str, Any], ...]:
    data = _report_model_section_data(report_model, section_id)
    return _snapshot_rows(data)


def _report_model_section_data(
    report_model: Mapping[str, Any],
    section_id: str,
) -> Mapping[str, Any] | None:
    raw_sections = report_model.get("sections")
    if not isinstance(raw_sections, Sequence) or isinstance(
        raw_sections,
        (str, bytes, bytearray),
    ):
        return None
    for raw in raw_sections:
        if not isinstance(raw, Mapping):
            continue
        if _text(raw.get("id")) != section_id:
            continue
        data = raw.get("data")
        return data if isinstance(data, Mapping) else None
    return None


def _mapping_has_fields(data: Mapping[str, Any], fields: frozenset[str]) -> bool:
    return fields <= set(data)


def _rows_have_fields(
    rows: Sequence[Mapping[str, Any]],
    fields: frozenset[str],
) -> bool:
    return all(fields <= set(row) for row in rows)


def _snapshot_safe_section_data(
    data: Mapping[str, Any],
    safe_fields: Sequence[str],
) -> dict[str, Any]:
    direct_fields = tuple(field for field in safe_fields if "." not in field)
    row_fields = tuple(
        field.removeprefix("rows.")
        for field in safe_fields
        if field.startswith("rows.")
    )
    out: dict[str, Any] = {
        field: _json_ready(data[field])
        for field in direct_fields
        if field in data
    }
    if row_fields:
        rows = data.get("rows")
        if isinstance(rows, Sequence) and not isinstance(
            rows,
            (str, bytes, bytearray),
        ):
            out["rows"] = [
                {
                    field: _json_ready(row[field])
                    for field in row_fields
                    if isinstance(row, Mapping) and field in row
                }
                for row in rows
                if isinstance(row, Mapping)
            ]
    return out


def _snapshot_rows(section_data: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(section_data, Mapping):
        return ()
    rows = section_data.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def deflection_snapshot_content_opportunities(
    snapshot: Mapping[str, Any],
    *,
    limit: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
) -> tuple[dict[str, Any], ...]:
    """Return structured, unpaid-safe opportunities from a free snapshot."""

    top_questions = snapshot.get("top_questions")
    if not isinstance(top_questions, Sequence) or isinstance(
        top_questions,
        (str, bytes, bytearray),
    ):
        return ()

    out: list[dict[str, Any]] = []
    for index, raw in enumerate(top_questions, start=1):
        if len(out) >= _positive_limit(limit):
            break
        if not isinstance(raw, Mapping):
            continue
        question = _text(raw.get("question"))
        if not question:
            continue
        frequency = _int(raw.get("weighted_frequency") or raw.get("frequency"))
        rank = _int(raw.get("rank")) or index
        out.append(
            {
                "rank": rank,
                "question": question,
                "ticket_count": _int(raw.get("ticket_count")),
                "weighted_frequency": frequency,
                "customer_wording": _text(raw.get("customer_wording")),
                "opportunity_score": _int(raw.get("opportunity_score")) or frequency,
                "coverage_status": _text(raw.get("coverage_status")) or "locked_snapshot",
                "recommended_content_action": "Create or improve an FAQ entry for this repeated customer question.",
                "unlock_hint": "Unlock the full report for detailed source-backed guidance.",
            }
        )
    return tuple(out)


def deflection_report_summary(faq_result: TicketFAQMarkdownResult) -> dict[str, Any]:
    """Return compact counts used by the report and CLI summary JSON."""

    items = tuple(_item(item) for item in faq_result.items)
    proven = tuple(
        item for item in items
        if _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS
    )
    needs_review = tuple(
        item for item in items
        if _text(item.get("answer_evidence_status")) != _RESOLUTION_EVIDENCE_STATUS
    )
    summary = {
        "generated": len(items),
        "source_count": int(faq_result.source_count),
        "ticket_source_count": int(faq_result.ticket_source_count),
        "non_repeat_ticket_count": int(faq_result.non_repeat_ticket_count),
        "non_repeat_question_count": int(faq_result.non_repeat_question_count),
        "drafted_answer_count": len(proven),
        "no_proven_answer_count": len(needs_review),
        "support_ticket_resolution_evidence_present": len(proven) > 0,
        "support_ticket_resolution_evidence_count": len(proven),
        "output_checks": dict(faq_result.output_checks),
        "top_question": _text(items[0].get("question")) if items else "",
        "top_opportunity_score": _int(items[0].get("opportunity_score")) if items else 0,
    }
    summary.update(_outcome_diagnostics_summary(items))
    source_date_window = _complete_source_date_window({}, items)
    if source_date_window:
        summary.update(source_date_window)
    return summary


def render_deflection_report(
    faq_result: TicketFAQMarkdownResult,
    *,
    title: str = "Support Ticket Deflection Report",
    source_label: str | None = None,
    summary: Mapping[str, Any] | None = None,
) -> str:
    """Render the customer-facing Markdown report."""

    return render_deflection_report_model(
        build_deflection_report_model(
            faq_result,
            title=title,
            source_label=source_label,
            summary=summary,
        )
    )


def build_deflection_report_model(
    faq_result: TicketFAQMarkdownResult,
    *,
    title: str = "Support Ticket Deflection Report",
    source_label: str | None = None,
    summary: Mapping[str, Any] | None = None,
) -> DeflectionStructuredReport:
    """Build the surface-neutral paid deflection report model."""

    resolved_summary = dict(summary or deflection_report_summary(faq_result))
    items = tuple(_item(item) for item in faq_result.items)
    sections: list[DeflectionReportSection] = [
        _report_section(
            section_id="support_tax",
            data=_support_tax_data(resolved_summary, items),
            markdown_lines=_support_tax_section(resolved_summary, items),
        )
    ]
    if source_label:
        sections.append(
            _report_section(
                section_id="source_file",
                data={"source_label": _text(source_label)},
                markdown_lines=["**Source file:**", "", _md(source_label), ""],
            )
        )
    sections.extend([
        _report_section(
            section_id="seo_targets",
            data=_seo_targets_data(items, limit=DEFAULT_DEFLECTION_SEO_TARGET_LIMIT),
            markdown_lines=_help_desk_seo_targeting_section(items),
        ),
        _report_section(
            section_id="ranked_questions",
            data={"rows": _ranked_question_rows(items)},
            markdown_lines=_ranked_opportunity_section(items),
        ),
        _report_section(
            section_id="priority_fix_queue",
            data=_priority_fix_queue_data(items),
            markdown_lines=[],
        ),
        _report_section(
            section_id="top_unresolved_repeats",
            data=_top_unresolved_repeats_data(items),
            markdown_lines=[],
        ),
        _report_section(
            section_id="drafted_resolutions",
            data=_drafted_resolutions_data(items),
            markdown_lines=[],
        ),
        _report_section(
            section_id="already_covered_still_recurring",
            data=_already_covered_still_recurring_data(items),
            markdown_lines=[],
        ),
        _report_section(
            section_id="backlog_table",
            data=_backlog_table_data(items),
            markdown_lines=[],
        ),
    ])
    diagnostics_lines = _outcome_diagnostics_section(items, resolved_summary)
    if diagnostics_lines:
        sections.append(
            _report_section(
                section_id="outcome_diagnostics",
                data=_outcome_diagnostics_data(items, resolved_summary),
                markdown_lines=diagnostics_lines,
            )
        )
    sections.extend([
        _report_section(
            section_id="question_details",
            data={"rows": _question_detail_rows(items)},
            markdown_lines=_question_detail_section(items),
        ),
        _report_section(
            section_id="complete_evidence",
            data=_complete_evidence_section_data(items),
            markdown_lines=[],
        ),
    ])
    return DeflectionStructuredReport(
        schema_version=DEFLECTION_REPORT_SCHEMA_VERSION,
        title=_text(title) or "Support Ticket Deflection Report",
        summary=resolved_summary,
        sections=tuple(sorted(sections, key=lambda section: section.priority)),
    )


def render_deflection_report_model(model: DeflectionStructuredReport) -> str:
    """Render the current Markdown surface from structured report sections."""

    lines: list[str] = [f"# {_md(model.title)}", ""]
    for section in sorted(model.sections, key=lambda item: item.priority):
        if "markdown" not in section.surfaces:
            continue
        lines.extend(section.markdown_lines)
    return "\n".join(lines).rstrip() + "\n"


def _report_section(
    *,
    section_id: str,
    data: Mapping[str, Any],
    markdown_lines: Sequence[str],
) -> DeflectionReportSection:
    try:
        definition = DEFLECTION_REPORT_SECTION_REGISTRY[section_id]
    except KeyError as exc:
        raise ValueError(f"unknown deflection report section: {section_id}") from exc

    missing_data = [
        key
        for key in definition.required_data
        if key not in data
    ]
    if missing_data:
        missing = ", ".join(missing_data)
        raise ValueError(
            f"section {section_id!r} missing required data keys: {missing}"
        )

    return DeflectionReportSection(
        id=definition.id,
        title=definition.title,
        priority=definition.priority,
        surfaces=definition.surfaces,
        default_limit=definition.default_limit,
        required_data=definition.required_data,
        snapshot_safe_fields=definition.snapshot_safe_fields,
        data=dict(data),
        markdown_lines=tuple(markdown_lines),
    )


def _support_tax_data(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repeat_ticket_count = _repeat_ticket_count(items)
    non_repeat_ticket_count = _non_repeat_ticket_count(summary, items)
    source_window = _complete_source_date_window(summary, items)
    data: dict[str, Any] = {
        "repeat_ticket_count": repeat_ticket_count,
        "non_repeat_ticket_count": non_repeat_ticket_count,
        "generated_question_count": _int(summary.get("generated")),
        "assisted_contact_cost": _ASSISTED_CONTACT_COST,
        "estimated_support_cost": _support_cost(repeat_ticket_count),
        "source_date_window": source_window or None,
        "drafted_answer_count": _int(summary.get("drafted_answer_count")),
        "no_proven_answer_count": _int(summary.get("no_proven_answer_count")),
        "ticket_source_count": _int(summary.get("ticket_source_count")),
    }
    if source_window:
        days = _int(source_window.get("source_window_days"))
        data["annualized_support_cost"] = _support_cost(
            repeat_ticket_count * 365 / days
        )
    else:
        data["annualized_run_rate_support_cost"] = _support_cost(
            repeat_ticket_count * 12
        )
    return data


def _seo_targets_data(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    phrases = _customer_phrase_list(items)
    display_limit = max(1, int(limit))
    displayed_phrases = phrases[:display_limit]
    return {
        "phrases": list(displayed_phrases),
        "total_phrase_count": len(phrases),
        "displayed_phrase_count": len(displayed_phrases),
        "omitted_phrase_count": max(len(phrases) - len(displayed_phrases), 0),
        "limit": display_limit,
    }


def _ranked_question_rows(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "question": _text(item.get("question")),
            "ticket_count": _ticket_count(item),
            "weighted_frequency": _int(
                item.get("weighted_frequency") or item.get("frequency")
            ),
            "customer_wording": _snapshot_customer_wording(
                item,
                _text(item.get("question")),
            ),
            "estimated_support_cost": _support_cost(_ticket_count(item)),
            "opportunity_score": _int(item.get("opportunity_score")),
            "answer_status": _status_label(item),
            "source_proof": _source_count_label(item),
        }
        for index, item in enumerate(items, start=1)
    ]


def _outcome_diagnostics_data(
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        diagnostics = _item_outcome_diagnostics(item)
        if not diagnostics:
            continue
        rows.append({
            "question": _text(item.get("question")),
            "status_mix": _status_mix_label(diagnostics),
            "reopened_ticket_count": _int(diagnostics.get("reopened_ticket_count")),
            "negative_csat_ticket_count": _int(
                diagnostics.get("negative_csat_ticket_count")
            ),
            "guidance": _outcome_guidance(diagnostics),
        })
    return {
        "outcome_diagnostic_ticket_count": _int(
            summary.get("outcome_diagnostic_ticket_count")
        ),
        "outcome_risk_ticket_count": _int(summary.get("outcome_risk_ticket_count")),
        "reopened_ticket_count": _int(summary.get("reopened_ticket_count")),
        "negative_csat_ticket_count": _int(summary.get("negative_csat_ticket_count")),
        "rows": rows,
    }


def _priority_fix_queue_data(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_items = _action_items(items)
    model_limit = max(_ACTION_RESULT_PAGE_LIMIT, _ACTION_PDF_LIMIT)
    return {
        "items": action_items[:model_limit],
        "status_counts": _action_status_counts(action_items),
        "result_page_limit": _ACTION_RESULT_PAGE_LIMIT,
        "pdf_limit": _ACTION_PDF_LIMIT,
        "backlog_limit": _ACTION_BACKLOG_LIMIT,
        "support_cost_basis": _action_support_cost_basis(),
    }


def _top_unresolved_repeats_data(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unresolved = [
        item for item in _action_items(items)
        if item["status"] in {"Needs answer", "Needs review"}
        and _int(item.get("ticket_count")) >= 2
    ]
    top_items = unresolved[:_ACTION_RESULT_PAGE_LIMIT]
    return {
        "items": top_items,
        "top_item_count": len(top_items),
        "support_cost_basis": _action_support_cost_basis(),
    }


def _drafted_resolutions_data(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    drafted = [
        item for item in _action_items(items)
        if item["status"] == "Draft ready"
    ][:_ACTION_RESULT_PAGE_LIMIT]
    return {
        "items": drafted,
        "top_item_count": len(drafted),
    }


def _already_covered_still_recurring_data(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recurring = [
        item for item in _action_items(items)
        if item["status"] == "Already covered but still recurring"
    ][:_ACTION_RESULT_PAGE_LIMIT]
    return {
        "items": recurring,
        "top_item_count": len(recurring),
    }


def _backlog_table_data(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_items = _action_items(items)
    return {
        "items": action_items[:_ACTION_BACKLOG_LIMIT],
        "total_item_count": len(action_items),
        "default_limit": _ACTION_BACKLOG_LIMIT,
    }


def _action_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        _action_item(rank, item)
        for rank, item in enumerate(items, start=1)
    ]
    return sorted(
        rows,
        key=lambda row: (
            -_int(row["priority_score"]),
            -float(row["estimated_support_cost"]),
            -_int(row["opportunity_score"]),
            _int(row["rank"]),
        ),
    )


def _action_item(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    status = _action_status(item)
    ticket_count = _ticket_count(item)
    return {
        "rank": rank,
        "question": _text(item.get("question")),
        "status": status,
        "owner_lane": _action_owner_lane(item),
        "fix_type": _action_fix_type(status),
        "csat_signal": _action_csat_signal(item),
        "confidence": _action_confidence(item),
        "priority_score": _action_priority_score(item, status),
        "priority_drivers": _action_priority_drivers(item, status),
        "recommended_title": _action_recommended_title(item),
        "recommended_action": _action_recommended_action(status),
        "representative_phrasing": _action_representative_phrasing(item),
        "ticket_count": ticket_count,
        "estimated_support_cost": _support_cost(ticket_count),
        "support_cost_formula": "ticket_count * assisted_contact_cost",
        "support_cost_source": "default_assisted_contact_benchmark",
        "opportunity_score": _int(item.get("opportunity_score")),
        "top_evidence": _action_top_evidence(item),
    }


def _action_status(item: Mapping[str, Any]) -> str:
    ticket_count = _ticket_count(item)
    has_question = bool(_text(item.get("question")))
    if ticket_count <= 0 or not has_question:
        return "Low confidence"
    if ticket_count < 2 or _source_count(item) < 2:
        return "Low confidence"
    has_resolution_evidence = (
        _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS
    )
    has_answer = bool(_text(item.get("answer")))
    if not has_resolution_evidence:
        return "Needs answer"
    diagnostics = _item_outcome_diagnostics(item)
    has_pain_after_answer = (
        _int(diagnostics.get("reopened_ticket_count")) > 0
        or _int(diagnostics.get("negative_csat_ticket_count")) > 0
    )
    if has_answer and has_pain_after_answer:
        return "Already covered but still recurring"
    if has_answer:
        return "Draft ready"
    return "Needs review"


def _action_owner_lane(item: Mapping[str, Any]) -> str:
    topic = _text(item.get("topic"))
    return topic or "Unknown"


def _action_fix_type(status: str) -> str:
    return {
        "Draft ready": "publish_help_center_answer",
        "Needs answer": "create_missing_answer",
        "Needs review": "review_resolution_evidence",
        "Low confidence": "review_cluster_quality",
        "Already covered but still recurring": "improve_discoverability_or_answer_quality",
    }.get(status, "Unknown")


def _action_csat_signal(item: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _item_outcome_diagnostics(item)
    present_count = _int(diagnostics.get("csat_present_count"))
    negative_count = _int(diagnostics.get("negative_csat_ticket_count"))
    if present_count <= 0:
        return {
            "status": "insufficient_data",
            "csat_present_count": 0,
            "negative_csat_ticket_count": 0,
            "numeric_average": None,
        }
    average: float | None = None
    if present_count >= 3 and diagnostics.get("csat_score_average") is not None:
        average = float(diagnostics.get("csat_score_average"))
    return {
        "status": "present" if average is not None else "sparse",
        "csat_present_count": present_count,
        "negative_csat_ticket_count": negative_count,
        "numeric_average": average,
    }


def _action_confidence(item: Mapping[str, Any]) -> str:
    ticket_count = _ticket_count(item)
    source_count = _source_count(item)
    if ticket_count < 2 or source_count < 2:
        return "low"
    if ticket_count >= 5 and source_count >= 3:
        return "high"
    return "medium"


def _action_priority_score(item: Mapping[str, Any], status: str) -> int:
    diagnostics = _item_outcome_diagnostics(item)
    ticket_count = _ticket_count(item)
    score = int(round(_support_cost(ticket_count)))
    score += _int(item.get("opportunity_score"))
    score += int(_ACTION_STATUS_PRIORITY_WEIGHTS.get(status, 0))
    score += _int(diagnostics.get("negative_csat_ticket_count")) * 30
    score += _int(diagnostics.get("reopened_ticket_count")) * 20
    if ticket_count >= 5:
        score += 15
    elif ticket_count >= 2:
        score += 5
    average = diagnostics.get("csat_score_average")
    present_count = _int(diagnostics.get("csat_present_count"))
    if present_count >= 3 and average is not None:
        try:
            parsed_average = float(average)
        except (TypeError, ValueError):
            parsed_average = 0.0
        if parsed_average < 3.0:
            score += int(round((3.0 - parsed_average) * 10))
    if _action_confidence(item) == "low":
        score -= 25
    return max(0, score)


def _action_priority_drivers(item: Mapping[str, Any], status: str) -> list[str]:
    diagnostics = _item_outcome_diagnostics(item)
    drivers: list[str] = []
    ticket_count = _ticket_count(item)
    if ticket_count >= 5:
        drivers.append("high_repeat_volume")
    elif ticket_count >= 2:
        drivers.append("repeat_volume")
    drivers.append({
        "Draft ready": "draft_ready",
        "Needs answer": "missing_answer",
        "Needs review": "review_resolution_evidence",
        "Low confidence": "low_confidence",
        "Already covered but still recurring": "already_covered_recurring",
    }.get(status, "review_before_action"))
    if _int(diagnostics.get("negative_csat_ticket_count")) > 0:
        drivers.append("negative_csat")
    if _int(diagnostics.get("reopened_ticket_count")) > 0:
        drivers.append("reopened_after_answer")
    if _action_confidence(item) == "low" and "low_confidence" not in drivers:
        drivers.append("low_confidence")
    if _support_cost(ticket_count) > 0:
        drivers.append("benchmark_cost")
    return drivers[:5]


def _action_recommended_title(item: Mapping[str, Any]) -> str:
    return _snapshot_customer_wording(item, _text(item.get("question"))) or _text(
        item.get("question")
    )


def _action_recommended_action(status: str) -> str:
    return {
        "Draft ready": "Publish or adapt the drafted resolution into help-center or macro copy.",
        "Needs answer": "Write and approve the missing answer for this repeated customer question.",
        "Needs review": "Review the available evidence before publishing customer-facing guidance.",
        "Low confidence": "Review the cluster before assigning a content fix.",
        "Already covered but still recurring": "Improve discoverability, search wording, macro use, or answer quality.",
    }.get(status, "Review before taking action.")


def _action_representative_phrasing(item: Mapping[str, Any]) -> list[str]:
    phrases: list[str] = []
    for value in (
        item.get("customer_wording"),
        item.get("question"),
        item.get("topic"),
    ):
        text = _text(value)
        if text and text not in phrases:
            phrases.append(text)
    return phrases[:3]


def _action_top_evidence(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_ids = _texts(item.get("source_ids"))
    quotes = _texts(item.get("evidence_quotes"))
    rows: list[dict[str, Any]] = []
    used_quotes: set[int] = set()
    for source_id in source_ids[:_SOURCE_EXAMPLE_LIMIT]:
        quote_index, quote = _evidence_quote_for_source(source_id, quotes, used_quotes)
        if quote_index is not None:
            used_quotes.add(quote_index)
        rows.append({
            "source_id": source_id,
            "evidence_quote": quote,
        })
    if rows:
        return rows
    return [
        {"source_id": "", "evidence_quote": quote}
        for quote in quotes[:_SOURCE_EXAMPLE_LIMIT]
    ]


def _action_support_cost_basis() -> dict[str, Any]:
    return {
        "status": "benchmark_only",
        "assisted_contact_cost": _ASSISTED_CONTACT_COST,
        "formula": "ticket_count * assisted_contact_cost",
        "source": "default_assisted_contact_benchmark",
    }


def _action_status_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = _text(item.get("status")) or "Unknown"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _question_detail_rows(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        rows.append({
            "rank": index,
            "question": _text(item.get("question")),
            "customer_wording": _text(item.get("customer_wording")),
            "topic": _text(item.get("topic")),
            "ticket_count": _ticket_count(item),
            "weighted_frequency": _int(
                item.get("weighted_frequency") or item.get("frequency")
            ),
            "source_count": _source_count(item),
            "estimated_support_cost": _support_cost(_ticket_count(item)),
            "answer_status": _status_label(item),
            "answer_evidence_status": _text(item.get("answer_evidence_status")),
            "resolution_evidence_scope": _text(item.get("resolution_evidence_scope")),
            "answer_linkage": _evidence_answer_linkage(item),
            "answer": _text(item.get("answer")),
            "steps": _texts(item.get("steps")),
            "term_mappings": [
                dict(mapping)
                for mapping in item.get("term_mappings") or ()
                if isinstance(mapping, Mapping)
            ],
            "source_ids": _texts(item.get("source_ids")),
            "evidence_quotes": _texts(item.get("evidence_quotes")),
            "outcome_diagnostics": dict(_item_outcome_diagnostics(item)),
        })
    return rows


def _complete_evidence_section_data(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_row_count = sum(
        len(_evidence_export_rows(rank, item))
        for rank, item in enumerate(items, start=1)
    )
    source_ids = {
        source_id
        for item in items
        for source_id in _texts(item.get("source_ids"))
    }
    return {
        "question_count": len(items),
        "evidence_row_count": evidence_row_count,
        "source_id_count": len(source_ids),
        "surfaces": ["export"],
    }


def _support_tax_section(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> list[str]:
    repeat_ticket_count = _repeat_ticket_count(items)
    non_repeat_ticket_count = _non_repeat_ticket_count(summary, items)
    batch_cost = _support_cost(repeat_ticket_count)
    source_window = _complete_source_date_window(summary, items)
    lines = [
        "## Support Tax Confirmation",
        "",
        (
            f"This report found {_count(repeat_ticket_count)} question-level "
            f"repeat tickets across {_count(_int(summary.get('generated')))} "
            "ranked questions. At the "
            f"Gartner {_ASSISTED_CONTACT_COST_LABEL} assisted-contact benchmark, "
            "that repeated-question work sizes to about "
            f"{_format_money(batch_cost)} of assisted-contact handling."
        ),
    ]
    if non_repeat_ticket_count:
        lines.extend([
            "",
            (
                f"{_count(non_repeat_ticket_count)} tickets asked a question that "
                "appeared only once in this upload; they are excluded from the "
                "repeat counts and cost sizing above."
            ),
        ])
    if source_window:
        annualized = _support_cost(
            repeat_ticket_count * 365 / _int(source_window.get("source_window_days"))
        )
        lines.extend([
            "",
            (
                "The source window is "
                f"{_source_window_label(source_window)}. At the same measured daily "
                f"pace, that is about {_format_money(annualized)} over 12 months."
            ),
        ])
    else:
        monthly_pace = _support_cost(repeat_ticket_count * 12)
        lines.extend([
            "",
            (
                "This report did not receive a complete source-date window for every "
                "contributing ticket, so this report does not infer a monthly or "
                "annual reporting period. If this uploaded batch is monthly pace, "
                f"the 12-month run-rate would be about {_format_money(monthly_pace)}."
            ),
        ])
    lines.extend([
        "",
        (
            "Estimate only. This is not a savings guarantee; adjust the "
            f"{_ASSISTED_CONTACT_COST_LABEL} benchmark to your own loaded support "
            "cost."
        ),
        "",
        (
            "The full unlocked report below gives you every ranked question, "
            "the estimated support cost by question, publishable help-center copy "
            "where your uploaded resolutions prove the answer, the no-proven-answer "
            "roadmap, and a complete evidence export for audit/detail review."
        ),
        "",
        f"- Publishable answers drafted from proven resolutions: {_count(_int(summary.get('drafted_answer_count')))}",
        f"- Questions still needing an approved resolution: {_count(_int(summary.get('no_proven_answer_count')))}",
        f"- Ticket sources represented: {_count(_int(summary.get('ticket_source_count')))}",
        "",
    ])
    return lines


def _help_desk_seo_targeting_section(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_DEFLECTION_SEO_TARGET_LIMIT,
) -> list[str]:
    phrases = _customer_phrase_list(items)
    display_limit = max(1, int(limit))
    displayed_phrases = phrases[:display_limit]
    lines = [
        "## Your Help-Desk SEO Targeting List",
        "",
        (
            "Use these source-backed phrases as help-center headings, "
            "internal-search synonyms, and FAQ wording. These were mined from "
            "the tickets you uploaded; this report does not claim keyword "
            "volume, search rank, or traffic."
        ),
        "",
    ]
    if not phrases:
        return [*lines, "No customer phrase targets were generated for this run.", ""]
    lines.extend(
        f"{index}. {_md(phrase)}"
        for index, phrase in enumerate(displayed_phrases, start=1)
    )
    omitted_count = len(phrases) - len(displayed_phrases)
    if omitted_count > 0:
        lines.extend([
            "",
            (
                f"SEO phrase index capped at {_count(display_limit)} entries for "
                f"readability; {_count(omitted_count)} additional source-backed "
                "phrases remain represented in the question detail blocks below."
            ),
        ])
    lines.append("")
    return lines


def _ranked_opportunity_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "## Ranked Question Opportunities",
        "",
        "| Rank | Customer question | Tickets | Estimated support cost | Opportunity | Answer status | Source proof |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    if not items:
        return [
            *lines,
            "| - | No ranked FAQ opportunities were generated. | 0 | $0 | 0 | - | - |",
            "",
        ]
    for index, item in enumerate(items, start=1):
        ticket_count = _ticket_count(item)
        lines.append(
            "| "
            f"{index} | {_cell(item.get('question'))} | "
            f"{ticket_count} | "
            f"{_format_money(_support_cost(ticket_count))} | "
            f"{_int(item.get('opportunity_score'))} | "
            f"{_status_label(item)} | "
            f"{_cell(_source_count_label(item))} |"
        )
    return [*lines, ""]


def _outcome_diagnostics_section(
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[str]:
    diagnostic_items = [
        item for item in items
        if _item_outcome_diagnostics(item)
    ]
    if not diagnostic_items:
        return []
    lines = [
        "## Resolution Outcome Diagnostics",
        "",
        (
            "These status and CSAT signals flag answers that may need review. "
            "They do not prove a publishable answer; only uploaded resolution "
            "evidence can do that."
        ),
        "",
        (
            f"- Tickets with outcome diagnostics: "
            f"{_count(_int(summary.get('outcome_diagnostic_ticket_count')))}"
        ),
        (
            f"- Tickets with reopened or negative-CSAT risk: "
            f"{_count(_int(summary.get('outcome_risk_ticket_count')))}"
        ),
        (
            f"- Reopened tickets: "
            f"{_count(_int(summary.get('reopened_ticket_count')))}"
        ),
        (
            f"- Negative CSAT tickets: "
            f"{_count(_int(summary.get('negative_csat_ticket_count')))}"
        ),
        "",
        "| Customer question | Status mix | Reopened | Negative CSAT | Guidance |",
        "|---|---|---:|---:|---|",
    ]
    for item in diagnostic_items:
        diagnostics = _item_outcome_diagnostics(item)
        lines.append(
            "| "
            f"{_cell(item.get('question'))} | "
            f"{_cell(_status_mix_label(diagnostics))} | "
            f"{_int(diagnostics.get('reopened_ticket_count'))} | "
            f"{_int(diagnostics.get('negative_csat_ticket_count'))} | "
            f"{_cell(_outcome_guidance(diagnostics))} |"
        )
    return [*lines, ""]


def _question_detail_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "## Question Details and Evidence",
        "",
        (
            "Each ranked question appears once below with its answer status, "
            "publishable copy or review guidance, vocabulary gaps, and complete "
            "source evidence."
        ),
        "",
        (
            "Questions without uploaded resolution evidence stay in review: "
            "outcome/status signals can prioritize them, but only resolution "
            "evidence can make an answer publishable."
        ),
        "",
    ]
    if not items:
        return [
            *lines,
            "No ranked FAQ opportunities were generated.",
            "",
        ]
    for index, item in enumerate(items, start=1):
        lines.extend([
            f"### {index}. {_md(item.get('question'))}",
            "",
            *_customer_wording_detail(item),
            f"**Answer status:** {_md(_status_label(item))}",
            "",
            (
                f"**Ticket/support-cost context:** {_count(_ticket_count(item))} "
                f"tickets, estimated at {_format_money(_support_cost(_ticket_count(item)))} "
                "of assisted-contact handling."
            ),
            "",
        ])
        if _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS:
            lines.extend(_publishable_answer_detail(item))
        else:
            lines.extend(_no_proven_answer_detail(item))
        lines.extend(_vocabulary_gap_detail(item))
        lines.extend(_complete_evidence_detail(item))
    return lines


def _customer_wording_detail(item: Mapping[str, Any]) -> list[str]:
    question = _text(item.get("question"))
    customer_wording = _text(item.get("customer_wording"))
    if not customer_wording or customer_wording == question:
        return []
    return [
        f"**Customer wording:** {_md(customer_wording)}",
        "",
    ]


def _publishable_answer_detail(item: Mapping[str, Any]) -> list[str]:
    steps = _texts(item.get("steps"))
    answer = _text(item.get("answer")) or (
        "This draft answer is backed by uploaded resolution evidence."
    )
    lines = [
        "**Publishable answer draft:**",
        "",
        _md(answer),
        "",
    ]
    if steps:
        lines.extend([
            "**Draft answer steps:**",
            "",
            *[
                f"{step_index}. {_md(step)}"
                for step_index, step in enumerate(steps, start=1)
            ],
            "",
        ])
    else:
        lines.extend([
            "**Draft answer steps:**",
            "",
            "No step list was generated for this answer.",
            "",
        ])
    lines.extend([
        f"**Evidence backing:** {_source_backing_summary(item, resolved=True)}",
        "",
    ])
    return lines


def _no_proven_answer_detail(item: Mapping[str, Any]) -> list[str]:
    return [
        "**No proven answer yet:**",
        "",
        "No uploaded resolution evidence was present for this question.",
        "",
        f"**Ticket backing:** {_source_backing_summary(item, resolved=False)}",
        "",
    ]


def _vocabulary_gap_detail(item: Mapping[str, Any]) -> list[str]:
    mappings = [
        mapping
        for mapping in item.get("term_mappings") or ()
        if isinstance(mapping, Mapping)
    ]
    if not mappings:
        return []
    lines = ["**Vocabulary gaps:**", ""]
    for mapping in mappings:
        lines.append(
            "- "
            f"{_md(mapping.get('customer_term'))} -> "
            f"{_md(mapping.get('documentation_term'))}: "
            f"{_md(mapping.get('suggestion'))} "
            f"({_count(_int(mapping.get('source_id_count')))} sources)"
        )
    lines.append("")
    return lines


def _complete_evidence_detail(item: Mapping[str, Any]) -> list[str]:
    quotes = _texts(item.get("evidence_quotes"))
    source_ids = _texts(item.get("source_ids"))
    lines = ["**Complete evidence:**", ""]
    if source_ids:
        lines.extend([
            f"**Source IDs (full list):** {_md(', '.join(source_ids))}",
            "",
        ])
    elif _source_count(item) > 0:
        lines.extend([
            (
                "**Source IDs (full list):** Not available in this export; "
                f"source count is {_source_count(item)}."
            ),
            "",
        ])
    if not quotes:
        lines.extend(["No source excerpts were rendered for this item.", ""])
        return lines
    for quote in quotes:
        lines.append(f"- {_md(quote)}")
    lines.append("")
    return lines


def _evidence_export_question(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    source_ids = _texts(item.get("source_ids"))
    return {
        "question_id": _evidence_question_id(rank),
        "rank": rank,
        "question": _text(item.get("question")),
        "customer_wording": _text(item.get("customer_wording")),
        "topic": _text(item.get("topic")),
        "ticket_count": _ticket_count(item),
        "weighted_frequency": _int(item.get("weighted_frequency") or item.get("frequency")),
        "opportunity_score": _int(item.get("opportunity_score")),
        "answer_evidence_status": _text(item.get("answer_evidence_status")),
        "resolution_evidence_scope": _text(item.get("resolution_evidence_scope")),
        "answer_linkage": _evidence_answer_linkage(item),
        "answer": _text(item.get("answer")),
        "steps": _texts(item.get("steps")),
        "source_ids": source_ids,
        "evidence_quote_count": len(_texts(item.get("evidence_quotes"))),
        "term_mappings": [
            dict(mapping)
            for mapping in item.get("term_mappings") or ()
            if isinstance(mapping, Mapping)
        ],
        "outcome_diagnostics": dict(_item_outcome_diagnostics(item)),
    }


def _evidence_export_rows(rank: int, item: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    question_id = _evidence_question_id(rank)
    source_ids = _texts(item.get("source_ids"))
    quotes = _texts(item.get("evidence_quotes"))
    rows: list[dict[str, Any]] = []
    used_quotes: set[int] = set()

    for source_index, source_id in enumerate(source_ids, start=1):
        quote_index, quote = _evidence_quote_for_source(source_id, quotes, used_quotes)
        if quote_index is not None:
            used_quotes.add(quote_index)
        rows.append(
            _evidence_row(
                question_id=question_id,
                rank=rank,
                item=item,
                row_index=source_index,
                source_id=source_id,
                quote=quote,
                source_field="evidence_quote" if quote else "source_id",
            )
        )

    for quote_index, quote in enumerate(quotes):
        if quote_index in used_quotes:
            continue
        rows.append(
            _evidence_row(
                question_id=question_id,
                rank=rank,
                item=item,
                row_index=len(rows) + 1,
                source_id="",
                quote=quote,
                source_field="evidence_quote",
            )
        )
    return tuple(rows)


def _evidence_row(
    *,
    question_id: str,
    rank: int,
    item: Mapping[str, Any],
    row_index: int,
    source_id: str,
    quote: str,
    source_field: str,
) -> dict[str, Any]:
    return {
        "row_id": f"{question_id}-e{row_index:03d}",
        "question_id": question_id,
        "rank": rank,
        "question": _text(item.get("question")),
        "source_id": source_id,
        "source_field": source_field,
        "evidence_quote": quote,
        "answer_evidence_status": _text(item.get("answer_evidence_status")),
        "resolution_evidence_scope": _text(item.get("resolution_evidence_scope")),
        "answer_linkage": _evidence_answer_linkage(item),
    }


def _evidence_question_id(rank: int) -> str:
    return f"q{rank:03d}"


def _evidence_answer_linkage(item: Mapping[str, Any]) -> str:
    if _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS:
        return "publishable_answer"
    return "needs_review"


def _evidence_quote_for_source(
    source_id: str,
    quotes: Sequence[str],
    used_quotes: set[int],
) -> tuple[int | None, str]:
    normalized_source = source_id.casefold()
    for index, quote in enumerate(quotes):
        if index in used_quotes:
            continue
        if normalized_source and normalized_source in quote.casefold():
            return index, quote
    return None, ""


def _status_label(item: Mapping[str, Any]) -> str:
    status = _text(item.get("answer_evidence_status"))
    if status == _RESOLUTION_EVIDENCE_STATUS:
        return "drafted from resolution evidence"
    if status == _DRAFT_NEEDS_REVIEW_STATUS:
        return "no proven answer yet"
    return status or "unknown"


def _item(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _item_outcome_diagnostics(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("outcome_diagnostics")
    return value if isinstance(value, Mapping) else {}


def _outcome_diagnostics_summary(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_summary: dict[str, int] = {}
    diagnostic_count = 0
    risk_count = 0
    reopened_count = 0
    negative_csat_count = 0
    csat_present_count = 0
    for item in items:
        diagnostics = _item_outcome_diagnostics(item)
        if not diagnostics:
            continue
        diagnostic_count += _int(diagnostics.get("diagnostic_ticket_count"))
        risk_count += _int(diagnostics.get("outcome_risk_ticket_count"))
        reopened_count += _int(diagnostics.get("reopened_ticket_count"))
        negative_csat_count += _int(diagnostics.get("negative_csat_ticket_count"))
        csat_present_count += _int(diagnostics.get("csat_present_count"))
        raw_summary = diagnostics.get("ticket_status_summary")
        if isinstance(raw_summary, Mapping):
            for status, count in raw_summary.items():
                key = _text(status)
                if key:
                    status_summary[key] = status_summary.get(key, 0) + _int(count)
    if diagnostic_count == 0:
        return {}
    return {
        "outcome_diagnostics_present": True,
        "outcome_diagnostic_ticket_count": diagnostic_count,
        "outcome_risk_ticket_count": risk_count,
        "reopened_ticket_count": reopened_count,
        "negative_csat_ticket_count": negative_csat_count,
        "csat_present_count": csat_present_count,
        "ticket_status_summary": dict(sorted(status_summary.items())),
    }


def _status_mix_label(diagnostics: Mapping[str, Any]) -> str:
    raw_summary = diagnostics.get("ticket_status_summary")
    if not isinstance(raw_summary, Mapping) or not raw_summary:
        return "No status supplied"
    parts = [
        f"{_text(status)}: {_count(_int(count))}"
        for status, count in sorted(raw_summary.items())
        if _text(status) and _int(count) > 0
    ]
    return ", ".join(parts) or "No status supplied"


def _outcome_guidance(diagnostics: Mapping[str, Any]) -> str:
    if _int(diagnostics.get("reopened_ticket_count")):
        return "Review the answer before publishing because at least one ticket reopened."
    if _int(diagnostics.get("negative_csat_ticket_count")):
        return "Review the answer before publishing because CSAT was negative."
    return "Outcome context only; use resolution evidence to decide publishability."


def _artifact_summary(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(artifact, DeflectionReportArtifact):
        return artifact.summary
    value = artifact.get("summary")
    return value if isinstance(value, Mapping) else {}


def _artifact_items(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(artifact, DeflectionReportArtifact):
        return tuple(_item(item) for item in artifact.faq_result.items)
    faq_result = artifact.get("faq_result")
    if not isinstance(faq_result, Mapping):
        return ()
    items = faq_result.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return ()
    return tuple(_item(item) for item in items if isinstance(item, Mapping))


def _resolution_evidence_count(summary: Mapping[str, Any]) -> int:
    if "support_ticket_resolution_evidence_count" in summary:
        return _int(summary.get("support_ticket_resolution_evidence_count"))
    return _int(summary.get("drafted_answer_count"))


def _resolution_evidence_present(summary: Mapping[str, Any]) -> bool:
    explicit = summary.get("support_ticket_resolution_evidence_present")
    if isinstance(explicit, bool):
        return explicit
    return _resolution_evidence_count(summary) > 0


def _snapshot_customer_wording(item: Mapping[str, Any], question: str) -> str:
    explicit = _text(item.get("customer_wording"))
    if explicit:
        return explicit
    if _text(item.get("question_source")) == "customer_wording":
        return question
    return ""


def _snapshot_teaser(
    items: Sequence[Mapping[str, Any]],
    *,
    preview_count: int,
) -> dict[str, Any]:
    eligible = tuple(
        (rank, item)
        for rank, item in enumerate(items, start=1)
        if _is_teaser_eligible(item)
    )
    if not eligible:
        return {"full_answer": None, "previews": []}
    full_rank, full_item = _select_full_teaser_item(eligible)
    previews = [
        _teaser_preview(rank, item)
        for rank, item in eligible
        if rank != full_rank
    ][:preview_count]
    return {
        "full_answer": _teaser_full_answer(full_rank, full_item),
        "previews": previews,
    }


def _teaser_full_answer_rank(teaser: Mapping[str, Any]) -> int | None:
    full_answer = teaser.get("full_answer")
    if not isinstance(full_answer, Mapping):
        return None
    rank = _int(full_answer.get("rank"))
    return rank if rank > 0 else None


def _is_teaser_eligible(item: Mapping[str, Any]) -> bool:
    return (
        _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS
        and _text(item.get("resolution_evidence_scope")) == _RESOLUTION_EVIDENCE_SCOPE_SCOPED
        and bool(_text(item.get("answer")))
    )


def _select_full_teaser_item(
    eligible: Sequence[tuple[int, Mapping[str, Any]]],
) -> tuple[int, Mapping[str, Any]]:
    return eligible[0]


def _teaser_full_answer(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "question": _text(item.get("question")),
        "answer": _text(item.get("answer")),
        "steps": _texts(item.get("steps")),
        "answer_evidence_status": _RESOLUTION_EVIDENCE_STATUS,
        "resolution_evidence_scope": _RESOLUTION_EVIDENCE_SCOPE_SCOPED,
        "weighted_frequency": _int(item.get("weighted_frequency") or item.get("frequency")),
        "source_count": _source_count(item),
    }


def _teaser_preview(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "question": _text(item.get("question")),
        "answer_evidence_status": _RESOLUTION_EVIDENCE_STATUS,
        "resolution_evidence_scope": _RESOLUTION_EVIDENCE_SCOPE_SCOPED,
        "weighted_frequency": _int(item.get("weighted_frequency") or item.get("frequency")),
        "step_count": len(_texts(item.get("steps"))),
        "source_count": _source_count(item),
        "body_withheld": True,
    }


def _source_count(item: Mapping[str, Any]) -> int:
    explicit = _int(item.get("source_count"))
    if explicit > 0:
        return explicit
    source_count = len(_texts(item.get("source_ids")))
    return source_count or _int(item.get("ticket_count"))


def _repeat_ticket_count(items: Sequence[Mapping[str, Any]]) -> int:
    # A question asked once is not a repeat: resolution-scoped items can
    # still carry a single ticket, and they stay in the report as drafted
    # answers, but they do not count as repeat work (#1481).
    return sum(
        _ticket_count(item)
        for item in items
        if _ticket_count(item) >= 2
    )


def _non_repeat_ticket_count(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> int:
    excluded = _int(summary.get("non_repeat_ticket_count"))
    singleton_items = sum(
        _ticket_count(item)
        for item in items
        if _ticket_count(item) == 1
    )
    return excluded + singleton_items


def _ticket_count(item: Mapping[str, Any]) -> int:
    ticket_count = _int(item.get("ticket_count"))
    if ticket_count > 0:
        return ticket_count
    source_count = len(_texts(item.get("source_ids")))
    return source_count if source_count > 0 else 0


def _complete_source_date_window(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary_window = _summary_source_date_window(summary)
    if summary_window:
        return summary_window
    return _items_source_date_window(items)


def _summary_source_date_window(summary: Mapping[str, Any]) -> dict[str, Any]:
    start = _iso_date_text(summary.get("source_date_start"))
    end = _iso_date_text(summary.get("source_date_end"))
    days = _int(summary.get("source_window_days"))
    if not start or not end or days < 1:
        return {}
    parsed_start = date.fromisoformat(start)
    parsed_end = date.fromisoformat(end)
    if parsed_end < parsed_start:
        return {}
    expected_days = (parsed_end - parsed_start).days + 1
    if days != expected_days:
        return {}
    return {
        "source_date_start": start,
        "source_date_end": end,
        "source_window_days": days,
    }


def _items_source_date_window(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    starts: list[date] = []
    ends: list[date] = []
    missing_source_count = 0
    saw_dated_span = False
    for item in items:
        raw_span = item.get("source_date_span")
        if not isinstance(raw_span, Mapping):
            if _ticket_count(item) > 0:
                missing_source_count += _ticket_count(item)
            continue
        start = _iso_date(raw_span.get("start"))
        end = _iso_date(raw_span.get("end"))
        if start is None or end is None or end < start:
            missing_source_count += 1
            continue
        saw_dated_span = True
        starts.append(start)
        ends.append(end)
        missing_source_count += _int(raw_span.get("missing_source_count"))
    if not saw_dated_span or missing_source_count > 0 or not starts or not ends:
        return {}
    start = min(starts)
    end = max(ends)
    return {
        "source_date_start": start.isoformat(),
        "source_date_end": end.isoformat(),
        "source_window_days": (end - start).days + 1,
    }


def _customer_phrase_list(items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    phrases: list[str] = []

    def add(value: Any) -> None:
        phrase = _text(value)
        if not phrase:
            return
        key = phrase.casefold()
        if key in seen:
            return
        seen.add(key)
        phrases.append(phrase)

    for item in items:
        add(
            item.get("customer_wording")
            or item.get("question")
        )
        for mapping in item.get("term_mappings") or ():
            if isinstance(mapping, Mapping):
                add(mapping.get("customer_term"))
    return tuple(phrases)


def _source_count_label(item: Mapping[str, Any]) -> str:
    count = _source_count(item)
    if count <= 0:
        return "No source tickets"
    if count == 1:
        return "1 source ticket"
    return f"{_count(count)} source tickets"


def _source_backing_summary(item: Mapping[str, Any], *, resolved: bool) -> str:
    count = _source_count(item)
    source_ids = _texts(item.get("source_ids"))
    examples = source_ids[:_SOURCE_EXAMPLE_LIMIT]
    if resolved:
        prefix = (
            f"Backed by {_count(count)} resolved ticket"
            f"{'' if count == 1 else 's'}"
        )
    else:
        prefix = (
            f"Seen in {_count(count)} repeated ticket"
            f"{'' if count == 1 else 's'}"
        )
    if not examples:
        return f"{prefix}. Complete source details are in this question detail block."
    more_count = max(len(source_ids) - len(examples), 0)
    if more_count > 0:
        example_text = f"{', '.join(examples)}, +{_count(more_count)} more"
    else:
        example_text = ", ".join(examples)
    return (
        f"{prefix} ({_md(example_text)}). Complete source IDs are in this question detail block."
    )


def _source_window_label(source_window: Mapping[str, Any]) -> str:
    start = _text(source_window.get("source_date_start"))
    end = _text(source_window.get("source_date_end"))
    days = _int(source_window.get("source_window_days"))
    return f"{start} to {end} ({_count(days)} days)"


def _support_cost(ticket_count: float | int) -> float:
    return max(0.0, float(ticket_count) * _ASSISTED_CONTACT_COST)


def _format_money(value: float | int) -> str:
    rounded = int(float(value) + 0.5)
    return f"${rounded:,}"


def _count(value: int) -> str:
    return f"{value:,}"


def _iso_date_text(value: Any) -> str:
    parsed = _iso_date(value)
    return parsed.isoformat() if parsed is not None else ""


def _iso_date(value: Any) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _texts(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    out: list[str] = []
    for item in values:
        text = _text(item)
        if text:
            out.append(text)
    return out


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_limit(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N
    return max(1, min(parsed, 25))


class _DeflectionPayloadScrubber:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._tokens_by_normalized_identifier: dict[str, str] = {}
        self._tokens_by_raw_identifier: dict[str, str] = {}
        self._source_links: set[str] = set()
        self._collect_identifier_tokens(payload)

    def scrub_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        scrubbed = self._scrub_value(value)
        if not isinstance(scrubbed, dict):
            return {}
        return scrubbed

    def _collect_identifier_tokens(self, value: Any, *, identifier_field: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _is_deflection_source_link_key(key):
                    self._collect_source_links(item)
                    continue
                self._collect_identifier_tokens(
                    item,
                    identifier_field=_is_deflection_identifier_key(key),
                )
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                self._collect_identifier_tokens(
                    item,
                    identifier_field=identifier_field,
                )
            return
        if identifier_field:
            self._token_for_identifier(value)

    def _collect_source_links(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                self._collect_source_links(item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                self._collect_source_links(item)
            return
        if isinstance(value, str) and _should_preserve_source_link(value):
            self._source_links.add(value)

    def _scrub_value(
        self,
        value: Any,
        *,
        identifier_field: bool = False,
        source_link_field: bool = False,
    ) -> Any:
        if source_link_field:
            return self._scrub_source_link_value(value)
        if isinstance(value, Mapping):
            return {
                self._scrub_key(key): self._scrub_value(
                    item,
                    identifier_field=_is_deflection_identifier_key(key),
                    source_link_field=_is_deflection_source_link_key(key),
                )
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                self._scrub_value(item, identifier_field=identifier_field)
                for item in value
            ]
        if identifier_field:
            return self._token_for_identifier(value)
        if isinstance(value, str):
            return self._scrub_text(value)
        return value

    def _scrub_source_link_value(self, value: Any) -> Any:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._scrub_source_link_value(item) for item in value]
        if isinstance(value, str):
            return _scrub_source_link_text(value)
        return value

    def _scrub_key(self, key: Any) -> str:
        return self._scrub_text(str(key))

    def _token_for_identifier(self, value: Any) -> str:
        text = _text(value)
        if not text:
            return ""
        existing = self._tokens_by_raw_identifier.get(text)
        if existing:
            return existing
        normalized = _normalize_deflection_identifier(text)
        token = self._tokens_by_normalized_identifier.get(normalized)
        if token is None:
            token = (
                f"{_DEFLECTION_IDENTIFIER_TOKEN_PREFIX}-"
                f"{_alpha_index(len(self._tokens_by_normalized_identifier))}"
            )
            self._tokens_by_normalized_identifier[normalized] = token
        self._tokens_by_raw_identifier[text] = token
        return token

    def _scrub_text(self, value: str) -> str:
        protected_text, source_link_placeholders = _protect_source_link_mentions(
            value,
            self._source_links,
        )
        scrubbed = _scrub_deflection_text(
            _scrub_known_identifier_text(
                protected_text,
                self._tokens_by_raw_identifier,
            )
        )
        return _restore_source_link_mentions(scrubbed, source_link_placeholders)


def _scrub_source_link_text(value: str) -> str:
    if _should_preserve_source_link(value):
        return value
    return _scrub_deflection_text(value)


def _should_preserve_source_link(value: str) -> bool:
    text = _text(value)
    if not text:
        return False
    return not _has_deflection_source_link_pii(text)


def _protect_source_link_mentions(
    value: str,
    source_links: set[str],
) -> tuple[str, dict[str, str]]:
    if not source_links:
        return value, {}
    placeholders: dict[str, str] = {}
    out_lines: list[str] = []
    for line in value.splitlines(keepends=True):
        if not _line_preserves_source_links(line):
            out_lines.append(line)
            continue
        protected = line
        for source_link in sorted(source_links, key=len, reverse=True):
            placeholder = f"DEFLECTIONSOURCELINK{_alpha_index(len(placeholders))}"
            pattern = re.compile(re.escape(source_link), re.IGNORECASE)
            if not pattern.search(protected):
                continue
            protected = pattern.sub(placeholder, protected)
            placeholders[placeholder] = source_link
        out_lines.append(protected)
    return "".join(out_lines), placeholders


def _line_preserves_source_links(value: str) -> bool:
    text = value.strip().casefold()
    return (
        text.startswith("`")
        or "source id" in text
        or "source ids" in text
        or text.startswith("**sources:**")
        or text.startswith("sources:")
    )


def _restore_source_link_mentions(
    value: str,
    placeholders: Mapping[str, str],
) -> str:
    text = value
    for placeholder, source_link in placeholders.items():
        text = text.replace(placeholder, source_link)
    return text


def _is_deflection_identifier_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_")
    return normalized in _DEFLECTION_IDENTIFIER_KEYS or normalized.endswith(
        _DEFLECTION_IDENTIFIER_KEY_SUFFIXES
    )


def _is_deflection_source_link_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_")
    return normalized in _DEFLECTION_SOURCE_LINK_KEYS or normalized.endswith(
        _DEFLECTION_SOURCE_LINK_KEY_SUFFIXES
    )


def _normalize_deflection_identifier(value: str) -> str:
    digit_groups = re.findall(r"\d{4,}", value)
    if digit_groups:
        return "|".join(digit_groups)
    return re.sub(r"\s+", " ", value.strip().casefold())


def _alpha_index(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, len(letters))
        chars.append(letters[remainder])
    return "".join(reversed(chars))


def _scrub_known_identifier_text(
    value: str,
    tokens_by_raw_identifier: Mapping[str, str],
) -> str:
    text = value
    for raw_identifier, token in sorted(
        tokens_by_raw_identifier.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if not _should_scrub_identifier_inside_text(raw_identifier):
            continue
        pattern = re.compile(
            rf"{_DEFLECTION_BOUNDARY_LEFT}{re.escape(raw_identifier)}{_DEFLECTION_BOUNDARY_RIGHT}",
            re.IGNORECASE,
        )
        text = pattern.sub(token, text)
    return text


def _should_scrub_identifier_inside_text(value: str) -> bool:
    return bool(re.search(r"\d{4,}", value) or re.search(r"[^A-Za-z0-9]", value))


def _scrub_deflection_text(value: str) -> str:
    text = _DEFLECTION_REDACTION_ARTIFACT_RE.sub("[redacted-text]", value)
    text = _DEFLECTION_SSN_RE.sub("[redacted-ssn]", text)
    text = _DEFLECTION_EMAIL_RE.sub("[redacted-email]", text)
    text = _DEFLECTION_PAYMENT_CARD_RE.sub(_redact_deflection_payment_card, text)
    text = _DEFLECTION_PHONE_RE.sub("[redacted-phone]", text)
    text = _DEFLECTION_STREET_ADDRESS_RE.sub(_redact_deflection_street_address, text)
    text = _DEFLECTION_PERSON_NAME_RE.sub(_redact_deflection_person_name, text)
    text = _DEFLECTION_STANDARD_PREFIXED_CONTEXTUAL_IDENTIFIER_RE.sub(
        _redact_deflection_standard_prefixed_contextual_identifier,
        text,
    )
    text = _DEFLECTION_CONTEXTUAL_OPAQUE_IDENTIFIER_RE.sub(
        _redact_deflection_contextual_opaque_identifier,
        text,
    )
    text = _DEFLECTION_OPAQUE_IDENTIFIER_RE.sub(
        _redact_deflection_opaque_identifier,
        text,
    )
    return _DEFLECTION_IDENTIFIER_RE.sub(_redact_deflection_labeled_identifier, text)


def _redact_deflection_payment_card(match: re.Match[str]) -> str:
    replacement = _deflection_payment_card_replacement(match.group(0))
    if replacement is None:
        return match.group(0)
    return replacement


def _deflection_payment_card_replacement(value: str) -> str | None:
    digit_positions = [index for index, char in enumerate(value) if char.isdigit()]
    digits = "".join(value[index] for index in digit_positions)
    if _looks_like_payment_card_number(digits):
        return "[redacted-payment-card]"

    for digit_count in range(min(19, len(digits) - 1), 12, -1):
        prefix_digits = digits[:digit_count]
        if not _looks_like_payment_card_number(prefix_digits):
            continue
        suffix = value[digit_positions[digit_count - 1] + 1 :]
        if suffix.startswith("-"):
            continue
        if suffix and not suffix[0].isspace():
            continue
        return f"[redacted-payment-card]{suffix}"

    return None


def _looks_like_payment_card_number(digits: str) -> bool:
    if not 13 <= len(digits) <= 19:
        return False
    if len(set(digits)) == 1:
        return False
    total = 0
    should_double = False
    for char in reversed(digits):
        value = int(char)
        if should_double:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        should_double = not should_double
    return total % 10 == 0


def _redact_deflection_street_address(match: re.Match[str]) -> str:
    if not _looks_like_deflection_street_address(match):
        return match.group(0)
    return "[redacted-address]"


def _looks_like_deflection_street_address(match: re.Match[str]) -> bool:
    body = match.group("street_body")
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z]+", body)
        if token.casefold() not in {"n", "s", "e", "w"}
    ]
    if any(token in _DEFLECTION_STREET_ADDRESS_REJECT_TOKENS for token in tokens):
        return False
    return bool(tokens or re.search(r"\d", body))


def _redact_deflection_person_name(match: re.Match[str]) -> str:
    decision = _deflection_person_name_match_decision(match)
    if not decision.redact:
        return match.group(0)
    return f"{match.group('prefix')}[redacted-name]{decision.trailing_text}"


@dataclass(frozen=True)
class _DeflectionPersonNameDecision:
    redact: bool
    trailing_text: str = ""


def _deflection_person_name_match_decision(
    match: re.Match[str],
) -> _DeflectionPersonNameDecision:
    candidate = match.group("name")
    shortened = _shortest_deflection_person_name_candidate(candidate)
    if _is_plain_role_deflection_person_name_cue(match.group("prefix")):
        if shortened is not None and _has_deflection_trailing_label_context(match):
            _, trailing = shortened
            return _DeflectionPersonNameDecision(redact=True, trailing_text=trailing)
        if _looks_like_plain_role_deflection_person_name(candidate):
            return _DeflectionPersonNameDecision(redact=True)
        if shortened is not None:
            _, trailing = shortened
            return _DeflectionPersonNameDecision(redact=True, trailing_text=trailing)
        return _DeflectionPersonNameDecision(redact=False)
    if _looks_like_deflection_person_name(candidate):
        return _DeflectionPersonNameDecision(redact=True)
    if shortened is not None:
        _, trailing = shortened
        return _DeflectionPersonNameDecision(redact=True, trailing_text=trailing)
    if _looks_like_deflection_cued_person_name(candidate, match.group("prefix")):
        return _DeflectionPersonNameDecision(redact=True)
    return _DeflectionPersonNameDecision(redact=False)


def _deflection_person_name_match_is_pii(match: re.Match[str]) -> bool:
    return _deflection_person_name_match_decision(match).redact


def _is_plain_role_deflection_person_name_cue(prefix: str) -> bool:
    normalized = " ".join(prefix.strip().casefold().split())
    return normalized in {
        "agent",
        "client",
        "contact",
        "customer",
        "member",
        "person",
        "requester",
        "user",
    }


def _looks_like_plain_role_deflection_person_name(value: str) -> bool:
    tokens = [
        token.strip(" .'-")
        for token in value.split()
        if token.strip(" .'-")
    ]
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    lowered = [token.casefold() for token in tokens]
    if any(token in _DEFLECTION_PERSON_NAME_REJECT_TOKENS for token in lowered):
        return False
    if _looks_like_deflection_role_phrase(tokens):
        return False
    return all(token[0].isupper() and len(token) >= 2 for token in tokens)


def _looks_like_deflection_cued_person_name(value: str, prefix: str) -> bool:
    if not _is_strong_deflection_person_name_cue(prefix):
        return False
    tokens = [
        token.strip(" .'-").casefold()
        for token in value.split()
        if token.strip(" .'-")
    ]
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    if all(token in _DEFLECTION_PERSON_NAME_REJECT_TOKENS for token in tokens):
        return False
    return all(len(token) >= 2 for token in tokens)


def _is_strong_deflection_person_name_cue(prefix: str) -> bool:
    normalized = " ".join(prefix.strip().casefold().split())
    return " name" in f" {normalized}" or normalized.endswith(" is")


def _shortest_deflection_person_name_candidate(value: str) -> tuple[str, str] | None:
    parts = value.split()
    if len(parts) != 3:
        return None
    first_two = " ".join(parts[:2])
    trailing = parts[2]
    if (
        not _looks_like_deflection_person_name(first_two)
        or _looks_like_deflection_role_phrase(parts[:2])
    ):
        return None
    return first_two, f" {trailing}"


def _has_deflection_trailing_label_context(match: re.Match[str]) -> bool:
    suffix = match.string[match.end() :]
    next_token = re.match(r"\s+([A-Za-z][A-Za-z'-]*)", suffix)
    return bool(
        next_token
        and next_token.group(1).strip(" .'-").casefold()
        in _DEFLECTION_TRAILING_LABEL_CONTEXT_TOKENS
    )


def _looks_like_deflection_role_phrase(tokens: Sequence[str]) -> bool:
    return any(
        token.strip(" .'-").casefold() in _DEFLECTION_ROLE_PHRASE_TOKENS
        for token in tokens
    )


def _redact_deflection_standard_prefixed_contextual_identifier(
    match: re.Match[str],
) -> str:
    identifier = match.group("identifier")
    if _looks_like_deflection_technical_identifier(identifier):
        return match.group(0)
    return f"{match.group('prefix')}[redacted-identifier]"


def _redact_deflection_contextual_opaque_identifier(match: re.Match[str]) -> str:
    token = match.group("identifier")
    if not _looks_like_deflection_opaque_identifier(token, require_prefix=False):
        return match.group(0)
    return f"{match.group('prefix')}[redacted-identifier]"


def _redact_deflection_labeled_identifier(match: re.Match[str]) -> str:
    if _identifier_match_has_technical_standard_prefix(match):
        return match.group(0)
    return "[redacted-identifier]"


def _identifier_match_has_technical_standard_prefix(match: re.Match[str]) -> bool:
    prefix = match.string[max(0, match.start() - 16) : match.start()]
    if not re.search(r"(?:ISO|HIPAA)[-_ ]?$", prefix, re.IGNORECASE):
        return False
    return bool(_DEFLECTION_STANDARD_NUMBER_LABEL_RE.fullmatch(match.group(0)))


def _looks_like_deflection_person_name(value: str) -> bool:
    tokens = [
        token.strip(" .'-").casefold()
        for token in value.split()
        if token.strip(" .'-")
    ]
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    if any(token in _DEFLECTION_PERSON_NAME_REJECT_TOKENS for token in tokens):
        return False
    return all(len(token) >= 2 for token in tokens)


def _redact_deflection_opaque_identifier(match: re.Match[str]) -> str:
    token = match.group(0)
    if not _looks_like_deflection_opaque_identifier(token, require_prefix=True):
        return token
    return "[redacted-identifier]"


def _looks_like_deflection_opaque_identifier(
    value: str,
    *,
    require_prefix: bool,
) -> bool:
    compact = re.sub(r"[-_]", "", value)
    if len(compact) < 8:
        return False
    if _looks_like_deflection_technical_identifier(value):
        return False
    if not any(char.isalpha() for char in compact):
        return not require_prefix
    digit_count = sum(1 for char in compact if char.isdigit())
    if digit_count < 1:
        return False
    lowered = value.casefold()
    if require_prefix and not lowered.startswith(_DEFLECTION_OPAQUE_IDENTIFIER_PREFIXES):
        return False
    return digit_count >= 1


def _looks_like_deflection_technical_identifier(value: str) -> bool:
    return bool(_DEFLECTION_TECHNICAL_IDENTIFIER_RE.fullmatch(value.strip()))


def _has_deflection_source_link_pii(value: str) -> bool:
    if (
        _DEFLECTION_EMAIL_RE.search(value)
        or _DEFLECTION_PHONE_RE.search(value)
        or _DEFLECTION_SSN_RE.search(value)
        or _DEFLECTION_REDACTION_ARTIFACT_RE.search(value)
        or _has_deflection_labeled_identifier_pii(value)
    ):
        return True
    if any(
        _deflection_payment_card_replacement(match.group(0)) is not None
        for match in _DEFLECTION_PAYMENT_CARD_RE.finditer(value)
    ):
        return True
    if any(
        _looks_like_deflection_street_address(match)
        for match in _DEFLECTION_STREET_ADDRESS_RE.finditer(value)
    ):
        return True
    if any(
        _deflection_person_name_match_is_pii(match)
        for match in _DEFLECTION_PERSON_NAME_RE.finditer(value)
    ):
        return True
    if any(
        not _looks_like_deflection_technical_identifier(match.group("identifier"))
        for match in _DEFLECTION_STANDARD_PREFIXED_CONTEXTUAL_IDENTIFIER_RE.finditer(
            value
        )
    ):
        return True
    if any(
        _looks_like_deflection_opaque_identifier(
            match.group("identifier"),
            require_prefix=False,
        )
        for match in _DEFLECTION_CONTEXTUAL_OPAQUE_IDENTIFIER_RE.finditer(value)
    ):
        return True
    return any(
        _looks_like_deflection_opaque_identifier(match.group(0), require_prefix=True)
        for match in _DEFLECTION_OPAQUE_IDENTIFIER_RE.finditer(value)
    )


def _has_deflection_labeled_identifier_pii(value: str) -> bool:
    return bool(
        _DEFLECTION_IDENTIFIER_RE.search(value)
        and not _looks_like_deflection_source_link_identifier(value)
    )


def _looks_like_deflection_source_link_identifier(value: str) -> bool:
    text = _text(value).strip().casefold()
    return bool(re.fullmatch(r"(?:ticket|source)-[a-z0-9][a-z0-9_-]*", text))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


def _cell(value: Any) -> str:
    text = _md(value).replace("\n", " ")
    return text.replace("|", "\\|")


def _md(value: Any) -> str:
    return _text(value).replace("\r", " ").strip()


__all__ = [
    "DEFAULT_DEFLECTION_SNAPSHOT_TOP_N",
    "DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS",
    "DEFAULT_DEFLECTION_SEO_TARGET_LIMIT",
    "DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT",
    "DEFLECTION_FULL_REPORT_QA_SCORECARD_SCHEMA_VERSION",
    "DEFLECTION_REPORT_MODEL_FIELDS",
    "DEFLECTION_REPORT_SCHEMA_VERSION",
    "DEFLECTION_REPORT_SECTION_FIELDS",
    "DEFLECTION_REPORT_SECTION_REGISTRY",
    "DeflectionSnapshot",
    "DeflectionReportArtifact",
    "DeflectionReportSection",
    "DeflectionReportSectionDefinition",
    "DeflectionStructuredReport",
    "FAQDeflectionReportService",
    "build_deflection_report_model",
    "build_deflection_snapshot",
    "build_deflection_report_artifact",
    "build_deflection_full_report_qa_deterministic_harness",
    "build_deflection_full_report_qa_scorecard",
    "deflection_report_summary",
    "deflection_report_model_contract_shape",
    "deflection_snapshot_content_opportunities",
    "render_deflection_report",
    "render_deflection_report_model",
    "scrub_deflection_report_payload",
]
