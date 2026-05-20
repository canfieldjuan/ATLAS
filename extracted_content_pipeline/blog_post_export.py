"""Read-only blog-post draft export helpers for AI Content Ops hosts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from io import StringIO
import json
import re
from typing import Any

from .blog_ports import BlogPostDraft, BlogPostRepository
from .campaign_ports import TenantScope


JsonDict = dict[str, Any]


_EXPORT_COLUMNS = (
    "slug",
    "title",
    "description",
    "topic_type",
    "tag_count",
    "chart_count",
    "generation_input_tokens",
    "generation_output_tokens",
    "generation_total_tokens",
    "generation_parse_attempts",
    "reasoning_context_used",
    "reasoning_wedge",
    "reasoning_confidence",
    "passed_output_checks",
    "output_checks",
    "seo_aeo_readiness",
    "geo_readiness",
    "tags",
    "content",
    "charts",
    "data_context",
    "metadata",
    "id",
    "status",
)


@dataclass(frozen=True)
class BlogPostDraftExportResult:
    rows: tuple[JsonDict, ...]
    limit: int
    filters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.rows),
            "limit": self.limit,
            "filters": dict(self.filters),
            "rows": [dict(row) for row in self.rows],
        }

    def as_csv(self) -> str:
        handle = StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(_EXPORT_COLUMNS))
        writer.writeheader()
        for row in self.rows:
            writer.writerow({
                column: _csv_value(row.get(column))
                for column in _EXPORT_COLUMNS
            })
        return handle.getvalue()


async def export_blog_post_drafts(
    repository: BlogPostRepository,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    status: str | None = "draft",
    topic_type: str | None = None,
    limit: int = 20,
) -> BlogPostDraftExportResult:
    """Return generated blog-post drafts for host review/export workflows."""

    tenant = _tenant_scope(scope)
    normalized_limit = _normalize_limit(limit)
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if tenant.account_id:
        filters["account_id"] = tenant.account_id
    if topic_type:
        filters["topic_type"] = topic_type
    drafts = await repository.list_drafts(
        scope=tenant,
        status=status,
        topic_type=topic_type,
        limit=normalized_limit,
    )
    return BlogPostDraftExportResult(
        rows=tuple(_draft_row(draft) for draft in drafts),
        limit=normalized_limit,
        filters=filters,
    )


def _draft_row(draft: BlogPostDraft) -> JsonDict:
    row = draft.as_dict()
    row["tag_count"] = len(draft.tags)
    row["chart_count"] = len(draft.charts)
    row.update(_metadata_summary(draft.metadata))
    readiness = _seo_aeo_readiness(draft)
    row["output_checks"] = readiness["checks"]
    row["passed_output_checks"] = readiness["passed"]
    row["seo_aeo_readiness"] = readiness
    row["geo_readiness"] = _geo_readiness(draft)
    return row


def _metadata_summary(value: Any) -> JsonDict:
    metadata = _metadata_mapping(value)
    usage = _metadata_mapping(metadata.get("generation_usage"))
    reasoning = _metadata_mapping(metadata.get("reasoning_context"))
    return {
        "generation_input_tokens": usage.get("input_tokens"),
        "generation_output_tokens": usage.get("output_tokens"),
        "generation_total_tokens": usage.get("total_tokens"),
        "generation_parse_attempts": metadata.get("generation_parse_attempts"),
        "reasoning_context_used": bool(reasoning),
        "reasoning_wedge": reasoning.get("wedge"),
        "reasoning_confidence": reasoning.get("confidence"),
    }


_QUESTION_H2_RE = re.compile(
    r"^##\s+(?:who|what|when|where|why|how|is|are|can|should|does|do|which)\b.*\?",
    re.IGNORECASE | re.MULTILINE,
)
_H2_RE = re.compile(r"^##\s+.+$", re.MULTILINE)
_UNRESOLVED_TOKEN_RE = re.compile(r"\{\{([^{}]+)\}\}")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s*(.+)$", re.MULTILINE)
_FRESHNESS_RE = re.compile(
    r"\b(?:20\d{2}|Q[1-4]\s+20\d{2}|last\s+\d+\s+(?:days?|weeks?|months?|years?)|"
    r"past\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)
_EVIDENCE_RE = re.compile(
    r"\b(?:\d+[\d,]*(?:\.\d+)?%?|\d+\s+(?:reviews?|tickets?|responses?|"
    r"customers?|users?|accounts?)|review(?:s| data| patterns)?|ticket(?:s| data)?|"
    r"customer wording|source(?:s)?|survey(?:s)?)\b",
    re.IGNORECASE,
)
_VAGUE_H2_RE = re.compile(
    r"^##\s+(?:overview|introduction|conclusion|key takeaways|final thoughts|summary)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _seo_aeo_readiness(draft: BlogPostDraft) -> JsonDict:
    metadata = _metadata_mapping(draft.metadata)
    checks = {
        "seo_title_ready": _text_len_between(metadata.get("seo_title"), max_len=60),
        "seo_description_ready": _text_len_between(
            metadata.get("seo_description"),
            max_len=155,
        ),
        "target_keyword_present": bool(_clean_text(metadata.get("target_keyword"))),
        "secondary_keywords_present": len(_metadata_list(metadata.get("secondary_keywords"))) >= 1,
        "faq_ready": len(_metadata_list(metadata.get("faq"))) >= 3,
        "aeo_structure_detected": _aeo_structure_detected(draft.content),
    }
    missing = [key for key, value in checks.items() if not value]
    passed = sum(1 for value in checks.values() if value)
    return {
        "status": "ready" if not missing else "needs_review",
        "passed": passed,
        "total": len(checks),
        "missing": missing,
        "checks": checks,
    }


def _geo_readiness(draft: BlogPostDraft) -> JsonDict:
    metadata = _metadata_mapping(draft.metadata)
    data_context = _metadata_mapping(draft.data_context)
    body = str(draft.content or "")
    topic_terms = _topic_terms(draft, metadata=metadata, data_context=data_context)
    checks = {
        "entity_clarity": _entity_clarity(draft, body, topic_terms=topic_terms),
        "answer_first_sections": _aeo_structure_detected(body),
        "citable_section_structure": _citable_section_structure(body, topic_terms=topic_terms),
        "evidence_specificity": _evidence_specificity(body),
        "freshness_context": _freshness_context(body, data_context),
        "faq_coverage": len(_metadata_list(metadata.get("faq"))) >= 3,
        "citation_safety": _citation_safety(body),
    }
    missing = [key for key, value in checks.items() if not value]
    passed = sum(1 for value in checks.values() if value)
    return {
        "status": "ready" if not missing else "needs_review",
        "passed": passed,
        "total": len(checks),
        "missing": missing,
        "checks": checks,
    }


def _text_len_between(value: Any, *, max_len: int) -> bool:
    text = _clean_text(value)
    return bool(text) and len(text) <= max_len


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _metadata_list(value: Any) -> list[Any]:
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        value = decoded
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _aeo_structure_detected(content: Any) -> bool:
    body = str(content or "")
    if _QUESTION_H2_RE.search(body):
        return True
    for match in _H2_RE.finditer(body):
        section_start = match.end()
        next_heading = _H2_RE.search(body, section_start)
        section = body[section_start:next_heading.start() if next_heading else None]
        first_paragraph = _first_paragraph(section)
        word_count = len(first_paragraph.split())
        if 40 <= word_count <= 80:
            return True
    return False


def _topic_terms(
    draft: BlogPostDraft,
    *,
    metadata: Mapping[str, Any],
    data_context: Mapping[str, Any],
) -> tuple[str, ...]:
    candidates = (
        data_context.get("vendor"),
        data_context.get("vendor_name"),
        data_context.get("product"),
        data_context.get("product_name"),
        data_context.get("category"),
        data_context.get("topic"),
        metadata.get("target_keyword"),
    )
    terms: list[str] = []
    for candidate in candidates:
        text = _clean_text(candidate)
        if len(text) >= 3 and text.lower() not in {item.lower() for item in terms}:
            terms.append(text)
    return tuple(terms)


def _entity_clarity(
    draft: BlogPostDraft,
    body: str,
    *,
    topic_terms: tuple[str, ...],
) -> bool:
    if _VAGUE_H2_RE.search(body):
        return False
    searchable = f"{draft.title}\n{body[:600]}".lower()
    if not topic_terms:
        return False
    return any(term.lower() in searchable for term in topic_terms)


def _citable_section_structure(body: str, *, topic_terms: tuple[str, ...]) -> bool:
    sections = _h2_sections(body)
    if len(sections) < 2:
        return False
    return sum(
        1
        for heading, section in sections
        if _self_contained_section(heading, section, topic_terms=topic_terms)
    ) >= 2


def _h2_sections(body: str) -> list[tuple[str, str]]:
    matches = list(_H2_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else None
        heading = match.group(0).removeprefix("##").strip()
        sections.append((heading, body[match.end():next_start]))
    return sections


def _self_contained_section(
    heading: str,
    section: str,
    *,
    topic_terms: tuple[str, ...],
) -> bool:
    first_paragraph = _first_paragraph(section)
    word_count = len(first_paragraph.split())
    if not 40 <= word_count <= 120:
        return False
    if not topic_terms:
        return True
    searchable = f"{heading} {first_paragraph}".lower()
    return any(term.lower() in searchable for term in topic_terms)


def _evidence_specificity(body: str) -> bool:
    if _BLOCKQUOTE_RE.search(body):
        return True
    return bool(_EVIDENCE_RE.search(body))


def _freshness_context(body: str, data_context: Mapping[str, Any]) -> bool:
    for key in ("review_period", "source_report_date", "published_at", "date"):
        if _clean_text(data_context.get(key)):
            return True
    return bool(_FRESHNESS_RE.search(body))


def _citation_safety(body: str) -> bool:
    if 'href="#"' in body or "href='#'" in body:
        return False
    unresolved_tokens = [
        token.strip()
        for token in _UNRESOLVED_TOKEN_RE.findall(body)
        if not token.strip().startswith("chart:")
    ]
    if unresolved_tokens:
        return False
    lower_body = body.lower()
    if "lorem ipsum" in lower_body or "/blog/placeholder" in lower_body:
        return False
    return True


def _first_paragraph(section: str) -> str:
    for chunk in re.split(r"\n\s*\n", section.strip()):
        text = re.sub(r"^[>\-\*\s]+", "", chunk.strip())
        if text and not text.startswith("{{chart:") and not text.startswith("<table"):
            return text
    return ""


def _metadata_mapping(value: Any) -> JsonDict:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, separators=(",", ":"))
    return "" if value is None else value


def _normalize_limit(value: Any) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return limit


def _tenant_scope(value: TenantScope | Mapping[str, Any] | None) -> TenantScope:
    if isinstance(value, TenantScope):
        return value
    if isinstance(value, Mapping):
        return TenantScope(
            account_id=str(value.get("account_id") or "") or None,
            user_id=str(value.get("user_id") or "") or None,
        )
    return TenantScope()


__all__ = [
    "BlogPostDraftExportResult",
    "export_blog_post_drafts",
]
