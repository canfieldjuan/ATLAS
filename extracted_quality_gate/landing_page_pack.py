"""Landing-page quality pack: deterministic validators for marketing pages.

Sibling to ``report_pack`` (PR-Reports-1b) and ``campaign_pack`` /
``blog_pack``. Validates the structured ``LandingPageDraft`` shape from
``extracted_content_pipeline.landing_page_ports``: title, slug, hero,
ordered sections, CTA, SEO meta. Pure-function discipline (no DB, no
LLM, no clock) -- sanitization belongs in the wrapper.

Public API:

    evaluate_landing_page(
        input: QualityInput,
        *,
        policy: QualityPolicy | None = None,
    ) -> QualityReport

The ``input`` carries the structured landing-page payload through
``input.context``. Recognised keys:

  - ``title`` (str)
  - ``slug`` (str)
  - ``hero`` (Mapping): expected keys ``headline``, ``subheadline``,
    ``cta_label``, ``cta_url``
  - ``sections`` (Sequence[Mapping]): each ``{"id", "title",
    "body_markdown", ...}``; optional ``metadata.kind``,
    ``metadata.primary_question``, and ``metadata.answer_summary`` are
    scored as warnings when malformed.
  - ``cta`` (Mapping): expected keys ``label``, ``url``
  - ``meta`` (Mapping): expected keys ``title_tag``, ``description``

Recognised ``policy.thresholds`` keys:
  - ``min_sections`` (int): default 1
  - ``min_meta_description_chars`` (int): warn when meta.description is
    below this; default 0 (no floor). 120-160 is the SEO sweet spot.
  - ``pass_score`` (int): default 70
  - ``blocking_penalty`` (int): default 18 (per blocker)
  - ``warning_penalty`` (int): default 6 (per warning)

Recognised ``policy.metadata`` keys:
  - ``blocked_phrasing`` (Sequence[str] | str): word-boundary blocked
    phrases. Bare string is auto-wrapped (mirrors report_pack).
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .types import (
    GateDecision,
    GateFinding,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)
from .landing_page_section_contract import (
    LANDING_PAGE_QUESTION_SECTION_KINDS,
    LANDING_PAGE_SECTION_KINDS,
    normalize_landing_page_section_kind,
)


_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_GENERIC_SLUGS = frozenset({"landing-page", "campaign", "demo", "offer", "page"})
_PLACEHOLDER_URLS = frozenset({"#", "/#", "javascript:void(0)", "javascript:;"})
_UNRESOLVED_TOKEN_RE = re.compile(
    r"\{\{[^{}]+\}\}|\{\{[^{}]+\}|\b(?:todo|tbd|lorem ipsum)\b",
    re.IGNORECASE,
)
_GENERIC_SECTION_TITLES = frozenset({
    "benefits",
    "conclusion",
    "features",
    "introduction",
    "overview",
    "summary",
})
_STOPWORDS = frozenset({
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "before",
    "business",
    "can",
    "company",
    "customer",
    "customers",
    "for",
    "from",
    "have",
    "into",
    "landing",
    "page",
    "problem",
    "problems",
    "same",
    "that",
    "the",
    "their",
    "this",
    "turn",
    "use",
    "with",
    "your",
})


_DEFAULT_THRESHOLDS: Mapping[str, Any] = {
    "min_sections": 1,
    "min_meta_description_chars": 0,
    "pass_score": 70,
    "blocking_penalty": 18,
    "warning_penalty": 6,
}


def _threshold_int(policy: QualityPolicy | None, key: str) -> int:
    if policy is not None:
        value = policy.thresholds.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
    return int(_DEFAULT_THRESHOLDS[key])


def _blocked_phrases(policy: QualityPolicy | None) -> tuple[str, ...]:
    """Mirror of ``report_pack._blocked_phrases``: bare string auto-wraps."""
    if policy is None:
        return ()
    raw = policy.metadata.get("blocked_phrasing")
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if not isinstance(raw, Sequence):
        return ()
    return tuple(str(item) for item in raw if str(item).strip())


def evaluate_landing_page(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Run the deterministic landing-page-quality validators."""

    context = dict(input.context or {})
    title = str(context.get("title") or "").strip()
    slug = str(context.get("slug") or "").strip()
    hero = context.get("hero") if isinstance(context.get("hero"), Mapping) else {}
    cta = context.get("cta") if isinstance(context.get("cta"), Mapping) else {}
    meta = context.get("meta") if isinstance(context.get("meta"), Mapping) else {}
    sections_raw = context.get("sections") or ()
    sections: list[Mapping[str, Any]] = [
        section for section in sections_raw if isinstance(section, Mapping)
    ]

    findings: list[GateFinding] = []

    # ---- Title / slug ----
    if not title:
        findings.append(GateFinding(code="no_title", message="no_title", severity=GateSeverity.BLOCKER))
    if not slug:
        findings.append(GateFinding(code="no_slug", message="no_slug", severity=GateSeverity.BLOCKER))
    elif not _slug_quality(slug):
        findings.append(
            GateFinding(
                code="invalid_slug",
                message=f"invalid_slug:{slug}",
                severity=GateSeverity.BLOCKER,
                metadata={"slug": slug},
            )
        )

    # ---- Hero ----
    headline = str(hero.get("headline") or "").strip()
    if not headline:
        findings.append(
            GateFinding(
                code="no_hero_headline",
                message="no_hero_headline",
                severity=GateSeverity.BLOCKER,
            )
        )
    subheadline = str(hero.get("subheadline") or "").strip()
    if not subheadline:
        findings.append(
            GateFinding(
                code="no_hero_subheadline",
                message="no_hero_subheadline",
                severity=GateSeverity.WARNING,
            )
        )

    # ---- CTA ----
    cta_label = str(cta.get("label") or "").strip()
    cta_url = str(cta.get("url") or "").strip()
    if not cta_label or not cta_url:
        findings.append(
            GateFinding(
                code="no_cta",
                message="no_cta",
                severity=GateSeverity.BLOCKER,
                metadata={"has_label": bool(cta_label), "has_url": bool(cta_url)},
            )
        )
    elif _placeholder_url(cta_url):
        findings.append(
            GateFinding(
                code="placeholder_cta_url",
                message=f"placeholder_cta_url:{cta_url}",
                severity=GateSeverity.BLOCKER,
                metadata={"url": cta_url},
            )
        )

    # ---- Sections ----
    min_sections = _threshold_int(policy, "min_sections")
    if len(sections) < min_sections:
        findings.append(
            GateFinding(
                code="no_sections",
                message=f"no_sections:{len(sections)}_below_min_{min_sections}",
                severity=GateSeverity.BLOCKER,
                metadata={"section_count": len(sections), "min_sections": min_sections},
            )
        )
    for index, section in enumerate(sections):
        section_title = str(section.get("title") or "").strip()
        if not section_title:
            findings.append(
                GateFinding(
                    code="section_missing_title",
                    message=f"section_missing_title:{index}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"section_index": index},
                )
            )
        elif section_title.lower() in _GENERIC_SECTION_TITLES:
            findings.append(
                GateFinding(
                    code="generic_section_title",
                    message=f"generic_section_title:{index}:{section_title}",
                    severity=GateSeverity.WARNING,
                    metadata={"section_index": index, "title": section_title},
                )
            )
        section_body = str(section.get("body_markdown") or "").strip()
        if not section_body:
            findings.append(
                GateFinding(
                    code="section_missing_body",
                    message=f"section_missing_body:{index}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"section_index": index},
                )
            )
        findings.extend(_section_metadata_warnings(index, section))

    # ---- SEO metadata ----
    title_tag = str(meta.get("title_tag") or "").strip()
    if not title_tag:
        findings.append(
            GateFinding(
                code="missing_meta_title_tag",
                message="missing_meta_title_tag",
                severity=GateSeverity.WARNING,
            )
        )
    elif len(title_tag) > 70:
        findings.append(
            GateFinding(
                code="meta_title_tag_too_long",
                message=f"meta_title_tag_too_long:{len(title_tag)}>70",
                severity=GateSeverity.WARNING,
                metadata={"length": len(title_tag), "max": 70},
            )
        )
    min_meta_chars = _threshold_int(policy, "min_meta_description_chars")
    description = str(meta.get("description") or "").strip()
    if min_meta_chars > 0:
        if not description:
            findings.append(
                GateFinding(
                    code="missing_meta_description",
                    message="missing_meta_description",
                    severity=GateSeverity.WARNING,
                )
            )
        elif len(description) < min_meta_chars:
            findings.append(
                GateFinding(
                    code="meta_description_too_short",
                    message=f"meta_description_too_short:{len(description)}<{min_meta_chars}",
                    severity=GateSeverity.WARNING,
                    metadata={"length": len(description), "min": min_meta_chars},
                )
            )
    if not _metadata_consistent(title=title, hero=hero, sections=sections, meta=meta):
        findings.append(
            GateFinding(
                code="metadata_inconsistent",
                message="metadata_inconsistent",
                severity=GateSeverity.WARNING,
            )
        )

    # ---- Placeholder/template safety ----
    unresolved_tokens = _unresolved_tokens(
        title=title,
        hero=hero,
        cta=cta,
        meta=meta,
        sections=sections,
    )
    for token in unresolved_tokens:
        findings.append(
            GateFinding(
                code="unresolved_placeholder",
                message=f"unresolved_placeholder:{token}",
                severity=GateSeverity.BLOCKER,
                metadata={"token": token},
            )
        )

    # ---- Blocked phrasing (case-insensitive word-boundary) ----
    phrases = _blocked_phrases(policy)
    if phrases:
        haystack_parts: list[str] = []
        if title:
            haystack_parts.append(title)
        if headline:
            haystack_parts.append(headline)
        if subheadline:
            haystack_parts.append(subheadline)
        # Hero CTA label is separate from the page-level CTA -- both get
        # scanned because either can land banned copy on the page.
        hero_cta_label = str(hero.get("cta_label") or "").strip()
        if hero_cta_label:
            haystack_parts.append(hero_cta_label)
        if cta_label:
            haystack_parts.append(cta_label)
        # SEO meta is the most public-facing surface (search snippets,
        # social cards). Banned phrases in title_tag / description hurt
        # most when they leak there.
        meta_title_tag = str(meta.get("title_tag") or "").strip()
        meta_description = str(meta.get("description") or "").strip()
        if meta_title_tag:
            haystack_parts.append(meta_title_tag)
        if meta_description:
            haystack_parts.append(meta_description)
        for section in sections:
            for key in ("title", "body_markdown"):
                value = section.get(key)
                if isinstance(value, str) and value.strip():
                    haystack_parts.append(value)
        haystack = "\n".join(haystack_parts)
        for phrase in phrases:
            phrase_str = str(phrase).strip()
            if not phrase_str:
                continue
            pattern = re.compile(rf"\b{re.escape(phrase_str)}\b", re.IGNORECASE)
            if pattern.search(haystack):
                findings.append(
                    GateFinding(
                        code="blocked_phrasing",
                        message=f"blocked_phrasing:{phrase}",
                        severity=GateSeverity.BLOCKER,
                        metadata={"phrase": phrase_str},
                    )
                )

    return _build_report(findings=findings, sections=sections, policy=policy)


def _slug_quality(value: str) -> bool:
    slug = str(value or "").strip()
    return bool(_SLUG_RE.fullmatch(slug)) and slug not in _GENERIC_SLUGS


def _placeholder_url(value: str) -> bool:
    url = str(value or "").strip().lower()
    return url in _PLACEHOLDER_URLS or url.startswith("javascript:")


def _section_metadata_warnings(
    index: int,
    section: Mapping[str, Any],
) -> tuple[GateFinding, ...]:
    metadata = section.get("metadata") if isinstance(section.get("metadata"), Mapping) else {}
    kind = normalize_landing_page_section_kind(metadata.get("kind"))
    findings: list[GateFinding] = []
    if not kind:
        findings.append(
            GateFinding(
                code="section_missing_kind",
                message=f"section_missing_kind:{index}",
                severity=GateSeverity.WARNING,
                metadata={"section_index": index},
            )
        )
    elif kind not in LANDING_PAGE_SECTION_KINDS:
        findings.append(
            GateFinding(
                code="section_invalid_kind",
                message=f"section_invalid_kind:{index}:{kind}",
                severity=GateSeverity.WARNING,
                metadata={"section_index": index, "kind": kind},
            )
        )
    primary_question = str(metadata.get("primary_question") or "").strip()
    if kind in LANDING_PAGE_QUESTION_SECTION_KINDS or primary_question:
        summary = str(metadata.get("answer_summary") or "").strip()
        if not summary:
            findings.append(
                GateFinding(
                    code="section_missing_answer_summary",
                    message=f"section_missing_answer_summary:{index}",
                    severity=GateSeverity.WARNING,
                    metadata={"section_index": index, "kind": kind},
                )
            )
        elif not _answer_summary_visible(summary, section.get("body_markdown")):
            findings.append(
                GateFinding(
                    code="section_answer_summary_not_visible",
                    message=f"section_answer_summary_not_visible:{index}",
                    severity=GateSeverity.WARNING,
                    metadata={"section_index": index, "kind": kind},
                )
            )
    return tuple(findings)


def _answer_summary_visible(summary: str, body_markdown: Any) -> bool:
    normalized_summary = _normalize_text(summary)
    normalized_body = _normalize_text(body_markdown)
    return bool(normalized_summary and normalized_body.startswith(normalized_summary))


def _metadata_consistent(
    *,
    title: str,
    hero: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> bool:
    metadata_text = _normalize_text(" ".join((
        str(meta.get("title_tag") or ""),
        str(meta.get("description") or ""),
        str(meta.get("og_title") or ""),
    )))
    if not metadata_text:
        return False
    visible_text = _normalize_text(" ".join((
        title,
        _mapping_text(hero),
        " ".join(
            f"{section.get('title') or ''} {section.get('body_markdown') or ''}"
            for section in sections
        ),
    )))
    return _contains_any(metadata_text, _key_terms(visible_text))


def _unresolved_tokens(
    *,
    title: str,
    hero: Mapping[str, Any],
    cta: Mapping[str, Any],
    meta: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    haystack_parts = [title, _mapping_text(hero), _mapping_text(cta), _mapping_text(meta)]
    for section in sections:
        haystack_parts.append(_mapping_text(section))
    haystack = "\n".join(part for part in haystack_parts if part)
    tokens: list[str] = []
    for match in _UNRESOLVED_TOKEN_RE.finditer(haystack):
        token = match.group(0).strip()
        if token and token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def _mapping_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_mapping_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_mapping_text(item) for item in value)
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _key_terms(value: Any) -> tuple[str, ...]:
    normalized = _normalize_text(value)
    if not normalized:
        return ()
    words = [
        word for word in normalized.split()
        if len(word) >= 4 and word not in _STOPWORDS
    ]
    terms: list[str] = []
    seen: set[str] = set()
    if len(words) > 1:
        phrase = " ".join(words)
        terms.append(phrase)
        seen.add(phrase)
    for word in words:
        if word not in seen:
            terms.append(word)
            seen.add(word)
    return tuple(terms)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _build_report(
    *,
    findings: list[GateFinding],
    sections: list[Mapping[str, Any]],
    policy: QualityPolicy | None,
) -> QualityReport:
    blockers = [f for f in findings if f.severity == GateSeverity.BLOCKER]
    warnings = [f for f in findings if f.severity == GateSeverity.WARNING]
    blocking_penalty = _threshold_int(policy, "blocking_penalty")
    warning_penalty = _threshold_int(policy, "warning_penalty")
    pass_score = _threshold_int(policy, "pass_score")
    score = max(
        0,
        100 - (blocking_penalty * len(blockers)) - (warning_penalty * len(warnings)),
    )
    passed = (not blockers) and score >= pass_score
    if blockers or score < pass_score:
        decision = GateDecision.BLOCK
    elif warnings:
        decision = GateDecision.WARN
    else:
        decision = GateDecision.PASS
    metadata = {
        "score": score,
        "threshold": pass_score,
        "status": "pass" if passed else "fail",
        "blocking_issues": tuple(f.message for f in blockers),
        "blocking_codes": tuple(f.code for f in blockers),
        "warnings": tuple(f.message for f in warnings),
        "warning_codes": tuple(f.code for f in warnings),
        "section_count": len(sections),
    }
    return QualityReport(
        passed=passed,
        decision=decision,
        findings=tuple(findings),
        metadata=metadata,
    )


__all__ = ["evaluate_landing_page"]
