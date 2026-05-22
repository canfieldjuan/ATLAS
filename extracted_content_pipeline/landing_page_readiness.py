"""Shared landing-page SEO/AEO/GEO readiness scoring."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from .landing_page_ports import LandingPageDraft, LandingPageSection
from extracted_quality_gate.landing_page_section_contract import (
    LANDING_PAGE_OBJECTION_SECTION_KINDS,
    LANDING_PAGE_PROBLEM_SECTION_KINDS,
    LANDING_PAGE_QUESTION_SECTION_KINDS,
    LANDING_PAGE_SECTION_KINDS,
    LANDING_PAGE_SOLUTION_SECTION_KINDS,
    normalize_landing_page_section_kind,
)


JsonDict = dict[str, Any]


_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_UNRESOLVED_TOKEN_RE = re.compile(
    r"\{\{[^{}]+\}\}|\b(?:todo|tbd|lorem ipsum)\b",
    re.IGNORECASE,
)
_NUMERIC_CLAIM_RE = re.compile(r"\b\d+(?:[\d,]*|\.\d+)?%?\b")
_EVIDENCE_RE = re.compile(
    r"\b(?:case stud(?:y|ies)|customer(?:s)?|testimonial(?:s)?|review(?:s)?|"
    r"source(?:s)?|data|survey(?:s)?|benchmark(?:s)?|ticket(?:s)?|"
    r"last\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)
_OBJECTION_SECTION_RE = re.compile(
    r"\b(?:faq|question(?:s)?|objection(?:s)?|concern(?:s)?|how it works|"
    r"pricing|compare|comparison|proof|risk|security|implementation)\b",
    re.IGNORECASE,
)
_PROBLEM_RE = re.compile(r"\b(?:problem|pain|challenge|risk|stuck|friction)\b", re.IGNORECASE)
_SOLUTION_RE = re.compile(r"\b(?:solution|solve|fix|how it works|approach|workflow|way)\b", re.IGNORECASE)
_GENERIC_SLUGS = frozenset({"landing-page", "campaign", "demo", "offer", "page"})
_GENERIC_AUDIENCES = frozenset({
    "business",
    "businesses",
    "companies",
    "company",
    "customers",
    "people",
    "teams",
    "users",
})
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


def landing_page_seo_aeo_readiness(draft: LandingPageDraft) -> JsonDict:
    meta = _metadata_mapping(draft.meta)
    text = _visible_text(draft)
    hero_text = _mapping_text(draft.hero)
    checks = {
        "title_tag": _text_len_between(meta.get("title_tag"), min_len=8, max_len=70),
        "meta_description": _text_len_between(
            meta.get("description"),
            min_len=50,
            max_len=180,
        ),
        "slug_quality": _slug_quality(draft.slug),
        "metadata_consistency": _metadata_consistency(draft, meta),
        "answer_first_hero": _answer_first_hero(draft, hero_text),
        "problem_solution_clarity": _problem_solution_clarity(draft),
        "audience_specificity": _audience_specificity(draft, text),
        "objection_coverage": _objection_coverage(draft),
    }
    return _readiness_summary(checks)


def landing_page_geo_readiness(draft: LandingPageDraft) -> JsonDict:
    text = _visible_text(draft)
    checks = {
        "offer_entity_clarity": _offer_entity_clarity(draft, text),
        "audience_entity_clarity": _audience_specificity(draft, text),
        "answer_extractability": _answer_extractability(draft),
        "section_semantics": _section_semantics(draft),
        "trust_signal_visibility": _trust_signal_visibility(draft, text),
        "conversion_path_clarity": _conversion_path_clarity(draft),
        "claim_safety": _claim_safety(draft, text),
    }
    return _readiness_summary(checks)


def landing_page_readiness_repair_issues(draft: LandingPageDraft) -> tuple[str, ...]:
    """Return stable repair issue names for missing SEO/AEO/GEO readiness checks."""

    seo = landing_page_seo_aeo_readiness(draft)
    geo = landing_page_geo_readiness(draft)
    return tuple(
        [f"seo_aeo_readiness:{item}" for item in seo.get("missing") or ()]
        + [f"geo_readiness:{item}" for item in geo.get("missing") or ()]
    )


def _readiness_summary(checks: Mapping[str, bool]) -> JsonDict:
    missing = [key for key, value in checks.items() if not value]
    passed = sum(1 for value in checks.values() if value)
    return {
        "status": "ready" if not missing else "needs_review",
        "passed": passed,
        "total": len(checks),
        "missing": missing,
        "checks": dict(checks),
    }


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


def _text_len_between(value: Any, *, min_len: int, max_len: int) -> bool:
    text = _clean_text(value)
    return min_len <= len(text) <= max_len and not _UNRESOLVED_TOKEN_RE.search(text)


def _slug_quality(value: Any) -> bool:
    slug = _clean_text(value)
    return bool(_SLUG_RE.fullmatch(slug)) and slug not in _GENERIC_SLUGS


def _metadata_consistency(draft: LandingPageDraft, meta: Mapping[str, Any]) -> bool:
    metadata_text = _normalize_text(" ".join((
        _clean_text(meta.get("title_tag")),
        _clean_text(meta.get("description")),
        _clean_text(meta.get("og_title")),
    )))
    if not metadata_text:
        return False
    return _contains_any(metadata_text, _key_terms(
        draft.campaign_name,
        draft.title,
        draft.value_prop,
        _mapping_text(draft.hero),
    ))


def _answer_first_hero(draft: LandingPageDraft, hero_text: str) -> bool:
    headline = _clean_text(_metadata_mapping(draft.hero).get("headline"))
    subheadline = _clean_text(_metadata_mapping(draft.hero).get("subheadline"))
    if not headline or not subheadline:
        return False
    normalized = _normalize_text(hero_text)
    return _contains_any(normalized, _key_terms(draft.persona, draft.value_prop))


def _problem_solution_clarity(draft: LandingPageDraft) -> bool:
    section_text = " ".join(
        f"{section.id} {section.title} {section.body_markdown} "
        f"{_mapping_text(section.metadata)}"
        for section in draft.sections
    )
    section_kinds = {_section_kind(section) for section in draft.sections}
    has_problem = bool(_PROBLEM_RE.search(section_text)) or bool(
        section_kinds.intersection(LANDING_PAGE_PROBLEM_SECTION_KINDS)
    )
    has_solution = bool(_SOLUTION_RE.search(section_text)) or bool(
        section_kinds.intersection(LANDING_PAGE_SOLUTION_SECTION_KINDS)
    )
    value_terms = _key_terms(draft.value_prop)
    return (
        has_problem
        and has_solution
        and _contains_any(_normalize_text(section_text), value_terms)
    )


def _audience_specificity(draft: LandingPageDraft, text: str) -> bool:
    persona = _clean_text(draft.persona)
    if len(persona) < 4 or persona.lower() in _GENERIC_AUDIENCES:
        return False
    return _contains_any(_normalize_text(text), _key_terms(persona))


def _objection_coverage(draft: LandingPageDraft) -> bool:
    for section in draft.sections:
        if _section_kind(section) in LANDING_PAGE_OBJECTION_SECTION_KINDS:
            return True
        if _OBJECTION_SECTION_RE.search(f"{section.id} {section.title}"):
            return True
    return False


def _offer_entity_clarity(draft: LandingPageDraft, text: str) -> bool:
    if not _clean_text(draft.campaign_name) and not _clean_text(draft.title):
        return False
    return _contains_any(_normalize_text(text), _key_terms(
        draft.campaign_name,
        draft.title,
        draft.value_prop,
    ))


def _answer_extractability(draft: LandingPageDraft) -> bool:
    hero_text = _mapping_text(draft.hero)
    words = hero_text.split()
    if 12 <= len(words) <= 90 and _answer_first_hero(draft, hero_text):
        return True
    if not draft.sections:
        return False
    summary = _section_answer_summary(draft.sections[0])
    if not _section_answer_summary_visible(draft.sections[0]):
        return False
    return _contains_any(
        _normalize_text(summary),
        _key_terms(draft.persona, draft.value_prop, draft.title),
    )


def _section_semantics(draft: LandingPageDraft) -> bool:
    if not draft.sections:
        return False
    for section in draft.sections:
        title = _clean_text(section.title)
        if len(title) < 4 or title.lower() in _GENERIC_SECTION_TITLES:
            return False
        kind = _section_kind(section)
        if kind not in LANDING_PAGE_SECTION_KINDS:
            return False
        if _section_requires_visible_answer(section) and not _section_answer_summary_visible(section):
            return False
    return True


def _section_kind(section: LandingPageSection) -> str:
    return normalize_landing_page_section_kind(
        _metadata_mapping(section.metadata).get("kind")
    )


def _section_primary_question(section: LandingPageSection) -> str:
    return _clean_text(_metadata_mapping(section.metadata).get("primary_question"))


def _section_answer_summary(section: LandingPageSection) -> str:
    return _clean_text(_metadata_mapping(section.metadata).get("answer_summary"))


def _section_requires_visible_answer(section: LandingPageSection) -> bool:
    return bool(_section_primary_question(section)) or (
        _section_kind(section) in LANDING_PAGE_QUESTION_SECTION_KINDS
    )


def _section_answer_summary_visible(section: LandingPageSection) -> bool:
    summary = _section_answer_summary(section)
    body = _clean_text(section.body_markdown)
    words = summary.split()
    if len(words) < 6 or len(words) > 90:
        return False
    normalized_summary = _normalize_text(summary)
    normalized_body = _normalize_text(body)
    return bool(normalized_summary and normalized_body.startswith(normalized_summary))


def _trust_signal_visibility(draft: LandingPageDraft, text: str) -> bool:
    return bool(draft.reference_ids) or bool(_EVIDENCE_RE.search(text))


def _conversion_path_clarity(draft: LandingPageDraft) -> bool:
    cta = _metadata_mapping(draft.cta)
    label = _clean_text(cta.get("label"))
    url = _clean_text(cta.get("url"))
    if not label or not url or url in {"#", "/#", "javascript:void(0)"}:
        return False
    hero = _metadata_mapping(draft.hero)
    hero_label = _clean_text(hero.get("cta_label"))
    if hero_label and not _shared_terms(label, hero_label):
        return False
    return True


def _claim_safety(draft: LandingPageDraft, text: str) -> bool:
    if _UNRESOLVED_TOKEN_RE.search(text):
        return False
    has_evidence = bool(draft.reference_ids) or bool(_EVIDENCE_RE.search(text))
    if _NUMERIC_CLAIM_RE.search(text) and not has_evidence:
        return False
    return True


def _visible_text(draft: LandingPageDraft) -> str:
    parts = [
        draft.title,
        _mapping_text(draft.hero),
        _mapping_text(draft.cta),
    ]
    for section in draft.sections:
        parts.extend((
            section.id,
            section.title,
            section.body_markdown,
            _mapping_text(section.metadata),
        ))
    return " ".join(_clean_text(part) for part in parts if _clean_text(part))


def _mapping_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_mapping_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_mapping_text(item) for item in value)
    return _clean_text(value)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(value).lower()).strip()


def _key_terms(*values: Any) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        words = [
            word for word in normalized.split()
            if len(word) >= 4 and word not in _STOPWORDS
        ]
        if len(words) > 1:
            phrase = " ".join(words)
            if phrase not in seen:
                seen.add(phrase)
                terms.append(phrase)
        for word in words:
            if word not in seen:
                seen.add(word)
                terms.append(word)
    return tuple(terms)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    return any(term in text for term in terms)


def _shared_terms(left: str, right: str) -> bool:
    left_terms = set(_key_terms(left))
    right_terms = set(_key_terms(right))
    return bool(left_terms and right_terms and left_terms.intersection(right_terms))


__all__ = [
    "landing_page_geo_readiness",
    "landing_page_readiness_repair_issues",
    "landing_page_seo_aeo_readiness",
]
