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
    "body_markdown", ...}``
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

    # ---- SEO meta description ----
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
        if cta_label:
            haystack_parts.append(cta_label)
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
