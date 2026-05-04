"""Blog quality pack: deterministic validators for long-form blog posts.

Owned by ``extracted_quality_gate`` (PR-B4a). The single public entry
point ``evaluate_blog_post`` is pure: no DB, no clock, no network. It
takes a :class:`QualityInput` (the cleaned body in ``content`` plus
domain context in ``context``) and returns a :class:`QualityReport`
whose findings can be inspected, surfaced to operators, or fed back
into a repair loop.

Sanitization (markdown cleanup, unmatched-quote removal) is NOT in
scope for the pack -- those mutate the body and so belong in the
Atlas-side wrapper. The pack only validates.

Specificity validation (witness-anchor support, evidence coverage)
is NOT in scope for the pack -- per the PR-B5 framing it ships as
its own pack later. The Atlas wrapper composes specificity findings
on top of the pack's report.

Public API:

    evaluate_blog_post(
        input: QualityInput,
        *,
        policy: QualityPolicy | None = None,
    ) -> QualityReport

Recognised ``input.context`` keys:
  - ``topic_type``: str -- routes the length check to a per-topic policy.
  - ``charts``: tuple of dicts ``{"chart_id": str, "data_labels_lower": set[str]}``
    used for placeholder presence/duplication and chart-scope ambiguity.
  - ``data_context``: dict with optional ``review_period``, ``vendor``,
    ``vendor_a``, ``vendor_b``, ``to_vendor``, ``category_winner``,
    ``category_loser``, ``_valid_internal_slugs`` (list[str]),
    ``_known_vendors`` (list[str]).
  - ``slug``: str -- used to whitelist the post's own slug from internal-
    link validation.
  - ``suggested_title``: str -- used for title/vendor mismatch warning.
  - ``source_quotes``: tuple[str, ...] -- presence triggers a stricter
    quote-count check.
  - ``required_vendors``: tuple[str, ...] -- vendors that MUST be
    mentioned in the body (caller resolves from blueprint).
  - ``grounded_vendors``: set[str] -- vendor names supported by chart
    data; sentences naming an ungrounded vendor become an
    ``unsupported_data_claim`` warning.

Recognised ``policy.thresholds`` keys (all optional, all have defaults):
  - ``min_words``: int (default 1500)
  - ``target_words``: int (default 2200)
  - ``pass_score``: int (default 70)
  - ``blocking_penalty``: int (default 18) -- score deducted per blocker
  - ``warning_penalty``: int (default 6) -- score deducted per warning
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


_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s*(.+)$")
_CHART_REF_RE = re.compile(r"\{\{chart:([a-zA-Z0-9\-_]+)\}\}")
_INTERNAL_LINK_RE = re.compile(r"/blog/([a-z0-9\-]+)")

# Markers that turn an otherwise-neutral sentence into a "data claim".
# A sentence containing one of these is subject to vendor-grounding
# checks. Mirrors the legacy ``_DATA_CLAIM_MARKERS`` plus the
# ``\d+%`` / ``\d+ reviews`` / ``\d+ stories`` numeric markers.
_DATA_CLAIM_MARKERS: tuple[str, ...] = (
    "most common",
    "top migration",
    "top source",
    "primary source",
    "primary driver",
    "leading source",
    "data shows",
    "reviews mention",
    "switched from",
    "stories analyzed",
)
_DATA_CLAIM_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(m) for m in _DATA_CLAIM_MARKERS)
    + r"|\d+\s*%|\d+\s+reviews?\b|\d+\s+stories?\b)",
    re.IGNORECASE,
)
# Multi-word capitalized name pattern. Matches PascalCase, mixedCase,
# all-caps, and CamelCase joined by spaces, up to 4 tokens.
_VENDORISH_NAME_PATTERN = re.compile(
    r"\b("
    r"(?:[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+|[A-Z]{2,}|[a-z]+[A-Z][A-Za-z0-9]*)"
    r"(?:\s+(?:[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+|[A-Z]{2,}|[a-z]+[A-Z][A-Za-z0-9]*)){0,3}"
    r")\b"
)

# Common English / structural words that match the capitalized-name
# pattern but are not vendor names. The wrapper can extend this via
# ``context['non_vendor_skip_words']`` for product-specific lists
# (e.g. Atlas's ReviewSource enum names).
_DEFAULT_NON_VENDOR_SKIP_WORDS: frozenset[str] = frozenset(
    " ".join(re.findall(r"[a-z0-9]+", w.lower())) for w in (
        # Common English words
        "The", "This", "That", "When", "What", "Where", "Which", "While",
        "Most", "Top", "Data", "Teams", "Users", "Some", "Each", "Both",
        "Many", "Other", "These", "Those", "After", "Before", "Between",
        "About", "Since", "Until", "During", "However", "Although",
        # Document structure
        "Introduction", "Conclusion", "Overview", "Analysis", "Guide",
        "Report", "Summary", "Review", "Reviews", "Reviewers", "Reviewer",
        "Section", "Chart", "Table", "Source", "Sources",
        "Note", "Key", "Figure", "Methodology", "Decision",
        "Rating", "Ratings", "Platform", "Platforms", "Price", "Pricing",
        "Feature", "Features", "Support", "Integration", "Performance",
        # Time/date
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        # Tech/domain (not vendors)
        "API", "SEO", "SaaS", "CRM", "ERP", "DNS", "SSL", "CSS",
        "HTML", "REST", "SDK", "ROI", "KPI", "B2B", "SMB",
    )
)
_DEFAULT_SKIP_SENTENCE_PREFIXES: tuple[str, ...] = (
    "#",
    "_methodology note:",
    "methodology note:",
    "analysis methodology:",
    "*analysis methodology:",
)
_DEFAULT_SKIP_SENTENCE_CONTAINS: tuple[str, ...] = (
    "this analysis draws on",
    "this post draws on",
    "analysis based on self-selected reviewer feedback",
    "analysis reflects self-selected feedback from",
)

_CHART_SCOPE_PHRASES: tuple[str, ...] = (
    "most common source",
    "top migration source",
    "top source",
    "primary source",
    "where users come from",
    "where teams come from",
    "most common migration",
)
_COUNT_PATTERN = re.compile(
    r"(\d[\d,]*)\s+(?:switching|churn|migration|displacement)\s+"
    r"(?:signals?|stories|reviews?|mentions?)",
    re.IGNORECASE,
)


def _normalized_vendor_text(text: str) -> str:
    """Lowercase + alnum-tokenize. Mirrors atlas's ``_normalized_vendor_text``."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


_DEFAULT_THRESHOLDS: Mapping[str, int] = {
    "min_words": 1500,
    "target_words": 2200,
    "pass_score": 70,
    "blocking_penalty": 18,
    "warning_penalty": 6,
}


def _threshold(policy: QualityPolicy | None, key: str) -> int:
    if policy is not None:
        value = policy.thresholds.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return int(_DEFAULT_THRESHOLDS[key])


def _extract_blockquotes(body: str, min_len: int = 12) -> list[str]:
    quotes: list[str] = []
    for line in body.splitlines():
        match = _BLOCKQUOTE_RE.match(line)
        if not match:
            continue
        text = match.group(1).strip()
        # Strip surrounding double-quote pairs and trailing attribution
        if text.startswith('"') and '"' in text[1:]:
            text = text[1:].split('"', 1)[0]
        if len(text) >= min_len:
            quotes.append(text)
    return quotes


def evaluate_blog_post(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Run the deterministic blog-quality validators.

    Returns a :class:`QualityReport`. ``decision`` is:
      * ``BLOCK`` when any blocker fires OR score < pass_score
      * ``WARN`` when warnings fire but no blockers
      * ``PASS`` otherwise

    The report's ``metadata`` mirrors the legacy dict shape so
    Atlas-side callers can preserve their existing telemetry.
    """
    body = str(input.content or "")
    context = dict(input.context or {})
    data_context: Mapping[str, Any] = context.get("data_context") or {}

    findings: list[GateFinding] = []

    # ---- Word count ----
    min_words = _threshold(policy, "min_words")
    target_words = _threshold(policy, "target_words")
    word_count = len(body.split())
    if word_count < min_words:
        # Legacy underscore-joined message format that downstream
        # consumers rely on for prefix matching, e.g.
        # ``_only_content_too_short_blockers`` in
        # ``b2b_blog_post_generation.py`` checks
        # ``startswith("content_too_short:")`` to gate the
        # deterministic-repair retry path.
        findings.append(
            GateFinding(
                code="content_too_short",
                message=f"content_too_short:{word_count}_words_need_{min_words}",
                severity=GateSeverity.BLOCKER,
                metadata={"word_count": word_count, "min_words": min_words},
            )
        )
    elif word_count < target_words:
        findings.append(
            GateFinding(
                code="content_below_seo_target",
                message=f"content_below_seo_target_{target_words}_words",
                severity=GateSeverity.WARNING,
                metadata={"word_count": word_count, "target_words": target_words},
            )
        )

    # ---- Chart placeholders ----
    charts: Sequence[Mapping[str, Any]] = context.get("charts") or ()
    chart_ids = [str(c.get("chart_id") or "") for c in charts if c.get("chart_id")]
    chart_id_set = set(chart_ids)
    chart_mentions = _CHART_REF_RE.findall(body)
    for chart_id in chart_ids:
        count = chart_mentions.count(chart_id)
        if count == 0:
            findings.append(
                GateFinding(
                    code="missing_chart_placeholder",
                    message=f"missing_chart_placeholder:{chart_id}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"chart_id": chart_id},
                )
            )
        elif count > 1:
            findings.append(
                GateFinding(
                    code="duplicate_chart_placeholder",
                    message=f"duplicate_chart_placeholder:{chart_id}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"chart_id": chart_id, "count": count},
                )
            )
    unknown_chart_ids = sorted({cid for cid in chart_mentions if cid not in chart_id_set})
    if unknown_chart_ids:
        findings.append(
            GateFinding(
                code="unknown_chart_placeholders",
                message=f"unknown_chart_placeholders:{','.join(unknown_chart_ids)}",
                severity=GateSeverity.BLOCKER,
                metadata={"chart_ids": tuple(unknown_chart_ids)},
            )
        )

    # ---- Unresolved {{token}} placeholders (excluding chart refs) ----
    unresolved_tokens = sorted(
        {
            token.strip()
            for token in _PLACEHOLDER_RE.findall(body)
            if not token.strip().startswith("chart:")
        }
    )
    if unresolved_tokens:
        findings.append(
            GateFinding(
                code="unresolved_placeholders",
                message=f"unresolved_placeholders:{','.join(unresolved_tokens[:6])}",
                severity=GateSeverity.BLOCKER,
                metadata={"tokens": tuple(unresolved_tokens)},
            )
        )

    # ---- Quote count ----
    source_quotes = context.get("source_quotes") or ()
    blockquotes = _extract_blockquotes(body)
    if source_quotes and len(blockquotes) < 2:
        findings.append(
            GateFinding(
                code="too_few_sourced_quotes",
                message=f"too_few_sourced_quotes:{len(blockquotes)}",
                severity=GateSeverity.BLOCKER,
                metadata={"quote_count": len(blockquotes)},
            )
        )
    if not source_quotes and not blockquotes:
        findings.append(
            GateFinding(
                code="no_quotes_present",
                message="no_quotes_present",
                severity=GateSeverity.WARNING,
            )
        )

    # ---- Review period mention ----
    review_period = str(data_context.get("review_period") or "").strip()
    if review_period and review_period not in body:
        findings.append(
            GateFinding(
                code="review_period_not_explicitly_mentioned",
                message="review_period_not_explicitly_mentioned",
                severity=GateSeverity.WARNING,
                metadata={"review_period": review_period},
            )
        )

    # ---- Methodology disclaimer ----
    body_lower = body.lower()
    if "self-selected" not in body_lower:
        findings.append(
            GateFinding(
                code="methodology_disclaimer_missing_self_selected",
                message="methodology_disclaimer_missing_self_selected",
                severity=GateSeverity.WARNING,
            )
        )

    # ---- Required vendor mentions ----
    required_vendors = context.get("required_vendors") or ()
    missing_vendors = [
        vendor
        for vendor in required_vendors
        if vendor and not re.search(rf"\b{re.escape(vendor)}\b", body, re.IGNORECASE)
    ]
    if missing_vendors:
        findings.append(
            GateFinding(
                code="missing_vendor_mentions",
                message=f"missing_vendor_mentions:{','.join(missing_vendors)}",
                severity=GateSeverity.BLOCKER,
                metadata={"vendors": tuple(missing_vendors)},
            )
        )

    # ---- Placeholder href="#" links ----
    if 'href="#"' in body or "href='#'" in body:
        findings.append(
            GateFinding(
                code="placeholder_links_href_hash",
                message="placeholder_links_href_hash",
                severity=GateSeverity.BLOCKER,
            )
        )

    # ---- Nonexistent internal blog links ----
    internal_links = _INTERNAL_LINK_RE.findall(body)
    if internal_links:
        own_slug = str(context.get("slug") or "")
        valid_slugs = set(data_context.get("_valid_internal_slugs") or [])
        fake = [link for link in internal_links if link not in valid_slugs and link != own_slug]
        if fake:
            findings.append(
                GateFinding(
                    code="nonexistent_internal_links",
                    message=f"nonexistent_internal_links:{','.join(fake[:4])}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"links": tuple(fake)},
                )
            )

    # ---- Title/vendor mismatch ----
    suggested_title = str(context.get("suggested_title") or "")
    title_lower = suggested_title.lower()
    for vk in ("vendor", "vendor_a", "vendor_b"):
        v = str(data_context.get(vk) or "").strip()
        if v and len(v) > 2 and v.lower() not in title_lower:
            findings.append(
                GateFinding(
                    code="title_missing_expected_vendor",
                    message=f"title_missing_expected_vendor:{v}",
                    severity=GateSeverity.WARNING,
                    metadata={"vendor": v},
                )
            )
            break

    # ---- Unsupported category outcome ----
    if "category winner" in body_lower or "category loser" in body_lower:
        if not (data_context.get("category_winner") or data_context.get("category_loser")):
            findings.append(
                GateFinding(
                    code="unsupported_category_outcome_assertion",
                    message="unsupported_category_outcome_assertion",
                    severity=GateSeverity.BLOCKER,
                )
            )

    # ---- Unsupported data claims (vendor names not in grounded set) ----
    grounded_vendors = set(context.get("grounded_vendors") or ())
    known_vendors = list(data_context.get("_known_vendors") or [])
    extra_skip_words = context.get("non_vendor_skip_words")
    if extra_skip_words:
        skip_words = _DEFAULT_NON_VENDOR_SKIP_WORDS | frozenset(extra_skip_words)
    else:
        skip_words = _DEFAULT_NON_VENDOR_SKIP_WORDS
    unsupported = _find_unsupported_claims(
        body,
        grounded_vendors,
        known_vendors,
        non_vendor_skip_words=skip_words,
    )
    for claim in unsupported[:3]:
        findings.append(
            GateFinding(
                code="unsupported_data_claim",
                message=f"unsupported_data_claim:{claim}",
                severity=GateSeverity.WARNING,
                metadata={"claim": claim},
            )
        )

    # ---- Chart-scope ambiguity ----
    chart_labels_lower: set[str] = set()
    for chart in charts:
        for label in chart.get("data_labels_lower") or ():
            if label:
                chart_labels_lower.add(str(label).lower())
    if chart_labels_lower and known_vendors:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", body)
        for sentence in sentences:
            s_lower = sentence.lower()
            if not any(phrase in s_lower for phrase in _CHART_SCOPE_PHRASES):
                continue
            for v in known_vendors:
                if (
                    len(v) > 2
                    and v.lower() not in chart_labels_lower
                    and re.search(r"\b" + re.escape(v) + r"\b", sentence, re.IGNORECASE)
                ):
                    findings.append(
                        GateFinding(
                            code="chart_scope_ambiguity",
                            message=f"chart_scope_ambiguity:{v}: {sentence.strip()[:100]}",
                            severity=GateSeverity.WARNING,
                            metadata={"vendor": v},
                        )
                    )
                    break

    # ---- Numeric consistency (sub-counts vs headline) ----
    count_values = [int(m.replace(",", "")) for m in _COUNT_PATTERN.findall(body)]
    if len(count_values) >= 2:
        headline = max(count_values)
        sub_total = sum(v for v in count_values if v != headline)
        if sub_total > headline and headline > 0:
            findings.append(
                GateFinding(
                    code="numeric_inconsistency",
                    message=(
                        f"numeric_inconsistency:sub-counts ({sub_total}) "
                        f"exceed headline ({headline})"
                    ),
                    severity=GateSeverity.WARNING,
                    metadata={"headline": headline, "sub_total": sub_total},
                )
            )

    # ---- Migration direction drift (migration_guide only) ----
    topic_type = str(context.get("topic_type") or "")
    if topic_type == "migration_guide":
        topic_vendor = str(
            data_context.get("vendor") or data_context.get("to_vendor") or ""
        ).strip()
        if topic_vendor:
            outbound_phrases = (
                f"switching from {topic_vendor}".lower(),
                f"leaving {topic_vendor}".lower(),
                f"moving away from {topic_vendor}".lower(),
                f"migrating from {topic_vendor}".lower(),
                f"abandoning {topic_vendor}".lower(),
            )
            outbound_count = sum(
                1 for phrase in outbound_phrases if phrase in body_lower
            )
            if outbound_count >= 2:
                findings.append(
                    GateFinding(
                        code="migration_direction_drift",
                        message=(
                            f"migration_direction_drift:too much outbound prose "
                            f"about leaving {topic_vendor} in a switch-to-{topic_vendor} "
                            f"article ({outbound_count} outbound phrases)"
                        ),
                        severity=GateSeverity.WARNING,
                        metadata={
                            "vendor": topic_vendor,
                            "outbound_count": outbound_count,
                        },
                    )
                )

    return _build_report(
        findings=findings,
        word_count=word_count,
        min_words=min_words,
        target_words=target_words,
        quote_count=len(blockquotes),
        policy=policy,
    )


def _find_unsupported_claims(
    body: str,
    grounded: set[str],
    known_vendors: list[str],
    *,
    non_vendor_skip_words: frozenset[str] = _DEFAULT_NON_VENDOR_SKIP_WORDS,
    skip_sentence_prefixes: tuple[str, ...] = _DEFAULT_SKIP_SENTENCE_PREFIXES,
    skip_sentence_contains: tuple[str, ...] = _DEFAULT_SKIP_SENTENCE_CONTAINS,
) -> list[str]:
    """Return sentences with data-claim markers naming ungrounded vendors.

    Two detection strategies:

      1. Known-vendor lookup. A ``known_vendors`` entry not in
         ``grounded`` is the primary signal -- catches single-word
         names like ``Magento`` or ``Trello`` that the regex
         strategy alone cannot disambiguate from English words.
      2. Regex fallback for multi-word capitalized names not in the
         known universe. Skips entries that hit
         ``non_vendor_skip_words`` so structural words like
         "Top Source" or "Decision Guide" do not flag.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", body)
    flagged: list[str] = []

    grounded_normalized = {_normalized_vendor_text(g) for g in grounded}

    ungrounded_known: list[tuple[str, re.Pattern[str]]] = []
    for v in known_vendors:
        if not v or len(v) <= 2:
            continue
        nv = _normalized_vendor_text(v)
        if nv and nv not in grounded_normalized:
            pattern = re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE)
            ungrounded_known.append((v, pattern))

    for sentence in sentences:
        if not _DATA_CLAIM_PATTERN.search(sentence):
            continue
        trimmed = sentence.strip()
        trimmed_lower = trimmed.lower()
        if any(trimmed_lower.startswith(prefix) for prefix in skip_sentence_prefixes):
            continue
        if any(fragment in trimmed_lower for fragment in skip_sentence_contains):
            continue

        # Strategy 1: known vendor lookup
        found_known = False
        for vendor_name, pattern in ungrounded_known:
            if pattern.search(sentence):
                flagged.append(f"{vendor_name}: {trimmed[:120]}")
                found_known = True
                break
        if found_known:
            continue

        # Strategy 2: multi-word capitalized name not in the known universe
        for name in _VENDORISH_NAME_PATTERN.findall(sentence):
            if " " not in name.strip():
                continue  # single words handled by known-vendor lookup
            normalized_name = _normalized_vendor_text(name)
            if not normalized_name or len(normalized_name) <= 2:
                continue
            if normalized_name in grounded_normalized:
                continue
            if normalized_name in non_vendor_skip_words:
                continue
            flagged.append(f"{name}: {trimmed[:120]}")
            break
    return flagged


def _build_report(
    *,
    findings: list[GateFinding],
    word_count: int,
    min_words: int,
    target_words: int,
    quote_count: int,
    policy: QualityPolicy | None,
) -> QualityReport:
    blockers = [f for f in findings if f.severity == GateSeverity.BLOCKER]
    warnings = [f for f in findings if f.severity == GateSeverity.WARNING]
    blocking_penalty = _threshold(policy, "blocking_penalty")
    warning_penalty = _threshold(policy, "warning_penalty")
    pass_score = _threshold(policy, "pass_score")
    score = max(0, 100 - (blocking_penalty * len(blockers)) - (warning_penalty * len(warnings)))
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
        # Legacy-shape mirrors so the wrapper has a 1-step conversion.
        # ``*_issues``/``warnings`` carry rendered messages; ``*_codes``
        # carry the structured ``GateFinding.code`` values so consumers
        # can match on a stable identifier instead of a brittle prefix
        # in the message string.
        "blocking_issues": tuple(f.message for f in blockers),
        "blocking_codes": tuple(f.code for f in blockers),
        "warnings": tuple(f.message for f in warnings),
        "warning_codes": tuple(f.code for f in warnings),
        "quote_count": quote_count,
        "word_count": word_count,
        "min_words_required": min_words,
        "target_words": target_words,
    }
    return QualityReport(
        passed=passed,
        decision=decision,
        findings=tuple(findings),
        metadata=metadata,
    )


__all__ = [
    "evaluate_blog_post",
]
