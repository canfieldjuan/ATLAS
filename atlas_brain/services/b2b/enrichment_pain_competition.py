from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentPainCompetitionDeps:
    normalize_text_list: Any
    normalize_pain_category: Any
    normalize_company_name: Any
    pain_patterns: dict[str, re.Pattern[str]]
    pain_derivation_fields: tuple[str, ...]
    competitor_recovery_patterns: tuple[str, ...]
    competitor_recovery_blocklist: set[str]
    generic_competitor_tokens: set[str]
    competitor_context_patterns: tuple[str, ...]


def _pain_scores(texts: list[str], *, deps: EnrichmentPainCompetitionDeps) -> dict[str, int]:
    scores = {category: 0 for category in deps.pain_patterns}
    for text in texts:
        for category, pattern in deps.pain_patterns.items():
            if pattern.search(text):
                scores[category] += 1
    return scores


def _primary_reason_category(*texts: str, deps: EnrichmentPainCompetitionDeps) -> str | None:
    normalized = [text for text in texts if text]
    if not normalized:
        return None
    scored = _pain_scores(normalized, deps=deps)
    ranked = sorted(
        ((score, category) for category, score in scored.items() if score > 0),
        reverse=True,
    )
    return ranked[0][1] if ranked else None


def subject_vendor_phrase_texts(
    result: dict[str, Any],
    field: str,
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> list[str]:
    raw = result.get(field) or []
    if not raw:
        return []

    from atlas_brain.autonomous.tasks._b2b_phrase_metadata import is_v2_tagged, phrase_metadata_map

    if not is_v2_tagged(result):
        return deps.normalize_text_list(raw)

    meta = phrase_metadata_map(result)
    texts: list[str] = []
    for index, value in enumerate(raw):
        phrase = str(value or "").strip()
        if not phrase:
            continue
        row = meta.get((field, index)) or {}
        if row.get("subject") == "subject_vendor":
            texts.append(phrase)
    return texts


def derive_pain_categories(
    result: dict[str, Any],
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> list[dict[str, str]]:
    from atlas_brain.autonomous.tasks._b2b_phrase_metadata import is_v2_tagged, phrase_metadata_map

    if is_v2_tagged(result):
        meta = phrase_metadata_map(result)
        weighted_items: list[tuple[str, float]] = []
        for field in deps.pain_derivation_fields:
            raw = result.get(field) or []
            for index, value in enumerate(raw):
                phrase = str(value or "").strip()
                if not phrase:
                    continue
                row = meta.get((field, index)) or {}
                if row.get("subject") != "subject_vendor":
                    continue
                polarity = row.get("polarity")
                if polarity == "negative":
                    weighted_items.append((phrase, 1.0))
                elif polarity == "mixed":
                    weighted_items.append((phrase, 0.5))
        if not weighted_items:
            return []
        scores: dict[str, float] = {category: 0.0 for category in deps.pain_patterns}
        for text, weight in weighted_items:
            for category, pattern in deps.pain_patterns.items():
                if pattern.search(text):
                    scores[category] += weight
        ranked: list[tuple[float, str]] = [
            (score, category)
            for category, score in scores.items()
            if score > 0
        ]
    else:
        texts = (
            deps.normalize_text_list(result.get("specific_complaints"))
            + deps.normalize_text_list(result.get("pricing_phrases"))
            + deps.normalize_text_list(result.get("feature_gaps"))
            + deps.normalize_text_list(result.get("quotable_phrases"))
        )
        if not texts:
            return []
        scored = _pain_scores(texts, deps=deps)
        ranked = [
            (float(score), category)
            for category, score in scored.items()
            if score > 0
        ]

    ranked.sort(reverse=True)
    if not ranked:
        return [{"category": "overall_dissatisfaction", "severity": "primary"}]
    categories = [{"category": ranked[0][1], "severity": "primary"}]
    for _score, category in ranked[1:3]:
        if category != categories[0]["category"]:
            categories.append({"category": category, "severity": "secondary"})
    return categories


def _count_corroborating_signals(result: dict[str, Any]) -> int:
    count = 0
    churn = result.get("churn_signals")
    if isinstance(churn, dict):
        for key in ("intent_to_leave", "actively_evaluating", "migration_in_progress"):
            if churn.get(key) is True:
                count += 1
    if result.get("would_recommend") is False:
        count += 1
    sentiment = result.get("sentiment_trajectory")
    if isinstance(sentiment, dict):
        direction = str(sentiment.get("direction") or "").strip().lower()
        if direction in ("consistently_negative", "declining"):
            count += 1
    return count


def _count_pain_phrase_matches(
    result: dict[str, Any],
    pain_category: str,
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> int:
    pattern = deps.pain_patterns.get(pain_category)
    if pattern is None:
        return 0

    from atlas_brain.autonomous.tasks._b2b_phrase_metadata import is_v2_tagged, phrase_metadata_map

    count = 0
    if is_v2_tagged(result):
        meta = phrase_metadata_map(result)
        for field in deps.pain_derivation_fields:
            for index, value in enumerate(result.get(field) or []):
                phrase = str(value or "").strip()
                if not phrase:
                    continue
                row = meta.get((field, index)) or {}
                if row.get("subject") != "subject_vendor":
                    continue
                if row.get("polarity") not in ("negative", "mixed"):
                    continue
                if pattern.search(phrase):
                    count += 1
    else:
        texts: list[str] = []
        for field in deps.pain_derivation_fields:
            texts.extend(deps.normalize_text_list(result.get(field)))
        for text in texts:
            if pattern.search(text):
                count += 1
    return count


def compute_pain_confidence(
    result: dict[str, Any],
    pain_category: str,
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> str:
    normalized = deps.normalize_pain_category(pain_category)
    signal_count = _count_corroborating_signals(result)

    if normalized == "overall_dissatisfaction":
        if signal_count >= 2:
            return "strong"
        if signal_count >= 1:
            return "weak"
        return "none"

    phrase_count = _count_pain_phrase_matches(result, normalized, deps=deps)
    if phrase_count >= 2:
        return "strong"
    if phrase_count >= 1 and signal_count >= 1:
        return "weak"
    return "none"


def demote_primary_pain(result: dict[str, Any], demoted_category: str) -> None:
    if demoted_category == "overall_dissatisfaction":
        return
    existing = result.get("pain_categories")
    if not isinstance(existing, list):
        existing = []
    new_list: list[dict[str, str]] = [
        {"category": "overall_dissatisfaction", "severity": "primary"}
    ]
    appended_demoted = False
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        category = str(entry.get("category") or "").strip().lower()
        if not category:
            continue
        if category == "overall_dissatisfaction":
            continue
        if category == demoted_category and not appended_demoted:
            new_list.append({"category": demoted_category, "severity": "secondary"})
            appended_demoted = True
        elif category != demoted_category:
            new_list.append(
                {"category": category, "severity": entry.get("severity", "secondary")}
            )
    if not appended_demoted:
        new_list.append({"category": demoted_category, "severity": "secondary"})
    result["pain_categories"] = new_list


def _is_generic_competitor_name(name: str, *, deps: EnrichmentPainCompetitionDeps) -> bool:
    normalized = deps.normalize_company_name(name) or str(name or "").strip().lower()
    if not normalized:
        return True
    if normalized in deps.competitor_recovery_blocklist:
        return True
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", str(name or ""))
        if token
    ]
    return bool(tokens) and all(token in deps.generic_competitor_tokens for token in tokens)


def _has_named_competitor_context(
    name: str,
    source_row: dict[str, Any],
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> bool:
    candidate = str(name or "").strip()
    if not candidate:
        return False
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    name_lower = candidate.lower()
    for match in re.finditer(re.escape(name_lower), review_blob):
        start = max(0, match.start() - 96)
        end = min(len(review_blob), match.end() + 96)
        window = review_blob[start:end]
        if any(pattern in window for pattern in deps.competitor_context_patterns):
            return True
    return False


def recover_competitor_mentions(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> list[dict[str, Any]]:
    existing = [
        dict(comp) for comp in (result.get("competitors_mentioned") or [])
        if isinstance(comp, dict) and str(comp.get("name") or "").strip()
    ]
    if not existing and not any(source_row.get(field) for field in ("summary", "review_text", "pros", "cons")):
        return existing

    incumbent_norm = deps.normalize_company_name(str(source_row.get("vendor_name") or "")) or ""
    seen = {
        (
            deps.normalize_company_name(str(comp.get("name") or ""))
            or str(comp.get("name") or "").strip().lower()
        ): comp
        for comp in existing
    }

    recovery_blob = " ".join(
        [str(source_row.get(field) or "") for field in ("summary", "review_text", "pros", "cons")]
        + deps.normalize_text_list(result.get("quotable_phrases"))
    )

    for pattern in deps.competitor_recovery_patterns:
        for match in re.finditer(pattern, recovery_blob):
            candidate = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9.]+$", "", match.group(1).strip())
            if not candidate:
                continue
            normalized = deps.normalize_company_name(candidate) or candidate.lower()
            if not normalized or normalized == incumbent_norm:
                continue
            if normalized in deps.competitor_recovery_blocklist:
                continue
            generic_tokens = [
                token.lower()
                for token in re.findall(r"[A-Za-z0-9]+", candidate)
                if token
            ]
            if generic_tokens and all(token in deps.generic_competitor_tokens for token in generic_tokens):
                continue
            if normalized in seen:
                continue
            seen[normalized] = {"name": candidate}

    return list(seen.values())


def derive_competitor_annotations(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentPainCompetitionDeps,
) -> list[dict[str, Any]]:
    comps = []
    churn = result.get("churn_signals") or {}
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    for comp in result.get("competitors_mentioned", []) or []:
        if not isinstance(comp, dict):
            continue
        merged = dict(comp)
        name = str(comp.get("name") or "").strip()
        if _is_generic_competitor_name(name, deps=deps):
            continue
        comp_blob = " ".join(
            [name]
            + deps.normalize_text_list(comp.get("features"))
            + [str(comp.get("reason_detail") or "")]
        ).lower()
        named_context = _has_named_competitor_context(name, source_row, deps=deps)
        switch_patterns = (
            f"switched to {name.lower()}",
            f"moved to {name.lower()}",
            f"replaced with {name.lower()}",
            f"migrating to {name.lower()}",
        )
        reverse_patterns = (
            f"moved from {name.lower()}",
            f"switched from {name.lower()}",
        )
        evaluation_patterns = (
            f"evaluating {name.lower()}",
            f"looking at {name.lower()}",
            f"considering {name.lower()}",
            f"shortlist {name.lower()}",
            f"poc with {name.lower()}",
        )
        if any(pattern in review_blob for pattern in reverse_patterns):
            evidence_type = "reverse_flow"
        elif any(pattern in review_blob for pattern in switch_patterns):
            evidence_type = "explicit_switch"
        elif any(pattern in review_blob for pattern in evaluation_patterns) or churn.get("actively_evaluating"):
            evidence_type = "active_evaluation"
        elif merged.get("reason_detail") or merged.get("features"):
            evidence_type = "implied_preference"
        elif named_context:
            evidence_type = "implied_preference"
        else:
            evidence_type = "neutral_mention"
        confidence = "low"
        if evidence_type == "explicit_switch":
            confidence = "high" if churn.get("migration_in_progress") or churn.get("renewal_timing") else "medium"
        elif evidence_type == "active_evaluation":
            confidence = "medium" if merged.get("reason_detail") else "low"
        elif evidence_type == "implied_preference" and merged.get("reason_detail"):
            confidence = "medium"
        merged["evidence_type"] = evidence_type
        merged["displacement_confidence"] = confidence
        merged["reason_category"] = _primary_reason_category(
            str(merged.get("reason_detail") or ""),
            comp_blob,
            deps=deps,
        )
        if (
            merged["evidence_type"] == "neutral_mention"
            and merged["displacement_confidence"] == "low"
            and not str(merged.get("reason_detail") or "").strip()
            and not str(merged.get("reason_category") or "").strip()
            and not deps.normalize_text_list(merged.get("features"))
            and not named_context
        ):
            continue
        comps.append(merged)
    return comps
