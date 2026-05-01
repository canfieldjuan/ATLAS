from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentBudgetDeps:
    contains_any: Any
    coerce_bool: Any
    normalize_compare_text: Any
    normalize_text_list: Any
    combined_source_text: Any
    normalized_low_fidelity_noisy_sources: Any
    text_mentions_name: Any
    has_commercial_context: Any
    has_strong_commercial_context: Any
    has_technical_context: Any
    has_consumer_context: Any
    timeline_ambiguous_vendor_tokens: set[str]
    timeline_ambiguous_vendor_product_context_patterns: tuple[str, ...]
    budget_any_amount_token_re: Any
    budget_price_per_seat_re: Any
    budget_annual_amount_re: Any
    budget_currency_token_re: Any
    budget_seat_count_re: Any
    budget_price_increase_re: Any
    budget_price_increase_detail_re: Any
    budget_annual_period_patterns: tuple[str, ...]
    budget_monthly_period_patterns: tuple[str, ...]
    budget_noise_patterns: tuple[str, ...]
    budget_per_unit_patterns: tuple[str, ...]
    budget_annual_context_patterns: tuple[str, ...]
    budget_commercial_context_patterns: tuple[str, ...]


def budget_match_window(text: str, match: re.Match[str], radius: int = 56) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].lower()


def normalize_budget_value_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.lower()
    text = re.sub(r"\busd\b\s*", "$", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"(?<=[0-9km])a(year|yr)\b", r" a \1", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\bper\s+", "per ", text)
    text = re.sub(r"\ba\s+(year|yr)\b", r"a \1", text)
    text = text.strip()
    return text or None


def normalize_budget_detail_text(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n'\".,;:()[]{}")
    return text or None


def extract_budget_currency_marker(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("usd") or "$" in text:
        return "$"
    if lowered.startswith("eur") or "\u20ac" in text:
        return "\u20ac"
    if lowered.startswith("gbp") or "\u00a3" in text:
        return "\u00a3"
    return None


def extract_single_budget_amount(value: Any, *, deps: EnrichmentBudgetDeps) -> tuple[str | None, float | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    matches = list(deps.budget_any_amount_token_re.finditer(text))
    if len(matches) != 1:
        return None, None
    raw_amount = matches[0].group(0)
    currency = extract_budget_currency_marker(raw_amount)
    amount = extract_numeric_amount(raw_amount)
    if currency is None or amount is None:
        return None, None
    return currency, amount


def extract_budget_period_multiplier(value: Any, *, deps: EnrichmentBudgetDeps) -> int | None:
    text = str(value or "").lower()
    if not text:
        return None
    if deps.contains_any(text, deps.budget_annual_period_patterns):
        return 1
    if deps.contains_any(text, deps.budget_monthly_period_patterns):
        return 12
    return None


def format_annual_budget_amount(currency: str, amount: float) -> str | None:
    if amount <= 0 or amount > 1_000_000_000_000:
        return None
    if amount >= 1_000_000:
        scaled = amount / 1_000_000
        suffix = "m"
    elif amount >= 1_000:
        scaled = amount / 1_000
        suffix = "k"
    else:
        scaled = amount
        suffix = ""

    if abs(scaled - round(scaled)) < 1e-9:
        value_text = str(int(round(scaled)))
    elif scaled >= 100:
        value_text = f"{scaled:.0f}"
    elif scaled >= 10:
        value_text = f"{scaled:.1f}".rstrip("0").rstrip(".")
    else:
        value_text = f"{scaled:.2f}".rstrip("0").rstrip(".")
    return f"{currency}{value_text}{suffix}/year"


def derive_annual_spend_from_unit_price(budget: dict[str, Any], *, deps: EnrichmentBudgetDeps) -> str | None:
    try:
        seat_count = int(budget.get("seat_count"))
    except (TypeError, ValueError):
        return None
    if not (1 <= seat_count <= 1_000_000):
        return None

    currency, unit_amount = extract_single_budget_amount(budget.get("price_per_seat"), deps=deps)
    if currency is None or unit_amount is None:
        return None

    period_multiplier = extract_budget_period_multiplier(budget.get("price_per_seat"), deps=deps)
    if period_multiplier is None:
        return None

    return format_annual_budget_amount(currency, unit_amount * seat_count * period_multiplier)


def has_budget_noise_context(text: str, *, deps: EnrichmentBudgetDeps) -> bool:
    return deps.contains_any(str(text or "").lower(), deps.budget_noise_patterns)


def has_budget_commercial_signal(
    result: dict[str, Any],
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentBudgetDeps,
) -> bool:
    churn = result.get("churn_signals") or {}
    pricing_phrases = deps.normalize_text_list(result.get("pricing_phrases"))
    summary_text = str((source_row or {}).get("summary") or "").strip().lower()
    review_blob = deps.combined_source_text(source_row)
    review_norm = deps.normalize_compare_text(review_blob)
    structured_churn = any((
        bool(churn.get("intent_to_leave")),
        bool(churn.get("actively_evaluating")),
        bool(churn.get("migration_in_progress")),
        bool(churn.get("contract_renewal_mentioned")),
    ))
    if not (pricing_phrases or structured_churn or deps.has_commercial_context(review_norm)):
        return False
    if source_row is None:
        return True

    noisy_sources = deps.normalized_low_fidelity_noisy_sources()
    source = str(source_row.get("source") or "").strip().lower()
    if source not in noisy_sources:
        return True

    vendor_norm = deps.normalize_compare_text(source_row.get("vendor_name"))
    product_norm = deps.normalize_compare_text(source_row.get("product_name"))
    product_hit = (
        bool(source_row.get("product_name"))
        and product_norm != vendor_norm
        and deps.text_mentions_name(review_norm, source_row.get("product_name"))
    )
    vendor_hit = (
        bool(source_row.get("vendor_name"))
        and deps.text_mentions_name(review_norm, source_row.get("vendor_name"))
    )
    if vendor_norm in deps.timeline_ambiguous_vendor_tokens and vendor_hit:
        vendor_hit = deps.contains_any(review_blob, deps.timeline_ambiguous_vendor_product_context_patterns)
    if deps.has_consumer_context(review_norm) and not (product_hit or vendor_hit or structured_churn):
        return False
    if deps.has_technical_context(summary_text, review_norm) and not structured_churn:
        return False
    return any((
        product_hit,
        vendor_hit,
        structured_churn,
        deps.has_strong_commercial_context(review_norm) and not has_budget_noise_context(review_blob, deps=deps),
    ))


def derive_budget_signals(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentBudgetDeps,
) -> dict[str, Any]:
    budget = result.get("budget_signals")
    if not isinstance(budget, dict):
        budget = {}
        result["budget_signals"] = budget

    if not has_budget_commercial_signal(result, source_row, deps=deps):
        return budget

    candidates: list[str] = []
    seen_candidates: set[str] = set()
    for phrase in deps.normalize_text_list(result.get("pricing_phrases")):
        lowered = phrase.lower()
        if lowered not in seen_candidates:
            seen_candidates.add(lowered)
            candidates.append(phrase)
    review_blob = deps.combined_source_text(source_row)
    if review_blob.strip():
        candidates.append(review_blob)

    if not budget.get("price_per_seat"):
        for text in candidates:
            match = deps.budget_price_per_seat_re.search(text)
            if not match:
                continue
            window = budget_match_window(text, match)
            if has_budget_noise_context(window, deps=deps):
                continue
            normalized = normalize_budget_value_text(match.group(0))
            if normalized:
                budget["price_per_seat"] = normalized
                break

    if not budget.get("annual_spend_estimate"):
        for text in candidates:
            match = deps.budget_annual_amount_re.search(text)
            if not match:
                continue
            window = budget_match_window(text, match)
            if has_budget_noise_context(window, deps=deps):
                continue
            normalized = normalize_budget_value_text(match.group(0))
            if normalized:
                budget["annual_spend_estimate"] = normalized
                break
        if not budget.get("annual_spend_estimate"):
            for text in candidates:
                for match in deps.budget_currency_token_re.finditer(text):
                    window = budget_match_window(text, match)
                    if has_budget_noise_context(window, deps=deps):
                        continue
                    if deps.contains_any(window, deps.budget_per_unit_patterns):
                        continue
                    if deps.contains_any(window, deps.budget_monthly_period_patterns):
                        continue
                    if not deps.contains_any(window, deps.budget_annual_context_patterns):
                        continue
                    normalized = normalize_budget_value_text(match.group("raw"))
                    if normalized:
                        budget["annual_spend_estimate"] = normalized
                        break
                if budget.get("annual_spend_estimate"):
                    break

    if not budget.get("seat_count"):
        for text in candidates:
            for match in deps.budget_seat_count_re.finditer(text):
                window = budget_match_window(text, match)
                if has_budget_noise_context(window, deps=deps):
                    continue
                if not deps.contains_any(window, deps.budget_commercial_context_patterns):
                    continue
                try:
                    count = int(match.group("count").replace(",", ""))
                except ValueError:
                    continue
                if 1 <= count <= 1_000_000:
                    budget["seat_count"] = count
                    break
            if budget.get("seat_count"):
                break

    if not budget.get("annual_spend_estimate"):
        derived_annual_spend = derive_annual_spend_from_unit_price(budget, deps=deps)
        if derived_annual_spend:
            budget["annual_spend_estimate"] = derived_annual_spend

    if not deps.coerce_bool(budget.get("price_increase_mentioned")):
        for text in candidates:
            match = deps.budget_price_increase_re.search(text)
            if not match:
                continue
            window = budget_match_window(text, match)
            if has_budget_noise_context(window, deps=deps):
                continue
            if not deps.contains_any(window, deps.budget_commercial_context_patterns):
                continue
            budget["price_increase_mentioned"] = True
            if not budget.get("price_increase_detail"):
                detail_match = deps.budget_price_increase_detail_re.search(text)
                detail = normalize_budget_detail_text(
                    detail_match.group(0) if detail_match else match.group(0)
                )
                if detail:
                    budget["price_increase_detail"] = detail
            break
    elif not budget.get("price_increase_detail"):
        for text in candidates:
            detail_match = deps.budget_price_increase_detail_re.search(text)
            if detail_match:
                detail = normalize_budget_detail_text(detail_match.group(0))
                if detail:
                    budget["price_increase_detail"] = detail
                    break

    return budget


def extract_numeric_amount(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)(?:\s*([km]))?", str(value).lower())
    if not match:
        return None
    amount = float(match.group(1).replace(",", ""))
    suffix = match.group(2)
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return amount


def derive_contract_value_signal(result: dict[str, Any]) -> str:
    budget = result.get("budget_signals") or {}
    reviewer_context = result.get("reviewer_context") or {}
    spend = extract_numeric_amount(budget.get("annual_spend_estimate"))
    seats = budget.get("seat_count")
    try:
        seat_count = int(seats) if seats is not None else 0
    except (TypeError, ValueError):
        seat_count = 0
    segment = str(reviewer_context.get("company_size_segment") or "unknown")
    if spend is not None and spend >= 100000:
        return "enterprise_high"
    if seat_count >= 500 or segment == "enterprise":
        return "enterprise_high"
    if spend is not None and spend >= 25000:
        return "enterprise_mid"
    if seat_count >= 200 or segment == "mid_market":
        return "enterprise_mid"
    if spend is not None and spend >= 5000:
        return "mid_market"
    if seat_count >= 25:
        return "mid_market"
    if segment in {"smb", "startup"}:
        return "smb"
    return "unknown"
