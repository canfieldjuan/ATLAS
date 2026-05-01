from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentUrgencyDeps:
    contains_any: Any
    normalize_text_list: Any


def derive_urgency_indicators(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentUrgencyDeps,
    price_complaint: bool = False,
) -> dict[str, bool]:
    churn = result.get("churn_signals") or {}
    budget = result.get("budget_signals") or {}
    timeline = result.get("timeline") or {}
    competitors = result.get("competitors_mentioned") or []
    complaints = result.get("specific_complaints") or []
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    price_text = " ".join(deps.normalize_text_list(result.get("pricing_phrases"))).lower()
    recommendation_text = " ".join(deps.normalize_text_list(result.get("recommendation_language"))).lower()
    named_alt_with_reason = any(
        isinstance(comp, dict) and comp.get("name") and comp.get("reason_detail")
        for comp in competitors
    )
    return {
        "intent_to_leave_signal": bool(churn.get("intent_to_leave")),
        "actively_evaluating_signal": bool(churn.get("actively_evaluating")),
        "migration_in_progress_signal": bool(churn.get("migration_in_progress")),
        "explicit_cancel_language": bool(churn.get("intent_to_leave")) and deps.contains_any(
            review_blob, ("cancel", "not renewing", "terminate", "ending our contract")
        ),
        "active_migration_language": bool(churn.get("migration_in_progress")) or "migrat" in review_blob,
        "active_evaluation_language": bool(churn.get("actively_evaluating")) or deps.contains_any(
            review_blob, ("evaluating", "shortlist", "poc", "comparing options")
        ),
        "completed_switch_language": deps.contains_any(
            review_blob, ("switched to", "moved to", "replaced with")
        ),
        "comparison_shopping_language": deps.contains_any(
            review_blob, ("vs ", "alternative", "which should", "looking for options")
        ),
        "named_alternative_with_reason": named_alt_with_reason,
        "frustration_without_alternative": bool(complaints) and not competitors,
        "price_pressure_language": bool(price_complaint) or deps.contains_any(
            review_blob + " " + price_text,
            (
                "price increase",
                "pricing policy",
                "too expensive",
                "costs will constantly increase",
                "forced to change provider",
                "unjustified expenses",
            ),
        ),
        "reconsideration_language": deps.contains_any(
            review_blob,
            (
                "reconsidering",
                "considering changing",
                "considering switching",
                "considering swtiching",
                "forced to change provider",
                "considering another tool",
            ),
        ),
        "dollar_amount_mentioned": bool(budget.get("annual_spend_estimate") or budget.get("price_per_seat"))
        or "$" in price_text,
        "timeline_mentioned": bool(
            churn.get("renewal_timing")
            or timeline.get("contract_end")
            or timeline.get("evaluation_deadline")
        ),
        "decision_maker_language": bool((result.get("reviewer_context") or {}).get("decision_maker"))
        or deps.contains_any(
            review_blob + " " + recommendation_text,
            ("i decided", "we approved", "signed off", "our team approved"),
        ),
    }
