from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentBuyerAuthorityDeps:
    sanitize_reviewer_title: Any
    coerce_bool: Any
    coerce_json_dict: Any
    contains_any: Any
    role_type_aliases: dict[str, str]
    role_level_aliases: dict[str, str]
    champion_reviewer_title_pattern: Any
    evaluator_reviewer_title_pattern: Any
    exec_role_text_pattern: Any
    director_role_text_pattern: Any
    manager_role_text_pattern: Any
    ic_role_text_pattern: Any
    commercial_decision_text_pattern: Any
    exec_reviewer_title_pattern: Any
    manager_decision_title_pattern: Any
    economic_buyer_text_patterns: tuple[Any, ...]
    champion_text_patterns: tuple[Any, ...]
    evaluator_text_patterns: tuple[Any, ...]
    end_user_text_patterns: tuple[Any, ...]
    post_purchase_review_sources: set[str]
    post_purchase_usage_patterns: tuple[str, ...]


def canonical_role_type(value: Any, *, deps: EnrichmentBuyerAuthorityDeps) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    if not raw:
        return "unknown"
    return deps.role_type_aliases.get(raw, "unknown")


def normalize_role_title_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.strip().lower())


def clean_reviewer_title_for_role_inference(value: Any, *, deps: EnrichmentBuyerAuthorityDeps) -> str:
    title = deps.sanitize_reviewer_title(value) or ""
    if not title or len(title) > 120:
        return ""
    return title


def canonical_role_level(value: Any, *, deps: EnrichmentBuyerAuthorityDeps) -> str:
    raw = normalize_role_title_key(value)
    return deps.role_level_aliases.get(raw, "unknown")


def combined_source_text(source_row: dict[str, Any] | None) -> str:
    if not isinstance(source_row, dict):
        return ""
    parts = [
        str(source_row.get("summary") or ""),
        str(source_row.get("review_text") or ""),
        str(source_row.get("pros") or ""),
        str(source_row.get("cons") or ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def infer_role_level_from_text(
    reviewer_title: Any,
    source_row: dict[str, Any] | None,
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> str:
    title = clean_reviewer_title_for_role_inference(reviewer_title, deps=deps)
    if title:
        canonical = canonical_role_level(title, deps=deps)
        if canonical != "unknown":
            return canonical
        if re.search(r"\b(cfo|ceo|coo|cio|cto|cro|cmo|chief|founder|owner|president)\b", title, re.I):
            return "executive"
        if re.search(r"\b(vp\b|vice president|svp|evp|director|head of)\b", title, re.I):
            return "director"
        if deps.champion_reviewer_title_pattern.search(title):
            return "manager"
        if deps.evaluator_reviewer_title_pattern.search(title):
            return "ic"
    source_text = combined_source_text(source_row)
    if not source_text:
        return "unknown"
    if deps.exec_role_text_pattern.search(source_text):
        return "executive"
    if deps.director_role_text_pattern.search(source_text):
        return "director"
    if deps.manager_role_text_pattern.search(source_text):
        return "manager"
    if deps.ic_role_text_pattern.search(source_text):
        return "ic"
    return "unknown"


def has_manager_level_decision_context(
    result: dict[str, Any],
    source_row: dict[str, Any] | None,
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> bool:
    buyer_authority = deps.coerce_json_dict(result.get("buyer_authority"))
    if deps.coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return True

    budget = deps.coerce_json_dict(result.get("budget_signals"))
    if any(budget.get(field) for field in ("annual_spend_estimate", "price_per_seat", "price_increase_detail")):
        return True
    if deps.coerce_bool(budget.get("price_increase_mentioned")) is True:
        return True

    timeline = deps.coerce_json_dict(result.get("timeline"))
    if timeline.get("contract_end") or timeline.get("evaluation_deadline"):
        return True

    churn = deps.coerce_json_dict(result.get("churn_signals"))
    if any(
        deps.coerce_bool(churn.get(field)) is True
        for field in ("actively_evaluating", "migration_in_progress", "contract_renewal_mentioned")
    ):
        return True

    return bool(deps.commercial_decision_text_pattern.search(combined_source_text(source_row)))


def infer_decision_maker(
    result: dict[str, Any],
    source_row: dict[str, Any] | None,
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> bool:
    reviewer_context = deps.coerce_json_dict(result.get("reviewer_context"))
    buyer_authority = deps.coerce_json_dict(result.get("buyer_authority"))
    role_level = canonical_role_level(reviewer_context.get("role_level"), deps=deps)
    if role_level in {"executive", "director"}:
        return True
    if deps.coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return True
    if canonical_role_type(buyer_authority.get("role_type"), deps=deps) == "economic_buyer":
        return True

    title = clean_reviewer_title_for_role_inference((source_row or {}).get("reviewer_title"), deps=deps)
    if title and deps.exec_reviewer_title_pattern.search(title):
        return True
    if title and deps.manager_decision_title_pattern.search(title):
        return has_manager_level_decision_context(result, source_row, deps=deps)
    return False


def infer_buyer_role_type_from_text(
    buyer_authority: dict[str, Any],
    source_row: dict[str, Any] | None,
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> str:
    if not isinstance(source_row, dict):
        return "unknown"
    if str(source_row.get("content_type") or "").strip().lower() == "insider_account":
        return "unknown"
    source_text = combined_source_text(source_row)
    if not source_text:
        return "unknown"
    for pattern in deps.economic_buyer_text_patterns:
        if pattern.search(source_text):
            return "economic_buyer"
    for pattern in deps.champion_text_patterns:
        if pattern.search(source_text):
            return "champion"
    for pattern in deps.evaluator_text_patterns:
        if pattern.search(source_text):
            return "evaluator"
    buying_stage = str(buyer_authority.get("buying_stage") or "").strip().lower()
    for pattern in deps.end_user_text_patterns:
        if pattern.search(source_text):
            return "evaluator" if buying_stage in {"evaluation", "active_purchase"} else "end_user"
    return "unknown"


def infer_buyer_role_type(
    buyer_authority: dict[str, Any],
    reviewer_context: dict[str, Any] | None,
    reviewer_title: Any,
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> str:
    ctx = reviewer_context if isinstance(reviewer_context, dict) else {}
    role_level = str(ctx.get("role_level") or "").strip().lower()
    buying_stage = str(buyer_authority.get("buying_stage") or "").strip().lower()
    if deps.coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return "economic_buyer"
    if deps.coerce_bool(ctx.get("decision_maker")) is True:
        return "economic_buyer"
    if role_level in {"executive", "director"}:
        return "economic_buyer"
    title = clean_reviewer_title_for_role_inference(reviewer_title, deps=deps)
    if title and deps.exec_reviewer_title_pattern.search(title):
        return "economic_buyer"
    if role_level == "manager":
        return "champion"
    if title and deps.champion_reviewer_title_pattern.search(title):
        return "champion"
    if role_level == "ic" and buying_stage in {"evaluation", "active_purchase"}:
        return "evaluator"
    if title and deps.evaluator_reviewer_title_pattern.search(title):
        return "evaluator" if buying_stage in {"evaluation", "active_purchase"} else "end_user"
    if role_level == "ic":
        return "end_user"
    return infer_buyer_role_type_from_text(buyer_authority, source_row, deps=deps)


def has_post_purchase_signal(
    source_row: dict[str, Any],
    review_blob: str,
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> bool:
    source = str(source_row.get("source") or "").strip().lower()
    if source in deps.post_purchase_review_sources:
        return True
    return deps.contains_any(review_blob, deps.post_purchase_usage_patterns)


def derive_buyer_authority_fields(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentBuyerAuthorityDeps,
) -> tuple[str, bool, str]:
    reviewer_context = result.get("reviewer_context") or {}
    churn = result.get("churn_signals") or {}
    role_level = str(reviewer_context.get("role_level") or "unknown")
    decision_maker = bool(reviewer_context.get("decision_maker"))
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    if decision_maker or role_level in {"executive", "director"}:
        role_type = "economic_buyer"
    elif churn.get("actively_evaluating"):
        role_type = "evaluator"
    elif role_level == "manager":
        role_type = "champion"
    elif role_level == "ic":
        role_type = "end_user"
    else:
        role_type = "unknown"
    executive_sponsor_mentioned = deps.contains_any(
        review_blob,
        ("ceo", "cfo", "cto", "coo", "leadership", "executive team", "vp approved", "signed off"),
    )
    if churn.get("contract_renewal_mentioned") or churn.get("renewal_timing"):
        buying_stage = "renewal_decision"
    elif churn.get("actively_evaluating") or churn.get("migration_in_progress"):
        buying_stage = "evaluation"
    elif decision_maker and deps.contains_any(review_blob, ("approved", "signed off", "purchased", "bought")):
        buying_stage = "active_purchase"
    elif has_post_purchase_signal(source_row, review_blob, deps=deps):
        buying_stage = "post_purchase"
    else:
        buying_stage = "unknown"
    return role_type, executive_sponsor_mentioned, buying_stage
