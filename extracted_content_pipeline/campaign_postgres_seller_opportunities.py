"""Build Amazon seller campaign opportunities from seller targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
import json
import re
from typing import Any


JsonDict = dict[str, Any]
DEFAULT_SELLER_TARGET_MODE = "amazon_seller"


@dataclass(frozen=True)
class SellerOpportunityPreparationResult:
    prepared: int = 0
    skipped: int = 0
    replaced: int = 0
    target_mode: str = DEFAULT_SELLER_TARGET_MODE
    target_ids: tuple[str, ...] = field(default_factory=tuple)
    categories: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prepared": self.prepared,
            "skipped": self.skipped,
            "replaced": self.replaced,
            "target_mode": self.target_mode,
            "target_ids": list(self.target_ids),
            "categories": list(self.categories),
        }


async def prepare_seller_campaign_opportunities(
    pool: Any,
    *,
    account_id: str | None = None,
    category: str | None = None,
    seller_status: str = "active",
    limit: int = 100,
    replace_existing: bool = False,
    target_mode: str = DEFAULT_SELLER_TARGET_MODE,
    seller_targets_table: str = "seller_targets",
    category_snapshots_table: str = "category_intelligence_snapshots",
    opportunities_table: str = "campaign_opportunities",
) -> SellerOpportunityPreparationResult:
    """Prepare normalized campaign opportunities for active seller targets."""

    normalized_limit = _normalize_limit(limit)
    rows = await _read_seller_intelligence_rows(
        pool,
        category=category,
        seller_status=seller_status,
        limit=normalized_limit,
        seller_targets_table=seller_targets_table,
        category_snapshots_table=category_snapshots_table,
    )
    opportunities = [
        opportunity
        for opportunity in (
            _opportunity_from_row(row, account_id=account_id, target_mode=target_mode)
            for row in rows
        )
        if opportunity
    ]
    skipped = max(0, len(rows) - len(opportunities))
    replaced = 0
    if replace_existing and opportunities:
        replaced = await _delete_existing_opportunities(
            pool,
            opportunities,
            account_id=account_id,
            target_mode=target_mode,
            opportunities_table=opportunities_table,
        )
    for opportunity in opportunities:
        await _insert_opportunity(
            pool,
            opportunity,
            opportunities_table=opportunities_table,
        )
    return SellerOpportunityPreparationResult(
        prepared=len(opportunities),
        skipped=skipped,
        replaced=replaced,
        target_mode=target_mode,
        target_ids=tuple(str(item["target_id"]) for item in opportunities),
        categories=tuple(_unique(item["product_category"] for item in opportunities)),
    )


async def _read_seller_intelligence_rows(
    pool: Any,
    *,
    category: str | None,
    seller_status: str,
    limit: int,
    seller_targets_table: str,
    category_snapshots_table: str,
) -> Sequence[Mapping[str, Any]]:
    params: list[Any] = []
    params.append(_clean_required(seller_status, "seller_status"))
    where = ["st.status = $1", "array_length(st.categories, 1) IS NOT NULL"]
    if category:
        params.append(_clean_required(category, "category"))
        where.append(f"${len(params)} = ANY(st.categories)")
    params.append(limit)
    rows = await pool.fetch(
        f"""
        SELECT
            st.id AS seller_target_id,
            st.seller_name,
            st.company_name,
            st.email,
            st.seller_type,
            st.categories,
            st.storefront_url,
            st.notes,
            cat.category,
            snap.snapshot_date,
            snap.total_reviews,
            snap.total_brands,
            snap.total_products,
            snap.top_pain_points,
            snap.feature_gaps,
            snap.competitive_flows,
            snap.brand_health,
            snap.safety_signals,
            snap.manufacturing_insights,
            snap.top_root_causes
          FROM {_identifier(seller_targets_table)} st
          JOIN LATERAL unnest(st.categories) AS cat(category) ON TRUE
          JOIN LATERAL (
              SELECT *
                FROM {_identifier(category_snapshots_table)} cis
               WHERE cis.category = cat.category
                 AND cis.subcategory IS NULL
               ORDER BY cis.snapshot_date DESC, cis.created_at DESC
               LIMIT 1
          ) snap ON TRUE
         WHERE {' AND '.join(where)}
         ORDER BY snap.total_reviews DESC NULLS LAST, st.created_at ASC
         LIMIT ${len(params)}
        """,
        *params,
    )
    return tuple(_row_dict(row) for row in rows)


def _opportunity_from_row(
    row: Mapping[str, Any],
    *,
    account_id: str | None,
    target_mode: str,
) -> JsonDict | None:
    seller_target_id = _clean(row.get("seller_target_id"))
    category = _clean(row.get("category"))
    seller_name = _clean(row.get("seller_name")) or _clean(row.get("company_name"))
    if not seller_target_id or not category or not seller_name:
        return None
    category_intelligence = _category_intelligence(row)
    raw_payload = {
        "seller_target": {
            "id": seller_target_id,
            "seller_name": _clean(row.get("seller_name")),
            "company_name": _clean(row.get("company_name")),
            "seller_type": _clean(row.get("seller_type")),
            "categories": _json_sequence(row.get("categories")),
            "storefront_url": _clean(row.get("storefront_url")),
            "notes": _clean(row.get("notes")),
        },
        "category": category,
        "product_category": category,
        "category_intelligence": category_intelligence,
    }
    return {
        "account_id": _clean(account_id) or None,
        "target_id": _seller_target_key(seller_target_id, category),
        "target_mode": _clean_required(target_mode, "target_mode"),
        "company_name": seller_name,
        "vendor_name": None,
        "contact_name": _clean(row.get("seller_name")) or None,
        "contact_email": _clean(row.get("email")) or None,
        "contact_title": _clean(row.get("seller_type")) or None,
        "product_category": category,
        "opportunity_score": _number_or_none(row.get("total_reviews")),
        "urgency_score": _urgency_score(category_intelligence),
        "pain_points": _pain_points(category_intelligence),
        "competitors": _competitors(category_intelligence),
        "evidence": _evidence(category_intelligence),
        "raw_payload": raw_payload,
    }


async def _delete_existing_opportunities(
    pool: Any,
    opportunities: Sequence[Mapping[str, Any]],
    *,
    account_id: str | None,
    target_mode: str,
    opportunities_table: str,
) -> int:
    target_ids = [str(item["target_id"]) for item in opportunities if item.get("target_id")]
    if not target_ids:
        return 0
    params: list[Any] = [target_mode, target_ids]
    where = ["target_mode = $1", "target_id = ANY($2::text[])"]
    if account_id:
        params.append(account_id)
        where.append(f"account_id = ${len(params)}")
    result = await pool.execute(
        f"""
        DELETE FROM {_identifier(opportunities_table)}
         WHERE {' AND '.join(where)}
        """,
        *params,
    )
    return _affected_rows(result)


async def _insert_opportunity(
    pool: Any,
    opportunity: Mapping[str, Any],
    *,
    opportunities_table: str,
) -> None:
    await pool.execute(
        f"""
        INSERT INTO {_identifier(opportunities_table)} (
            account_id, target_id, target_mode, company_name, vendor_name,
            contact_name, contact_email, contact_title, opportunity_score,
            urgency_score, pain_points, competitors, evidence, raw_payload
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb
        )
        """,
        opportunity.get("account_id"),
        opportunity.get("target_id"),
        opportunity.get("target_mode"),
        opportunity.get("company_name"),
        opportunity.get("vendor_name"),
        opportunity.get("contact_name"),
        opportunity.get("contact_email"),
        opportunity.get("contact_title"),
        opportunity.get("opportunity_score"),
        opportunity.get("urgency_score"),
        _jsonb(opportunity.get("pain_points") or []),
        _jsonb(opportunity.get("competitors") or []),
        _jsonb(opportunity.get("evidence") or []),
        _jsonb(opportunity.get("raw_payload") or {}),
    )


def _category_intelligence(row: Mapping[str, Any]) -> JsonDict:
    return {
        "category_stats": {
            "total_reviews": int(row.get("total_reviews") or 0),
            "total_brands": int(row.get("total_brands") or 0),
            "total_products": int(row.get("total_products") or 0),
            "snapshot_date": _json_ready(row.get("snapshot_date")),
        },
        "top_pain_points": _json_sequence(row.get("top_pain_points")),
        "feature_gaps": _json_sequence(row.get("feature_gaps")),
        "competitive_flows": _json_sequence(row.get("competitive_flows")),
        "brand_health": _json_sequence(row.get("brand_health")),
        "safety_signals": _json_sequence(row.get("safety_signals")),
        "manufacturing_insights": _json_sequence(row.get("manufacturing_insights")),
        "top_root_causes": _json_sequence(row.get("top_root_causes")),
    }


def _seller_target_key(seller_target_id: str, category: str) -> str:
    return f"seller:{seller_target_id}:{_slug(category)}"


def _urgency_score(category_intelligence: Mapping[str, Any]) -> int:
    pain_points = _json_sequence(category_intelligence.get("top_pain_points"))
    feature_gaps = _json_sequence(category_intelligence.get("feature_gaps"))
    safety_signals = _json_sequence(category_intelligence.get("safety_signals"))
    return sum(
        _count_from_item(item)
        for item in [*pain_points[:5], *feature_gaps[:5], *safety_signals[:3]]
    )


def _pain_points(category_intelligence: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for item in _json_sequence(category_intelligence.get("top_pain_points")):
        if isinstance(item, Mapping):
            text = _clean(item.get("complaint") or item.get("cause") or item.get("request"))
        else:
            text = _clean(item)
        if text and text not in out:
            out.append(text)
    for item in _json_sequence(category_intelligence.get("top_root_causes")):
        text = _clean(item.get("cause") if isinstance(item, Mapping) else item)
        if text and text not in out:
            out.append(text)
    return out


def _competitors(category_intelligence: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for item in _json_sequence(category_intelligence.get("competitive_flows")):
        if isinstance(item, Mapping):
            values = (
                item.get("from_brand"),
                item.get("to_brand"),
                item.get("competitor"),
                item.get("mentioned_brand"),
            )
        else:
            values = (item,)
        for value in values:
            text = _clean(value)
            if text and text not in out:
                out.append(text)
    return out


def _evidence(category_intelligence: Mapping[str, Any]) -> list[JsonDict]:
    evidence: list[JsonDict] = []
    for source_key in (
        "top_pain_points",
        "feature_gaps",
        "safety_signals",
        "manufacturing_insights",
        "competitive_flows",
    ):
        for item in _json_sequence(category_intelligence.get(source_key))[:5]:
            if isinstance(item, Mapping):
                evidence.append({"source": source_key, **dict(item)})
            else:
                evidence.append({"source": source_key, "text": _clean(item)})
    return evidence


def _count_from_item(item: Any) -> int:
    if not isinstance(item, Mapping):
        return 1 if _clean(item) else 0
    for key in ("count", "flagged_count", "review_count", "mentions"):
        value = _number_or_none(item.get(key))
        if value is not None:
            return int(value)
    return 1


def _json_sequence(value: Any) -> list[Any]:
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if value in (None, "", {}, []):
        return []
    return [value]


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _unique(values: Sequence[Any] | Any) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _clean_required(value: Any, field_name: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _normalize_limit(value: Any) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return limit


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    text = _clean(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _slug(value: Any) -> str:
    text = _clean(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "category"


def _affected_rows(result: Any) -> int:
    parts = str(result or "").split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


__all__ = [
    "DEFAULT_SELLER_TARGET_MODE",
    "SellerOpportunityPreparationResult",
    "prepare_seller_campaign_opportunities",
]
