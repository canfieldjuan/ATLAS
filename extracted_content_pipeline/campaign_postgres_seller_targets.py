"""Postgres seller target helpers for the standalone campaign product."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


JsonDict = dict[str, Any]

_SELLER_TYPES = frozenset(
    {"private_label", "manufacturer", "agency", "wholesale_reseller"}
)
_SELLER_STATUSES = frozenset({"active", "paused", "unsubscribed", "bounced"})


@dataclass(frozen=True)
class SellerTargetListResult:
    targets: tuple[JsonDict, ...]
    total: int
    limit: int
    offset: int
    filters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "targets": [dict(row) for row in self.targets],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "filters": dict(self.filters),
        }


async def list_seller_targets(
    pool: Any,
    *,
    status: str | None = None,
    seller_type: str | None = None,
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
    seller_targets_table: str = "seller_targets",
) -> SellerTargetListResult:
    table = _identifier(seller_targets_table)
    normalized_limit = _normalize_limit(limit)
    normalized_offset = _normalize_offset(offset)
    where: list[str] = []
    params: list[Any] = []
    filters: dict[str, Any] = {}

    if status:
        filters["status"] = _normalize_status(status)
        params.append(filters["status"])
        where.append(f"status = ${len(params)}")
    if seller_type:
        filters["seller_type"] = _normalize_seller_type(seller_type)
        params.append(filters["seller_type"])
        where.append(f"seller_type = ${len(params)}")
    if category:
        filters["category"] = _clean(category)
        params.append(filters["category"])
        where.append(f"${len(params)} = ANY(categories)")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = await pool.fetch(
        f"""
        SELECT *
          FROM {table}
          {where_sql}
         ORDER BY created_at DESC
         LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *params,
        normalized_limit,
        normalized_offset,
    )
    total = await pool.fetchval(
        f"SELECT COUNT(*) FROM {table} {where_sql}",
        *params,
    )
    return SellerTargetListResult(
        targets=tuple(_serializable_row(_row_dict(row)) for row in rows),
        total=int(total or 0),
        limit=normalized_limit,
        offset=normalized_offset,
        filters=filters,
    )


async def create_seller_target(
    pool: Any,
    *,
    seller_name: str | None = None,
    company_name: str | None = None,
    email: str | None = None,
    seller_type: str = "private_label",
    categories: Sequence[str] = (),
    storefront_url: str | None = None,
    notes: str | None = None,
    source: str = "manual",
    seller_targets_table: str = "seller_targets",
) -> JsonDict:
    clean_seller_name = _clean(seller_name) or None
    clean_company_name = _clean(company_name) or None
    if not clean_seller_name and not clean_company_name:
        raise ValueError("at least one of seller_name or company_name is required")
    row = await pool.fetchrow(
        f"""
        INSERT INTO {_identifier(seller_targets_table)} (
            seller_name, company_name, email, seller_type,
            categories, storefront_url, notes, source
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *
        """,
        clean_seller_name,
        clean_company_name,
        _clean(email) or None,
        _normalize_seller_type(seller_type),
        list(_clean_sequence(categories)),
        _clean(storefront_url) or None,
        _clean(notes) or None,
        _clean_required(source, "source"),
    )
    return _serializable_row(_row_dict(row))


async def get_seller_target(
    pool: Any,
    *,
    target_id: str,
    seller_targets_table: str = "seller_targets",
) -> JsonDict | None:
    row = await pool.fetchrow(
        f"SELECT * FROM {_identifier(seller_targets_table)} WHERE id = $1",
        _normalize_uuid(target_id),
    )
    return _serializable_row(_row_dict(row)) if row else None


async def update_seller_target(
    pool: Any,
    *,
    target_id: str,
    values: Mapping[str, Any],
    seller_targets_table: str = "seller_targets",
) -> JsonDict | None:
    normalized = _normalized_update_values(values)
    if not normalized:
        raise ValueError("no fields to update")

    updates: list[str] = []
    params: list[Any] = []
    for field, value in normalized.items():
        params.append(value)
        updates.append(f"{field} = ${len(params)}")
    params.append(_normalize_uuid(target_id))
    result = await pool.execute(
        f"""
        UPDATE {_identifier(seller_targets_table)}
           SET {', '.join(updates)}, updated_at = NOW()
         WHERE id = ${len(params)}
        """,
        *params,
    )
    if _affected_rows(result) == 0:
        return None
    return await get_seller_target(
        pool,
        target_id=target_id,
        seller_targets_table=seller_targets_table,
    )


async def delete_seller_target(
    pool: Any,
    *,
    target_id: str,
    seller_targets_table: str = "seller_targets",
) -> bool:
    result = await pool.execute(
        f"DELETE FROM {_identifier(seller_targets_table)} WHERE id = $1",
        _normalize_uuid(target_id),
    )
    return _affected_rows(result) > 0


def _normalized_update_values(values: Mapping[str, Any]) -> JsonDict:
    out: JsonDict = {}
    for field in ("seller_name", "company_name", "email", "storefront_url", "notes"):
        if field in values:
            out[field] = _clean(values.get(field)) or None
    if "seller_type" in values:
        out["seller_type"] = _normalize_seller_type(values.get("seller_type"))
    if "status" in values:
        out["status"] = _normalize_status(values.get("status"))
    if "categories" in values:
        out["categories"] = list(_clean_sequence(values.get("categories") or ()))
    return out


def _normalize_seller_type(value: Any) -> str:
    seller_type = _clean_required(value, "seller_type")
    if seller_type not in _SELLER_TYPES:
        allowed = ", ".join(sorted(_SELLER_TYPES))
        raise ValueError(f"unsupported seller_type: {seller_type}; expected one of {allowed}")
    return seller_type


def _normalize_status(value: Any) -> str:
    status = _clean_required(value, "status")
    if status not in _SELLER_STATUSES:
        allowed = ", ".join(sorted(_SELLER_STATUSES))
        raise ValueError(f"unsupported status: {status}; expected one of {allowed}")
    return status


def _normalize_uuid(value: Any) -> str:
    try:
        return str(UUID(_clean_required(value, "target_id")))
    except ValueError as exc:
        raise ValueError("invalid target_id UUID") from exc


def _normalize_limit(value: Any) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return limit


def _normalize_offset(value: Any) -> int:
    offset = int(value)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    return offset


def _clean_sequence(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(item for item in (_clean(value) for value in values) if item)


def _clean_required(value: Any, field_name: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _affected_rows(result: Any) -> int:
    parts = str(result or "").split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def _serializable_row(row: Mapping[str, Any]) -> JsonDict:
    return {key: _json_ready(value) for key, value in row.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


__all__ = [
    "SellerTargetListResult",
    "create_seller_target",
    "delete_seller_target",
    "get_seller_target",
    "list_seller_targets",
    "update_seller_target",
]
