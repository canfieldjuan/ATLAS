"""Shared helpers for DB-backed Apollo company override records."""

import json
from typing import Any

from asyncpg.exceptions import UndefinedTableError

from ..config import settings
from .company_normalization import normalize_company_name


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def _normalize_domains(domains: list[str]) -> list[str]:
    cleaned = [d.strip().lower().removeprefix("www.") for d in domains if d and d.strip()]
    return _dedupe_strings(cleaned)


def normalize_override_record(
    company_name_raw: str,
    search_names: list[str],
    domains: list[str],
) -> dict[str, Any]:
    company_name_norm = normalize_company_name(company_name_raw)
    normalized_search_names = _dedupe_strings([company_name_raw, *search_names])
    normalized_domains = _normalize_domains(domains)
    if not company_name_norm:
        raise ValueError("company_name_raw must normalize to a non-empty value")
    return {
        "company_name_raw": company_name_raw.strip(),
        "company_name_norm": company_name_norm,
        "search_names": normalized_search_names,
        "domains": normalized_domains,
    }


def _settings_override_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_name, payload in (settings.apollo.company_enrichment_overrides or {}).items():
        if not isinstance(payload, dict):
            continue
        search_names = [str(v) for v in payload.get("search_names", []) if str(v).strip()]
        domains = [str(v) for v in payload.get("domains", []) if str(v).strip()]
        if payload.get("domain"):
            domains.append(str(payload["domain"]))
        records.append(normalize_override_record(str(raw_name), search_names, domains))
    return records


async def fetch_company_override_map(pool) -> dict[str, dict[str, Any]]:
    try:
        rows = await pool.fetch(
            """
            SELECT id, company_name_raw, company_name_norm, search_names, domains, created_at, updated_at
            FROM apollo_company_overrides
            ORDER BY company_name_norm
            """,
        )
    except UndefinedTableError:
        rows = []
    if not rows:
        return {r["company_name_norm"]: r for r in _settings_override_records()}
    return {
        row["company_name_norm"]: {
            "id": row["id"],
            "company_name_raw": row["company_name_raw"],
            "company_name_norm": row["company_name_norm"],
            "search_names": list(row["search_names"] or []),
            "domains": list(row["domains"] or []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    }


async def upsert_company_override(pool, company_name_raw: str, search_names: list[str], domains: list[str]) -> dict[str, Any]:
    record = normalize_override_record(company_name_raw, search_names, domains)
    row = await pool.fetchrow(
        """
        INSERT INTO apollo_company_overrides
            (company_name_raw, company_name_norm, search_names, domains, updated_at)
        VALUES ($1, $2, $3::jsonb, $4::jsonb, NOW())
        ON CONFLICT (company_name_norm) DO UPDATE SET
            company_name_raw = EXCLUDED.company_name_raw,
            search_names = EXCLUDED.search_names,
            domains = EXCLUDED.domains,
            updated_at = NOW()
        RETURNING id, company_name_raw, company_name_norm, search_names, domains, created_at, updated_at
        """,
        record["company_name_raw"],
        record["company_name_norm"],
        json.dumps(record["search_names"]),
        json.dumps(record["domains"]),
    )
    return dict(row)


async def delete_company_override(pool, override_id) -> bool:
    result = await pool.execute("DELETE FROM apollo_company_overrides WHERE id = $1", override_id)
    return result.endswith("DELETE 1")


async def bootstrap_company_overrides_from_settings(pool) -> dict[str, int]:
    imported = 0
    for record in _settings_override_records():
        await upsert_company_override(pool, record["company_name_raw"], record["search_names"], record["domains"])
        imported += 1
    return {"imported": imported}
