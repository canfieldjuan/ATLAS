"""Conservative ownership backfill for legacy vendor_targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .tracked_vendor_sources import (
    VENDOR_TARGET_SOURCE_TYPE,
    upsert_tracked_vendor_source,
)

CLAIM_REASON_DIRECT_SOURCE = "direct_vendor_target_source"
CLAIM_REASON_EXACT_OWN = "exact_own_vendor_match"
CLAIM_REASON_EXACT_COMPETITOR = "exact_competitor_vendor_match"
CLAIM_REASON_COMPETITOR_OVERLAP = "challenger_competitor_overlap"


@dataclass(frozen=True)
class VendorTargetClaimCandidate:
    target_id: str
    company_name: str
    target_mode: str
    contact_email: str | None
    account_id: str
    tracked_vendor_name: str
    tracked_vendor_names: list[str]
    track_mode: str
    claim_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "company_name": self.company_name,
            "target_mode": self.target_mode,
            "contact_email": self.contact_email,
            "account_id": self.account_id,
            "tracked_vendor_name": self.tracked_vendor_name,
            "tracked_vendor_names": self.tracked_vendor_names,
            "track_mode": self.track_mode,
            "claim_reason": self.claim_reason,
        }


def _build_legacy_scope(
    alias: str,
    *,
    company_name: str | None,
    target_mode: str | None,
) -> tuple[str, list[Any]]:
    clauses = [f"{alias}.account_id IS NULL"]
    params: list[Any] = []
    if company_name:
        params.append(company_name)
        clauses.append(f"LOWER({alias}.company_name) = LOWER(${len(params)})")
    if target_mode:
        params.append(target_mode)
        clauses.append(f"{alias}.target_mode = ${len(params)}")
    return " AND ".join(clauses), params


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    claim_reason: str,
) -> VendorTargetClaimCandidate:
    tracked_vendor_name = str(row.get("tracked_vendor_name") or "")
    tracked_vendor_names_raw = row.get("tracked_vendor_names") or []
    tracked_vendor_names: list[str] = []
    if isinstance(tracked_vendor_names_raw, list):
        for value in tracked_vendor_names_raw:
            vendor_name = str(value or "").strip()
            if vendor_name and vendor_name not in tracked_vendor_names:
                tracked_vendor_names.append(vendor_name)
    if tracked_vendor_name and tracked_vendor_name not in tracked_vendor_names:
        tracked_vendor_names.insert(0, tracked_vendor_name)
    return VendorTargetClaimCandidate(
        target_id=str(row.get("target_id") or ""),
        company_name=str(row.get("company_name") or ""),
        target_mode=str(row.get("target_mode") or ""),
        contact_email=str(row.get("contact_email")) if row.get("contact_email") else None,
        account_id=str(row.get("account_id") or ""),
        tracked_vendor_name=tracked_vendor_name or (tracked_vendor_names[0] if tracked_vendor_names else ""),
        tracked_vendor_names=tracked_vendor_names,
        track_mode=str(row.get("track_mode") or ""),
        claim_reason=claim_reason,
    )


async def _fetch_legacy_targets(
    pool,
    *,
    company_name: str | None,
    target_mode: str | None,
) -> list[dict[str, Any]]:
    where_sql, params = _build_legacy_scope(
        "vt",
        company_name=company_name,
        target_mode=target_mode,
    )
    rows = await pool.fetch(
        f"""
        SELECT vt.id::text AS target_id,
               vt.company_name,
               vt.target_mode,
               vt.contact_email
        FROM vendor_targets vt
        WHERE {where_sql}
        ORDER BY LOWER(vt.company_name), vt.target_mode, vt.id
        """,
        *params,
    )
    return [dict(row) for row in rows]


async def _fetch_direct_source_candidates(
    pool,
    *,
    company_name: str | None,
    target_mode: str | None,
) -> list[dict[str, Any]]:
    where_sql, params = _build_legacy_scope(
        "vt",
        company_name=company_name,
        target_mode=target_mode,
    )
    source_idx = len(params) + 1
    rows = await pool.fetch(
        f"""
        WITH matches AS (
            SELECT vt.id::text AS target_id,
                   vt.company_name,
                   vt.target_mode,
                   vt.contact_email,
                   COUNT(DISTINCT tvs.account_id) AS account_count,
                   MIN(tvs.account_id::text) AS account_id,
                   MIN(tvs.vendor_name) AS tracked_vendor_name,
                   MIN(tvs.track_mode) AS track_mode
            FROM vendor_targets vt
            JOIN tracked_vendor_sources tvs
              ON tvs.source_type = ${source_idx}
             AND tvs.source_key = vt.id::text
             AND LOWER(tvs.vendor_name) = LOWER(vt.company_name)
            WHERE {where_sql}
            GROUP BY vt.id, vt.company_name, vt.target_mode, vt.contact_email
        )
        SELECT target_id, company_name, target_mode, contact_email,
               account_id, tracked_vendor_name, track_mode
        FROM matches
        WHERE account_count = 1
        ORDER BY LOWER(company_name), target_mode, target_id
        """,
        *params,
        VENDOR_TARGET_SOURCE_TYPE,
    )
    return [dict(row) for row in rows]


async def _fetch_exact_track_candidates(
    pool,
    *,
    target_mode: str,
    track_mode: str,
    company_name: str | None,
) -> list[dict[str, Any]]:
    where_sql, params = _build_legacy_scope(
        "vt",
        company_name=company_name,
        target_mode=target_mode,
    )
    track_idx = len(params) + 1
    rows = await pool.fetch(
        f"""
        WITH matches AS (
            SELECT vt.id::text AS target_id,
                   vt.company_name,
                   vt.target_mode,
                   vt.contact_email,
                   COUNT(DISTINCT tv.account_id) AS account_count,
                   MIN(tv.account_id::text) AS account_id,
                   MIN(tv.vendor_name) AS tracked_vendor_name
            FROM vendor_targets vt
            JOIN tracked_vendors tv
              ON LOWER(tv.vendor_name) = LOWER(vt.company_name)
             AND tv.track_mode = ${track_idx}
            WHERE {where_sql}
            GROUP BY vt.id, vt.company_name, vt.target_mode, vt.contact_email
        )
        SELECT target_id, company_name, target_mode, contact_email,
               account_id, tracked_vendor_name
        FROM matches
        WHERE account_count = 1
        ORDER BY LOWER(company_name), target_mode, target_id
        """,
        *params,
        track_mode,
    )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["track_mode"] = track_mode
        normalized.append(item)
    return normalized


async def _fetch_competitor_overlap_candidates(
    pool,
    *,
    company_name: str | None,
) -> list[dict[str, Any]]:
    where_sql, params = _build_legacy_scope(
        "vt",
        company_name=company_name,
        target_mode="challenger_intel",
    )
    track_idx = len(params) + 1
    rows = await pool.fetch(
        f"""
        WITH overlap_matches AS (
            SELECT vt.id::text AS target_id,
                   vt.company_name,
                   vt.target_mode,
                   vt.contact_email,
                   tv.account_id::text AS account_id,
                   ARRAY_AGG(DISTINCT tv.vendor_name ORDER BY tv.vendor_name) AS tracked_vendor_names
            FROM vendor_targets vt
            JOIN tracked_vendors tv
              ON tv.track_mode = ${track_idx}
             AND EXISTS (
                 SELECT 1
                 FROM unnest(COALESCE(vt.competitors_tracked, ARRAY[]::text[])) AS c(name)
                 WHERE LOWER(c.name) = LOWER(tv.vendor_name)
            )
            WHERE {where_sql}
            GROUP BY vt.id, vt.company_name, vt.target_mode, vt.contact_email, tv.account_id
        ),
        account_counts AS (
            SELECT target_id, COUNT(*) AS account_count
            FROM overlap_matches
            GROUP BY target_id
        )
        SELECT o.target_id, o.company_name, o.target_mode, o.contact_email,
               o.account_id, o.tracked_vendor_names
        FROM overlap_matches o
        JOIN account_counts ac ON ac.target_id = o.target_id
        WHERE ac.account_count = 1
        ORDER BY LOWER(o.company_name), o.target_mode, o.target_id
        """,
        *params,
        "competitor",
    )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        names = item.get("tracked_vendor_names") or []
        item["tracked_vendor_name"] = names[0] if isinstance(names, list) and names else ""
        item["track_mode"] = "competitor"
        normalized.append(item)
    return normalized


def _select_candidates(
    legacy_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    own_rows: list[dict[str, Any]],
    competitor_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> list[VendorTargetClaimCandidate]:
    ordered_sources = [
        (direct_rows, CLAIM_REASON_DIRECT_SOURCE),
        (own_rows, CLAIM_REASON_EXACT_OWN),
        (competitor_rows, CLAIM_REASON_EXACT_COMPETITOR),
        (overlap_rows, CLAIM_REASON_COMPETITOR_OVERLAP),
    ]
    candidates_by_id: dict[str, VendorTargetClaimCandidate] = {}
    legacy_ids = {str(row.get("target_id") or "") for row in legacy_rows}
    for rows, reason in ordered_sources:
        for row in rows:
            candidate = _candidate_from_row(row, claim_reason=reason)
            if candidate.target_id not in legacy_ids:
                continue
            candidates_by_id.setdefault(candidate.target_id, candidate)
    return sorted(
        candidates_by_id.values(),
        key=lambda item: (
            item.company_name.lower(),
            item.target_mode.lower(),
            item.target_id,
        ),
    )


async def plan_legacy_vendor_target_account_backfill(
    pool,
    *,
    company_name: str | None = None,
    target_mode: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    legacy_rows = await _fetch_legacy_targets(
        pool,
        company_name=company_name,
        target_mode=target_mode,
    )
    direct_rows = await _fetch_direct_source_candidates(
        pool,
        company_name=company_name,
        target_mode=target_mode,
    )
    own_rows = await _fetch_exact_track_candidates(
        pool,
        target_mode="vendor_retention",
        track_mode="own",
        company_name=company_name,
    )
    competitor_rows = await _fetch_exact_track_candidates(
        pool,
        target_mode="challenger_intel",
        track_mode="competitor",
        company_name=company_name,
    )
    overlap_rows = await _fetch_competitor_overlap_candidates(
        pool,
        company_name=company_name,
    )
    candidates = _select_candidates(
        legacy_rows,
        direct_rows,
        own_rows,
        competitor_rows,
        overlap_rows,
    )
    selected = candidates if limit is None else candidates[: max(limit, 0)]
    return {
        "filters": {
            "company_name": company_name,
            "target_mode": target_mode,
            "limit": limit,
        },
        "legacy_targets": len(legacy_rows),
        "direct_source_matches": len(direct_rows),
        "exact_own_matches": len(own_rows),
        "exact_competitor_matches": len(competitor_rows),
        "challenger_overlap_matches": len(overlap_rows),
        "claimable_targets_total": len(candidates),
        "claimable_targets_selected": len(selected),
        "skipped_targets": len(legacy_rows) - len(candidates),
        "candidates": [candidate.to_dict() for candidate in selected],
    }


async def apply_legacy_vendor_target_account_backfill(
    pool,
    *,
    company_name: str | None = None,
    target_mode: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    plan = await plan_legacy_vendor_target_account_backfill(
        pool,
        company_name=company_name,
        target_mode=target_mode,
        limit=limit,
    )
    if not plan["candidates"]:
        plan["applied"] = 0
        plan["already_claimed"] = 0
        return plan

    applied: list[dict[str, Any]] = []
    already_claimed = 0
    async with pool.transaction() as conn:
        for candidate in plan["candidates"]:
            row = await conn.fetchrow(
                """
                UPDATE vendor_targets
                SET account_id = $1,
                    updated_at = NOW()
                WHERE id = $2::uuid
                  AND account_id IS NULL
                RETURNING id
                """,
                candidate["account_id"],
                candidate["target_id"],
            )
            if not row:
                already_claimed += 1
                continue
            tracked_vendor_names = candidate.get("tracked_vendor_names") or []
            for tracked_vendor_name in tracked_vendor_names:
                await upsert_tracked_vendor_source(
                    conn,
                    candidate["account_id"],
                    tracked_vendor_name,
                    source_type=VENDOR_TARGET_SOURCE_TYPE,
                    source_key=candidate["target_id"],
                    track_mode=candidate["track_mode"],
                )
            applied.append(candidate)

    plan["applied"] = len(applied)
    plan["already_claimed"] = already_claimed
    plan["applied_candidates"] = applied
    return plan
