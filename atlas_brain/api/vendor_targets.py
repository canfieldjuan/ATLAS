"""
REST endpoints for Vendor/Challenger target management.

Vendor targets represent our actual customers -- the companies we sell
churn intelligence (vendor_retention) or intent leads (challenger_intel) to.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, optional_auth, require_auth
from ..services.scraping.target_provisioning import (
    provision_vendor_onboarding_targets,
)
from ..services.tracked_vendor_sources import (
    delete_vendor_target_sources_for_all_accounts,
    replace_vendor_target_sources,
)
from ..services.vendor_registry import (
    resolve_known_vendor_name,
    resolve_vendor_name,
)
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.vendor_targets")

router = APIRouter(prefix="/b2b/vendor-targets", tags=["b2b-vendor-targets"])


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _normalize_account_id(value) -> str | None:
    return str(value) if value else None


def _shape_vendor_target(row, user: AuthUser | None = None) -> dict[str, object]:
    target = dict(row)
    account_id = _normalize_account_id(target.get("account_id"))
    target["account_id"] = account_id
    if user:
        target["ownership_scope"] = (
            "owned" if account_id == user.account_id else "legacy_global"
        )
    else:
        target["ownership_scope"] = "account_owned" if account_id else "legacy_global"
    return target


def _append_list_scope(
    conditions: list[str],
    params: list,
    idx: int,
    user: AuthUser | None,
    *,
    include_legacy_global: bool,
) -> tuple[int, int | None]:
    if not user:
        return idx, None
    scope_idx = idx
    if include_legacy_global:
        conditions.append(f"(account_id = ${idx} OR account_id IS NULL)")
    else:
        conditions.append(f"account_id = ${idx}")
    params.append(user.account_id)
    return idx + 1, scope_idx


async def _sync_vendor_target_tracking(
    pool,
    user: AuthUser | None,
    target_id: str,
    company_name: str,
    competitors_tracked: list[str] | None,
) -> dict[str, object] | None:
    if not user:
        return None

    canonical_company = await resolve_vendor_name(company_name)
    vendors_to_track: list[dict[str, str]] = [
        {
            "vendor_name": canonical_company,
            "track_mode": "own",
        }
    ]
    skipped_competitors: list[str] = []
    seen_vendor_names: set[str] = {canonical_company.lower()}
    for raw_competitor in competitors_tracked or []:
        canonical_competitor = await resolve_known_vendor_name(raw_competitor)
        if not canonical_competitor:
            skipped_competitors.append(raw_competitor)
            continue
        lowered = canonical_competitor.lower()
        if lowered in seen_vendor_names:
            continue
        seen_vendor_names.add(lowered)
        vendors_to_track.append(
            {
                "vendor_name": canonical_competitor,
                "track_mode": "competitor",
            }
        )

    sync_result = await replace_vendor_target_sources(
        pool,
        user.account_id,
        target_id,
        vendors_to_track,
    )

    logger.info(
        "Synced %d vendors to tracked_vendors for account %s (skipped_competitors=%d)",
        len(sync_result["synced_vendors"]),
        user.account_id,
        len(skipped_competitors),
    )
    sync_result["skipped_competitors"] = skipped_competitors
    return sync_result


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VendorTargetCreate(BaseModel):
    company_name: str = Field(..., max_length=200)
    target_mode: str = Field(..., max_length=30)
    product_category: str | None = Field(None, max_length=200)
    scrape_target_slugs: dict[str, str] = Field(default_factory=dict)
    contact_name: str | None = Field(None, max_length=200)
    contact_email: str | None = Field(None, max_length=254)
    contact_role: str | None = Field(None, max_length=100)
    products_tracked: list[str] | None = Field(None, max_length=50)
    competitors_tracked: list[str] | None = Field(None, max_length=50)
    tier: str = Field("report", max_length=30)
    status: str = Field("active", max_length=30)
    notes: str | None = Field(None, max_length=5000)


class VendorTargetUpdate(BaseModel):
    company_name: str | None = Field(None, max_length=200)
    target_mode: str | None = Field(None, max_length=30)
    contact_name: str | None = Field(None, max_length=200)
    contact_email: str | None = Field(None, max_length=254)
    contact_role: str | None = Field(None, max_length=100)
    products_tracked: list[str] | None = Field(None, max_length=50)
    competitors_tracked: list[str] | None = Field(None, max_length=50)
    tier: str | None = Field(None, max_length=30)
    status: str | None = Field(None, max_length=30)
    notes: str | None = Field(None, max_length=5000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_vendor_targets(
    target_mode: Optional[str] = Query(None, description="Filter by target mode"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search company name"),
    include_legacy_global: bool = Query(
        True,
        description="Include legacy global targets when authenticated",
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: AuthUser | None = Depends(optional_auth),
):
    """List all vendor/challenger targets."""
    pool = _pool_or_503()

    conditions = []
    params: list = []
    idx = 1

    if target_mode:
        conditions.append(f"target_mode = ${idx}")
        params.append(target_mode)
        idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if search:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(search)
        idx += 1

    idx, scope_idx = _append_list_scope(
        conditions,
        params,
        idx,
        user,
        include_legacy_global=include_legacy_global,
    )

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order_by = "ORDER BY created_at DESC"
    if scope_idx is not None:
        order_by = (
            f"ORDER BY CASE WHEN account_id = ${scope_idx} THEN 0 ELSE 1 END, "
            "created_at DESC"
        )

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, target_mode, contact_name, contact_email,
               contact_role, products_tracked, competitors_tracked, tier,
               status, notes, account_id, created_at, updated_at
        FROM vendor_targets
        {where}
        {order_by}
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params, limit, offset,
    )

    count = await pool.fetchval(
        f"SELECT COUNT(*) FROM vendor_targets {where}",
        *params,
    )

    return {
        "targets": [_shape_vendor_target(r, user) for r in rows],
        "count": count,
    }


@router.post("", status_code=201)
async def create_vendor_target(
    body: VendorTargetCreate,
    user: AuthUser | None = Depends(optional_auth),
):
    """Add a new vendor or challenger target."""
    pool = _pool_or_503()

    if body.target_mode not in ("vendor_retention", "challenger_intel"):
        raise HTTPException(status_code=400, detail="target_mode must be 'vendor_retention' or 'challenger_intel'")

    row = await pool.fetchrow(
        """
        INSERT INTO vendor_targets (
            account_id, company_name, target_mode, contact_name, contact_email,
            contact_role, products_tracked, competitors_tracked,
            tier, status, notes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id, company_name, target_mode, contact_name, contact_email,
                  contact_role, products_tracked, competitors_tracked, tier,
                  status, notes, account_id, created_at, updated_at
        """,
        user.account_id if user else None,
        body.company_name,
        body.target_mode,
        body.contact_name,
        body.contact_email,
        body.contact_role,
        body.products_tracked,
        body.competitors_tracked,
        body.tier,
        body.status,
        body.notes,
    )

    canonical_company = await resolve_vendor_name(body.company_name)
    tracked_vendor_sync = await _sync_vendor_target_tracking(
        pool,
        user,
        str(row.get("id") or ""),
        canonical_company,
        body.competitors_tracked,
    )

    try:
        scrape_provisioning = await provision_vendor_onboarding_targets(
            pool,
            canonical_company,
            product_category=body.product_category,
            source_slug_overrides=body.scrape_target_slugs,
            dry_run=False,
        )
    except Exception as exc:  # pragma: no cover - defensive operational guard
        logger.warning(
            "Auto scrape provisioning failed for vendor target %s: %s",
            canonical_company,
            exc,
        )
        scrape_provisioning = {
            "status": "error",
            "requested": 0,
            "applied": 0,
            "matched_vendors": [],
            "unmatched_vendors": [canonical_company.lower()],
            "actions": [],
        }

    result = _shape_vendor_target(row, user)
    if tracked_vendor_sync is not None:
        result["tracked_vendor_sync"] = tracked_vendor_sync
    result["scrape_provisioning"] = scrape_provisioning
    return result


@router.get("/{target_id}")
async def get_vendor_target(
    target_id: UUID,
    user: AuthUser | None = Depends(optional_auth),
):
    """Get a single vendor/challenger target with intelligence summary."""
    pool = _pool_or_503()

    if user:
        row = await pool.fetchrow(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier,
                   status, notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE id = $1
              AND (account_id = $2 OR account_id IS NULL)
            """,
            target_id,
            user.account_id,
        )
    else:
        row = await pool.fetchrow(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier,
                   status, notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE id = $1
            """,
            target_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    target = _shape_vendor_target(row, user)

    # Fetch campaign stats for this target
    campaign_stats = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_campaigns,
            COUNT(*) FILTER (WHERE status = 'draft') AS drafts,
            COUNT(*) FILTER (WHERE status = 'sent') AS sent,
            COUNT(*) FILTER (WHERE status = 'approved') AS approved,
            MAX(created_at) AS last_campaign_at
        FROM b2b_campaigns
        WHERE LOWER(company_name) = LOWER($1)
          AND target_mode = $2
        """,
        target["company_name"],
        target["target_mode"],
    )
    target["campaign_stats"] = dict(campaign_stats) if campaign_stats else {}

    # Fetch recent reports for this target
    report_type = "vendor_retention" if target["target_mode"] == "vendor_retention" else "challenger_intel"
    reports = await pool.fetch(
        """
        SELECT id, report_date, report_type, executive_summary, created_at
        FROM b2b_intelligence
        WHERE vendor_filter ILIKE '%' || $1 || '%'
          AND report_type = $2
        ORDER BY report_date DESC
        LIMIT 5
        """,
        target["company_name"],
        report_type,
    )
    target["recent_reports"] = [dict(r) for r in reports]

    return target


@router.put("/{target_id}")
async def update_vendor_target(
    target_id: UUID,
    body: VendorTargetUpdate,
    user: AuthUser | None = Depends(optional_auth),
):
    """Update a vendor/challenger target."""
    pool = _pool_or_503()

    # Build dynamic SET clause -- model_fields_set tracks which fields were
    # explicitly provided in the request body (including explicit nulls).
    updates = []
    params: list = []
    idx = 1

    for field in [
        "company_name", "target_mode", "contact_name", "contact_email",
        "contact_role", "products_tracked", "competitors_tracked",
        "tier", "status", "notes",
    ]:
        if field in body.model_fields_set:
            updates.append(f"{field} = ${idx}")
            params.append(getattr(body, field))
            idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "target_mode" in body.model_fields_set:
        if body.target_mode is None:
            raise HTTPException(status_code=400, detail="target_mode cannot be null")
        if body.target_mode not in ("vendor_retention", "challenger_intel"):
            raise HTTPException(status_code=400, detail="target_mode must be 'vendor_retention' or 'challenger_intel'")

    if user:
        existing = await pool.fetchrow(
            """
            SELECT id, account_id
            FROM vendor_targets
            WHERE id = $1
              AND (account_id = $2 OR account_id IS NULL)
            """,
            target_id,
            user.account_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Target not found")
        if existing.get("account_id") is None:
            updates.append(f"account_id = ${idx}")
            params.append(user.account_id)
            idx += 1

    updates.append(f"updated_at = NOW()")

    row = await pool.fetchrow(
        f"""
        UPDATE vendor_targets
        SET {', '.join(updates)}
        WHERE id = ${idx}
        RETURNING id, company_name, target_mode, contact_name, contact_email,
                  contact_role, products_tracked, competitors_tracked, tier,
                  status, notes, account_id, created_at, updated_at
        """,
        *params, target_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    result = _shape_vendor_target(row, user)
    tracked_vendor_sync = await _sync_vendor_target_tracking(
        pool,
        user,
        str(result.get("id") or ""),
        str(result.get("company_name") or ""),
        result.get("competitors_tracked"),
    )
    if tracked_vendor_sync is not None:
        result["tracked_vendor_sync"] = tracked_vendor_sync
    return result


@router.post("/{target_id}/claim")
async def claim_vendor_target(
    target_id: UUID,
    user: AuthUser = Depends(require_auth),
):
    """Explicitly claim a legacy global vendor target into account scope."""
    pool = _pool_or_503()

    async with pool.transaction() as conn:
        existing = await conn.fetchrow(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier,
                   status, notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE id = $1
              AND (account_id = $2 OR account_id IS NULL)
            """,
            target_id,
            user.account_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Target not found")

        already_claimed = existing.get("account_id") == user.account_id
        row = existing
        if not already_claimed:
            duplicate = await conn.fetchrow(
                """
                SELECT id
                FROM vendor_targets
                WHERE account_id = $1
                  AND LOWER(company_name) = LOWER($2)
                  AND target_mode = $3
                  AND id <> $4
                LIMIT 1
                """,
                user.account_id,
                existing["company_name"],
                existing["target_mode"],
                target_id,
            )
            if duplicate:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "An owned target already exists for this company and "
                        "target_mode. Resolve the duplicate before claiming "
                        "the legacy global row."
                    ),
                )
            row = await conn.fetchrow(
                """
                UPDATE vendor_targets
                SET account_id = $1,
                    updated_at = NOW()
                WHERE id = $2
                  AND account_id IS NULL
                RETURNING id, company_name, target_mode, contact_name,
                          contact_email, contact_role, products_tracked,
                          competitors_tracked, tier, status, notes,
                          account_id, created_at, updated_at
                """,
                user.account_id,
                target_id,
            )
            if not row:
                raise HTTPException(
                    status_code=409,
                    detail="Legacy global target was claimed concurrently.",
                )

    result = _shape_vendor_target(row, user)
    tracked_vendor_sync = await _sync_vendor_target_tracking(
        pool,
        user,
        str(result.get("id") or ""),
        str(result.get("company_name") or ""),
        result.get("competitors_tracked"),
    )
    result["already_claimed"] = already_claimed
    if tracked_vendor_sync is not None:
        result["tracked_vendor_sync"] = tracked_vendor_sync
    return result


@router.delete("/{target_id}")
async def delete_vendor_target(
    target_id: UUID,
    user: AuthUser | None = Depends(optional_auth),
):
    """Remove a vendor/challenger target."""
    pool = _pool_or_503()

    if user:
        existing = await pool.fetchrow(
            """
            SELECT id, account_id
            FROM vendor_targets
            WHERE id = $1
              AND (account_id = $2 OR account_id IS NULL)
            """,
            target_id,
            user.account_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Target not found")
        if existing.get("account_id") is None:
            raise HTTPException(
                status_code=409,
                detail="Legacy global target must be claimed by an authenticated update before deletion.",
            )
        row = await pool.fetchrow(
            """
            DELETE FROM vendor_targets
            WHERE id = $1 AND account_id = $2
            RETURNING id
            """,
            target_id,
            user.account_id,
        )
    else:
        row = await pool.fetchrow(
            """
            DELETE FROM vendor_targets
            WHERE id = $1
            RETURNING id
            """,
            target_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    tracked_vendor_cleanup = await delete_vendor_target_sources_for_all_accounts(
        pool,
        str(target_id),
    )
    return {
        "deleted": True,
        "tracked_vendor_cleanup": tracked_vendor_cleanup,
    }


@router.post("/{target_id}/generate-report")
async def generate_target_report(
    target_id: UUID,
    user: AuthUser | None = Depends(optional_auth),
):
    """Generate an intelligence report for this target (vendor or challenger)."""
    pool = _pool_or_503()

    if user:
        row = await pool.fetchrow(
            """
            SELECT company_name, target_mode
            FROM vendor_targets
            WHERE id = $1
              AND (account_id = $2 OR account_id IS NULL)
            """,
            target_id,
            user.account_id,
        )
    else:
        row = await pool.fetchrow(
            "SELECT company_name, target_mode FROM vendor_targets WHERE id = $1",
            target_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    if row["target_mode"] == "vendor_retention":
        from ..autonomous.tasks.b2b_churn_intelligence import generate_vendor_report

        report = await generate_vendor_report(pool, row["company_name"])
        if not report:
            raise HTTPException(status_code=404, detail="No churn signals found for this vendor")
    elif row["target_mode"] == "challenger_intel":
        from ..autonomous.tasks.b2b_churn_intelligence import generate_challenger_report

        report = await generate_challenger_report(pool, row["company_name"])
        if not report:
            raise HTTPException(status_code=404, detail="No competitor mentions found for this challenger")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown target_mode: {row['target_mode']}")

    return report
