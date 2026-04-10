"""Competitive-set repository for scoped synthesis control."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError
from ..models import CompetitiveSet, CompetitiveSetRun

logger = logging.getLogger("atlas.storage.competitive_set")


class CompetitiveSetRepository:
    """CRUD and scheduler helpers for B2B competitive sets."""

    async def create(
        self,
        *,
        account_id: UUID,
        name: str,
        focal_vendor_name: str,
        competitor_vendor_names: list[str],
        active: bool = True,
        refresh_mode: str = "manual",
        refresh_interval_hours: Optional[int] = None,
        vendor_synthesis_enabled: bool = True,
        pairwise_enabled: bool = True,
        category_council_enabled: bool = False,
        asymmetry_enabled: bool = False,
    ) -> CompetitiveSet:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("create competitive set")

        set_id = uuid4()
        now = datetime.now(timezone.utc)
        competitor_vendor_names = self._normalize_vendor_names(competitor_vendor_names)

        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    """
                    INSERT INTO b2b_competitive_sets (
                        id, account_id, name, focal_vendor_name, active,
                        refresh_mode, refresh_interval_hours,
                        vendor_synthesis_enabled, pairwise_enabled,
                        category_council_enabled, asymmetry_enabled,
                        created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $12)
                    """,
                    set_id,
                    account_id,
                    name,
                    focal_vendor_name,
                    active,
                    refresh_mode,
                    refresh_interval_hours,
                    vendor_synthesis_enabled,
                    pairwise_enabled,
                    category_council_enabled,
                    asymmetry_enabled,
                    now,
                )
                if competitor_vendor_names:
                    await conn.executemany(
                        """
                        INSERT INTO b2b_competitive_set_vendors (
                            competitive_set_id, vendor_name, sort_order
                        )
                        VALUES ($1, $2, $3)
                        """,
                        [
                            (set_id, vendor_name, idx)
                            for idx, vendor_name in enumerate(competitor_vendor_names)
                        ],
                    )
            created = await self.get_by_id(set_id)
            if not created:
                raise DatabaseOperationError("create competitive set", Exception("No row returned"))
            return created
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as exc:
            raise DatabaseOperationError("create competitive set", exc)

    async def get_by_id(self, competitive_set_id: UUID) -> Optional[CompetitiveSet]:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get competitive set")
        try:
            rows = await pool.fetch(
                """
                SELECT cs.*,
                       csv.vendor_name AS competitor_vendor_name,
                       csv.sort_order AS competitor_sort_order
                FROM b2b_competitive_sets cs
                LEFT JOIN b2b_competitive_set_vendors csv
                  ON csv.competitive_set_id = cs.id
                WHERE cs.id = $1
                ORDER BY csv.sort_order ASC, csv.vendor_name ASC
                """,
                competitive_set_id,
            )
            return self._rows_to_competitive_set(rows)
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("get competitive set", exc)

    async def get_by_id_for_account(
        self,
        competitive_set_id: UUID,
        account_id: UUID,
    ) -> Optional[CompetitiveSet]:
        result = await self.get_by_id(competitive_set_id)
        if not result or result.account_id != account_id:
            return None
        return result

    async def get_by_name_for_account(
        self,
        account_id: UUID,
        name: str,
    ) -> Optional[CompetitiveSet]:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get competitive set by name")
        try:
            rows = await pool.fetch(
                """
                SELECT cs.*,
                       csv.vendor_name AS competitor_vendor_name,
                       csv.sort_order AS competitor_sort_order
                FROM b2b_competitive_sets cs
                LEFT JOIN b2b_competitive_set_vendors csv
                  ON csv.competitive_set_id = cs.id
                WHERE cs.account_id = $1
                  AND LOWER(cs.name) = LOWER($2)
                ORDER BY csv.sort_order ASC, csv.vendor_name ASC
                """,
                account_id,
                name,
            )
            return self._rows_to_competitive_set(rows)
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("get competitive set by name", exc)

    async def list_for_account(
        self,
        account_id: UUID,
        *,
        include_inactive: bool = False,
    ) -> list[CompetitiveSet]:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("list competitive sets")
        try:
            rows = await pool.fetch(
                f"""
                SELECT cs.*,
                       csv.vendor_name AS competitor_vendor_name,
                       csv.sort_order AS competitor_sort_order
                FROM b2b_competitive_sets cs
                LEFT JOIN b2b_competitive_set_vendors csv
                  ON csv.competitive_set_id = cs.id
                WHERE cs.account_id = $1
                {' ' if include_inactive else 'AND cs.active = TRUE'}
                ORDER BY cs.created_at DESC, csv.sort_order ASC, csv.vendor_name ASC
                """,
                account_id,
            )
            return self._group_rows(rows)
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("list competitive sets", exc)

    async def update(
        self,
        competitive_set_id: UUID,
        *,
        name: Optional[str] = None,
        focal_vendor_name: Optional[str] = None,
        competitor_vendor_names: Optional[list[str]] = None,
        active: Optional[bool] = None,
        refresh_mode: Optional[str] = None,
        refresh_interval_hours: Optional[int] = None,
        vendor_synthesis_enabled: Optional[bool] = None,
        pairwise_enabled: Optional[bool] = None,
        category_council_enabled: Optional[bool] = None,
        asymmetry_enabled: Optional[bool] = None,
    ) -> Optional[CompetitiveSet]:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update competitive set")
        try:
            existing = await self.get_by_id(competitive_set_id)
            if not existing:
                return None
            now = datetime.now(timezone.utc)
            next_name = name if name is not None else existing.name
            next_focal = focal_vendor_name if focal_vendor_name is not None else existing.focal_vendor_name
            next_active = active if active is not None else existing.active
            next_refresh_mode = refresh_mode if refresh_mode is not None else existing.refresh_mode
            next_interval = refresh_interval_hours if refresh_interval_hours is not None or next_refresh_mode == "manual" else existing.refresh_interval_hours
            next_vendor_enabled = (
                vendor_synthesis_enabled
                if vendor_synthesis_enabled is not None
                else existing.vendor_synthesis_enabled
            )
            next_pairwise = pairwise_enabled if pairwise_enabled is not None else existing.pairwise_enabled
            next_council = (
                category_council_enabled
                if category_council_enabled is not None
                else existing.category_council_enabled
            )
            next_asymmetry = asymmetry_enabled if asymmetry_enabled is not None else existing.asymmetry_enabled
            next_competitors = (
                self._normalize_vendor_names(competitor_vendor_names)
                if competitor_vendor_names is not None
                else existing.competitor_vendor_names
            )
            if next_refresh_mode == "manual":
                next_interval = None

            async with pool.transaction() as conn:
                await conn.execute(
                    """
                    UPDATE b2b_competitive_sets
                    SET name = $2,
                        focal_vendor_name = $3,
                        active = $4,
                        refresh_mode = $5,
                        refresh_interval_hours = $6,
                        vendor_synthesis_enabled = $7,
                        pairwise_enabled = $8,
                        category_council_enabled = $9,
                        asymmetry_enabled = $10,
                        updated_at = $11
                    WHERE id = $1
                    """,
                    competitive_set_id,
                    next_name,
                    next_focal,
                    next_active,
                    next_refresh_mode,
                    next_interval,
                    next_vendor_enabled,
                    next_pairwise,
                    next_council,
                    next_asymmetry,
                    now,
                )
                if competitor_vendor_names is not None:
                    await conn.execute(
                        "DELETE FROM b2b_competitive_set_vendors WHERE competitive_set_id = $1",
                        competitive_set_id,
                    )
                    if next_competitors:
                        await conn.executemany(
                            """
                            INSERT INTO b2b_competitive_set_vendors (
                                competitive_set_id, vendor_name, sort_order
                            )
                            VALUES ($1, $2, $3)
                            """,
                            [
                                (competitive_set_id, vendor_name, idx)
                                for idx, vendor_name in enumerate(next_competitors)
                            ],
                        )
            return await self.get_by_id(competitive_set_id)
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("update competitive set", exc)

    async def delete(self, competitive_set_id: UUID) -> bool:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("delete competitive set")
        try:
            result = await pool.execute(
                "DELETE FROM b2b_competitive_sets WHERE id = $1",
                competitive_set_id,
            )
            return result.endswith("1")
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("delete competitive set", exc)

    async def list_due_scheduled(self, *, limit: int = 25) -> list[CompetitiveSet]:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("list due competitive sets")
        try:
            rows = await pool.fetch(
                """
                WITH due_sets AS (
                    SELECT cs.id
                    FROM b2b_competitive_sets cs
                    WHERE cs.active = TRUE
                      AND cs.refresh_mode = 'scheduled'
                      AND cs.refresh_interval_hours IS NOT NULL
                      AND (
                            cs.last_run_status IS NULL
                            OR cs.last_run_status != 'running'
                          )
                      AND COALESCE(cs.last_run_at, cs.last_success_at, cs.created_at)
                          <= NOW() - make_interval(hours => cs.refresh_interval_hours)
                    ORDER BY COALESCE(cs.last_run_at, cs.last_success_at, cs.created_at) ASC
                    LIMIT $1
                )
                SELECT cs.*,
                       csv.vendor_name AS competitor_vendor_name,
                       csv.sort_order AS competitor_sort_order
                FROM b2b_competitive_sets cs
                JOIN due_sets ds ON ds.id = cs.id
                LEFT JOIN b2b_competitive_set_vendors csv
                  ON csv.competitive_set_id = cs.id
                ORDER BY COALESCE(cs.last_run_at, cs.last_success_at, cs.created_at) ASC,
                         csv.sort_order ASC,
                         csv.vendor_name ASC
                """,
                limit,
            )
            return self._group_rows(rows)
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("list due competitive sets", exc)

    async def mark_run_started(
        self,
        competitive_set_id: UUID,
        *,
        run_id: str | None,
        trigger: str,
        execution_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark competitive set run started")
        existing = await self.get_by_id(competitive_set_id)
        if not existing:
            raise DatabaseOperationError("mark competitive set run started", Exception("Competitive set not found"))
        run_summary = {
            "run_id": run_id,
            "trigger": trigger,
            "execution_id": execution_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(summary, dict):
            run_summary.update(summary)
        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    """
                    INSERT INTO b2b_competitive_set_runs (
                        competitive_set_id, account_id, run_id, trigger,
                        status, execution_id, summary, started_at, created_at
                    )
                    VALUES ($1, $2, $3, $4, 'running', $5, $6::jsonb, NOW(), NOW())
                    ON CONFLICT (competitive_set_id, run_id)
                    DO UPDATE SET
                        trigger = EXCLUDED.trigger,
                        status = 'running',
                        execution_id = EXCLUDED.execution_id,
                        summary = EXCLUDED.summary,
                        started_at = NOW()
                    """,
                    competitive_set_id,
                    existing.account_id,
                    str(run_id or ""),
                    trigger,
                    execution_id,
                    json.dumps(run_summary, default=str),
                )
                await conn.execute(
                    """
                    UPDATE b2b_competitive_sets
                    SET last_run_at = NOW(),
                        last_run_status = 'running',
                        last_run_summary = $2::jsonb,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    competitive_set_id,
                    json.dumps(run_summary, default=str),
                )
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("mark competitive set run started", exc)

    async def mark_run_completed(
        self,
        competitive_set_id: UUID,
        *,
        status: str,
        summary: dict[str, Any],
    ) -> None:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark competitive set run completed")
        try:
            success_ts = datetime.now(timezone.utc) if status in {"succeeded", "partial"} else None
            run_id = str(summary.get("run_id") or "").strip()
            existing = await self.get_by_id(competitive_set_id)
            if not existing:
                raise DatabaseOperationError("mark competitive set run completed", Exception("Competitive set not found"))
            async with pool.transaction() as conn:
                if run_id:
                    await conn.execute(
                        """
                        INSERT INTO b2b_competitive_set_runs (
                            competitive_set_id, account_id, run_id, trigger,
                            status, execution_id, summary, started_at,
                            completed_at, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW(), NOW(), NOW())
                        ON CONFLICT (competitive_set_id, run_id)
                        DO UPDATE SET
                            status = EXCLUDED.status,
                            execution_id = COALESCE(EXCLUDED.execution_id, b2b_competitive_set_runs.execution_id),
                            summary = EXCLUDED.summary,
                            completed_at = NOW()
                        """,
                        competitive_set_id,
                        existing.account_id,
                        run_id,
                        str(summary.get("trigger") or "manual"),
                        status,
                        str(summary.get("execution_id") or "") or None,
                        json.dumps(summary, default=str),
                    )
                await conn.execute(
                    """
                    UPDATE b2b_competitive_sets
                    SET last_run_status = $2,
                        last_run_summary = $3::jsonb,
                        last_success_at = CASE
                            WHEN $4::timestamptz IS NOT NULL THEN $4::timestamptz
                            ELSE last_success_at
                        END,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    competitive_set_id,
                    status,
                    json.dumps(summary, default=str),
                    success_ts,
                )
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("mark competitive set run completed", exc)

    async def list_runs_for_account_set(
        self,
        competitive_set_id: UUID,
        account_id: UUID,
        *,
        limit: int = 5,
    ) -> list[CompetitiveSetRun]:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("list competitive set runs")
        try:
            rows = await pool.fetch(
                """
                SELECT *
                FROM b2b_competitive_set_runs
                WHERE competitive_set_id = $1
                  AND account_id = $2
                ORDER BY started_at DESC, created_at DESC
                LIMIT $3
                """,
                competitive_set_id,
                account_id,
                limit,
            )
            return [self._row_to_competitive_set_run(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as exc:
            raise DatabaseOperationError("list competitive set runs", exc)

    @staticmethod
    def _normalize_vendor_names(vendor_names: list[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_name in vendor_names or []:
            name = str(raw_name or "").strip()
            key = name.lower()
            if not name or key in seen:
                continue
            normalized.append(name)
            seen.add(key)
        return normalized

    def _group_rows(self, rows) -> list[CompetitiveSet]:
        grouped: dict[UUID, list[Any]] = {}
        for row in rows:
            grouped.setdefault(row["id"], []).append(row)
        return [self._rows_to_competitive_set(group) for group in grouped.values() if group]

    def _rows_to_competitive_set(self, rows) -> Optional[CompetitiveSet]:
        if not rows:
            return None
        row = rows[0]
        summary = row["last_run_summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        elif summary is None:
            summary = {}
        competitors: list[str] = []
        for item in rows:
            vendor_name = item.get("competitor_vendor_name")
            if vendor_name and vendor_name not in competitors:
                competitors.append(vendor_name)
        return CompetitiveSet(
            id=row["id"],
            account_id=row["account_id"],
            name=row["name"],
            focal_vendor_name=row["focal_vendor_name"],
            competitor_vendor_names=competitors,
            active=bool(row["active"]),
            refresh_mode=row["refresh_mode"],
            refresh_interval_hours=row["refresh_interval_hours"],
            vendor_synthesis_enabled=bool(row["vendor_synthesis_enabled"]),
            pairwise_enabled=bool(row["pairwise_enabled"]),
            category_council_enabled=bool(row["category_council_enabled"]),
            asymmetry_enabled=bool(row["asymmetry_enabled"]),
            last_run_at=row["last_run_at"],
            last_success_at=row["last_success_at"],
            last_run_status=row["last_run_status"],
            last_run_summary=summary if isinstance(summary, dict) else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_competitive_set_run(self, row) -> CompetitiveSetRun:
        summary = row["summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        elif summary is None:
            summary = {}
        return CompetitiveSetRun(
            id=row["id"],
            competitive_set_id=row["competitive_set_id"],
            account_id=row["account_id"],
            run_id=str(row["run_id"] or ""),
            trigger=str(row["trigger"] or ""),
            status=str(row["status"] or ""),
            execution_id=row["execution_id"],
            summary=summary if isinstance(summary, dict) else {},
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            created_at=row["created_at"],
        )


_competitive_set_repo: Optional[CompetitiveSetRepository] = None


def get_competitive_set_repo() -> CompetitiveSetRepository:
    """Get the global competitive-set repository."""
    global _competitive_set_repo
    if _competitive_set_repo is None:
        _competitive_set_repo = CompetitiveSetRepository()
    return _competitive_set_repo
