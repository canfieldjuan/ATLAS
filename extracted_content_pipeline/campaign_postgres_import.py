"""Postgres importer for standalone campaign opportunity data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from .campaign_customer_data import (
    CampaignOpportunityLoadResult,
    CampaignOpportunityWarning,
    normalize_campaign_opportunity_rows,
)
from .campaign_opportunities import opportunity_target_id
from .campaign_ports import TenantScope


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class CampaignOpportunityImportResult:
    inserted: int
    skipped: int
    dry_run: bool
    replace_existing: bool
    target_ids: tuple[str, ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "inserted": self.inserted,
            "skipped": self.skipped,
            "dry_run": self.dry_run,
            "replace_existing": self.replace_existing,
            "target_ids": list(self.target_ids),
            "warnings": [warning.as_dict() for warning in self.warnings],
        }
        if self.source:
            data["source"] = self.source
        return data


async def import_campaign_opportunities(
    db: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: TenantScope | None = None,
    target_mode: str = "vendor_retention",
    opportunity_table: str = "campaign_opportunities",
    replace_existing: bool = False,
    dry_run: bool = False,
    normalize: bool = True,
    warnings: Sequence[CampaignOpportunityWarning] = (),
    source: str | None = None,
) -> CampaignOpportunityImportResult:
    """Load customer opportunity rows into the product-owned Postgres table."""

    table = _identifier(opportunity_table)
    loaded = _loaded_rows(rows, target_mode=target_mode, normalize=normalize)
    all_warnings = [*warnings, *loaded.warnings]
    prepared: list[JsonDict] = []
    skipped = 0
    for index, opportunity in enumerate(loaded.opportunities, start=1):
        target_id = opportunity_target_id(opportunity)
        if not target_id:
            skipped += 1
            all_warnings.append(
                CampaignOpportunityWarning(
                    code="missing_target_id",
                    message="Skipped row because it does not contain a stable target id.",
                    row_index=index,
                    field="target_id",
                )
            )
            continue
        row = dict(opportunity)
        row["target_id"] = target_id
        row["target_mode"] = str(row.get("target_mode") or target_mode or "").strip()
        prepared.append(row)

    target_ids = tuple(str(row["target_id"]) for row in prepared)
    if dry_run:
        return CampaignOpportunityImportResult(
            inserted=len(prepared),
            skipped=skipped,
            dry_run=True,
            replace_existing=replace_existing,
            target_ids=target_ids,
            warnings=tuple(all_warnings),
            source=source or loaded.source,
        )

    account_id = (scope or TenantScope()).account_id
    if replace_existing and target_ids:
        await db.execute(
            f"""
            DELETE FROM {table}
             WHERE account_id IS NOT DISTINCT FROM $1
               AND target_mode = $2
               AND target_id = ANY($3::text[])
            """,
            account_id,
            target_mode,
            list(target_ids),
        )
    for row in prepared:
        await db.execute(
            f"""
            INSERT INTO {table} (
                account_id, target_id, target_mode, company_name, vendor_name,
                contact_name, contact_email, contact_title, opportunity_score,
                urgency_score, pain_points, competitors, evidence, raw_payload,
                status
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb,
                'active'
            )
            """,
            account_id,
            row.get("target_id"),
            row.get("target_mode") or target_mode,
            _clean(row.get("company_name")),
            _clean(row.get("vendor_name")),
            _clean(row.get("contact_name")),
            _clean(row.get("contact_email")),
            _clean(row.get("contact_title")),
            row.get("opportunity_score"),
            row.get("urgency_score"),
            _jsonb(row.get("pain_points") or []),
            _jsonb(row.get("competitors") or []),
            _jsonb(row.get("evidence") or []),
            _jsonb(row),
        )
    return CampaignOpportunityImportResult(
        inserted=len(prepared),
        skipped=skipped,
        dry_run=False,
        replace_existing=replace_existing,
        target_ids=target_ids,
        warnings=tuple(all_warnings),
        source=source or loaded.source,
    )


def _loaded_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_mode: str,
    normalize: bool,
) -> CampaignOpportunityLoadResult:
    if normalize:
        return normalize_campaign_opportunity_rows(rows, target_mode=target_mode)
    return CampaignOpportunityLoadResult(
        opportunities=tuple(dict(row) for row in rows if isinstance(row, Mapping)),
    )


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


__all__ = [
    "CampaignOpportunityImportResult",
    "import_campaign_opportunities",
]
