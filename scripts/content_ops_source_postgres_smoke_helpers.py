"""Shared helpers for Content Ops source-row Postgres smoke commands."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from extracted_content_pipeline.campaign_example import (
    DeterministicCampaignLLM,
    StaticCampaignSkillStore,
)
from extracted_content_pipeline.campaign_generation import CampaignGenerationConfig
from extracted_content_pipeline.campaign_postgres_generation import (
    generate_campaign_drafts_from_postgres,
)


async def generate_imported_target_drafts(
    *,
    pool: Any,
    account_id: str,
    user_id: str | None,
    target_mode: str,
    channels: Sequence[str],
    target_ids: Sequence[str],
    opportunity_table: str,
    llm: Any | None = None,
    skills: Any | None = None,
) -> dict[str, Any]:
    resolved_llm = llm or DeterministicCampaignLLM()
    resolved_skills = skills or StaticCampaignSkillStore()
    saved_ids: list[str] = []
    errors: list[Mapping[str, Any]] = []
    requested = 0
    generated = 0
    skipped = 0
    reasoning_contexts_used = 0
    for target_id in target_ids:
        result = await generate_campaign_drafts_from_postgres(
            pool,
            scope={"account_id": account_id, "user_id": user_id},
            target_mode=target_mode,
            channel=channels[0],
            channels=tuple(channels),
            limit=1,
            filters={"target_id": target_id},
            llm=resolved_llm,
            skills=resolved_skills,
            config=CampaignGenerationConfig(channels=tuple(channels), limit=1),
            opportunity_table=opportunity_table,
        )
        data = result.as_dict()
        requested += int(data.get("requested") or 0)
        generated += int(data.get("generated") or 0)
        skipped += int(data.get("skipped") or 0)
        reasoning_contexts_used += int(data.get("reasoning_contexts_used") or 0)
        saved_ids.extend(str(item) for item in data.get("saved_ids") or ())
        errors.extend(error for error in data.get("errors") or () if isinstance(error, Mapping))
    return {
        "requested": requested,
        "generated": generated,
        "skipped": skipped,
        "reasoning_contexts_used": reasoning_contexts_used,
        "saved_ids": saved_ids,
        "errors": [dict(error) for error in errors],
    }


def generation_errors(result: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(result, Mapping):
        return ["generation result missing"]
    errors: list[str] = []
    reported_errors = result.get("errors") or []
    if reported_errors:
        errors.append(f"generation reported {len(reported_errors)} error(s)")
    if int(result.get("skipped") or 0):
        errors.append(f"generation skipped {result.get('skipped')} draft(s)")
    return errors


async def schema_readiness_errors(pool: Any, *, opportunity_table: str) -> list[str]:
    missing: list[str] = []
    for table_name in (opportunity_table, "b2b_campaigns"):
        if not await relation_exists(pool, table_name):
            missing.append(table_name)
    if not missing:
        return []
    command = (
        "python scripts/run_extracted_content_pipeline_migrations.py "
        "--database-url \"$EXTRACTED_DATABASE_URL\""
    )
    return [
        "required Content Ops table(s) missing: "
        f"{', '.join(missing)}. Run {command} before this smoke."
    ]


async def relation_exists(pool: Any, table_name: str) -> bool:
    value = await pool.fetchval("SELECT to_regclass($1)::text", table_name)
    return bool(value)


async def fetch_saved_drafts(pool: Any, saved_ids: Sequence[Any]) -> list[dict[str, Any]]:
    if not isinstance(saved_ids, Sequence) or isinstance(saved_ids, (str, bytes)):
        return []
    ids = [str(item) for item in saved_ids if str(item or "").strip()]
    if not ids:
        return []
    rows = await pool.fetch(
        """
        SELECT id, subject, body, target_mode, channel, metadata
          FROM b2b_campaigns
         WHERE id::text = ANY($1::text[])
        """,
        ids,
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        metadata = metadata_object(row_value(row, "metadata"))
        source_opportunity = metadata.get("source_opportunity")
        if not isinstance(source_opportunity, Mapping):
            source_opportunity = {}
        target_id = metadata.get("target_id") or source_opportunity.get("target_id")
        out.append({
            "id": str(row_value(row, "id") or ""),
            "target_id": str(target_id or ""),
            "subject": row_value(row, "subject"),
            "body": row_value(row, "body"),
            "target_mode": row_value(row, "target_mode"),
            "channel": row_value(row, "channel"),
        })
    return out


def draft_errors(
    result: Mapping[str, Any],
    *,
    min_drafts: int,
    forbidden_phrases: Sequence[str],
) -> list[str]:
    drafts = result.get("drafts")
    if not isinstance(drafts, list):
        return ["result.drafts is missing or not a list"]
    if len(drafts) < min_drafts:
        return [f"expected at least {min_drafts} draft(s), got {len(drafts)}"]
    errors: list[str] = []
    forbidden = [phrase.lower() for phrase in forbidden_phrases if phrase]
    for index, draft in enumerate(drafts[:min_drafts], start=1):
        if not isinstance(draft, Mapping):
            errors.append(f"draft {index} is not an object")
            continue
        for field in ("subject", "body", "target_id", "channel"):
            if not str(draft.get(field) or "").strip():
                errors.append(f"draft {index} missing {field}")
        body = str(draft.get("body") or "").lower()
        for phrase in forbidden:
            if phrase in body:
                errors.append(f"draft {index} contains forbidden phrase: {phrase}")
    return errors


def saved_draft_target_errors(
    saved_drafts: Sequence[Mapping[str, Any]],
    imported_target_ids: Sequence[str],
) -> list[str]:
    imported = {str(target_id) for target_id in imported_target_ids}
    errors: list[str] = []
    for draft in saved_drafts:
        draft_id_value = draft.get("id")
        draft_id = str(draft_id_value) if draft_id_value is not None else ""
        target_id_value = draft.get("target_id")
        target_id = str(target_id_value) if target_id_value is not None else ""
        if not target_id:
            errors.append(f"persisted draft missing target_id metadata: {draft_id or '<unknown>'}")
            continue
        if target_id not in imported:
            errors.append(f"persisted draft target_id was not imported: {target_id}")
    return errors


def row_value(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return row[key]


def metadata_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}
