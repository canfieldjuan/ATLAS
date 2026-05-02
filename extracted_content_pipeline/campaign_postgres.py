"""Postgres repository adapters for the standalone campaign product."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Mapping, Sequence

from .campaign_opportunities import normalize_campaign_opportunity
from .campaign_ports import (
    CampaignDraft,
    SendResult,
    TenantScope,
    WebhookEvent,
)


JsonDict = dict[str, Any]


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _source_opportunity(draft: CampaignDraft) -> JsonDict:
    raw = draft.metadata.get("source_opportunity") if isinstance(draft.metadata, Mapping) else None
    return dict(raw) if isinstance(raw, Mapping) else {}


def _campaign_company_name(draft: CampaignDraft, opportunity: Mapping[str, Any]) -> str:
    return (
        _clean(opportunity.get("company_name"))
        or _clean(opportunity.get("name"))
        or _clean(opportunity.get("seller_name"))
        or draft.target_id
    )


def _campaign_vendor_name(draft: CampaignDraft, opportunity: Mapping[str, Any]) -> str | None:
    return (
        _clean(opportunity.get("vendor_name"))
        or _clean(opportunity.get("vendor"))
        or _clean(draft.metadata.get("vendor_name"))
        or None
    )


def _json_mapping(value: Any) -> JsonDict:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _json_sequence(value: Any) -> list[Any]:
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _opportunity_from_row(row: Mapping[str, Any], *, target_mode: str) -> JsonDict:
    raw = _json_mapping(row.get("raw_payload"))
    opportunity = dict(raw)
    for key in (
        "target_id",
        "company_name",
        "vendor_name",
        "contact_name",
        "contact_email",
        "contact_title",
        "opportunity_score",
        "urgency_score",
    ):
        value = row.get(key)
        if value not in (None, "", [], {}):
            opportunity[key] = value
    if row.get("pain_points") not in (None, "", [], {}):
        opportunity["pain_points"] = _json_sequence(row.get("pain_points"))
    if row.get("competitors") not in (None, "", [], {}):
        opportunity["competitors"] = _json_sequence(row.get("competitors"))
    if row.get("evidence") not in (None, "", [], {}):
        opportunity["evidence"] = _json_sequence(row.get("evidence"))
    if row.get("account_id"):
        opportunity["account_id"] = row.get("account_id")
    return normalize_campaign_opportunity(opportunity, target_mode=target_mode)


def _safe_json_key(value: Any) -> str:
    key = _clean(value)
    if not key:
        raise ValueError("filter key cannot be empty")
    if not all(char.isalnum() or char == "_" for char in key):
        raise ValueError(f"unsupported filter key: {key}")
    return key


@dataclass(frozen=True)
class PostgresIntelligenceRepository:
    """Async Postgres adapter for customer campaign opportunity rows."""

    pool: Any
    opportunity_table: str = "campaign_opportunities"
    vendor_targets_table: str = "vendor_targets"

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[JsonDict]:
        params: list[Any] = [target_mode]
        where = [
            "status = 'active'",
            "(target_mode = $1 OR target_mode IS NULL)",
        ]
        if scope.account_id:
            params.append(scope.account_id)
            where.append(f"account_id = ${len(params)}")
        self._append_filters(where, params, filters)
        params.append(int(limit))
        rows = await self.pool.fetch(
            f"""
            SELECT
                id, account_id, target_id, target_mode, company_name, vendor_name,
                contact_name, contact_email, contact_title, opportunity_score,
                urgency_score, pain_points, competitors, evidence, raw_payload
              FROM {self._identifier(self.opportunity_table)}
             WHERE {' AND '.join(where)}
             ORDER BY urgency_score DESC NULLS LAST,
                      opportunity_score DESC NULLS LAST,
                      updated_at DESC NULLS LAST,
                      created_at DESC NULLS LAST
             LIMIT ${len(params)}
            """,
            *params,
        )
        return tuple(
            _opportunity_from_row(_row_dict(row), target_mode=target_mode)
            for row in rows
        )

    async def read_vendor_targets(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        vendor_name: str | None = None,
    ) -> Sequence[JsonDict]:
        del scope
        params: list[Any] = [target_mode]
        where = ["status = 'active'", "target_mode = $1"]
        if vendor_name:
            params.append(vendor_name)
            where.append(f"LOWER(company_name) = LOWER(${len(params)})")
        rows = await self.pool.fetch(
            f"""
            SELECT *
              FROM {self._identifier(self.vendor_targets_table)}
             WHERE {' AND '.join(where)}
             ORDER BY company_name ASC
            """,
            *params,
        )
        return tuple(_row_dict(row) for row in rows)

    def _append_filters(
        self,
        where: list[str],
        params: list[Any],
        filters: Mapping[str, Any] | None,
    ) -> None:
        for key, value in (filters or {}).items():
            if value in (None, "", [], {}):
                continue
            if key == "vendor_name":
                params.append(value)
                where.append(f"LOWER(vendor_name) = LOWER(${len(params)})")
            elif key == "company_name":
                params.append(value)
                where.append(f"LOWER(company_name) = LOWER(${len(params)})")
            elif key == "contact_email":
                params.append(value)
                where.append(f"LOWER(contact_email) = LOWER(${len(params)})")
            elif key == "target_id":
                params.append(value)
                where.append(f"target_id = ${len(params)}")
            elif key == "min_urgency":
                params.append(value)
                where.append(f"urgency_score >= ${len(params)}")
            elif key == "min_opportunity_score":
                params.append(value)
                where.append(f"opportunity_score >= ${len(params)}")
            else:
                json_key = _safe_json_key(key)
                params.append(json_key)
                key_position = len(params)
                params.append(value)
                where.append(
                    f"LOWER(raw_payload ->> ${key_position}) = LOWER(${len(params)})"
                )

    def _identifier(self, value: str) -> str:
        parts = value.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid SQL identifier: {value}")
        for part in parts:
            if not all(char.isalnum() or char == "_" for char in part):
                raise ValueError(f"invalid SQL identifier: {value}")
        return ".".join(f'"{part}"' for part in parts)


@dataclass(frozen=True)
class PostgresCampaignRepository:
    """Async Postgres adapter for generated campaigns and webhook events."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[CampaignDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        saved: list[str] = []
        for draft in drafts:
            opportunity = _source_opportunity(draft)
            metadata = {
                **dict(draft.metadata or {}),
                "target_id": draft.target_id,
                "target_mode": draft.target_mode,
                "scope": {
                    "account_id": scope.account_id,
                    "user_id": scope.user_id,
                },
            }
            campaign_id = await self.pool.fetchval(
                """
                INSERT INTO b2b_campaigns (
                    company_name, vendor_name, product_category, target_mode,
                    channel, subject, body, cta, status, recipient_email,
                    metadata, llm_model
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'draft', $9, $10::jsonb, $11)
                RETURNING id
                """,
                _campaign_company_name(draft, opportunity),
                _campaign_vendor_name(draft, opportunity),
                _clean(opportunity.get("product_category") or opportunity.get("category")) or None,
                draft.target_mode,
                draft.channel,
                draft.subject,
                draft.body,
                draft.metadata.get("cta"),
                (
                    _clean(opportunity.get("recipient_email"))
                    or _clean(opportunity.get("contact_email"))
                    or _clean(opportunity.get("email"))
                    or None
                ),
                _jsonb(metadata),
                draft.metadata.get("generation_model"),
            )
            saved.append(str(campaign_id))
        return tuple(saved)

    async def list_due_sends(
        self,
        *,
        limit: int,
        now: datetime,
    ) -> Sequence[JsonDict]:
        rows = await self.pool.fetch(
            """
            SELECT
                id, sequence_id, recipient_email, from_email, subject, body,
                metadata, company_name, vendor_name, channel, step_number
            FROM b2b_campaigns
            WHERE status = 'queued'
              AND recipient_email IS NOT NULL
            ORDER BY created_at ASC
            LIMIT $1
            """,
            int(limit),
        )
        return tuple(_row_dict(row) for row in rows)

    async def mark_sent(
        self,
        *,
        campaign_id: str,
        result: SendResult,
        sent_at: datetime,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE b2b_campaigns
               SET status = 'sent',
                   sent_at = $2,
                   esp_message_id = $3,
                   sent_message_id = $3,
                   metadata = COALESCE(metadata, '{}'::jsonb) || $4::jsonb,
                   updated_at = NOW()
             WHERE id = $1
            """,
            campaign_id,
            sent_at,
            result.message_id,
            _jsonb({"send_provider": result.provider, "send_raw": result.raw}),
        )

    async def mark_cancelled(
        self,
        *,
        campaign_id: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE b2b_campaigns
               SET status = 'cancelled',
                   metadata = COALESCE(metadata, '{}'::jsonb) || $3::jsonb,
                   updated_at = NOW()
             WHERE id = $1
            """,
            campaign_id,
            reason,
            _jsonb({"cancel_reason": reason, **dict(metadata or {})}),
        )

    async def mark_send_failed(
        self,
        *,
        campaign_id: str,
        error: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE b2b_campaigns
               SET status = 'cancelled',
                   metadata = COALESCE(metadata, '{}'::jsonb) || $3::jsonb,
                   updated_at = NOW()
             WHERE id = $1
            """,
            campaign_id,
            error,
            _jsonb({"send_error": error, **dict(metadata or {})}),
        )

    async def record_webhook_event(self, event: WebhookEvent) -> None:
        if event.event_type in {"opened", "clicked"}:
            column = "opened_at" if event.event_type == "opened" else "clicked_at"
            await self.pool.execute(
                f"""
                UPDATE b2b_campaigns
                   SET {column} = COALESCE({column}, $2),
                       metadata = COALESCE(metadata, '{{}}'::jsonb) || $3::jsonb,
                       updated_at = NOW()
                 WHERE esp_message_id = $1 OR sent_message_id = $1
                """,
                event.message_id,
                event.occurred_at,
                _jsonb({"last_webhook_event": event.event_type}),
            )
        await self.pool.execute(
            """
            INSERT INTO campaign_audit_log (
                event_type, esp_message_id, metadata, source, created_at
            )
            VALUES ($1, $2, $3::jsonb, $4, COALESCE($5, NOW()))
            """,
            f"webhook_{event.event_type}",
            event.message_id,
            _jsonb({"provider": event.provider, "email": event.email, "payload": event.payload}),
            event.provider or "webhook",
            event.occurred_at,
        )

    async def refresh_analytics(self) -> None:
        await self.pool.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY campaign_funnel_stats")


@dataclass(frozen=True)
class PostgresCampaignSequenceRepository:
    """Async Postgres adapter for campaign sequence progression."""

    pool: Any

    async def list_due_sequences(
        self,
        *,
        limit: int,
        now: datetime,
    ) -> Sequence[JsonDict]:
        rows = await self.pool.fetch(
            """
            SELECT *
              FROM campaign_sequences
             WHERE status = 'active'
               AND next_step_after IS NOT NULL
               AND next_step_after <= $1
               AND recipient_email IS NOT NULL
             ORDER BY next_step_after ASC
             LIMIT $2
            """,
            now,
            int(limit),
        )
        return tuple(_row_dict(row) for row in rows)

    async def list_previous_campaigns(
        self,
        *,
        sequence_id: str,
        limit: int,
    ) -> Sequence[JsonDict]:
        rows = await self.pool.fetch(
            """
            SELECT *
              FROM b2b_campaigns
             WHERE sequence_id = $1
             ORDER BY step_number ASC NULLS LAST, created_at ASC
             LIMIT $2
            """,
            sequence_id,
            int(limit),
        )
        return tuple(_row_dict(row) for row in rows)

    async def queue_sequence_step(
        self,
        *,
        sequence: JsonDict,
        content: JsonDict,
        from_email: str,
        queued_at: datetime,
    ) -> str:
        campaign_id = await self.pool.fetchval(
            """
            INSERT INTO b2b_campaigns (
                sequence_id, company_name, batch_id, channel, subject, body,
                cta, status, step_number, recipient_email, from_email,
                target_mode, product_category, metadata, created_at, updated_at
            )
            VALUES (
                $1, $2, $3, 'email_followup', $4, $5, $6, 'queued', $7, $8, $9,
                $10, $11, $12::jsonb, $13, $13
            )
            RETURNING id
            """,
            sequence.get("id"),
            sequence.get("company_name"),
            sequence.get("batch_id"),
            content.get("subject"),
            content.get("body"),
            content.get("cta"),
            content.get("step_number"),
            sequence.get("recipient_email"),
            from_email,
            content.get("target_mode"),
            content.get("product_category"),
            _jsonb({
                "angle_reasoning": content.get("angle_reasoning"),
                "sequence_context": {
                    "current_step": sequence.get("current_step"),
                    "max_steps": sequence.get("max_steps"),
                },
            }),
            queued_at,
        )
        return str(campaign_id)

    async def mark_sequence_step(
        self,
        *,
        sequence_id: str,
        current_step: int,
        updated_at: datetime,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE campaign_sequences
               SET current_step = $2,
                   updated_at = $3
             WHERE id = $1
            """,
            sequence_id,
            int(current_step),
            updated_at,
        )


@dataclass(frozen=True)
class PostgresSuppressionRepository:
    """Async Postgres adapter for campaign email/domain suppressions."""

    pool: Any

    async def is_suppressed(
        self,
        *,
        email: str | None = None,
        domain: str | None = None,
    ) -> bool:
        if email:
            row = await self.pool.fetchrow(
                """
                SELECT id
                  FROM campaign_suppressions
                 WHERE LOWER(email) = LOWER($1)
                   AND (expires_at IS NULL OR expires_at > NOW())
                 LIMIT 1
                """,
                email,
            )
            return row is not None
        if domain:
            row = await self.pool.fetchrow(
                """
                SELECT id
                  FROM campaign_suppressions
                 WHERE LOWER(domain) = LOWER($1)
                   AND (expires_at IS NULL OR expires_at > NOW())
                 LIMIT 1
                """,
                domain,
            )
            return row is not None
        return False

    async def add_suppression(
        self,
        *,
        reason: str,
        email: str | None = None,
        domain: str | None = None,
        source: str = "system",
        campaign_id: str | None = None,
        notes: str | None = None,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if email:
            await self.pool.execute(
                """
                INSERT INTO campaign_suppressions (
                    email, reason, source, campaign_id, notes, expires_at
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (LOWER(email)) WHERE email IS NOT NULL
                DO UPDATE SET
                    reason = EXCLUDED.reason,
                    source = EXCLUDED.source,
                    campaign_id = COALESCE(EXCLUDED.campaign_id, campaign_suppressions.campaign_id),
                    notes = COALESCE(EXCLUDED.notes, campaign_suppressions.notes),
                    expires_at = EXCLUDED.expires_at
                """,
                email,
                reason,
                source,
                campaign_id,
                notes or _clean((metadata or {}).get("notes")) or None,
                expires_at,
            )
            return
        await self.pool.execute(
            """
            INSERT INTO campaign_suppressions (
                domain, reason, source, campaign_id, notes, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            domain,
            reason,
            source,
            campaign_id,
            notes or _clean((metadata or {}).get("notes")) or None,
            expires_at,
        )


@dataclass(frozen=True)
class PostgresCampaignAuditSink:
    """AuditSink adapter backed by campaign_audit_log."""

    pool: Any

    async def record(
        self,
        event_type: str,
        *,
        campaign_id: str | None = None,
        sequence_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO campaign_audit_log (
                campaign_id, sequence_id, event_type, metadata, source
            )
            VALUES ($1, $2, $3, $4::jsonb, 'product')
            """,
            campaign_id,
            sequence_id,
            event_type,
            _jsonb(dict(metadata or {})),
        )
