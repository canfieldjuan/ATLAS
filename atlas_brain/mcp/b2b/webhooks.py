"""B2B Churn MCP -- webhook tools."""
import json
import uuid as _uuid
from typing import Optional

from ._shared import _is_uuid, logger, get_pool
from .server import mcp


@mcp.tool()
async def list_webhook_subscriptions(
    account_id: Optional[str] = None,
    enabled_only: bool = True,
) -> str:
    """
    List webhook subscriptions for debugging and monitoring.

    account_id: Optional UUID to filter by specific account
    enabled_only: If true, only show enabled subscriptions (default true)
    """
    try:
        pool = get_pool()
        conditions = []
        params: list = []
        idx = 1

        if account_id:
            if not _is_uuid(account_id):
                return json.dumps({"error": "account_id must be a valid UUID"})
            conditions.append(f"ws.account_id = ${idx}::uuid")
            params.append(account_id)
            idx += 1

        if enabled_only:
            conditions.append("ws.enabled = true")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT ws.id, ws.account_id, ws.url, ws.event_types,
                   COALESCE(ws.channel, 'generic') AS channel,
                   ws.enabled, ws.description, ws.created_at,
                   sa.name AS account_name,
                   (SELECT COUNT(*) FROM b2b_webhook_delivery_log dl
                    WHERE dl.subscription_id = ws.id AND dl.delivered_at > NOW() - INTERVAL '7 days') AS recent_deliveries,
                   (SELECT COUNT(*) FILTER (WHERE dl2.success) FROM b2b_webhook_delivery_log dl2
                    WHERE dl2.subscription_id = ws.id AND dl2.delivered_at > NOW() - INTERVAL '7 days') AS recent_successes
            FROM b2b_webhook_subscriptions ws
            JOIN saas_accounts sa ON sa.id = ws.account_id
            {where}
            ORDER BY ws.created_at DESC
            """,
            *params,
        )

        subs = []
        for r in rows:
            recent_total = r["recent_deliveries"] or 0
            subs.append({
                "id": str(r["id"]),
                "account_id": str(r["account_id"]),
                "account_name": r["account_name"],
                "url": r["url"],
                "event_types": r["event_types"],
                "channel": r["channel"],
                "enabled": r["enabled"],
                "description": r["description"],
                "created_at": r["created_at"].isoformat(),
                "recent_deliveries_7d": recent_total,
                "recent_success_rate_7d": round(r["recent_successes"] / max(recent_total, 1), 3) if recent_total else None,
            })

        return json.dumps({"subscriptions": subs, "count": len(subs)}, default=str)
    except Exception:
        logger.exception("list_webhook_subscriptions error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def send_test_webhook_tool(
    subscription_id: str,
) -> str:
    """
    Send a test payload to a webhook subscription to verify connectivity.

    subscription_id: UUID of the webhook subscription to test
    """
    if not _is_uuid(subscription_id):
        return json.dumps({"error": "subscription_id must be a valid UUID"})
    try:
        import uuid as _u

        from ...services.b2b.webhook_dispatcher import send_test_webhook

        pool = get_pool()
        result = await send_test_webhook(pool, _u.UUID(subscription_id))
        return json.dumps(result, default=str)
    except Exception:
        logger.exception("send_test_webhook error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def update_webhook(
    subscription_id: str,
    enabled: Optional[bool] = None,
    event_types: Optional[str] = None,
    url: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Update a webhook subscription (toggle enabled, change event_types/url/description).

    subscription_id: UUID of the webhook to update
    enabled: Set to true/false to enable/disable the webhook
    event_types: Comma-separated list of event types (change_event, churn_alert, report_generated, signal_update)
    url: New webhook URL (must start with https:// or http://)
    description: New description
    """
    if not _is_uuid(subscription_id):
        return json.dumps({"error": "subscription_id must be a valid UUID"})

    _valid_events = {"change_event", "churn_alert", "report_generated", "signal_update"}

    try:
        import uuid as _u

        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        sets: list[str] = []
        params: list = []
        idx = 1

        if enabled is not None:
            sets.append(f"enabled = ${idx}")
            params.append(enabled)
            idx += 1
        if event_types is not None:
            et_list = [e.strip() for e in event_types.split(",") if e.strip()]
            invalid = set(et_list) - _valid_events
            if invalid:
                return json.dumps({"error": f"Invalid event_types: {sorted(invalid)}. Valid: {sorted(_valid_events)}"})
            if not et_list:
                return json.dumps({"error": "event_types must not be empty"})
            sets.append(f"event_types = ${idx}")
            params.append(et_list)
            idx += 1
        if url is not None:
            if not url.startswith(("https://", "http://")):
                return json.dumps({"error": "url must begin with https:// or http://"})
            sets.append(f"url = ${idx}")
            params.append(url)
            idx += 1
        if description is not None:
            sets.append(f"description = ${idx}")
            params.append(description)
            idx += 1

        if not sets:
            return json.dumps({"error": "No fields to update"})

        sets.append("updated_at = NOW()")
        params.append(_u.UUID(subscription_id))

        row = await pool.fetchrow(
            f"""
            UPDATE b2b_webhook_subscriptions
            SET {', '.join(sets)}
            WHERE id = ${idx}
            RETURNING id, url, event_types, COALESCE(channel, 'generic') AS channel,
                      enabled, description, updated_at
            """,
            *params,
        )
        if not row:
            return json.dumps({"error": "Webhook not found"})

        return json.dumps({
            "success": True,
            "webhook": {
                "id": str(row["id"]),
                "url": row["url"],
                "event_types": row["event_types"],
                "channel": row["channel"],
                "enabled": row["enabled"],
                "description": row["description"],
                "updated_at": row["updated_at"].isoformat(),
            },
        }, default=str)
    except Exception:
        logger.exception("update_webhook error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_webhook_delivery_summary(
    account_id: Optional[str] = None,
    days: int = 7,
) -> str:
    """Aggregate webhook delivery health across all subscriptions.

    account_id: Optional UUID to filter by account (shows all if omitted)
    days: Window in days for delivery stats (default 7, max 90)
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        days = max(1, min(days, 90))
        conditions = ["ws.enabled = true"]
        params: list = [str(days)]
        idx = 2

        if account_id:
            if not _is_uuid(account_id):
                return json.dumps({"error": "account_id must be a valid UUID"})
            conditions.append(f"ws.account_id = ${idx}::uuid")
            params.append(account_id)
            idx += 1

        where = " AND ".join(conditions)

        row = await pool.fetchrow(
            f"""
            SELECT
                COUNT(DISTINCT ws.id) AS active_subscriptions,
                COUNT(dl.id) AS total_deliveries,
                COUNT(dl.id) FILTER (WHERE dl.success) AS successful,
                COUNT(dl.id) FILTER (WHERE NOT dl.success) AS failed,
                AVG(dl.duration_ms) FILTER (WHERE dl.success) AS avg_success_duration_ms,
                MAX(dl.delivered_at) AS last_delivery_at
            FROM b2b_webhook_subscriptions ws
            LEFT JOIN b2b_webhook_delivery_log dl
                ON dl.subscription_id = ws.id
                AND dl.delivered_at > NOW() - ($1 || ' days')::interval
            WHERE {where}
            """,
            *params,
        )

        total = row["total_deliveries"] or 0
        return json.dumps({
            "success": True,
            "window_days": days,
            "active_subscriptions": row["active_subscriptions"] or 0,
            "total_deliveries": total,
            "successful": row["successful"] or 0,
            "failed": row["failed"] or 0,
            "success_rate": round((row["successful"] or 0) / max(total, 1), 3) if total else None,
            "avg_success_duration_ms": round(row["avg_success_duration_ms"], 1) if row["avg_success_duration_ms"] else None,
            "last_delivery_at": str(row["last_delivery_at"]) if row["last_delivery_at"] else None,
        }, default=str)
    except Exception:
        logger.exception("get_webhook_delivery_summary error")
        return json.dumps({"success": False, "error": "Internal error"})
