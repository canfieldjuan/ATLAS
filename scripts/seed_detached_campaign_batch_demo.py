#!/usr/bin/env python3
"""Seed a provider-free detached campaign batch demo into the local DB.

Creates a synthetic `task_execution` plus Anthropic batch job/item rows so the
Operations -> Costs surfaces can be exercised without making provider calls.

Usage:
  python scripts/seed_detached_campaign_batch_demo.py seed
  python scripts/seed_detached_campaign_batch_demo.py cleanup --run-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.database import close_database, get_db_pool, init_database


TASK_NAME = "b2b_campaign_generation"
STAGE_ID = "b2b_campaign_generation.content"
TASK_DESCRIPTION = "Synthetic detached batch demo seed for Operations -> Costs"
REPLAY_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class DemoItem:
    custom_id: str
    artifact_id: str
    vendor_name: str
    status: str
    cache_prefiltered: bool
    fallback_single_call: bool
    input_tokens: int
    billable_input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: Decimal
    provider_request_id: str | None
    error_text: str | None
    response_text: str | None
    request_metadata: dict[str, Any]
    completed_at: datetime | None


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, default=str)


def _replay_contract(
    *,
    artifact_id: str,
    company_name: str,
    target_mode: str,
    channel: str = "email",
) -> dict[str, Any]:
    return {
        "contract_version": REPLAY_CONTRACT_VERSION,
        "entry": {
            "artifact_id": artifact_id,
            "campaign_batch_id": "demo-batch-1",
            "company_name": company_name,
            "payload": {"channel": channel, "target_mode": target_mode},
            "best": {"vendor_name": company_name, "opportunity_score": 91},
            "review_ids": ["demo-review-1"],
            "vendor_ctx": {"signal_summary": {"pain_distribution": [], "competitor_distribution": []}},
            "target": {"contact_email": "owner@example.com"},
            "followup_payload": {"channel": "email_followup", "target_mode": target_mode},
            "sequence_context": {"recipient_name": "Pat Demo"},
        },
    }


async def _ensure_task(pool) -> str:
    row = await pool.fetchrow(
        "SELECT id FROM scheduled_tasks WHERE name = $1 LIMIT 1",
        TASK_NAME,
    )
    if row:
        return str(row["id"])

    task_id = str(uuid4())
    now = datetime.now(timezone.utc)
    await pool.execute(
        """
        INSERT INTO scheduled_tasks (
            id, name, description, task_type, prompt, agent_type,
            schedule_type, cron_expression, interval_seconds, run_at,
            timezone, enabled, max_retries, retry_delay_seconds,
            timeout_seconds, metadata, created_at, updated_at
        ) VALUES (
            $1::uuid, $2, $3, 'builtin', NULL, 'atlas',
            'interval', NULL, 300, NULL,
            'America/Chicago', TRUE, 0, 60,
            900, $4::jsonb, $5, $5
        )
        """,
        task_id,
        TASK_NAME,
        TASK_DESCRIPTION,
        _json({"seeded_demo_task": True}),
        now,
    )
    return task_id


async def _delete_run(pool, run_id: str) -> None:
    await pool.execute("DELETE FROM pipeline_visibility_events WHERE run_id = $1", run_id)
    await pool.execute("DELETE FROM artifact_attempts WHERE run_id = $1", run_id)
    await pool.execute("DELETE FROM llm_usage WHERE run_id = $1", run_id)
    await pool.execute("DELETE FROM anthropic_message_batches WHERE run_id = $1", run_id)
    try:
        await pool.execute("DELETE FROM task_executions WHERE id = $1::uuid", run_id)
    except Exception:
        # If the caller used a non-UUID-like run id in the future, ignore here.
        pass


def _demo_items(now: datetime, *, stale_minutes: int) -> tuple[list[DemoItem], list[DemoItem]]:
    applied_success = now - timedelta(minutes=7)
    applied_failure = now - timedelta(minutes=6)
    stale_apply = now - timedelta(minutes=stale_minutes + 11)

    ended_items = [
        DemoItem(
            custom_id="campaign:salesforce:email",
            artifact_id="demo:salesforce:email",
            vendor_name="Salesforce",
            status="batch_succeeded",
            cache_prefiltered=False,
            fallback_single_call=False,
            input_tokens=1480,
            billable_input_tokens=620,
            cached_tokens=700,
            cache_write_tokens=160,
            output_tokens=288,
            cost_usd=Decimal("0.071200"),
            provider_request_id="msg_demo_salesforce",
            error_text=None,
            response_text="Subject: Renewal risk signal for Salesforce\n\nBody: Synthetic demo copy.",
            request_metadata={
                "channel": "email",
                "target_mode": "vendor_retention",
                "tier": "report",
                "replay_handler": "campaign_generation",
                "replay_entry": _replay_contract(
                    artifact_id="demo:salesforce:email",
                    company_name="Acme Manufacturing",
                    target_mode="vendor_retention",
                ),
                "applied_at": applied_success.isoformat(),
                "applied_status": "succeeded",
            },
            completed_at=now - timedelta(minutes=8),
        ),
        DemoItem(
            custom_id="campaign:hubspot:email",
            artifact_id="demo:hubspot:email",
            vendor_name="HubSpot",
            status="cache_hit",
            cache_prefiltered=True,
            fallback_single_call=False,
            input_tokens=1110,
            billable_input_tokens=0,
            cached_tokens=1110,
            cache_write_tokens=0,
            output_tokens=0,
            cost_usd=Decimal("0.000000"),
            provider_request_id=None,
            error_text=None,
            response_text=None,
            request_metadata={
                "channel": "email",
                "target_mode": "vendor_retention",
                "tier": "report",
                "replay_handler": "campaign_generation",
                "replay_entry": _replay_contract(
                    artifact_id="demo:hubspot:email",
                    company_name="Northwind Labs",
                    target_mode="vendor_retention",
                ),
                "applied_at": applied_success.isoformat(),
                "applied_status": "succeeded",
            },
            completed_at=now - timedelta(minutes=8),
        ),
        DemoItem(
            custom_id="campaign:shopify:email",
            artifact_id="demo:shopify:email",
            vendor_name="Shopify",
            status="fallback_succeeded",
            cache_prefiltered=False,
            fallback_single_call=True,
            input_tokens=920,
            billable_input_tokens=920,
            cached_tokens=0,
            cache_write_tokens=0,
            output_tokens=214,
            cost_usd=Decimal("0.053100"),
            provider_request_id="msg_demo_shopify_fallback",
            error_text=None,
            response_text="Subject: Migration timing signal for Shopify\n\nBody: Synthetic fallback demo copy.",
            request_metadata={
                "channel": "email",
                "target_mode": "vendor_retention",
                "tier": "report",
                "replay_handler": "campaign_generation",
                "replay_entry": _replay_contract(
                    artifact_id="demo:shopify:email",
                    company_name="Blue Harbor Retail",
                    target_mode="vendor_retention",
                ),
                "applied_at": applied_success.isoformat(),
                "applied_status": "succeeded",
            },
            completed_at=now - timedelta(minutes=7),
        ),
        DemoItem(
            custom_id="campaign:clickup:email",
            artifact_id="demo:clickup:email",
            vendor_name="ClickUp",
            status="batch_errored",
            cache_prefiltered=False,
            fallback_single_call=False,
            input_tokens=960,
            billable_input_tokens=960,
            cached_tokens=0,
            cache_write_tokens=0,
            output_tokens=0,
            cost_usd=Decimal("0.031000"),
            provider_request_id=None,
            error_text="Synthetic provider-side validation error",
            response_text=None,
            request_metadata={
                "channel": "email",
                "target_mode": "vendor_retention",
                "tier": "report",
                "replay_handler": "campaign_generation",
                "replay_entry": _replay_contract(
                    artifact_id="demo:clickup:email",
                    company_name="Contour Ops",
                    target_mode="vendor_retention",
                ),
                "applied_at": applied_failure.isoformat(),
                "applied_status": "failed",
                "applied_error": "batch_errored",
            },
            completed_at=now - timedelta(minutes=7),
        ),
    ]

    stale_items = [
        DemoItem(
            custom_id="campaign:azure:email",
            artifact_id="demo:azure:email",
            vendor_name="Azure",
            status="batch_succeeded",
            cache_prefiltered=False,
            fallback_single_call=False,
            input_tokens=1360,
            billable_input_tokens=820,
            cached_tokens=420,
            cache_write_tokens=120,
            output_tokens=240,
            cost_usd=Decimal("0.064500"),
            provider_request_id="msg_demo_azure",
            error_text=None,
            response_text="Subject: Azure support pressure signal\n\nBody: Synthetic stale-claim demo copy.",
            request_metadata={
                "channel": "email",
                "target_mode": "vendor_retention",
                "tier": "report",
                "replay_handler": "campaign_generation",
                "replay_entry": _replay_contract(
                    artifact_id="demo:azure:email",
                    company_name="Cobalt Security",
                    target_mode="vendor_retention",
                ),
                "applying_at": stale_apply.isoformat(),
                "applying_by": "reconcile:demo-seed:staleclaim",
            },
            completed_at=now - timedelta(minutes=stale_minutes + 12),
        ),
    ]
    return ended_items, stale_items


async def seed_demo(run_id: str, *, include_stale_health: bool) -> dict[str, str]:
    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database not initialized")

    task_id = await _ensure_task(pool)
    await _delete_run(pool, run_id)

    now = datetime.now(timezone.utc)
    stale_minutes = 30
    execution_started_at = now - timedelta(minutes=12)
    execution_completed_at = now - timedelta(minutes=4)

    await pool.execute(
        """
        INSERT INTO task_executions (
            id, task_id, status, started_at, completed_at, duration_ms,
            result_text, error, retry_count, metadata
        ) VALUES (
            $1::uuid, $2::uuid, 'completed', $3, $4, $5,
            $6, NULL, 0, $7::jsonb
        )
        """,
        run_id,
        task_id,
        execution_started_at,
        execution_completed_at,
        int((execution_completed_at - execution_started_at).total_seconds() * 1000),
        _json(
            {
                "mode": "synthetic_demo",
                "submitted_items": 4,
                "completed_items": 3,
                "failed_items": 1,
                "cache_prefiltered_items": 1,
                "fallback_single_call_items": 1,
            }
        ),
        _json({"seeded_demo": True, "surface": "operations_costs"}),
    )

    ended_batch_id = str(uuid4())
    stale_batch_id = str(uuid4())
    ended_items, stale_items = _demo_items(now, stale_minutes=stale_minutes)

    await pool.execute(
        """
        INSERT INTO anthropic_message_batches (
            id, stage_id, task_name, run_id, provider_batch_id, status,
            total_items, submitted_items, cache_prefiltered_items,
            fallback_single_call_items, completed_items, failed_items,
            estimated_sequential_cost_usd, estimated_batch_cost_usd,
            provider_error, metadata, created_at, updated_at, submitted_at, completed_at
        ) VALUES (
            $1::uuid, $2, $3, $4, $5, 'ended',
            4, 3, 1, 1, 3, 1,
            0.2123, 0.1243,
            NULL, $6::jsonb, $7, $8, $9, $10
        )
        """,
        ended_batch_id,
        STAGE_ID,
        TASK_NAME,
        run_id,
        f"msgbatch_demo_{run_id[:8]}",
        _json({"seeded_demo": True, "batch_kind": "ended"}),
        now - timedelta(minutes=10),
        now - timedelta(minutes=4),
        now - timedelta(minutes=9),
        now - timedelta(minutes=6),
    )

    if include_stale_health:
        await pool.execute(
            """
            INSERT INTO anthropic_message_batches (
                id, stage_id, task_name, run_id, provider_batch_id, status,
                total_items, submitted_items, cache_prefiltered_items,
                fallback_single_call_items, completed_items, failed_items,
                estimated_sequential_cost_usd, estimated_batch_cost_usd,
                provider_error, metadata, created_at, updated_at, submitted_at, completed_at
            ) VALUES (
                $1::uuid, $2, $3, $4, $5, 'in_progress',
                1, 1, 0, 0, 0, 0,
                0.0645, 0.0645,
                NULL, $6::jsonb, $7, $8, $9, NULL
            )
            """,
            stale_batch_id,
            STAGE_ID,
            TASK_NAME,
            run_id,
            f"msgbatch_demo_stale_{run_id[:8]}",
            _json({"seeded_demo": True, "batch_kind": "stale_claim"}),
            now - timedelta(minutes=stale_minutes + 18),
            now - timedelta(minutes=stale_minutes + 18),
            now - timedelta(minutes=stale_minutes + 16),
        )

    async def _insert_item(batch_id: str, item: DemoItem, *, created_at: datetime) -> None:
        await pool.execute(
            """
            INSERT INTO anthropic_message_batch_items (
                id, batch_id, custom_id, stage_id, artifact_type, artifact_id,
                vendor_name, status, cache_prefiltered, fallback_single_call,
                response_text, input_tokens, billable_input_tokens, cached_tokens,
                cache_write_tokens, output_tokens, cost_usd, provider_request_id,
                error_text, request_metadata, created_at, completed_at
            ) VALUES (
                gen_random_uuid(), $1::uuid, $2, $3, 'campaign', $4,
                $5, $6, $7, $8,
                $9, $10, $11, $12,
                $13, $14, $15, $16,
                $17, $18::jsonb, $19, $20
            )
            """,
            batch_id,
            item.custom_id,
            STAGE_ID,
            item.artifact_id,
            item.vendor_name,
            item.status,
            item.cache_prefiltered,
            item.fallback_single_call,
            item.response_text,
            item.input_tokens,
            item.billable_input_tokens,
            item.cached_tokens,
            item.cache_write_tokens,
            item.output_tokens,
            item.cost_usd,
            item.provider_request_id,
            item.error_text,
            _json(item.request_metadata),
            created_at,
            item.completed_at,
        )

    for index, item in enumerate(ended_items):
        await _insert_item(
            ended_batch_id,
            item,
            created_at=now - timedelta(minutes=9, seconds=30 - (index * 10)),
        )

    if include_stale_health:
        for item in stale_items:
            await _insert_item(
                stale_batch_id,
                item,
                created_at=now - timedelta(minutes=stale_minutes + 17),
            )

    return {
        "run_id": run_id,
        "ended_batch_id": ended_batch_id,
        "stale_batch_id": stale_batch_id if include_stale_health else "",
    }


def _parse_run_id(value: str | None) -> str:
    if not value:
        return str(uuid4())
    return str(UUID(str(value)))


async def _main_async(args: argparse.Namespace) -> int:
    await init_database()
    try:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise RuntimeError("Database not ready")

        if args.command == "cleanup":
            run_id = _parse_run_id(args.run_id)
            await _delete_run(pool, run_id)
            print(f"Cleaned synthetic detached campaign demo for run_id={run_id}")
            return 0

        run_id = _parse_run_id(args.run_id)
        result = await seed_demo(run_id, include_stale_health=bool(args.include_stale_health))
        print("Seeded synthetic detached campaign batch demo")
        print(f"run_id={result['run_id']}")
        print(f"ended_batch_id={result['ended_batch_id']}")
        if result["stale_batch_id"]:
            print(f"stale_batch_id={result['stale_batch_id']}")
        print("Paste the run_id into Operations -> Costs -> Run Detail")
        return 0
    finally:
        await close_database()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="Seed a synthetic detached campaign batch run")
    seed.add_argument("--run-id", help="Optional UUID to use as the synthetic run_id / task_execution id")
    seed.add_argument(
        "--include-stale-health",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also seed a stale in-progress batch and stale claimed item for queue/cost visibility",
    )

    cleanup = sub.add_parser("cleanup", help="Delete a previously seeded synthetic run")
    cleanup.add_argument("--run-id", required=True, help="UUID run_id returned by the seed command")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
