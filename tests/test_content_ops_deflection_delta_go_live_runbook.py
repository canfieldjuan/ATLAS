from __future__ import annotations

import json
import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "docs"
    / "extraction"
    / "validation"
    / "content_ops_deflection_delta_go_live_runbook.md"
)
MIGRATION = ROOT / "atlas_brain" / "storage" / "migrations" / "341_content_ops_deflection_delta_deliveries.sql"


def _section(title: str) -> str:
    doc = RUNBOOK.read_text(encoding="utf-8")
    start = doc.index(f"## {title}")
    next_heading = doc.find("\n## ", start + 1)
    return doc[start:] if next_heading == -1 else doc[start:next_heading]


def _curl_body(section_title: str) -> dict[str, object]:
    section = _section(section_title)
    match = re.search(r"-d '([^']+)'", section)
    assert match is not None
    return json.loads(match.group(1))


def test_deflection_delta_go_live_runbook_pins_migration_and_paid_pair() -> None:
    doc = RUNBOOK.read_text(encoding="utf-8")
    paid_pair = _section("Paid Pair Check")

    assert "WHERE name = '341_content_ops_deflection_delta_deliveries';" in doc
    assert "WHERE name = '341_content_ops_deflection_delta_deliveries.sql';" not in doc
    assert "schema_migrations" in doc
    assert "content_ops_deflection_reports" in doc
    assert "paid IS TRUE" in doc
    assert "current_request_id" in paid_pair
    assert "current_delivery_email" in paid_pair
    assert "baseline_report_count" in paid_pair
    assert "baseline_delivery_email_sample" in paid_pair
    assert "has_reserved_test_email" in paid_pair
    assert "report_rank = 1" in paid_pair
    assert "baseline.created_at < current_reports.created_at" in paid_pair
    assert "@example.com" in paid_pair
    assert "@test" in paid_pair
    assert "operator/test address" in paid_pair
    assert MIGRATION.exists()


def test_deflection_delta_go_live_runbook_rehearses_before_live_send() -> None:
    dry_run = _section("Dry-Run Activation")
    live = _section("Live Activation")
    rollback = _section("Rollback")

    assert "ATLAS_DEFLECTION_DELTA_ENABLED=true" in dry_run
    assert "ATLAS_DEFLECTION_DELIVERY_DRY_RUN=true" not in dry_run
    assert "shared with the already-live" in dry_run
    assert "ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL" in dry_run
    assert "ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false" not in dry_run
    assert "delivery_dry_run" in dry_run
    assert "delivery_dry_run_enabled" in dry_run
    assert "current_request_id" in dry_run
    assert "`reports_scanned` is 1" in dry_run
    assert "delivery_missing_config" in dry_run
    assert "delta_deliveries_enqueued" in dry_run
    assert "`delivery_dry_run` is 1" in dry_run
    assert "delivery_failed" in dry_run
    assert "ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false" in live
    assert "ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY" in live
    assert "target_account_id" in live
    assert "current_request_id" in live
    assert "Omitting" in live
    assert "`target_account_id` scans" in live
    assert "paid accounts globally" in live
    assert "omitting `current_request_id` scopes to the account" in live
    assert "/run" in live
    assert "ATLAS_DEFLECTION_DELIVERY_DRY_RUN=true" not in rollback
    assert "paid report delivery drain" in rollback


def test_deflection_delta_go_live_runbook_manual_run_uses_safe_override() -> None:
    body = _curl_body("Dry-Run Activation")

    assert body == {
        "delivery_dry_run": True,
        "target_account_id": "<account-id>",
        "current_request_id": "<current-request-id>",
    }


def test_deflection_delta_go_live_runbook_autonomous_command_targets_delta_task() -> None:
    dry_run = _section("Dry-Run Activation")
    command_start = dry_run.index("curl -fsS -X POST")
    command_end = dry_run.index("```", command_start)
    command = dry_run[command_start:command_end].replace("\\\n", " ")
    parts = shlex.split(command)

    assert parts[:4] == ["curl", "-fsS", "-X", "POST"]
    assert any(
        part.endswith("/api/v1/autonomous/content_ops_deflection_delta_automation/run")
        for part in parts
    )


def test_deflection_delta_go_live_runbook_exposes_task_id_for_polling() -> None:
    dry_run = _section("Dry-Run Activation")

    assert "{id, enabled, cron_expression, next_run_at, metadata}" in dry_run
    assert 'DELTA_TASK_ID="$(' in dry_run
    assert "$ATLAS_API_BASE_URL/api/v1/autonomous/$DELTA_TASK_ID/executions?limit=1" in dry_run
