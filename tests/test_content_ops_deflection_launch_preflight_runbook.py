from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "docs"
    / "extraction"
    / "validation"
    / "content_ops_deflection_launch_preflight_runbook.md"
)
PAID_EMAIL_MIGRATION = (
    ROOT
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "331_content_ops_deflection_report_delivery_email.sql"
)
PAID_QUEUE_MIGRATION = (
    ROOT
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "332_content_ops_deflection_report_deliveries.sql"
)
CHECKOUT_AUTHORIZATION_MIGRATION = (
    ROOT
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "342_content_ops_deflection_checkout_authorization.sql"
)


def _doc() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _section(title: str) -> str:
    doc = _doc()
    start = doc.index(f"## {title}")
    next_heading = doc.find("\n## ", start + 1)
    return doc[start:] if next_heading == -1 else doc[start:next_heading]


def test_launch_runbook_names_required_surfaces_and_trackers() -> None:
    doc = _doc()

    assert "#1921" in doc
    assert "#1440" in doc
    assert "#1386" in doc
    assert "Snapshot email" in doc
    assert "Snapshot PDF" in doc
    assert "paid report email" in doc
    assert "paid report PDF attachment" in doc
    assert "emailed hosted result URL" in doc
    assert "not complete until both buyer-facing email surfaces" in doc


def test_launch_runbook_pins_deployed_config_and_scheduler_gates() -> None:
    section = _section("Deployed Config Check")

    for required in (
        "ATLAS_API_BASE_URL",
        "ATLAS_B2B_SERVICE_TOKEN",
        "GAP_REPORT_NOTIFICATION_RESEND_API_KEY",
        "GAP_REPORT_NOTIFICATION_FROM_EMAIL",
        "ATLAS_DEFLECTION_DELIVERY_ENABLED=false",
        "ATLAS_DEFLECTION_DELIVERY_DRY_RUN=true",
        "ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL",
        "ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY",
        "ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL",
        "ATLAS_DEFLECTION_DELIVERY_RESULT_URL_TEMPLATE",
        "content_ops_deflection_report_delivery",
        "include_disabled=true",
        "{id, enabled, next_run_at, metadata}",
    ):
        assert required in section
    assert "sendSnapshotEmail" in section
    assert "atlas-portfolio" in section
    assert "legacy in-repo `portfolio-ui` SPA route" in section
    assert "/services/faq-deflection/results/{request_id}" in section
    assert "/systems/support-ticket-deflection/results/{request_id}" in section
    assert "will not auto-send before the rehearsal" in section
    assert "bypass the dry-run proof" in section


def test_launch_runbook_requires_paid_delivery_schema() -> None:
    section = _section("Database Gate")

    assert "schema_migrations" in section
    assert "331_content_ops_deflection_report_delivery_email" in section
    assert "332_content_ops_deflection_report_deliveries" in section
    assert "342_content_ops_deflection_checkout_authorization" in section
    assert "All three rows must be present" in section
    assert "checkout authorization columns" in section
    assert PAID_EMAIL_MIGRATION.exists()
    assert PAID_QUEUE_MIGRATION.exists()
    assert CHECKOUT_AUTHORIZATION_MIGRATION.exists()


def test_snapshot_email_proof_requires_attachment_and_blocks_skip_log() -> None:
    section = _section("Snapshot Email And PDF Proof")

    assert "deployed portfolio intake" in section
    assert "fetch the free Snapshot" in section
    assert "Body says the free Snapshot is ready" in section
    assert "/systems/support-ticket-deflection/results/<request-id>" in section
    assert "A Snapshot PDF is attached" in section
    assert "deflection.record.snapshot_pdf_attachment_skipped" in section
    assert "email without the Snapshot PDF is not launch proof" in section
    assert "excludes source IDs" in section
    assert "raw ticket bodies" in section
    assert "paid report markdown" in section


def test_paid_unlock_and_delivery_proof_require_real_queue_and_live_send() -> None:
    unlock = _section("Paid Unlock Gate")
    delivery = _section("Paid Report Email And PDF Proof")

    assert "real Stripe Checkout" in unlock
    assert "Do not replay a synthetic webhook" in unlock
    assert "Checkout price authorization" in unlock
    assert "content_ops_deflection_reports" in unlock
    assert "content_ops_deflection_report_deliveries" in unlock
    assert "r.request_id = '<request-id>'" in unlock
    assert "`paid` is true" in unlock
    assert "`delivery_status` is `pending`" in unlock
    assert "queue-wide" in delivery
    assert "does not accept a request/account filter" in delivery
    assert "claimable_rows" in delivery
    assert "target_rows" in delivery
    assert "Proceed only if `claimable_rows` is 1 and `target_rows` is 1" in delivery
    assert "scripts/send_content_ops_deflection_report_deliveries.py" in delivery
    assert "--json" in delivery
    assert "--send" in delivery
    assert "--resend-api-key" in delivery
    assert "dry-run JSON must show at least one scanned row" in delivery
    assert "queue selection" in delivery
    assert "paid/email gating" in delivery
    assert "does not render the PDF" in delivery
    assert "build the email body" in delivery
    assert "render_deflection_full_report_pdf" in delivery
    assert "paid-artifact.json" in delivery
    assert "paid-report.pdf" in delivery
    assert "first exercise" in delivery
    assert "paid PDF rendering" in delivery
    assert "live buyer send" in delivery
    assert "ATLAS_DEFLECTION_DELIVERY_ENABLED=true" in delivery
    assert "ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false" in delivery
    assert "deploy or restart ATLAS" in delivery
    assert "hosted scheduler configured for" in delivery
    assert "live paid delivery" in delivery
    assert "enabled config value is live" in delivery
    assert "manual one-off email is not enough" in delivery
    assert "live JSON has `sent` 1 and `failed` 0" in delivery
    assert "link-only paid email is not launch proof" in delivery


def test_paid_pdf_shape_is_curated_with_toc_and_export_pointer() -> None:
    section = _section("Paid PDF Shape Check")

    assert "curated/shareable report" in section
    assert "not the full evidence archive" in section
    assert "Table of contents" in section
    assert "not a 600+ page raw evidence dump" in section
    assert re.search(r"capped at 25 rows", section)
    assert re.search(r"capped at 10 questions", section)
    assert "complete evidence export" in section
    assert "hosted paid result page" in section


def test_hosted_url_cleanup_and_closeout_are_required() -> None:
    section = _section("Hosted URL, Cleanup, And Tracker Closeout")

    assert "exact URL from each email" in section
    assert "locked/free Snapshot state" in section
    assert "unlocked paid report" in section
    assert "must not fall back to demo data" in section
    assert "CRON_SECRET" in section
    assert "Privacy, Security, Terms" in section
    assert "refund" in section
    assert "#1921, #1440, and #1386" in section
