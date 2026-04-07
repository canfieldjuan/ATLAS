ALTER TABLE b2b_report_subscription_delivery_log
    DROP CONSTRAINT IF EXISTS b2b_report_subscription_delivery_log_status_check;

ALTER TABLE b2b_report_subscription_delivery_log
    ADD CONSTRAINT b2b_report_subscription_delivery_log_status_check
    CHECK (status IN ('processing', 'sent', 'partial', 'skipped', 'failed', 'dry_run'));
