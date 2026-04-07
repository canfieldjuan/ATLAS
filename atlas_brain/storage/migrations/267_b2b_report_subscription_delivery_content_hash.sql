ALTER TABLE b2b_report_subscription_delivery_log
    ADD COLUMN IF NOT EXISTS content_hash TEXT;
