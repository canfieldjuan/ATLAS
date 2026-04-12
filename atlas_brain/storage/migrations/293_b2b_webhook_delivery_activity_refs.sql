-- Persist canonical activity references alongside webhook delivery attempts
ALTER TABLE b2b_webhook_delivery_log
    ADD COLUMN IF NOT EXISTS signal_id UUID,
    ADD COLUMN IF NOT EXISTS review_id UUID,
    ADD COLUMN IF NOT EXISTS report_id UUID,
    ADD COLUMN IF NOT EXISTS vendor_name TEXT,
    ADD COLUMN IF NOT EXISTS company_name TEXT;

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_signal_id
    ON b2b_webhook_delivery_log (signal_id)
    WHERE signal_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_review_id
    ON b2b_webhook_delivery_log (review_id)
    WHERE review_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_report_id
    ON b2b_webhook_delivery_log (report_id)
    WHERE report_id IS NOT NULL;
