-- Queue post-purchase delivery work for paid FAQ deflection reports.

CREATE TABLE IF NOT EXISTS content_ops_deflection_report_deliveries (
    account_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    payment_reference TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    delivery_error TEXT,
    provider_message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    PRIMARY KEY (account_id, request_id)
);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_deliveries_pending
    ON content_ops_deflection_report_deliveries (created_at)
    WHERE delivery_status = 'pending';
