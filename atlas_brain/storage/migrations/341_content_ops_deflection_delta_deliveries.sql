-- Queue monthly delivery work for paid FAQ deflection report deltas.

CREATE TABLE IF NOT EXISTS content_ops_deflection_delta_deliveries (
    account_id TEXT NOT NULL,
    current_request_id TEXT NOT NULL,
    baseline_request_id TEXT NOT NULL,
    delivery_email TEXT NOT NULL,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    delivery_error TEXT,
    provider_message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    PRIMARY KEY (account_id, current_request_id, baseline_request_id),
    CHECK (current_request_id <> baseline_request_id),
    FOREIGN KEY (account_id, current_request_id, baseline_request_id)
        REFERENCES content_ops_deflection_deltas (
            account_id,
            current_request_id,
            baseline_request_id
        )
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_delta_deliveries_pending
    ON content_ops_deflection_delta_deliveries (created_at)
    WHERE delivery_status = 'pending';
