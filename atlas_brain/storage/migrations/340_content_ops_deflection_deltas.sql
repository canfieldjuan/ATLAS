-- Persist paid deflection report deltas by tenant/current/baseline report pair.

CREATE TABLE IF NOT EXISTS content_ops_deflection_deltas (
    account_id TEXT NOT NULL,
    current_request_id TEXT NOT NULL,
    baseline_request_id TEXT NOT NULL,
    delta JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, current_request_id, baseline_request_id),
    CHECK (current_request_id <> baseline_request_id),
    FOREIGN KEY (account_id, current_request_id)
        REFERENCES content_ops_deflection_reports (account_id, request_id)
        ON DELETE CASCADE,
    FOREIGN KEY (account_id, baseline_request_id)
        REFERENCES content_ops_deflection_reports (account_id, request_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_deltas_account_created
    ON content_ops_deflection_deltas (account_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_deltas_current
    ON content_ops_deflection_deltas (account_id, current_request_id);
