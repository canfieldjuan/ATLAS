-- Store paid-gated FAQ deflection report artifacts by tenant/request.

CREATE TABLE IF NOT EXISTS content_ops_deflection_reports (
    account_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    snapshot JSONB NOT NULL,
    artifact JSONB NOT NULL,
    paid BOOLEAN NOT NULL DEFAULT FALSE,
    payment_reference TEXT,
    paid_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, request_id)
);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_reports_account_created
    ON content_ops_deflection_reports (account_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_reports_paid
    ON content_ops_deflection_reports (account_id, paid)
    WHERE paid = true;
