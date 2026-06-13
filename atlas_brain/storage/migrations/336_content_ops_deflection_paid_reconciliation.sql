-- Paid-but-report-missing reconciliation ledger (#1462).
--
-- When a Stripe deflection checkout completes (paid) but no report row exists
-- and the event has aged past the write-ordering race window, the webhook
-- returns 2xx -- to stop Stripe retrying a non-2xx for hours -- and records the
-- paid-but-undelivered case here for manual reconciliation, instead of letting
-- the buyer's payment retry into the void. account_id is TEXT to match
-- content_ops_deflection_reports (migration 328).

CREATE TABLE IF NOT EXISTS content_ops_deflection_paid_reconciliation (
    id                BIGSERIAL PRIMARY KEY,
    account_id        TEXT NOT NULL,
    request_id        TEXT NOT NULL,
    stripe_session_id TEXT,
    event_type        TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT 'paid_report_missing',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, request_id, stripe_session_id)
);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_paid_reconciliation_created
    ON content_ops_deflection_paid_reconciliation (created_at DESC);
