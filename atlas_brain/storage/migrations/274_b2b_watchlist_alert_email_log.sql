-- Migration 274: Persist saved-view watchlist alert email deliveries.

CREATE TABLE IF NOT EXISTS b2b_watchlist_alert_email_log (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    watchlist_view_id UUID NOT NULL REFERENCES b2b_watchlist_views(id) ON DELETE CASCADE,
    event_ids UUID[] NOT NULL DEFAULT '{}',
    recipient_emails TEXT[] NOT NULL DEFAULT '{}',
    message_ids TEXT[] NOT NULL DEFAULT '{}',
    event_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    error TEXT,
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_b2b_watchlist_alert_email_log_status
        CHECK (status IN ('sent', 'partial', 'failed', 'no_events'))
);

CREATE INDEX IF NOT EXISTS idx_b2b_watchlist_alert_email_log_view_created
    ON b2b_watchlist_alert_email_log (watchlist_view_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_watchlist_alert_email_log_account_created
    ON b2b_watchlist_alert_email_log (account_id, created_at DESC);
