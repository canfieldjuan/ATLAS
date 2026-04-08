-- Migration 273: Persist watchlist alert events for saved-view threshold tracking.

CREATE TABLE IF NOT EXISTS b2b_watchlist_alert_events (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    watchlist_view_id UUID NOT NULL REFERENCES b2b_watchlist_views(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    threshold_field TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    vendor_name TEXT,
    company_name TEXT,
    category TEXT,
    source TEXT,
    threshold_value NUMERIC(6, 2),
    summary TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'open',
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_b2b_watchlist_alert_events_event_type
        CHECK (event_type IN ('vendor_alert', 'account_alert', 'stale_data')),
    CONSTRAINT chk_b2b_watchlist_alert_events_threshold_field
        CHECK (threshold_field IN ('vendor_alert_threshold', 'account_alert_threshold', 'stale_days_threshold')),
    CONSTRAINT chk_b2b_watchlist_alert_events_entity_type
        CHECK (entity_type IN ('vendor', 'account', 'signal_cluster')),
    CONSTRAINT chk_b2b_watchlist_alert_events_status
        CHECK (status IN ('open', 'resolved'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_watchlist_alert_events_view_entity
    ON b2b_watchlist_alert_events (watchlist_view_id, event_type, entity_key);

CREATE INDEX IF NOT EXISTS idx_b2b_watchlist_alert_events_account_status
    ON b2b_watchlist_alert_events (account_id, status, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_watchlist_alert_events_view_status
    ON b2b_watchlist_alert_events (watchlist_view_id, status, last_seen_at DESC);
