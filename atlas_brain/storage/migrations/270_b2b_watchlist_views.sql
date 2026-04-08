-- Migration 270: Tenant-scoped watchlist saved views and thresholds

CREATE TABLE IF NOT EXISTS b2b_watchlist_views (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    vendor_name TEXT,
    category TEXT,
    source TEXT,
    min_urgency NUMERIC(4, 2),
    include_stale BOOLEAN NOT NULL DEFAULT TRUE,
    named_accounts_only BOOLEAN NOT NULL DEFAULT FALSE,
    changed_wedges_only BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_b2b_watchlist_views_min_urgency
        CHECK (min_urgency IS NULL OR (min_urgency >= 0 AND min_urgency <= 10))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_watchlist_views_account_name
    ON b2b_watchlist_views (account_id, LOWER(name));

CREATE INDEX IF NOT EXISTS idx_b2b_watchlist_views_account_updated
    ON b2b_watchlist_views (account_id, updated_at DESC, created_at DESC);
