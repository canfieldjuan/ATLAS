-- Per-account ASIN tracking for multi-tenant data scoping
CREATE TABLE IF NOT EXISTS tracked_asins (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id  UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    asin        TEXT NOT NULL,
    label       VARCHAR(256),
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, asin)
);

CREATE INDEX IF NOT EXISTS idx_tracked_asins_account ON tracked_asins(account_id);
CREATE INDEX IF NOT EXISTS idx_tracked_asins_asin ON tracked_asins(asin);
