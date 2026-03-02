-- Migration 079: B2B product discriminator + tracked vendors
-- Extends saas_accounts for P5 (Vendor Retention) and P6 (Challenger Lead Gen)

-- Product discriminator on saas_accounts
ALTER TABLE saas_accounts
  ADD COLUMN IF NOT EXISTS product VARCHAR(32) NOT NULL DEFAULT 'consumer';
-- Values: consumer (P1) | b2b_retention (P5) | b2b_challenger (P6)

ALTER TABLE saas_accounts
  ADD COLUMN IF NOT EXISTS vendor_limit INT NOT NULL DEFAULT 1;

-- Per-account vendor tracking (parallel to tracked_asins)
CREATE TABLE IF NOT EXISTS tracked_vendors (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id  UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    vendor_name TEXT NOT NULL,
    track_mode  VARCHAR(32) NOT NULL DEFAULT 'own',  -- own (P5) | competitor (P6)
    label       VARCHAR(256),
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, vendor_name)
);
CREATE INDEX IF NOT EXISTS idx_tracked_vendors_account ON tracked_vendors(account_id);
