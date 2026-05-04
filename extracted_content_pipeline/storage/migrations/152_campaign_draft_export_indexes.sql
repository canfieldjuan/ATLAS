-- Product-owned indexes for host campaign draft review/export filters.

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_scope_account_created
    ON b2b_campaigns ((metadata -> 'scope' ->> 'account_id'), created_at DESC)
    WHERE metadata ? 'scope';

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_company_created
    ON b2b_campaigns (company_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_vendor_created
    ON b2b_campaigns (vendor_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_status_created
    ON b2b_campaigns (status, created_at DESC);
