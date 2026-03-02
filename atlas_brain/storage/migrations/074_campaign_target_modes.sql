-- Migration 074: Campaign target modes + vendor/challenger targets
--
-- Adds target_mode branching so the B2B pipeline can serve multiple products:
--   vendor_retention: sell churn intelligence to the vendor losing customers
--   challenger_intel: sell intent leads to the challenger gaining customers
--   churning_company: legacy mode (outreach to the company itself)

-- Add target_mode to campaigns
ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS target_mode VARCHAR(30) DEFAULT 'churning_company';

-- Vendor/challenger targets -- our actual customers (the people we sell intelligence TO)
CREATE TABLE IF NOT EXISTS vendor_targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL,
    target_mode VARCHAR(30) NOT NULL CHECK (target_mode IN ('vendor_retention', 'challenger_intel')),
    contact_name VARCHAR(255),
    contact_email VARCHAR(255),
    contact_role VARCHAR(100),
    products_tracked TEXT[],          -- which of their products we monitor
    competitors_tracked TEXT[],       -- which competitors they care about
    tier VARCHAR(20) DEFAULT 'report', -- report, dashboard, api
    status VARCHAR(20) DEFAULT 'active',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vendor_targets_mode ON vendor_targets(target_mode);
CREATE INDEX IF NOT EXISTS idx_vendor_targets_status ON vendor_targets(status);
CREATE INDEX IF NOT EXISTS idx_vendor_targets_company ON vendor_targets(LOWER(company_name));
