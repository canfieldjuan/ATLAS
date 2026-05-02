-- Product-owned customer opportunity source for standalone campaign generation.

CREATE TABLE IF NOT EXISTS campaign_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT,
    target_id TEXT NOT NULL,
    target_mode TEXT DEFAULT 'vendor_retention',
    company_name TEXT,
    vendor_name TEXT,
    contact_name TEXT,
    contact_email TEXT,
    contact_title TEXT,
    opportunity_score NUMERIC,
    urgency_score NUMERIC,
    pain_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    competitors JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaign_opportunities_account_mode
    ON campaign_opportunities (account_id, target_mode, status);

CREATE INDEX IF NOT EXISTS idx_campaign_opportunities_target_id
    ON campaign_opportunities (target_id);

CREATE INDEX IF NOT EXISTS idx_campaign_opportunities_vendor
    ON campaign_opportunities (LOWER(vendor_name));

CREATE INDEX IF NOT EXISTS idx_campaign_opportunities_contact_email
    ON campaign_opportunities (LOWER(contact_email));

CREATE INDEX IF NOT EXISTS idx_campaign_opportunities_raw_payload
    ON campaign_opportunities USING GIN(raw_payload);
